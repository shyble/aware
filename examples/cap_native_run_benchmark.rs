// Cap-native benchmark runner.
//
// The cap input layer uses the same primitive as the cap-augmented
// transformer. What's distinctive to cap-native is downstream: per-cap
// weight stacks at every block, and the cap-keyed output projection.
//
// ENV (cap layer config mirrors the cap-augmented transformer):
// AWARE_BENCH_ID, AWARE_BENCH_CORPUS, AWARE_BENCH_VAL_CORPUS,
// AWARE_BENCH_OUTPUT_DIR, AWARE_BENCH_SEED,
// AWARE_BENCH_D_MODEL, AWARE_BENCH_N_BLOCKS, AWARE_BENCH_N_HEADS,
// AWARE_BENCH_D_FF, AWARE_BENCH_SEQ_LEN, AWARE_BENCH_BATCH_SIZE,
// AWARE_BENCH_STEPS, AWARE_BENCH_EVAL_EVERY, AWARE_BENCH_N_EVAL_BATCHES,
// AWARE_BENCH_VAL_RATIO, AWARE_BENCH_LR,
// AWARE_BENCH_CAP_DISCOVERY (kmeans | random | nodiscovery | hybrid)
// AWARE_BENCH_CAP_N_TARGET (default 330)
// AWARE_BENCH_CAP_WINDOW (default 4)
//
// ENV (cap-native-specific knobs):
// AWARE_CN_TOP_K top-K for cap-keyed routing (default 0 = full softmax)
// AWARE_CN_CAP_INDEXED_MASK toggle cap-overlap attention bias (default false)
// AWARE_CN_ROUTING soft | sparse (default soft)

use std::env;
use std::fs;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::Instant;

use aware::aware::cap_native::{CapLayer1Config, CapNativeConfig, CapNativeSubstrate, RoutingMode};
use aware::aware::config::CapConfig;
use aware::aware::discover::DiscoveryKind;
use aware::aware::train::LossKind;
use aware::aware::train::StreamingFeeder;
use aware::aware::OptimizerKind;
use aware::data::bpe::{ensure_tokenized, load_bpe, BPETokenizer};
use candle_core::{Device, Result, Tensor};
use candle_nn::Optimizer;

/// Sample a bootstrap set from a tokenized .bin file without loading the
/// whole corpus into memory. Returns `n_samples * window` u32 tokens (for
/// window > 1, each sample is a contiguous window; for window == 1, just
/// random tokens).
fn sample_bootstrap_from_file(
    path: &Path,
    n_samples: usize,
    window: usize,
    seed: u64,
) -> std::io::Result<Vec<u32>> {
    let mut file = File::open(path)?;
    let n_tokens = (file.metadata()?.len() as usize) / 4;
    let max_start = n_tokens.saturating_sub(window.max(1) + 1).max(1);
    let mut rng = if seed == 0 { 0xC0FFEE_BABE } else { seed };
    let mut out = Vec::with_capacity(n_samples * window.max(1));
    let chunk_bytes = window.max(1) * 4;
    let mut buf = vec![0u8; chunk_bytes];
    for _ in 0..n_samples {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let start = ((rng >> 33) as usize) % max_start;
        file.seek(SeekFrom::Start((start * 4) as u64))?;
        file.read_exact(&mut buf)?;
        for chunk in buf.chunks_exact(4) {
            out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
    }
    Ok(out)
}

fn env_str(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}
fn env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
fn env_u64(key: &str, default: u64) -> u64 {
    env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
fn env_f64(key: &str, default: f64) -> f64 {
    env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
fn env_bool(key: &str, default: bool) -> bool {
    env::var(key)
        .ok()
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "y"))
        .unwrap_or(default)
}

fn parse_discovery(s: &str) -> DiscoveryKind {
    match s {
        "kmeans" => DiscoveryKind::KMeans,
        "kmeans_pp" | "kmeanspp" | "kmeans++" => DiscoveryKind::KMeansPP,
        "random" => DiscoveryKind::Random,
        "hybrid" => DiscoveryKind::Hybrid,
        _ => DiscoveryKind::NoDiscovery,
    }
}
fn parse_routing(s: &str) -> RoutingMode {
    match s {
        "sparse" | "hard_top1" | "hard" => RoutingMode::HardTop1Sparse,
        _ => RoutingMode::SoftTopK,
    }
}

fn compute_val_loss(
    model: &CapNativeSubstrate,
    val_feeder: &mut StreamingFeeder,
    device: &Device,
    n_batches: usize,
) -> Result<f32> {
    let mut total = 0.0f32;
    let mut count = 0;
    for _ in 0..n_batches {
        let (inp, tgt) = val_feeder.next_batch(device)?;
        let logits = model.forward(&inp)?;
        let (b, t, v) = logits.dims3()?;
        let logits_flat = logits.reshape((b * t, v))?;
        let tgt_flat = tgt.reshape((b * t,))?;
        let loss = candle_nn::loss::cross_entropy(&logits_flat, &tgt_flat)?;
        total += loss.to_scalar::<f32>()?;
        count += 1;
    }
    Ok(total / count.max(1) as f32)
}

fn main() -> Result<()> {
    // ── Parse env config ──
    let run_id = env_str("AWARE_BENCH_ID", "cap_native_unnamed");
    let corpus_path = env_str(
        "AWARE_BENCH_CORPUS",
        "data/tinystories/tinystories_train.txt",
    );
    let val_corpus_path = env::var("AWARE_BENCH_VAL_CORPUS").ok();
    let output_dir = env_str("AWARE_BENCH_OUTPUT_DIR", "data/bench");
    let seed = env_u64("AWARE_BENCH_SEED", 42);

    let d_model = env_usize("AWARE_BENCH_D_MODEL", 128);
    let n_blocks = env_usize("AWARE_BENCH_N_BLOCKS", 4);
    let n_heads = env_usize("AWARE_BENCH_N_HEADS", 4);
    let d_ff = env_usize("AWARE_BENCH_D_FF", 512);
    let seq_len = env_usize("AWARE_BENCH_SEQ_LEN", 128);
    let batch_size = env_usize("AWARE_BENCH_BATCH_SIZE", 32);
    let steps = env_usize("AWARE_BENCH_STEPS", 5000);
    let eval_every = env_usize("AWARE_BENCH_EVAL_EVERY", 100);
    let n_eval_batches = env_usize("AWARE_BENCH_N_EVAL_BATCHES", 8);
    let val_ratio = env_f64("AWARE_BENCH_VAL_RATIO", 0.05).clamp(0.0, 0.5);
    let lr = env_f64("AWARE_BENCH_LR", 3e-4);

    // Cap layer config (mirrors the cap-augmented transformer).
    // Continual-learning plumbing. LOAD builds the model without
    // discovery (the loaded checkpoint supplies keys, ids and weights);
    // SAVE writes a full checkpoint at the end of the run.
    let load_weights = env::var("AWARE_BENCH_LOAD_WEIGHTS").ok();
    let save_weights = env::var("AWARE_BENCH_SAVE_WEIGHTS").ok();

    let discovery_str = env_str("AWARE_BENCH_CAP_DISCOVERY", "kmeans");
    let discovery = if load_weights.is_some() {
        // Whatever the preset says, discovery is pointless before a
        // load: the checkpoint overwrites keys, and running KMeans
        // first would waste time and imply an identity that is about
        // to be replaced.
        DiscoveryKind::NoDiscovery
    } else {
        parse_discovery(&discovery_str)
    };
    let n_caps_target = env_usize("AWARE_BENCH_CAP_N_TARGET", 330);
    let cap_window = env_usize("AWARE_BENCH_CAP_WINDOW", 4);

    // Cap-native-specific
    let top_k = env_usize("AWARE_CN_TOP_K", 0);
    let cap_indexed_mask = env_bool("AWARE_CN_CAP_INDEXED_MASK", false);
    let routing_str = env_str("AWARE_CN_ROUTING", "soft");
    let routing = parse_routing(&routing_str);

    // Hierarchical (Phase E): two stacked discovered CapLayers.
    let hierarchical = env_bool("AWARE_CN_HIERARCHICAL", false);
    let l1_n_caps = env_usize("AWARE_CN_L1_N_CAPS", 128);
    let l1_budget = env_usize("AWARE_CN_L1_BUDGET", l1_n_caps.max(1) * 2);
    let l1_window = env_usize("AWARE_CN_L1_WINDOW", 1);
    let l1_discovery_str = env_str("AWARE_CN_L1_DISCOVERY", "kmeans");
    let l1_discovery = if load_weights.is_some() {
        DiscoveryKind::NoDiscovery
    } else {
        parse_discovery(&l1_discovery_str)
    };
    let l1_gradient_train = env_bool("AWARE_CN_L1_GRADIENT_TRAIN", false);

    let run_dir = format!("{}/{}", output_dir, run_id);
    fs::create_dir_all(&run_dir).ok();

    println!("=== Cap-Native Benchmark Runner ===");
    println!(" run_id: {}", run_id);
    println!(" corpus: {}", corpus_path);
    if let Some(vp) = &val_corpus_path {
        println!(" val_corpus: {}", vp);
    } else {
        println!(
            " val_corpus: (slicing last {}%)",
            (val_ratio * 100.0) as usize
        );
    }
    println!(" seed: {}", seed);
    println!(" d_model: {}", d_model);
    println!(" n_blocks: {}", n_blocks);
    println!(" n_heads: {}", n_heads);
    println!(" d_ff: {}", d_ff);
    println!(" cap_discovery: {}", discovery_str);
    println!(" n_caps_target: {}", n_caps_target);
    println!(" cap_window: {}", cap_window);
    println!(
        " top_k (cap-keyed): {} ({})",
        top_k,
        if top_k == 0 {
            "soft full softmax"
        } else {
            "top-K softmax"
        }
    );
    println!(" cap_indexed_mask: {}", cap_indexed_mask);
    println!(" routing: {} ({:?})", routing_str, routing);
    let tokens_per_step = batch_size * seq_len;
    println!(" steps: {} (eval every {})", steps, eval_every);
    println!();

    // ── Load BPE (train from a brief corpus pass if missing) ──
    // A tokenizer trained on one corpus fragments another badly, and the
    // .bin cache is not keyed to the tokenizer - so the BPE directory must
    // be explicit when the corpus is not TinyStories.
    let bpe_dir = env_str("AWARE_BENCH_BPE_DIR", "data/brain_tinystories");
    let bpe = match load_bpe(&bpe_dir) {
        Ok(b) => b,
        Err(_) => {
            let sample = fs::read_to_string(&corpus_path).unwrap_or_else(|e| {
                eprintln!("ERROR: read {}: {}", corpus_path, e);
                std::process::exit(1)
            });
            let trained = BPETokenizer::train(&sample, 256);
            drop(sample);
            trained
        }
    };

    // ── Tokenize once and cache as .bin (disk-streamed for training) ──
    let read_t = Instant::now();
    let train_bin = ensure_tokenized(Path::new(&corpus_path), &bpe).unwrap_or_else(|e| {
        eprintln!("ERROR: tokenize train: {}", e);
        std::process::exit(1)
    });
    let val_bin = match val_corpus_path.as_ref() {
        Some(vp) => ensure_tokenized(Path::new(vp), &bpe).unwrap_or_else(|e| {
            eprintln!("ERROR: tokenize val: {}", e);
            std::process::exit(1)
        }),
        None => {
            eprintln!(
                "ERROR: AWARE_BENCH_VAL_CORPUS required for disk-streaming mode \
                 (val_ratio={}; pass an explicit val file)",
                val_ratio
            );
            std::process::exit(1)
        }
    };

    let n_train_tokens = (fs::metadata(&train_bin).unwrap().len() / 4) as usize;
    let n_val_tokens = (fs::metadata(&val_bin).unwrap().len() / 4) as usize;
    println!(
        "[cn-bench] tokenized in {:.1}s: train={} val={}",
        read_t.elapsed().as_secs_f64(),
        n_train_tokens,
        n_val_tokens
    );

    let total_train_tokens = (steps * tokens_per_step) as f64;
    let effective_epochs = total_train_tokens / n_train_tokens as f64;
    println!(
        "[cn-bench] will train on {} tokens over {} steps = {:.2} effective epochs",
        total_train_tokens as usize, steps, effective_epochs
    );

    // Bootstrap sample for KMeans cap discovery — sampled directly from
    // train.bin without holding the corpus in memory.
    let bootstrap_tokens = if load_weights.is_some() {
        Vec::new() // keys come from the checkpoint, not from discovery
    } else {
        sample_bootstrap_from_file(&train_bin, 2000, cap_window.max(1), seed)
            .unwrap_or_else(|e| {
                eprintln!("ERROR: bootstrap sample: {}", e);
                std::process::exit(1)
            })
    };

    // ── Build cap-native substrate ──
    let device = aware::aware::default_device()?;
    println!(" device: {}", aware::aware::device_label(&device));
    let mut cap_config = CapConfig::default();
    cap_config.discovery = discovery;
    cap_config.n_caps_target = n_caps_target;
    cap_config.n_caps_budget = (n_caps_target * 2).max(512);
    cap_config.gradient_train = matches!(
        discovery,
        DiscoveryKind::NoDiscovery | DiscoveryKind::Hybrid
    );
    cap_config.cap_window = cap_window;

    let mut config = CapNativeConfig::default();
    config.vocab = bpe.vocab_size();
    config.d_model = d_model;
    config.n_blocks = n_blocks;
    config.n_heads = n_heads;
    config.d_ff = d_ff;
    config.max_seq_len = seq_len.max(128);
    config.cap_config = cap_config;
    config.top_k = top_k;
    config.cap_indexed_mask = cap_indexed_mask;
    config.routing = routing;
    config.hierarchical = hierarchical;
    config.cap_layer_1 = CapLayer1Config {
        n_caps_target: l1_n_caps,
        n_caps_budget: l1_budget,
        cap_window: l1_window,
        discovery: l1_discovery,
        gradient_train: l1_gradient_train,
    };
    if hierarchical {
        println!(
            " hierarchical: true (layer 1: n_caps={}, window={}, disc={:?})",
            l1_n_caps, l1_window, l1_discovery,
        );
    }

    let mut model = CapNativeSubstrate::builder()
        .with_config(config.clone())
        .with_device(device.clone())
        .with_seed(seed)
        .with_bootstrap_sample_tokens(bootstrap_tokens)
        .build()?;
    if let Some(ckpt) = &load_weights {
        aware::aware::cap_native::load_checkpoint(&mut model, ckpt)?;
        println!("[cn-bench] loaded checkpoint: {}", ckpt);
    }
    let model = model; // immutable from here
    let n_params = model.n_params();
    println!(
        "[cn-bench] params: {} ({:.2} MB)",
        n_params,
        (n_params * 4) as f64 / (1024.0 * 1024.0)
    );
    println!("[cn-bench] config: {}", config.label());

    // ── Feeders (disk-streamed, constant memory) + optimizer ──
    let mut train_feeder = StreamingFeeder::from_file(&train_bin, batch_size, seq_len)
        .unwrap_or_else(|e| {
            eprintln!("ERROR: open train feeder: {}", e);
            std::process::exit(1)
        })
        .with_seed(seed);
    let mut val_feeder = StreamingFeeder::from_file(&val_bin, batch_size, seq_len)
        .unwrap_or_else(|e| {
            eprintln!("ERROR: open val feeder: {}", e);
            std::process::exit(1)
        })
        .with_seed(seed.wrapping_add(1));
    let mut opt = {
        let vm = model.varmap.lock().unwrap();
        OptimizerKind::AdamW { lr }.build(&*vm)?
    };
    let loss_kind = LossKind::CrossEntropy;

    // ── Training loop ──
    let start = Instant::now();
    let mut trajectory: Vec<(usize, f32, f32, f64)> = Vec::new();
    let mut last_train_loss = 0.0f32;
    let mut final_val_loss = 0.0f32;
    // Track best val ppl across the run so the report captures the
    // true convergence point even when training continues past the
    // plateau into overfitting.
    let mut best_val_loss = f32::INFINITY;
    let mut best_val_step: usize = 0;

    let trace_every_step = std::env::var("AWARE_TRACE_STEPS").ok().as_deref() == Some("1");
    // Write report.json every N eval points so long runs survive
    // mid-run kills. Default 5 evals (e.g. 500 steps at eval_every=100).
    let checkpoint_every_evals = env_usize("AWARE_BENCH_CHECKPOINT_EVERY_EVALS", 5);
    let mut evals_since_checkpoint = 0usize;

    let report_path = format!("{}/report.json", run_dir);
    let val_source_str_owned = val_corpus_path
        .as_deref()
        .unwrap_or("(sliced from train)")
        .to_string();
    let routing_label_owned = format!("{:?}", routing);

    let write_report = |trajectory: &Vec<(usize, f32, f32, f64)>,
                        last_train_loss: f32,
                        final_val_loss: f32,
                        best_val_loss: f32,
                        best_val_step: usize,
                        total_seconds: f64,
                        completed: bool|
     -> Result<()> {
        let trajectory_json: Vec<String> = trajectory.iter().map(|(s, tr, va, el)| {
            format!(
                " {{\"step\":{},\"train_loss\":{:.4},\"val_loss\":{:.4},\"val_perplexity\":{:.2},\"elapsed_s\":{:.1}}}",
                s, tr, va, (*va as f64).exp(), el,
            )
        }).collect();
        let trajectory_block = trajectory_json.join(",\n");
        let best_val_ppl_str = if best_val_loss.is_finite() {
            format!("{:.2}", (best_val_loss as f64).exp())
        } else {
            "null".to_string()
        };
        let final_val_ppl_str = if final_val_loss.is_finite() && final_val_loss > 0.0 {
            format!("{:.2}", (final_val_loss as f64).exp())
        } else {
            "null".to_string()
        };
        let report_json = format!(
            "{{\n \"run_id\": \"{}\",\n \"architecture\": \"cap_native\",\n \"completed\": {},\n \"seed\": {},\n \"train_corpus\": \"{}\",\n \"val_corpus\": \"{}\",\n \"d_model\": {},\n \"n_blocks\": {},\n \"n_heads\": {},\n \"d_ff\": {},\n \"cap_discovery\": \"{}\",\n \"n_caps_target\": {},\n \"cap_window\": {},\n \"top_k\": {},\n \"cap_indexed_mask\": {},\n \"routing\": \"{}\",\n \"steps\": {},\n \"effective_epochs\": {:.2},\n \"tokens_trained\": {},\n \"final_train_loss\": {:.4},\n \"final_val_loss\": {:.4},\n \"final_val_perplexity\": {},\n \"best_val_loss\": {:.4},\n \"best_val_perplexity\": {},\n \"best_val_step\": {},\n \"params\": {},\n \"wall_clock_seconds\": {:.1},\n \"trajectory\": [\n{}\n ]\n}}\n",
            run_id, completed, seed,
            corpus_path, val_source_str_owned,
            d_model, n_blocks, n_heads, d_ff,
            discovery_str, n_caps_target, cap_window,
            top_k, cap_indexed_mask, routing_label_owned,
            steps, effective_epochs, total_train_tokens as usize,
            last_train_loss, final_val_loss, final_val_ppl_str,
            best_val_loss, best_val_ppl_str, best_val_step,
            n_params, total_seconds,
            trajectory_block,
        );
        fs::write(&report_path, report_json)
            .map_err(|e| candle_core::Error::Msg(format!("write report: {}", e)))?;
        Ok(())
    };

    // ── Eval-only mode (continual-learning instrumentation) ──
    // Loads happen above; here we run a deterministic pass over val,
    // optionally dumping per-token winning caps, and exit without
    // touching the training path. Determinism: the val feeder is seeded
    // from AWARE_BENCH_SEED, so two eval-only runs with the same seed
    // score the SAME token stream — required for winner-dump comparison.
    if env_bool("AWARE_BENCH_EVAL_ONLY", false) {
        let n_batches = env_usize("AWARE_BENCH_EVAL_BATCHES", 64);
        let dump_path = env::var("AWARE_CN_DUMP_WINNERS").ok();
        let mut total_loss = 0.0f32;
        let mut l0_parts: Vec<Tensor> = Vec::new();
        let mut l1_parts: Vec<Tensor> = Vec::new();

        for _ in 0..n_batches {
            let (inp, tgt) = val_feeder.next_batch(&device)?;
            let logits = model.forward(&inp)?;
            let (b, t, v) = logits.dims3()?;
            let loss = candle_nn::loss::cross_entropy(
                &logits.reshape((b * t, v))?,
                &tgt.reshape((b * t,))?,
            )?;
            total_loss += loss.to_scalar::<f32>()?;

            if dump_path.is_some() {
                // Winners stay on the device; one transfer per pass at
                // the end, never per batch.
                let (l0, l1) = model.cap_winners(&inp)?;
                l0_parts.push(l0.flatten_all()?);
                if let Some(l1) = l1 {
                    l1_parts.push(l1.flatten_all()?);
                }
            }
        }

        let avg_loss = total_loss / n_batches.max(1) as f32;
        println!(
            "[cn-bench] EVAL-ONLY over {} batches: val_loss={:.4} val_ppl={:.2}",
            n_batches,
            avg_loss,
            (avg_loss as f64).exp()
        );

        if let Some(path) = dump_path {
            let write_winners = |parts: &[Tensor], suffix: &str| -> Result<()> {
                if parts.is_empty() {
                    return Ok(());
                }
                let refs: Vec<&Tensor> = parts.iter().collect();
                let all = Tensor::cat(&refs, 0)?;
                let host: Vec<u32> = all.to_vec1()?; // the single transfer
                let mut buf = Vec::with_capacity(host.len() * 4);
                for w in &host {
                    buf.extend_from_slice(&w.to_le_bytes());
                }
                let out = format!("{}.{}.bin", path, suffix);
                fs::write(&out, buf)
                    .map_err(|e| candle_core::Error::Msg(format!("write winners: {}", e)))?;
                println!("[cn-bench] winners: {} ({} tokens)", out, host.len());
                Ok(())
            };
            write_winners(&l0_parts, "l0")?;
            write_winners(&l1_parts, "l1")?;
        }

        final_val_loss = avg_loss;
        write_report(&trajectory, 0.0, final_val_loss, avg_loss, 0, start.elapsed().as_secs_f64(), true)?;
        println!("[cn-bench] report: {}", report_path);
        return Ok(());
    }

    for step in 1..=steps {
        let step_start = Instant::now();
        if trace_every_step {
            print!("[trace] step={} batch…", step);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
        let (inp, tgt) = train_feeder.next_batch(&device)?;
        if trace_every_step {
            print!(" fwd…");
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
        let logits = model.forward(&inp)?;
        let (b, t, v) = logits.dims3()?;
        let logits_flat = logits.reshape((b * t, v))?;
        let tgt_flat = tgt.reshape((b * t,))?;
        let loss = loss_kind.compute(&logits_flat, &tgt_flat)?;
        if trace_every_step {
            print!(" bwd…");
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
        opt.backward_step(&loss)?;
        last_train_loss = loss.to_scalar::<f32>()?;
        if trace_every_step {
            println!(
                " done loss={:.3} [{:.2}s]",
                last_train_loss,
                step_start.elapsed().as_secs_f64()
            );
        }

        if step % eval_every == 0 || step == steps {
            let val_loss = compute_val_loss(&model, &mut val_feeder, &device, n_eval_batches)?;
            final_val_loss = val_loss;
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                best_val_step = step;
            }
            let elapsed = start.elapsed().as_secs_f64();
            println!(
                " step={:>5} train={:.3} (ppl={:.1}) val={:.3} (ppl={:.1}) best={:.1}@{} [{:.1}s]",
                step,
                last_train_loss,
                (last_train_loss as f64).exp(),
                val_loss,
                (val_loss as f64).exp(),
                (best_val_loss as f64).exp(),
                best_val_step,
                elapsed,
            );
            trajectory.push((step, last_train_loss, val_loss, elapsed));
            evals_since_checkpoint += 1;
            if evals_since_checkpoint >= checkpoint_every_evals {
                write_report(
                    &trajectory,
                    last_train_loss,
                    final_val_loss,
                    best_val_loss,
                    best_val_step,
                    elapsed,
                    false,
                )?;
                println!("[cn-bench] checkpoint: {}", report_path);
                evals_since_checkpoint = 0;
            }
        }
    }

    let total_seconds = start.elapsed().as_secs_f64();
    println!(
        "[cn-bench] done. final val_ppl = {:.2}, best val_ppl = {:.2} @ step {}, time = {:.1}s",
        (final_val_loss as f64).exp(),
        (best_val_loss as f64).exp(),
        best_val_step,
        total_seconds
    );

    write_report(
        &trajectory,
        last_train_loss,
        final_val_loss,
        best_val_loss,
        best_val_step,
        total_seconds,
        true,
    )?;
    println!("[cn-bench] report: {}", report_path);

    if let Some(ckpt) = &save_weights {
        if let Some(parent) = std::path::Path::new(ckpt).parent() {
            let _ = fs::create_dir_all(parent);
        }
        aware::aware::cap_native::save_checkpoint(&model, ckpt)?;
        println!("[cn-bench] weights: {} (+ .meta)", ckpt);
    }
    Ok(())
}
