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
use std::time::Instant;

use aware::aware::cap_native::{CapNativeConfig, CapNativeSubstrate, RoutingMode};
use aware::aware::config::CapConfig;
use aware::aware::discover::DiscoveryKind;
use aware::aware::train::LossKind;
use aware::aware::train::StreamingFeeder;
use aware::aware::OptimizerKind;
use aware::data::bpe::load_bpe;
use aware::data::bpe::BPETokenizer;
use candle_core::{Device, Result};
use candle_nn::Optimizer;

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
    let discovery_str = env_str("AWARE_BENCH_CAP_DISCOVERY", "kmeans");
    let discovery = parse_discovery(&discovery_str);
    let n_caps_target = env_usize("AWARE_BENCH_CAP_N_TARGET", 330);
    let cap_window = env_usize("AWARE_BENCH_CAP_WINDOW", 4);

    // Cap-native-specific
    let top_k = env_usize("AWARE_CN_TOP_K", 0);
    let cap_indexed_mask = env_bool("AWARE_CN_CAP_INDEXED_MASK", false);
    let routing_str = env_str("AWARE_CN_ROUTING", "soft");
    let routing = parse_routing(&routing_str);

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

    // ── Load + tokenize corpus ──
    let read_t = Instant::now();
    let corpus = fs::read_to_string(&corpus_path).unwrap_or_else(|e| {
        eprintln!("ERROR: read {}: {}", corpus_path, e);
        std::process::exit(1)
    });

    let bpe = match load_bpe("data/brain_tinystories") {
        Ok(b) => b,
        Err(_) => BPETokenizer::train(&corpus, 256),
    };
    let train_all: Vec<u32> = bpe.encode(&corpus).into_iter().map(|t| t as u32).collect();
    drop(corpus);
    println!(
        "[cn-bench] tokenize train: {} tokens in {:.1}s",
        train_all.len(),
        read_t.elapsed().as_secs_f64()
    );

    let (train_tokens, val_tokens) = if let Some(vp) = &val_corpus_path {
        let vt = Instant::now();
        let val_corpus = fs::read_to_string(vp).unwrap_or_else(|e| {
            eprintln!("ERROR: read val {}: {}", vp, e);
            std::process::exit(1)
        });
        let val_toks: Vec<u32> = bpe
            .encode(&val_corpus)
            .into_iter()
            .map(|t| t as u32)
            .collect();
        println!(
            "[cn-bench] tokenize val: {} tokens in {:.1}s (from {})",
            val_toks.len(),
            vt.elapsed().as_secs_f64(),
            vp
        );
        (train_all, val_toks)
    } else {
        let split = (train_all.len() as f64 * (1.0 - val_ratio)) as usize;
        let mut all = train_all;
        let val_toks = all.split_off(split);
        eprintln!("[cn-bench] WARNING: val sliced from train tail; set AWARE_BENCH_VAL_CORPUS for proper holdout");
        (all, val_toks)
    };
    println!(
        "[cn-bench] split: train={} val={}",
        train_tokens.len(),
        val_tokens.len()
    );

    let total_train_tokens = (steps * tokens_per_step) as f64;
    let effective_epochs = total_train_tokens / train_tokens.len() as f64;
    println!(
        "[cn-bench] will train on {} tokens over {} steps = {:.2} effective epochs",
        total_train_tokens as usize, steps, effective_epochs
    );

    // Bootstrap sample for KMeans cap discovery (windowed when cap_window > 1).
    let target_samples = 2000usize;
    let mut bs_rng = seed;
    let bootstrap_tokens: Vec<u32> = if cap_window > 1 {
        let n_windows = target_samples;
        let max_start = train_tokens.len().saturating_sub(cap_window + 1);
        let mut toks = Vec::with_capacity(n_windows * cap_window);
        for _ in 0..n_windows {
            bs_rng = bs_rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let start = ((bs_rng >> 33) as usize) % max_start.max(1);
            toks.extend_from_slice(&train_tokens[start..start + cap_window]);
        }
        toks
    } else {
        (0..target_samples.min(train_tokens.len()))
            .map(|_| {
                bs_rng = bs_rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                train_tokens[((bs_rng >> 33) as usize) % train_tokens.len()]
            })
            .collect()
    };

    // ── Build cap-native substrate ──
    let device = Device::Cpu;
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

    let model = CapNativeSubstrate::builder()
        .with_config(config.clone())
        .with_device(device.clone())
        .with_seed(seed)
        .with_bootstrap_sample_tokens(bootstrap_tokens)
        .build()?;
    let n_params = model.n_params();
    println!(
        "[cn-bench] params: {} ({:.2} MB)",
        n_params,
        (n_params * 4) as f64 / (1024.0 * 1024.0)
    );
    println!("[cn-bench] config: {}", config.label());

    // ── Feeders + optimizer ──
    let mut train_feeder =
        StreamingFeeder::from_tokens(train_tokens, batch_size, seq_len).with_seed(seed);
    let mut val_feeder = StreamingFeeder::from_tokens(val_tokens, batch_size, seq_len)
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

    for step in 1..=steps {
        let (inp, tgt) = train_feeder.next_batch(&device)?;
        let logits = model.forward(&inp)?;
        let (b, t, v) = logits.dims3()?;
        let logits_flat = logits.reshape((b * t, v))?;
        let tgt_flat = tgt.reshape((b * t,))?;
        let loss = loss_kind.compute(&logits_flat, &tgt_flat)?;
        opt.backward_step(&loss)?;
        last_train_loss = loss.to_scalar::<f32>()?;

        if step % eval_every == 0 || step == steps {
            let val_loss = compute_val_loss(&model, &mut val_feeder, &device, n_eval_batches)?;
            final_val_loss = val_loss;
            let elapsed = start.elapsed().as_secs_f64();
            println!(
                " step={:>5} train={:.3} (ppl={:.1}) val={:.3} (ppl={:.1}) [{:.1}s]",
                step,
                last_train_loss,
                (last_train_loss as f64).exp(),
                val_loss,
                (val_loss as f64).exp(),
                elapsed,
            );
            trajectory.push((step, last_train_loss, val_loss, elapsed));
        }
    }

    let total_seconds = start.elapsed().as_secs_f64();
    println!(
        "[cn-bench] done. final val_ppl = {:.2}, time = {:.1}s",
        (final_val_loss as f64).exp(),
        total_seconds
    );

    // ── Write report.json ──
    let trajectory_json: Vec<String> = trajectory.iter().map(|(s, tr, va, el)| {
 format!(
 " {{\"step\":{},\"train_loss\":{:.4},\"val_loss\":{:.4},\"val_perplexity\":{:.2},\"elapsed_s\":{:.1}}}",
 s, tr, va, (*va as f64).exp(), el,
 )
 }).collect();
    let trajectory_block = trajectory_json.join(",\n");

    let val_source_str = val_corpus_path.as_deref().unwrap_or("(sliced from train)");
    let routing_label = format!("{:?}", routing);

    let report_json = format!(
 "{{\n \"run_id\": \"{}\",\n \"architecture\": \"cap_native\",\n \"seed\": {},\n \"train_corpus\": \"{}\",\n \"val_corpus\": \"{}\",\n \"d_model\": {},\n \"n_blocks\": {},\n \"n_heads\": {},\n \"d_ff\": {},\n \"cap_discovery\": \"{}\",\n \"n_caps_target\": {},\n \"cap_window\": {},\n \"top_k\": {},\n \"cap_indexed_mask\": {},\n \"routing\": \"{}\",\n \"steps\": {},\n \"effective_epochs\": {:.2},\n \"tokens_trained\": {},\n \"final_train_loss\": {:.4},\n \"final_val_loss\": {:.4},\n \"final_val_perplexity\": {:.2},\n \"params\": {},\n \"wall_clock_seconds\": {:.1},\n \"trajectory\": [\n{}\n ]\n}}\n",
 run_id, seed,
 corpus_path, val_source_str,
 d_model, n_blocks, n_heads, d_ff,
 discovery_str, n_caps_target, cap_window,
 top_k, cap_indexed_mask, routing_label,
 steps, effective_epochs, total_train_tokens as usize,
 last_train_loss, final_val_loss, (final_val_loss as f64).exp(),
 n_params, total_seconds,
 trajectory_block,
 );

    let report_path = format!("{}/report.json", run_dir);
    fs::write(&report_path, report_json)
        .map_err(|e| candle_core::Error::Msg(format!("write report: {}", e)))?;
    println!("[cn-bench] report: {}", report_path);
    Ok(())
}
