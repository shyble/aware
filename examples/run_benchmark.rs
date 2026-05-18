// Unified benchmark runner.
//
// Configurable via environment variables. Trains one substrate
// configuration, writes a report.json with config + trajectory + final
// metrics. Driven by scripts/run_benchmarks.sh for the full sweep.
//
// ENV (all optional with sensible defaults):
//   AWARE_BENCH_ID            short run id (used in output paths)
//   AWARE_BENCH_CORPUS        train corpus path
//   AWARE_BENCH_VAL_CORPUS    held-out val corpus path. If set, used as val
//                             instead of slicing from train. STRONGLY
//                             recommended - slicing train leaves the val
//                             "holdout" in the training distribution.
//   AWARE_BENCH_OUTPUT_DIR    where to write report.json (default data/bench/)
//   AWARE_BENCH_SEED          PRNG seed for the feeder (default 42)
//
//   AWARE_BENCH_D_MODEL       (default 128)
//   AWARE_BENCH_N_BLOCKS      (default 4)
//   AWARE_BENCH_N_HEADS       (default 4)
//   AWARE_BENCH_D_FF          (default 512)
//   AWARE_BENCH_SEQ_LEN       (default 128)
//   AWARE_BENCH_BATCH_SIZE    (default 32)
//   AWARE_BENCH_STEPS         (default 5000)
//   AWARE_BENCH_EVAL_EVERY    eval cadence in steps  (default 100)
//   AWARE_BENCH_N_EVAL_BATCHES batches per eval pass (default 8)
//   AWARE_BENCH_VAL_RATIO     held-out fraction (default 0.05)
//   AWARE_BENCH_LR            (default 3e-4)
//
//   AWARE_BENCH_ATTENTION      standard | cap_memory | cap_pair    (default standard)
//   AWARE_BENCH_CAP_SOURCE     local | shared                       (default local)
//   AWARE_BENCH_CAP_N          n_caps for attention-side cap matrix (default 512)
//   AWARE_BENCH_SHARED_N       n_caps for substrate-level shared matrix (default 512)
//
//   AWARE_BENCH_INCLUDE_CAP_LAYER  true | false                     (default true)
//   AWARE_BENCH_CAP_DISCOVERY      kmeans | random | nodiscovery    (default nodiscovery)
//   AWARE_BENCH_CAP_N_TARGET       initial cap count                (default 330)
//   AWARE_BENCH_CAP_WINDOW         input window length              (default 1)

use std::env;
use std::fs;
use std::time::Instant;

use aware::aware::attention::CapMatrixSource;
use aware::aware::{
    AttentionKind, BlockBuilder, CapConfig, DiscoveryKind, LossKind, OptimizerKind,
    StreamingFeeder, Substrate,
};
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

fn parse_attention(s: &str) -> AttentionKind {
    let source = match env_str("AWARE_BENCH_CAP_SOURCE", "local").as_str() {
        "shared" => CapMatrixSource::Shared,
        _ => CapMatrixSource::Local {
            n_caps: env_usize("AWARE_BENCH_CAP_N", 512),
        },
    };
    match s {
        "cap_memory" => AttentionKind::CapMemory { source },
        "cap_pair" => AttentionKind::CapPair { source },
        _ => AttentionKind::Standard,
    }
}

fn parse_discovery(s: &str) -> DiscoveryKind {
    match s {
        "kmeans" => DiscoveryKind::KMeans,
        "random" => DiscoveryKind::Random,
        "hybrid" => DiscoveryKind::Hybrid,
        _ => DiscoveryKind::NoDiscovery,
    }
}

/// Compute average cross-entropy over N sampled val batches.
fn compute_val_loss(
    model: &Substrate,
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
    let run_id = env_str("AWARE_BENCH_ID", "unnamed");
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

    let attn_kind_str = env_str("AWARE_BENCH_ATTENTION", "standard");
    let attention = parse_attention(&attn_kind_str);
    let cap_source = env_str("AWARE_BENCH_CAP_SOURCE", "local");
    let shared_n = env_usize("AWARE_BENCH_SHARED_N", 512);

    let include_cap_layer = env_bool("AWARE_BENCH_INCLUDE_CAP_LAYER", true);
    let discovery_str = env_str("AWARE_BENCH_CAP_DISCOVERY", "nodiscovery");
    let discovery = parse_discovery(&discovery_str);
    let n_caps_target = env_usize("AWARE_BENCH_CAP_N_TARGET", 330);
    let cap_window = env_usize("AWARE_BENCH_CAP_WINDOW", 1);

    let run_dir = format!("{}/{}", output_dir, run_id);
    fs::create_dir_all(&run_dir).ok();

    println!("=== Benchmark Runner ===");
    println!("  run_id:          {}", run_id);
    println!("  corpus:          {}", corpus_path);
    if let Some(vp) = &val_corpus_path {
        println!("  val_corpus:      {}", vp);
    } else {
        println!("  val_corpus:      (slicing last {}% of train - set AWARE_BENCH_VAL_CORPUS for proper holdout)",
            (val_ratio * 100.0) as usize);
    }
    println!("  seed:            {}", seed);
    println!("  d_model:         {}", d_model);
    println!("  n_blocks:        {}", n_blocks);
    println!(
        "  attention:       {} (source={})",
        attn_kind_str, cap_source
    );
    println!("  cap_layer:       {}", include_cap_layer);
    if include_cap_layer {
        println!("    discovery:     {}", discovery_str);
        println!("    n_caps_target: {}", n_caps_target);
        println!("    cap_window:    {}", cap_window);
    }
    let tokens_per_step = batch_size * seq_len;
    // train_tokens_len computed after tokenization; we'll print epochs after
    println!("  steps:           {}  (eval every {})", steps, eval_every);
    println!();

    // ── Load train corpus + BPE ──
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
        "[bench] tokenize train: {} tokens in {:.1}s",
        train_all.len(),
        read_t.elapsed().as_secs_f64()
    );

    // ── Val split: from separate file if AWARE_BENCH_VAL_CORPUS set,
    //              otherwise slice from train (deprecated path) ──
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
            "[bench] tokenize val:   {} tokens in {:.1}s (from {})",
            val_toks.len(),
            vt.elapsed().as_secs_f64(),
            vp
        );
        (train_all, val_toks)
    } else {
        let split = (train_all.len() as f64 * (1.0 - val_ratio)) as usize;
        let mut all = train_all;
        let val_toks = all.split_off(split);
        eprintln!("[bench] WARNING: val sliced from train tail (size {}); set AWARE_BENCH_VAL_CORPUS for proper held-out val", val_toks.len());
        (all, val_toks)
    };
    println!(
        "[bench] split: train={} val={}",
        train_tokens.len(),
        val_tokens.len()
    );

    // ── Compute effective epochs ──
    let total_train_tokens = (steps * tokens_per_step) as f64;
    let effective_epochs = total_train_tokens / train_tokens.len() as f64;
    println!("[bench] will train on {} tokens over {} steps = {:.2} effective epochs over the {} train corpus",
        total_train_tokens as usize, steps, effective_epochs, train_tokens.len());

    // ── Bootstrap sample for KMeans discovery ──
    // Target ~2000 sample vectors. When cap_window > 1, sample consecutive
    // token windows (so the bootstrap sample matches the cap layer's d_in).
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

    // ── Build substrate ──
    let device = Device::Cpu;
    let mut builder = Substrate::builder()
        .with_vocab(bpe.vocab_size())
        .with_d_model(d_model)
        .with_max_seq_len(seq_len.max(128))
        .with_device(device.clone());

    if cap_source == "shared" {
        builder = builder
            .with_shared_cap_matrix(shared_n)
            .with_shared_cap_d_v(d_model);
    }
    if include_cap_layer {
        builder = builder.with_cap_layer(CapConfig {
            discovery,
            n_caps_target,
            n_caps_budget: 1024,
            gradient_train: matches!(
                discovery,
                DiscoveryKind::NoDiscovery | DiscoveryKind::Hybrid
            ),
            cap_window,
            ..Default::default()
        });
        // KMeans / Hybrid actually use the sample; others ignore it.
        builder = builder.with_bootstrap_sample_tokens(bootstrap_tokens);
    }
    let block = BlockBuilder::new()
        .with_attention(attention)
        .with_heads(n_heads)
        .with_ffn(d_ff)
        .build();
    builder = builder.with_block_repeated(n_blocks, block);
    let model = builder.build()?;
    let n_params = model.n_params();
    println!(
        "[bench] params: {} ({:.2} MB)",
        n_params,
        (n_params * 4) as f64 / (1024.0 * 1024.0)
    );

    // ── Cap stats at start of training ──
    let cap_stats_initial = model.cap_stats();
    if let Some(s) = &cap_stats_initial.input_layer {
        println!(
            "[bench] input cap layer: n_caps={}  kind={}  trainable={}  frozen={}  dormant={}",
            s.n_caps, s.kind, s.trainable, s.n_frozen, s.n_dormant
        );
    }
    if let Some(s) = &cap_stats_initial.shared {
        println!(
            "[bench] shared cap matrix: n_caps={}  kind={}  trainable={}",
            s.n_caps, s.kind, s.trainable
        );
    }

    // ── Feeders (seeded) ──
    let mut train_feeder =
        StreamingFeeder::from_tokens(train_tokens, batch_size, seq_len).with_seed(seed);
    let mut val_feeder = StreamingFeeder::from_tokens(val_tokens, batch_size, seq_len)
        .with_seed(seed.wrapping_add(1));

    // ── Optimizer ──
    let mut opt = OptimizerKind::AdamW { lr }.build(&model.varmap)?;
    let loss_kind = LossKind::CrossEntropy;

    // ── Training loop with val + trajectory ──
    let start = Instant::now();
    let mut trajectory: Vec<(usize, f32, f32, f64)> = Vec::new(); // (step, train_loss, val_loss, elapsed)
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
                "  step={:>5}  train={:.3} (ppl={:.1})  val={:.3} (ppl={:.1})  [{:.1}s]",
                step,
                last_train_loss,
                (last_train_loss as f64).exp(),
                val_loss,
                (val_loss as f64).exp(),
                elapsed,
            );
            trajectory.push((step, last_train_loss, val_loss, elapsed));
            let _ = model.save_checkpoint(&format!("{}/substrate.safetensors", run_dir));
        }
    }

    let total_seconds = start.elapsed().as_secs_f64();
    let cap_stats_final = model.cap_stats();
    println!(
        "[bench] done. final val_ppl = {:.2}, time = {:.1}s",
        (final_val_loss as f64).exp(),
        total_seconds
    );

    // ── Write report.json with full trajectory + cap stats + epochs ──
    let trajectory_json: Vec<String> = trajectory.iter().map(|(s, tr, va, el)| {
        format!(
            "    {{\"step\":{},\"train_loss\":{:.4},\"val_loss\":{:.4},\"val_perplexity\":{:.2},\"elapsed_s\":{:.1}}}",
            s, tr, va, (*va as f64).exp(), el,
        )
    }).collect();
    let trajectory_block = trajectory_json.join(",\n");

    let cap_stats_json = |stats: &Option<aware::aware::substrate::LayerCapStats>| -> String {
        match stats {
            Some(s) => format!(
                "{{\"n_caps\":{},\"kind\":\"{}\",\"trainable\":{},\"n_frozen\":{},\"n_dormant\":{}}}",
                s.n_caps, s.kind, s.trainable, s.n_frozen, s.n_dormant,
            ),
            None => "null".to_string(),
        }
    };

    let val_source_str = val_corpus_path.as_deref().unwrap_or("(sliced from train)");
    let report_json = format!(
        "{{\n  \"run_id\": \"{}\",\n  \"seed\": {},\n  \"train_corpus\": \"{}\",\n  \"val_corpus\": \"{}\",\n  \"attention\": \"{}\",\n  \"cap_source\": \"{}\",\n  \"include_cap_layer\": {},\n  \"discovery\": \"{}\",\n  \"n_caps_target\": {},\n  \"cap_window\": {},\n  \"d_model\": {},\n  \"n_blocks\": {},\n  \"n_heads\": {},\n  \"d_ff\": {},\n  \"steps\": {},\n  \"effective_epochs\": {:.2},\n  \"tokens_trained\": {},\n  \"final_train_loss\": {:.4},\n  \"final_val_loss\": {:.4},\n  \"final_val_perplexity\": {:.2},\n  \"params\": {},\n  \"wall_clock_seconds\": {:.1},\n  \"cap_stats_initial\": {{\"input_layer\": {}, \"shared\": {}}},\n  \"cap_stats_final\":   {{\"input_layer\": {}, \"shared\": {}}},\n  \"trajectory\": [\n{}\n  ]\n}}\n",
        run_id, seed,
        corpus_path, val_source_str,
        attn_kind_str, cap_source, include_cap_layer,
        discovery_str, n_caps_target, cap_window,
        d_model, n_blocks, n_heads, d_ff,
        steps, effective_epochs, total_train_tokens as usize,
        last_train_loss, final_val_loss, (final_val_loss as f64).exp(),
        n_params, total_seconds,
        cap_stats_json(&cap_stats_initial.input_layer),
        cap_stats_json(&cap_stats_initial.shared),
        cap_stats_json(&cap_stats_final.input_layer),
        cap_stats_json(&cap_stats_final.shared),
        trajectory_block,
    );

    let report_path = format!("{}/report.json", run_dir);
    fs::write(&report_path, report_json)
        .map_err(|e| candle_core::Error::Msg(format!("write report: {}", e)))?;
    println!("[bench] report: {}", report_path);
    Ok(())
}
