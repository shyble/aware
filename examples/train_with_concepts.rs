// Train a Substrate model with both the kmeans_w4 winning architecture
// AND a concept layer in parallel.
//
// Architecture:
//   tokens -> embeddings
//          ↓
//          ├─-> Cap input layer (window=4, KMeans, n_caps=330)
//          │      -> standard transformer × 4 -> final_hidden
//          │
//          └─-> Concept layer (window=1, KMeans, n_concepts=128)
//                 frozen keys, gradient-trained values
//                 top-K=8 sparse activation
//                 contributes to final_hidden before unembed
//          ↓
//          tied unembed -> logits
//
// This is the next-paper architecture: identifiable per-token concepts
// with rich gradient-trained values, riding alongside the standard
// prediction pipeline.
//
// USAGE:
//   cargo run --release --features candle --example train_with_concepts \
//     -- data/tinystories_small/tinystories_train.txt
//
// ENV (uses same conventions as run_benchmark):
//   AWARE_BENCH_STEPS, AWARE_BENCH_BATCH_SIZE, AWARE_BENCH_SEQ_LEN,
//   AWARE_BENCH_LR, AWARE_BENCH_SEED,
//   AWARE_CONCEPT_N (default 128)
//   AWARE_CONCEPT_TOPK (default 8)
//   AWARE_CONCEPT_DISCOVERY (default kmeans)

use std::env;
use std::fs;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

use aware::aware::{
    AttentionKind, BlockBuilder, CapConfig, ConceptConfig, DiscoveryKind, LossKind, OptimizerKind,
    StreamingFeeder, Substrate,
};
use aware::data::bpe::{ensure_tokenized, load_bpe, BPETokenizer};

/// Sample a bootstrap set from a tokenized .bin file without loading the
/// whole corpus into memory.
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

fn parse_discovery(s: &str) -> DiscoveryKind {
    match s {
        "kmeans" => DiscoveryKind::KMeans,
        "random" => DiscoveryKind::Random,
        "hybrid" => DiscoveryKind::Hybrid,
        _ => DiscoveryKind::NoDiscovery,
    }
}

fn main() -> Result<()> {
    let paths: Vec<String> = env::args()
        .skip(1)
        .filter(|a| !a.starts_with("--"))
        .collect();
    if paths.is_empty() {
        eprintln!("Usage: train_with_concepts <corpus.txt>");
        std::process::exit(1);
    }

    let seed = env_u64("AWARE_BENCH_SEED", 42);
    let steps = env_usize("AWARE_BENCH_STEPS", 3000);
    let eval_every = env_usize("AWARE_BENCH_EVAL_EVERY", 100);
    let batch_size = env_usize("AWARE_BENCH_BATCH_SIZE", 32);
    let seq_len = env_usize("AWARE_BENCH_SEQ_LEN", 128);
    let lr = env_f64("AWARE_BENCH_LR", 3e-4);

    let n_concepts = env_usize("AWARE_CONCEPT_N", 128);
    let top_k = env_usize("AWARE_CONCEPT_TOPK", 8);
    let concept_discovery = parse_discovery(&env_str("AWARE_CONCEPT_DISCOVERY", "kmeans"));

    println!("=== Train with Concepts ===");
    println!("  base architecture:  kmeans_w4 (the prior winner)");
    println!(
        "  concept layer:      n_concepts={}, top_k={}, discovery={:?}",
        n_concepts, top_k, concept_discovery
    );
    println!();

    // ── Load BPE (train from a brief corpus pass if missing) ──
    let bpe = match load_bpe("data/brain_tinystories") {
        Ok(b) => b,
        Err(_) => {
            let mut sample = String::new();
            for p in &paths {
                sample.push_str(&fs::read_to_string(p).unwrap_or_else(|e| {
                    eprintln!("ERROR: read {}: {}", p, e);
                    std::process::exit(1)
                }));
                sample.push('\n');
            }
            let trained = BPETokenizer::train(&sample, 256);
            drop(sample);
            trained
        }
    };

    // ── Tokenize each input path to .bin (disk-streamed for training) ──
    // Multi-file input: we tokenize each separately and use the first as the
    // canonical train.bin. Multi-corpus streaming would require concatenation
    // at the file level; for now we just train on the first corpus and warn
    // if multiple were passed.
    if paths.len() > 1 {
        eprintln!(
            "WARN: multi-corpus input not supported in disk-streaming mode; \
             only the first corpus will be used"
        );
    }
    let train_path: PathBuf = PathBuf::from(&paths[0]);
    let train_bin = ensure_tokenized(&train_path, &bpe).unwrap_or_else(|e| {
        eprintln!("ERROR: tokenize train: {}", e);
        std::process::exit(1)
    });

    // Val: matching _val convention next to the train file.
    let val_path_str = paths[0].replace("_train", "_val");
    let val_path = PathBuf::from(&val_path_str);
    let val_bin = if val_path.exists() {
        ensure_tokenized(&val_path, &bpe).unwrap_or_else(|e| {
            eprintln!("ERROR: tokenize val: {}", e);
            std::process::exit(1)
        })
    } else {
        eprintln!(
            "ERROR: no val file at {} (disk-streaming requires explicit val)",
            val_path_str
        );
        std::process::exit(1)
    };

    let n_train_tokens = (fs::metadata(&train_bin).unwrap().len() / 4) as usize;
    let n_val_tokens = (fs::metadata(&val_bin).unwrap().len() / 4) as usize;
    println!(
        "[concepts] train tokens: {}, val tokens: {}, vocab: {}",
        n_train_tokens,
        n_val_tokens,
        bpe.vocab_size()
    );

    // Bootstrap sample for KMeans (cap_window=4 here; concept layer is
    // window=1 and uses the same bootstrap tokens).
    let cap_window = 4;
    let bootstrap_tokens = sample_bootstrap_from_file(
        &train_bin,
        2000,
        cap_window,
        seed,
    )
    .unwrap_or_else(|e| {
        eprintln!("ERROR: bootstrap sample: {}", e);
        std::process::exit(1)
    });

    let device = aware::aware::default_device()?;
    println!("  device: {}", aware::aware::device_label(&device));
    let model = Substrate::builder()
        .with_vocab(bpe.vocab_size())
        .with_d_model(128)
        .with_max_seq_len(seq_len.max(128))
        .with_device(device.clone())
        .with_cap_layer(CapConfig {
            kind: aware::aware::CapKind::Discovered,
            discovery: DiscoveryKind::KMeans,
            n_caps_target: 330,
            n_caps_budget: 1024,
            gradient_train: false,
            cap_window: 4,
            ..Default::default()
        })
        .with_bootstrap_sample_tokens(bootstrap_tokens)
        .with_block_repeated(
            4,
            BlockBuilder::new()
                .with_attention(AttentionKind::Standard)
                .with_heads(4)
                .with_ffn(512)
                .build(),
        )
        .with_concept_layer(ConceptConfig {
            n_concepts,
            top_k,
            discovery: concept_discovery,
            trainable_values: true,
            trainable_keys: false,
        })
        .build()?;

    let n_params = model.n_params();
    println!(
        "[concepts] params: {} ({:.2} MB)",
        n_params,
        (n_params * 4) as f64 / (1024.0 * 1024.0)
    );
    let cs = model.cap_stats();
    if let Some(s) = &cs.input_layer {
        println!(
            "[concepts] cap input layer: n={}, kind={}, trainable={}",
            s.n_caps, s.kind, s.trainable
        );
    }
    if let Some(cl) = &model.concept_layer {
        println!(
            "[concepts] concept layer: n_concepts={}, top_k={}",
            cl.n_concepts(),
            cl.config.top_k
        );
    }

    // Training loop
    let mut train_feeder =
        StreamingFeeder::from_file(&train_bin, batch_size, seq_len)
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

    let mut opt = OptimizerKind::AdamW { lr }.build(&model.varmap)?;
    let loss_kind = LossKind::CrossEntropy;
    let start = Instant::now();

    for step in 1..=steps {
        let (inp, tgt) = train_feeder.next_batch(&device)?;
        let logits = model.forward(&inp)?;
        let (b, t, v) = logits.dims3()?;
        let logits_flat = logits.reshape((b * t, v))?;
        let tgt_flat = tgt.reshape((b * t,))?;
        let loss = loss_kind.compute(&logits_flat, &tgt_flat)?;
        opt.backward_step(&loss)?;
        let train_loss = loss.to_scalar::<f32>()?;

        if step % eval_every == 0 || step == steps {
            // Val loss
            let mut val_total = 0.0f32;
            let n_val_batches = 4;
            for _ in 0..n_val_batches {
                let (vi, vt) = val_feeder.next_batch(&device)?;
                let vl = model.forward(&vi)?;
                let (b, t, v) = vl.dims3()?;
                let vl_flat = vl.reshape((b * t, v))?;
                let vt_flat = vt.reshape((b * t,))?;
                val_total += loss_kind.compute(&vl_flat, &vt_flat)?.to_scalar::<f32>()?;
            }
            let val_loss = val_total / n_val_batches as f32;
            let elapsed = start.elapsed().as_secs_f64();

            // Concept activation stats: how many concepts fire on average
            let (_, concept_acts) = model.forward_with_concepts(&inp)?;
            let n_active_avg: f32 = if let Some(acts) = concept_acts {
                let mask = acts.gt(0.0f32)?.to_dtype(candle_core::DType::F32)?;
                mask.sum_all()?.to_scalar::<f32>()? / ((b * t) as f32)
            } else {
                0.0
            };

            println!(
                "  step={:>5}  train={:.3} (ppl={:.1})  val={:.3} (ppl={:.1})  active_concepts={:.1}/pos  [{:.1}s]",
                step, train_loss, (train_loss as f64).exp(),
                val_loss, (val_loss as f64).exp(),
                n_active_avg, elapsed,
            );
        }
    }

    println!("\n[concepts] training complete.");
    Ok(())
}
