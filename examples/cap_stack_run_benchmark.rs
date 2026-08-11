// Cap stack: does hierarchical composition substitute for attention?
//
// The cap-input layer is the strongest measured result in this project --
// 51.22 -> 30.55 perplexity for 42,240 extra trainable parameters, with
// the keys frozen. It has only ever been used ONCE, at the input, in
// front of a transformer.
//
// This model removes the transformer entirely and stacks the cap layer
// instead. Range comes from composing local windows rather than from
// attending:
//
//     tokens -> embed
//            -> CapLayer(window w0)      receptive field w0
//            -> CapLayer(window w1)      receptive field w0 + w1 - 1
//            -> ...
//            -> norm -> output
//
// No attention anywhere, no recurrence, fully parallel. The question is
// narrow and worth answering before designing anything on top of it:
// does a second cap layer buy substantially more than the first, i.e.
// is hierarchical composition carrying real weight?
//
// What this CANNOT do, stated in advance so the result is read honestly:
// two positions sharing a cap get the same receptive field, so exact
// long-range binding ("John ... he") is out of reach. Range flows only
// through the hierarchy. The PMI measurement predicts that cost is small
// locally (41.9% of entropy at delta 1) and real at distance (0.7% at
// delta 32). If step 2 works, cap-conditioned VARIABLE offsets are the
// next step, and those come from the same PMI table.
//
// ENV: AWARE_STACK_WINDOWS (comma list, e.g. "3" or "3,9"), plus the
// AWARE_BENCH_* variables shared with the other runners.

use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

use aware::aware::config::CapConfig;
use aware::aware::discover::DiscoveryKind;
use aware::aware::layer::CapLayer;
use aware::aware::norm::RmsNorm;
use aware::aware::train::{LossKind, StreamingFeeder};
use aware::data::bpe::ensure_tokenized;
use candle_core::{Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, Module, VarBuilder, VarMap};

fn env_str(k: &str, d: &str) -> String {
    env::var(k).unwrap_or_else(|_| d.to_string())
}
fn env_usize(k: &str, d: usize) -> usize {
    env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}
fn env_u64(k: &str, d: u64) -> u64 {
    env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}
fn env_f64(k: &str, d: f64) -> f64 {
    env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}

struct CapStack {
    embed: Embedding,
    layers: Vec<CapLayer>,
    norms: Vec<RmsNorm>,
    final_norm: RmsNorm,
    head: Linear,
}

impl CapStack {
    /// Each layer is residual: the cap block is a refinement of the
    /// running representation, not a replacement. Without this a stack
    /// deeper than two trains poorly, for the same reason it does in any
    /// deep net -- and the question here is about DEPTH, so the residual
    /// must not be the thing that limits it.
    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let mut h = self.embed.forward(tokens)?;
        for (layer, norm) in self.layers.iter().zip(self.norms.iter()) {
            let normed = norm.forward(&h)?;
            h = (h + layer.forward(&normed)?)?;
        }
        self.head.forward(&self.final_norm.forward(&h)?)
    }
}

fn main() -> Result<()> {
    let corpus = env_str("AWARE_BENCH_CORPUS", "data/tinystories_small/tinystories_train.txt");
    let val_corpus = env_str("AWARE_BENCH_VAL_CORPUS", "data/tinystories_small/tinystories_val.txt");
    let bpe_dir = env_str("AWARE_BENCH_BPE_DIR", "data/brain_tinystories");
    let out_dir = env_str("AWARE_BENCH_OUTPUT_DIR", "data/bench_capstack");
    let run_id = env_str("AWARE_BENCH_ID", "cap_stack");
    let seed = env_u64("AWARE_BENCH_SEED", 42);

    let d_model = env_usize("AWARE_BENCH_D_MODEL", 128);
    let n_caps = env_usize("AWARE_BENCH_CAP_N_TARGET", 330);
    let seq_len = env_usize("AWARE_BENCH_SEQ_LEN", 128);
    let batch_size = env_usize("AWARE_BENCH_BATCH_SIZE", 32);
    let steps = env_usize("AWARE_BENCH_STEPS", 5000);
    let eval_every = env_usize("AWARE_BENCH_EVAL_EVERY", 100);
    let n_eval = env_usize("AWARE_BENCH_N_EVAL_BATCHES", 8);
    let lr = env_f64("AWARE_BENCH_LR", 3e-4);

    // Escape hatch so the frozen-deep behaviour remains reproducible:
    // the first depth ladder ran with every layer frozen, and that run
    // has to stay comparable.
    let frozen_deep = env_str("AWARE_STACK_FROZEN_DEEP", "false") == "true";
    let windows: Vec<usize> = env_str("AWARE_STACK_WINDOWS", "3")
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if windows.is_empty() {
        eprintln!("ERROR: AWARE_STACK_WINDOWS parsed empty");
        std::process::exit(1);
    }
    // Contiguous windows compose additively: stacking w0 then w1 sees
    // w0 + w1 - 1 tokens, not w0 * w1. Reported so the number is never
    // silently overstated.
    let rf: usize = windows.iter().sum::<usize>() - (windows.len() - 1);

    let device = aware::aware::device::default_device()?;
    let bpe = aware::data::bpe::load_bpe(&bpe_dir)
        .map_err(|e| candle_core::Error::Msg(format!("load bpe: {e}")))?;
    let vocab = bpe.vocab_size();
    let train_bin = ensure_tokenized(Path::new(&corpus), &bpe)
        .map_err(|e| candle_core::Error::Msg(format!("tokenize train: {e}")))?;
    let val_bin = ensure_tokenized(Path::new(&val_corpus), &bpe)
        .map_err(|e| candle_core::Error::Msg(format!("tokenize val: {e}")))?;

    println!("=== Cap Stack (no attention, no recurrence) ===");
    println!(" windows: {windows:?}  -> receptive field {rf} tokens");
    println!(" d_model: {d_model}  n_caps: {n_caps}  vocab: {vocab}");

    // Bootstrap sample for KMeans on the FIRST layer only. Deeper layers
    // discover over the previous layer's output, which does not exist
    // until the model does, so they take random keys -- and the ablation
    // says random keys already carry ~85% of what discovery gives.
    let sample_toks = {
        let bytes = fs::read(&train_bin)
            .map_err(|e| candle_core::Error::Msg(format!("read train bin: {e}")))?;
        let toks: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .take(200_000)
            .collect();
        toks
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let embed = embedding(vocab, d_model, vb.pp("embed"))?;

    // The sample must match what the layer's keys will be matched
    // AGAINST, which is the WINDOWED input (window x d_emb), not a single
    // embedding. Clustering raw d_emb vectors produces keys of the wrong
    // shape and the first matmul fails -- or worse, would silently
    // cluster the wrong thing if the dimensions happened to line up.
    let w0 = windows[0].max(1);
    let n_win = sample_toks.len() / w0;
    let sample_emb = {
        let t = Tensor::from_vec(
            sample_toks[..n_win * w0].to_vec(),
            (n_win, w0),
            &device,
        )?;
        embed.forward(&t)?.reshape((n_win, w0 * d_model))?
    };

    let mut layers = Vec::new();
    let mut norms = Vec::new();
    for (i, w) in windows.iter().enumerate() {
        let cfg = CapConfig {
            discovery: if i == 0 { DiscoveryKind::KMeans } else { DiscoveryKind::Random },
            n_caps_target: n_caps,
            n_caps_budget: n_caps,
            cap_window: *w,
            // Layer 0's keys stay frozen: they are discovered over
            // embeddings, they carry the identity the audit lifecycle
            // depends on, and freezing them is what makes the input
            // layer stable and cheap.
            //
            // Deeper layers are NOT frozen. Their keys would otherwise be
            // random directions over representations that did not exist
            // when they were chosen -- pinning a guess about a moving
            // target. They adapt via a zero-initialised delta on the
            // discovered base, so they begin exactly where discovery put
            // them and follow the layer below as it trains.
            adapt_keys: i > 0 && !frozen_deep,
            ..Default::default()
        };
        let sample = if i == 0 { Some(&sample_emb) } else { None };
        layers.push(CapLayer::new(
            cfg, d_model, d_model, &device, sample, vb.pp(format!("cap{i}")),
        )?);
        norms.push(RmsNorm::new(d_model, 1e-5, vb.pp(format!("norm{i}")))?);
    }
    let final_norm = RmsNorm::new(d_model, 1e-5, vb.pp("final_norm"))?;
    let head = linear_no_bias(d_model, vocab, vb.pp("head"))?;
    let model = CapStack { embed, layers, norms, final_norm, head };

    let n_params: usize = varmap
        .all_vars()
        .iter()
        .map(|v| v.elem_count())
        .sum();
    println!(
        " trainable params: {n_params} (layer-0 keys frozen; deeper keys adaptable: {})",
        !frozen_deep && windows.len() > 1
    );

    let mut train_feeder = StreamingFeeder::from_file(&train_bin, batch_size, seq_len)
        .map_err(|e| candle_core::Error::Msg(format!("train feeder: {e}")))?
        .with_seed(seed);
    let mut val_feeder = StreamingFeeder::from_file(&val_bin, batch_size, seq_len)
        .map_err(|e| candle_core::Error::Msg(format!("val feeder: {e}")))?
        .with_seed(seed.wrapping_add(1));

    let params = candle_nn::optim::ParamsAdamW { lr, ..Default::default() };
    let mut opt = aware::aware::PersistentAdamW::new(&varmap, params)?;
    let loss_kind = LossKind::CrossEntropy;

    let start = Instant::now();
    let mut best = f32::INFINITY;
    let mut traj: Vec<(usize, f32, f32)> = Vec::new();
    for step in 1..=steps {
        let (inp, tgt) = train_feeder.next_batch(&device)?;
        let logits = model.forward(&inp)?;
        let (b, t, v) = logits.dims3()?;
        let loss = loss_kind.compute(&logits.reshape((b * t, v))?, &tgt.reshape((b * t,))?)?;
        opt.backward_step(&loss)?;

        if step % eval_every == 0 || step == steps {
            let mut tot = 0.0f32;
            for _ in 0..n_eval {
                let (vi, vt) = val_feeder.next_batch(&device)?;
                let vl = model.forward(&vi)?;
                let (b, t, v) = vl.dims3()?;
                tot += candle_nn::loss::cross_entropy(
                    &vl.reshape((b * t, v))?,
                    &vt.reshape((b * t,))?,
                )?
                .to_scalar::<f32>()?;
            }
            let val = tot / n_eval as f32;
            let tr = loss.to_scalar::<f32>()?;
            best = best.min(val);
            traj.push((step, tr, val));
            println!(
                " step={step:>6} train={tr:.3} (ppl={:.1}) val={val:.3} (ppl={:.1}) best_ppl={:.1} [{:.1}s]",
                (tr as f64).exp(), (val as f64).exp(), (best as f64).exp(),
                start.elapsed().as_secs_f64()
            );
        }
    }

    let final_ppl = (traj.last().map(|x| x.2).unwrap_or(f32::NAN) as f64).exp();
    println!(
        "[cap-stack] done. windows={windows:?} rf={rf} params={n_params} final val_ppl={final_ppl:.2} best={:.2}",
        (best as f64).exp()
    );

    let dir = format!("{out_dir}/{run_id}");
    fs::create_dir_all(&dir).ok();
    let traj_json: Vec<String> = traj
        .iter()
        .map(|(s, tr, v)| format!("{{\"step\":{s},\"train_loss\":{tr:.4},\"val_loss\":{v:.4}}}"))
        .collect();
    fs::write(
        format!("{dir}/report.json"),
        format!(
            "{{\"run_id\":\"{run_id}\",\"architecture\":\"cap_stack\",\"windows\":{windows:?},\
             \"receptive_field\":{rf},\"d_model\":{d_model},\"n_caps\":{n_caps},\
             \"params\":{n_params},\"steps\":{steps},\"final_val_perplexity\":{final_ppl:.4},\
             \"best_val_perplexity\":{:.4},\"trajectory\":[{}]}}",
            (best as f64).exp(),
            traj_json.join(",")
        ),
    )
    .ok();
    println!("[cap-stack] report: {dir}/report.json");
    Ok(())
}
