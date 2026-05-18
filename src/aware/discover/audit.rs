// Shared audit logic used by KMeans and Hybrid discovery strategies.
//
// Identifies dead caps (activation_ema below threshold) and novel inputs
// (low max-cosine against any existing cap centroid), then pairs and
// replaces deadest caps with novel-input directions.

use candle_core::Result;

use super::{AuditReport, DiscoveryCtx};
use crate::aware::cap::{CapMatrix, CapMeta};

pub fn run_audit(caps: &mut CapMatrix, ctx: &DiscoveryCtx) -> AuditReport {
    let audit_cfg = ctx.audit;
    let n_caps = caps.n_caps();
    if n_caps == 0 {
        return AuditReport::empty();
    }

    // 1. Identify dead caps by activation_ema threshold.
    let dead_indices: Vec<usize> = caps
        .metadata
        .iter()
        .enumerate()
        .filter(|(_, m)| {
            !m.frozen && !m.dormant && m.activation_ema < audit_cfg.dead_rate_threshold
        })
        .map(|(i, _)| i)
        .collect();
    let dead_fraction = dead_indices.len() as f64 / n_caps as f64;

    // 2. Identify novel inputs from the sample.
    let novel_inputs = if let Some(sample) = ctx.sample {
        detect_novel_inputs(sample, &caps.keys, audit_cfg.novelty_threshold).unwrap_or_default()
    } else {
        Vec::new()
    };
    let novel_fraction = if let Some(sample) = ctx.sample {
        novel_inputs.len() as f64 / sample.dim(0).unwrap_or(1).max(1) as f64
    } else {
        0.0
    };

    // 3. Gate: require both imbalance conditions.
    if dead_fraction < audit_cfg.min_dead_fraction
        || novel_fraction < audit_cfg.min_novelty_fraction
        || novel_inputs.is_empty()
    {
        return AuditReport {
            n_dead_detected: dead_indices.len(),
            n_novel_inputs_found: novel_inputs.len(),
            n_replaced: 0,
            n_grown: 0,
        };
    }

    // 4. Pair-and-replace, capped by budget.
    let n_replace = dead_indices
        .len()
        .min(novel_inputs.len())
        .min(audit_cfg.replace_budget_per_call);

    let mut new_id_seed = caps.metadata.iter().map(|m| m.id).max().unwrap_or(0);
    for k in 0..n_replace {
        let slot = dead_indices[k];
        new_id_seed += 1;
        let new_key = &novel_inputs[k];
        let _ = caps.replace_row(slot, new_id_seed, new_key, None);
        caps.metadata[slot] = CapMeta::new(new_id_seed);
    }

    AuditReport {
        n_dead_detected: dead_indices.len(),
        n_novel_inputs_found: novel_inputs.len(),
        n_replaced: n_replace,
        n_grown: 0,
    }
}

/// Find input rows whose max-cosine against existing keys is below threshold.
/// Returns up to a budget worth of novel input rows as Vec<Vec<f32>>.
fn detect_novel_inputs(
    sample: &candle_core::Tensor,
    keys: &candle_core::Tensor,
    novelty_threshold: f64,
) -> Result<Vec<Vec<f32>>> {
    let (n, d) = sample.dims2()?;
    let (k, _) = keys.dims2()?;

    // Normalize sample and keys for cosine similarity.
    let sample_norm = sample
        .sqr()?
        .sum_keepdim(1)?
        .sqrt()?
        .clamp(1e-8f32, f32::INFINITY)?;
    let sample_n = sample.broadcast_div(&sample_norm)?;

    let keys_norm = keys
        .sqr()?
        .sum_keepdim(1)?
        .sqrt()?
        .clamp(1e-8f32, f32::INFINITY)?;
    let keys_n = keys.broadcast_div(&keys_norm)?;

    // cosine matrix: [n, k]
    let keys_t = keys_n.transpose(0, 1)?.contiguous()?;
    let cos = sample_n.matmul(&keys_t)?;
    let max_cos = cos.max(1)?; // [n]

    let max_cos_vec = max_cos.to_vec1::<f32>()?;
    let sample_data = sample.to_vec2::<f32>()?;
    let mut novel: Vec<Vec<f32>> = Vec::new();
    for i in 0..n {
        if (max_cos_vec[i] as f64) < novelty_threshold {
            novel.push(sample_data[i].clone());
        }
    }
    let _ = (d, k);
    Ok(novel)
}
