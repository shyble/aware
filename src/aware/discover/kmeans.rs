// K-means discovery: bootstrap via clustering on a sample of inputs.
// Audit detects dead caps (low activation EMA), finds novel inputs (low max
// cosine vs existing centroids), and replaces dead with novel.

use candle_core::{DType, Device, Result, Tensor};

use super::{AuditReport, Discovery, DiscoveryCtx};
use crate::aware::cap::{CapKind, CapMatrix};

pub struct KMeansDiscovery {
    pub max_iters: usize,
    pub seed: u64,
    pub hybrid: bool,
}

impl Default for KMeansDiscovery {
    fn default() -> Self {
        Self {
            max_iters: 25,
            seed: 42,
            hybrid: false,
        }
    }
}

impl KMeansDiscovery {
    pub fn hybrid() -> Self {
        Self {
            hybrid: true,
            ..Self::default()
        }
    }
}

impl Discovery for KMeansDiscovery {
    fn bootstrap(&self, ctx: &DiscoveryCtx) -> Result<CapMatrix> {
        let kind = if self.hybrid {
            CapKind::Hybrid
        } else {
            CapKind::Discovered
        };
        let trainable = self.hybrid;

        // If a sample is provided, run k-means. Otherwise fall back to random
        // unit-vector init (caller is expected to bootstrap from data later).
        let keys = if let Some(sample) = ctx.sample {
            kmeans(sample, ctx.n_caps_target, self.max_iters, self.seed)?
        } else {
            // No sample -> random unit vectors as a stand-in. Caller can
            // re-bootstrap once data is available.
            let raw = Tensor::randn(0.0f32, 1.0f32, (ctx.n_caps_target, ctx.d_in), ctx.device)?;
            let norm = raw.sqr()?.sum_keepdim(1)?.sqrt()?;
            raw.broadcast_div(&norm)?
        };

        let mut m = CapMatrix::empty(
            kind,
            ctx.d_in,
            ctx.d_out,
            ctx.n_caps_target,
            trainable,
            ctx.device.clone(),
        )?;
        m.keys = keys;
        Ok(m)
    }

    fn step(&self, caps: &mut CapMatrix, ctx: &DiscoveryCtx) -> AuditReport {
        // Audit lives in audit.rs; we delegate.
        super::audit::run_audit(caps, ctx)
    }
}

/// Mini-batch-ish k-means on a [n_samples, d] tensor. Returns centroids
/// of shape [k, d], normalized to unit length.
fn kmeans(sample: &Tensor, k: usize, max_iters: usize, seed: u64) -> Result<Tensor> {
    let (n_samples, d) = sample.dims2()?;
    let device = sample.device().clone();

    // Initialize centroids by picking k random sample rows.
    let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut indices: Vec<usize> = Vec::with_capacity(k);
    for _ in 0..k {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        indices.push((rng >> 33) as usize % n_samples.max(1));
    }
    // Gather: pick rows by index.
    let mut centroids = gather_rows(sample, &indices)?; // [k, d]

    for _iter in 0..max_iters {
        // Compute distances: sample [n, d] vs centroids [k, d]
        // dist[i, j] = ||sample[i] - centroids[j]||^2
        // = ||sample[i]||^2 + ||centroids[j]||^2 - 2 * sample[i] · centroids[j]
        let centroids_t = centroids.transpose(0, 1)?.contiguous()?;
        let dots = sample.matmul(&centroids_t)?; // [n, k]
        let s_sq = sample.sqr()?.sum_keepdim(1)?; // [n, 1]
        let c_sq = centroids.sqr()?.sum_keepdim(1)?; // [k, 1]
        let c_sq_row = c_sq.transpose(0, 1)?; // [1, k]
        let dists = s_sq
            .broadcast_add(&c_sq_row)?
            .broadcast_sub(&(dots * 2.0f64)?)?;
        // Assign: argmin over k
        let assignments = dists.argmin(1)?; // [n]

        // Update centroids: mean of assigned points per cluster.
        let new_centroids = update_centroids(sample, &assignments, k, d, &device)?;

        // Check convergence: if centroids barely move, break.
        let delta = (&new_centroids - &centroids)?
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()?;
        centroids = new_centroids;
        if delta < 1e-5 {
            break;
        }
    }

    // Normalize centroids to unit length.
    let norm = centroids
        .sqr()?
        .sum_keepdim(1)?
        .sqrt()?
        .clamp(1e-8f32, f32::INFINITY)?;
    centroids.broadcast_div(&norm)
}

fn gather_rows(sample: &Tensor, indices: &[usize]) -> Result<Tensor> {
    let (_n, _d) = sample.dims2()?;
    let device = sample.device();
    let idx_vec: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
    let idx_t = Tensor::from_vec(idx_vec, (indices.len(),), device)?;
    sample.index_select(&idx_t, 0)
}

fn update_centroids(
    sample: &Tensor,
    assignments: &Tensor,
    k: usize,
    d: usize,
    device: &Device,
) -> Result<Tensor> {
    // Convert assignments to CPU to do the bucket-mean simply.
    let assigns = assignments.to_vec1::<u32>()?;
    let n = sample.dim(0)?;
    let sample_data = sample.to_vec2::<f32>()?;

    let mut sums = vec![vec![0.0f32; d]; k];
    let mut counts = vec![0usize; k];
    for i in 0..n {
        let a = assigns[i] as usize;
        if a >= k {
            continue;
        }
        for j in 0..d {
            sums[a][j] += sample_data[i][j];
        }
        counts[a] += 1;
    }
    for a in 0..k {
        if counts[a] > 0 {
            let c = counts[a] as f32;
            for j in 0..d {
                sums[a][j] /= c;
            }
        }
    }
    let flat: Vec<f32> = sums.into_iter().flatten().collect();
    Tensor::from_vec(flat, (k, d), device).map(|t| t.to_dtype(DType::F32).unwrap())
}
