//! Grouped matmul: variable-size per-cap projections without padding.
//!
//! # The problem this solves
//!
//! Every dispatch path in `sparse_routing` and `blocksparse` pads each
//! cap's token bucket to a uniform width so one batched matmul can cover
//! all caps at once. Padding is arithmetic the GPU performs and then
//! throws away: with 330 caps over 4096 tokens the mean bucket holds ~12
//! tokens, so a capacity sized for the *largest* bucket means most of the
//! multiply is against zero rows.
//!
//! The primitive that removes the waste is a **grouped GEMM** — one
//! launch that multiplies `n_caps` independent `(m_k, d_in) @ (d_in,
//! d_out)` problems with *different* `m_k`. This is what Megablocks
//! approximates with block-sparsity, and what
//! `cublasGemmGroupedBatchedEx` provides directly.
//!
//! # Structure
//!
//! Two layers, deliberately separated:
//!
//! 1. [`grouped_matmul`] — the entry point. Portable, differentiable via
//!    candle's own autograd, correct on CPU, CUDA and Metal. It slices
//!    each cap's contiguous span and multiplies it by that cap's weight
//!    slab. No padding, but one kernel launch per non-empty group.
//!
//! 2. [`GroupedMatmulCuda`] — a `CustomOp2` that collapses those launches
//!    into a single grouped GEMM. CUDA only, opt-in, and it plugs in
//!    underneath the same entry point so no call site changes.
//!
//! Layer 1 is the correctness reference: layer 2 is verified by comparing
//! against it on the same inputs.
//!
//! # Input contract
//!
//! `xs` must be **sorted by winning cap**, so each cap owns a contiguous
//! row span. `offsets[k]..offsets[k+1]` is cap `k`'s span; `offsets` has
//! length `n_caps + 1` and is host-side metadata, not a tensor — the
//! grouped GEMM needs the sizes on the host to build its problem
//! descriptors regardless.

use candle_core::{CpuStorage, CustomOp2, Layout, Result, Shape, Tensor};

/// Is the fused CUDA grouped GEMM enabled? Off by default: the portable
/// path is always correct, and the fused path changes floating-point
/// reduction order, so it must be opted into and verified.
pub fn fused_enabled() -> bool {
    std::env::var("AWARE_CN_GROUPED_GEMM")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Multiply each cap's contiguous span of `xs` by that cap's weight slab.
///
/// - `xs`: `[total, d_in]`, rows sorted by winning cap
/// - `weights`: `[n_caps, d_in, d_out]`
/// - `offsets`: length `n_caps + 1`, prefix-summed bucket boundaries
///
/// Returns `[total, d_out]` in the same sorted row order.
///
/// Empty groups are skipped entirely — that is the whole point relative
/// to the padded paths, which multiply through them.
pub fn grouped_matmul(xs: &Tensor, weights: &Tensor, offsets: &[u32]) -> Result<Tensor> {
    let (total, d_in) = xs.dims2()?;
    let (n_caps, d_in_w, d_out) = weights.dims3()?;
    if d_in_w != d_in {
        candle_core::bail!("grouped_matmul: xs d_in {d_in} != weights d_in {d_in_w}");
    }
    if offsets.len() != n_caps + 1 {
        candle_core::bail!(
            "grouped_matmul: offsets len {} != n_caps + 1 ({})",
            offsets.len(),
            n_caps + 1
        );
    }
    if offsets[n_caps] as usize != total {
        candle_core::bail!(
            "grouped_matmul: offsets cover {} rows but xs has {total}",
            offsets[n_caps]
        );
    }

    if fused_enabled() && xs.device().is_cuda() {
        let op = GroupedMatmulCuda {
            offsets: offsets.to_vec(),
            d_out,
        };
        return xs.apply_op2(weights, op);
    }

    portable_grouped_matmul(xs, weights, offsets, d_out)
}

/// Reference implementation: one matmul per non-empty group.
///
/// Differentiable through candle's ordinary autograd — `narrow` and
/// `matmul` both have gradients — so no manual backward is needed here.
/// This is what the fused path is checked against.
fn portable_grouped_matmul(
    xs: &Tensor,
    weights: &Tensor,
    offsets: &[u32],
    d_out: usize,
) -> Result<Tensor> {
    let n_caps = weights.dim(0)?;
    let mut pieces: Vec<Tensor> = Vec::new();

    for k in 0..n_caps {
        let start = offsets[k] as usize;
        let n_k = (offsets[k + 1] - offsets[k]) as usize;
        if n_k == 0 {
            continue;
        }
        let xs_k = xs.narrow(0, start, n_k)?;
        let w_k = weights.narrow(0, k, 1)?.squeeze(0)?;
        pieces.push(xs_k.matmul(&w_k)?);
    }

    if pieces.is_empty() {
        return Tensor::zeros((0, d_out), xs.dtype(), xs.device());
    }
    let refs: Vec<&Tensor> = pieces.iter().collect();
    Tensor::cat(&refs, 0)
}

/// Fused grouped GEMM as a candle custom op.
///
/// The forward collapses every group into one `cublasGemmGroupedBatchedEx`
/// launch. The backward is expressed with ordinary tensor ops rather than
/// a second custom kernel: both gradients are themselves grouped
/// problems, and correctness matters more than saving launches on the
/// backward pass until the forward is proven.
pub struct GroupedMatmulCuda {
    offsets: Vec<u32>,
    d_out: usize,
}

impl CustomOp2 for GroupedMatmulCuda {
    fn name(&self) -> &'static str {
        "grouped-matmul"
    }

    /// CPU never takes this path — [`grouped_matmul`] routes non-CUDA
    /// devices to the portable implementation before constructing the op.
    /// Implemented anyway so the op is total rather than panicking if a
    /// future call site forgets the guard.
    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!(
            "grouped-matmul: no cpu path; call grouped_matmul(), which \
             routes cpu/metal to the portable implementation"
        )
    }

    // NOTE: cuda_fwd is implemented in `grouped_cuda.rs`, behind the
    // `cuda` feature, so this file compiles unchanged on a Mac.

    fn bwd(
        &self,
        xs: &Tensor,
        weights: &Tensor,
        _res: &Tensor,
        grad_out: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        // Both gradients are grouped problems over the same partition:
        //   grad_xs[g] = grad_out[g] @ W[k]^T
        //   grad_W[k]  = xs[g]^T @ grad_out[g]
        // Per-cap weight gradients do not accumulate across groups —
        // each cap's slab is touched by exactly its own tokens — so they
        // stack rather than sum.
        let n_caps = weights.dim(0)?;
        let d_in = xs.dim(1)?;

        let mut grad_x_pieces: Vec<Tensor> = Vec::new();
        let mut grad_w_slabs: Vec<Tensor> = Vec::new();

        for k in 0..n_caps {
            let start = self.offsets[k] as usize;
            let n_k = (self.offsets[k + 1] - self.offsets[k]) as usize;
            let w_k = weights.narrow(0, k, 1)?.squeeze(0)?;

            if n_k == 0 {
                // Untouched cap: zero gradient of the right shape, so the
                // stacked result still lines up with `weights`.
                grad_w_slabs.push(Tensor::zeros(
                    (d_in, self.d_out),
                    weights.dtype(),
                    weights.device(),
                )?);
                continue;
            }

            let xs_k = xs.narrow(0, start, n_k)?;
            let go_k = grad_out.narrow(0, start, n_k)?;

            grad_x_pieces.push(go_k.matmul(&w_k.t()?.contiguous()?)?);
            grad_w_slabs.push(xs_k.t()?.contiguous()?.matmul(&go_k)?);
        }

        let grad_xs = if grad_x_pieces.is_empty() {
            Tensor::zeros_like(xs)?
        } else {
            let refs: Vec<&Tensor> = grad_x_pieces.iter().collect();
            Tensor::cat(&refs, 0)?
        };
        let refs: Vec<&Tensor> = grad_w_slabs.iter().collect();
        let grad_w = Tensor::stack(&refs, 0)?;

        Ok((Some(grad_xs), Some(grad_w)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// The portable path must agree with an explicit per-group matmul,
    /// including when a cap receives no tokens at all — the empty-bucket
    /// case is what the padded paths handle implicitly and this one has
    /// to handle by construction.
    #[test]
    fn grouped_matches_per_group_reference() -> Result<()> {
        let dev = Device::Cpu;
        let (d_in, d_out) = (4, 3);
        // cap 0: 2 rows, cap 1: 0 rows, cap 2: 3 rows
        let offsets = vec![0u32, 2, 2, 5];
        let total = 5;
        let n_caps = 3;

        let xs = Tensor::randn(0f32, 1f32, (total, d_in), &dev)?;
        let weights = Tensor::randn(0f32, 1f32, (n_caps, d_in, d_out), &dev)?;

        let got = grouped_matmul(&xs, &weights, &offsets)?;
        assert_eq!(got.dims2()?, (total, d_out));

        for k in 0..n_caps {
            let start = offsets[k] as usize;
            let n_k = (offsets[k + 1] - offsets[k]) as usize;
            if n_k == 0 {
                continue;
            }
            let want = xs
                .narrow(0, start, n_k)?
                .matmul(&weights.narrow(0, k, 1)?.squeeze(0)?)?;
            let diff = (got.narrow(0, start, n_k)? - want)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(diff < 1e-5, "cap {k} mismatch: {diff}");
        }
        Ok(())
    }

    /// Offsets that do not cover exactly `total` rows are a routing bug;
    /// fail loudly rather than silently dropping tokens.
    #[test]
    fn rejects_inconsistent_offsets() -> Result<()> {
        let dev = Device::Cpu;
        let xs = Tensor::zeros((5, 4), DType::F32, &dev)?;
        let weights = Tensor::zeros((2, 4, 3), DType::F32, &dev)?;
        assert!(grouped_matmul(&xs, &weights, &[0, 2, 4]).is_err());
        Ok(())
    }
}
