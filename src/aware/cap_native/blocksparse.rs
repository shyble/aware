//! Block-sparse cap dispatch: routing index construction on the device.
//!
//! # Why this exists
//!
//! `sparse_routing::bounded_grouped_routing` builds its dispatch indices on
//! the CPU. Every cap-keyed component pulls the per-token winners off the
//! accelerator (`to_vec1`), stable-sorts `total` tokens single-threaded in
//! Rust, materialises a `n_caps * bound` padding permutation, and uploads
//! the result. On a CPU backend that costs nothing. On CUDA it is a full
//! pipeline stall, and the padding permutation grows as `n_caps * bound`,
//! which is what made the 330-cap single-discovery config slower on an
//! RTX 4060 (30.4 s/step) than on an M1 Pro CPU (15.1 s/step). Hoisting
//! routing to once per forward recovered 35 % of that; the remaining cost
//! is the construction itself.
//!
//! This module removes the host round trip entirely. Bucket positions come
//! from a cumulative sum over a one-hot of the winners — the assignment
//! trick used by capacity-based mixture-of-experts dispatch — so the
//! indices are produced by tensor ops on whatever device the model already
//! lives on.
//!
//! # What it does *not* do
//!
//! This is not Megablocks. True block-sparse matmul needs custom kernels
//! that candle does not expose, so the compute here is still a dense
//! batched matmul over padded capacity blocks. What changes is where the
//! indices are computed, not how the multiplication is tiled.
//!
//! # Token preservation
//!
//! Capacity-based MoE dispatch usually *drops* tokens whose bucket
//! overflows. Dropping would change what the model computes, not just how
//! fast it computes it, so this implementation instead iterates: pass `p`
//! handles bucket positions `[p*C, (p+1)*C)`, and passes continue until
//! every token has been processed. Skewed cap distributions cost extra
//! passes rather than accuracy. With a balanced distribution one pass
//! covers everything.
//!
//! # Expected numerics
//!
//! Each token's output is `x_t @ W_{cap(t)}`; the dot product for one
//! output element reduces over `d_in` regardless of how tokens are grouped
//! into blocks, and zero padding contributes nothing to other rows. Output
//! is therefore expected to match the CPU-routed path exactly. That is an
//! argument, not a proof — kernel tiling is free to reassociate — so
//! `VERIFY.md` treats it as a claim to be measured, not assumed.

use candle_core::{DType, Device, Result, Tensor, D};

/// Device-resident dispatch plan for one forward pass.
///
/// Built once from the shared cap activations and reused by every
/// cap-keyed component in the substrate, exactly like
/// `BoundedGroupedRouting`.
pub struct BlockSparseRouting {
    /// Per-token winning cap, `[total]`, `u32`, on device.
    pub winners_t: Tensor,
    /// Position of each token inside its own cap's bucket, `[total]`,
    /// `u32`. Token `t` is the `pos[t]`-th token routed to `winners[t]`.
    pub pos_t: Tensor,
    /// Capacity of one block. Bucket positions are handled `capacity` at
    /// a time.
    pub capacity: usize,
    /// Number of passes needed so that every token is covered:
    /// `ceil(max_bucket_size / capacity)`.
    pub n_passes: usize,
    pub n_caps: usize,
    pub total: usize,
}

/// Round a capacity up to a multiple of `TILE` so the padded blocks line
/// up with GPU matmul tiles instead of straddling them.
const TILE: usize = 32;

fn round_to_tile(n: usize) -> usize {
    if n == 0 {
        return TILE;
    }
    n.div_ceil(TILE) * TILE
}

/// Choose the per-block capacity.
///
/// Larger capacity means fewer passes but a larger padded tensor; the
/// padded block is `n_caps * capacity * d`, so this is the same memory
/// tradeoff `choose_bound` makes in the CPU path. The default targets the
/// uniform mean, tile-aligned, which covers a balanced distribution in a
/// single pass. `AWARE_CN_BS_CAP_MULT` scales it.
fn choose_capacity(total: usize, n_caps: usize) -> usize {
    if n_caps == 0 {
        return TILE;
    }
    let uniform = total.div_ceil(n_caps);
    let mult: usize = std::env::var("AWARE_CN_BS_CAP_MULT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    round_to_tile(uniform * mult.max(1))
}

/// Is the block-sparse path enabled? Opt-in, so the default dispatch and
/// every previously published number stay exactly as they were.
pub fn enabled() -> bool {
    std::env::var("AWARE_CN_BLOCKSPARSE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Build the dispatch plan from the shared cap activations.
///
/// The only host transfer is a single scalar — the largest bucket size —
/// which decides how many passes are required. That is unavoidable
/// without device-side control flow, and it is four bytes rather than the
/// `total + n_caps * bound` indices the CPU path moves.
pub fn blocksparse_routing(
    cap_acts: &Tensor,
    n_caps: usize,
    device: &Device,
) -> Result<BlockSparseRouting> {
    let dims = cap_acts.dims();
    let total: usize = dims[..dims.len() - 1].iter().product();
    let cap_flat = cap_acts.reshape((total, n_caps))?;

    // Winning cap per token. Stays on device — this is the transfer the
    // CPU path makes and this module exists to avoid.
    let winners_t = cap_flat.argmax(D::Minus1)?;

    // One-hot the winners, then take a running sum down the token axis.
    // Column k of the cumulative sum counts, at each token, how many
    // tokens so far have chosen cap k. Reading the winner's own column
    // therefore gives that token's index inside its bucket.
    // candle generates broadcast_{add,mul,sub,div,maximum,minimum} but no
    // broadcast comparison, so build the one-hot by scattering a 1.0 into
    // each row's winning column instead.
    let winners_i64 = winners_t.to_dtype(DType::I64)?;
    let winners_col = winners_i64.reshape((total, 1))?.contiguous()?;
    let ones_col = Tensor::ones((total, 1), DType::F32, device)?;
    let one_hot = Tensor::zeros((total, n_caps), DType::F32, device)?
        .scatter_add(&winners_col, &ones_col, 1)?;
    let running = one_hot.cumsum(0)?.contiguous()?;

    // `gather` along the cap axis picks each token's own column; subtract
    // one to turn a count into a zero-based position.
    let pos_f = running.gather(&winners_col, 1)?;
    let pos_t = (pos_f.reshape((total,))? - 1.0)?.to_dtype(DType::U32)?;

    // Bucket sizes are the final row of the running sum. The largest one
    // decides how many capacity-sized passes are needed to cover every
    // token without dropping any.
    let counts = running.narrow(0, total.saturating_sub(1), 1)?.reshape((n_caps,))?;
    let max_bucket = counts.max(0)?.to_scalar::<f32>()? as usize;

    let capacity = choose_capacity(total, n_caps);
    let n_passes = max_bucket.div_ceil(capacity).max(1);

    Ok(BlockSparseRouting {
        winners_t,
        pos_t,
        capacity,
        n_passes,
        n_caps,
        total,
    })
}

/// Apply a cap-keyed projection using the block-sparse plan.
///
/// `xs` is in original token order, `[total, d_in]`; `weights` is the
/// per-cap stack `[n_caps, d_in, d_out]`. Returns `[total, d_out]`, also
/// in original token order — unlike the CPU path, no sort-by-winner
/// permutation is needed on either side.
pub fn apply_blocksparse_projection(
    xs: &Tensor,
    weights: &Tensor,
    routing: &BlockSparseRouting,
) -> Result<Tensor> {
    let (total, d_in) = xs.dims2()?;
    let (n_caps_w, d_in_w, d_out) = weights.dims3()?;
    debug_assert_eq!(total, routing.total);
    debug_assert_eq!(n_caps_w, routing.n_caps);
    debug_assert_eq!(d_in_w, d_in);

    let device = xs.device();
    let dtype = xs.dtype();
    let capacity = routing.capacity;
    let n_caps = routing.n_caps;
    let slots = n_caps * capacity;

    // Row 0 is a zero sentinel: padding slots read it, so unused capacity
    // multiplies to zero and never disturbs a real token's row.
    let zero_row = Tensor::zeros((1, d_in), dtype, device)?;
    let xs_with_zero = Tensor::cat(&[&zero_row, xs], 0)?;

    let winners_i64 = routing.winners_t.to_dtype(DType::I64)?;
    let pos_i64 = routing.pos_t.to_dtype(DType::I64)?;
    let token_ids = Tensor::arange(1i64, (total + 1) as i64, device)?;

    let mut acc: Option<Tensor> = None;

    // Scalar arithmetic on a Tensor routes through `affine`, which is
    // float-only, so integer constants have to arrive as tensors and go
    // through the broadcast ops instead.
    let capacity_t = Tensor::new(&[capacity as i64], device)?;

    for pass in 0..routing.n_passes {
        let lo_t = Tensor::new(&[(pass * capacity) as i64], device)?;

        // Tokens whose bucket position falls in this pass's window.
        let local = pos_i64.broadcast_sub(&lo_t)?;
        let in_pass = local
            .ge(0i64)?
            .mul(&local.lt(capacity as i64)?)?
            .to_dtype(DType::I64)?;

        // Destination slot within the padded block layout. Tokens outside
        // this pass are parked at slot 0 and masked to the sentinel.
        let slot = winners_i64.broadcast_mul(&capacity_t)?.add(&local)?;
        let slot = (slot * &in_pass)?;
        let src = (&token_ids * &in_pass)?;

        // Scatter token ids into their slots: `gather_idx[s]` is the token
        // occupying slot `s`, or 0 for empty padding.
        let mut gather_idx = Tensor::zeros((slots,), DType::I64, device)?;
        gather_idx = gather_idx.scatter_add(&slot, &src, 0)?;

        // One batched matmul over `[n_caps, capacity, d_in]`.
        let block = xs_with_zero
            .index_select(&gather_idx, 0)?
            .reshape((n_caps, capacity, d_in))?;
        let out_block = block.matmul(weights)?.reshape((slots, d_out))?;

        // Read each token's row back from its slot. Tokens not handled in
        // this pass read slot 0, which is zero, so summing passes is safe.
        let take = (&slot * &in_pass)?;
        let rows = out_block.index_select(&take, 0)?;
        let mask = in_pass.to_dtype(dtype)?.reshape((total, 1))?;
        let contrib = rows.broadcast_mul(&mask)?;

        acc = Some(match acc {
            None => contrib,
            Some(prev) => (prev + contrib)?,
        });
    }

    acc.ok_or_else(|| candle_core::Error::Msg("blocksparse: no passes executed".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// The projection must equal the naive definition: every token
    /// multiplied by its own winning cap's slab. Checked token by token so
    /// a failure localises to a token rather than to an aggregate.
    #[test]
    fn projection_matches_naive_reference() -> Result<()> {
        let dev = Device::Cpu;
        let (total, n_caps, d_in, d_out) = (37, 5, 6, 4);

        // Deterministic, deliberately skewed cap activations so some
        // buckets overflow one capacity block and force multiple passes.
        let mut acts = vec![0f32; total * n_caps];
        for t in 0..total {
            let winner = if t < 20 { 0 } else { t % n_caps };
            acts[t * n_caps + winner] = 1.0;
        }
        let cap_acts = Tensor::from_vec(acts, (total, n_caps), &dev)?;
        let xs = Tensor::randn(0f32, 1f32, (total, d_in), &dev)?;
        let weights = Tensor::randn(0f32, 1f32, (n_caps, d_in, d_out), &dev)?;

        let routing = blocksparse_routing(&cap_acts, n_caps, &dev)?;
        let got = apply_blocksparse_projection(&xs, &weights, &routing)?;

        let winners: Vec<u32> = routing.winners_t.to_vec1()?;
        for t in 0..total {
            let k = winners[t] as usize;
            let want = xs
                .narrow(0, t, 1)?
                .matmul(&weights.narrow(0, k, 1)?.squeeze(0)?)?;
            let diff = (got.narrow(0, t, 1)? - want)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(diff < 1e-4, "token {t} (cap {k}) differs by {diff}");
        }
        Ok(())
    }
}
