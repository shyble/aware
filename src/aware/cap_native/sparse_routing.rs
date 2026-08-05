// Shared helpers for memory-efficient sparse (HardTop1) cap routing.
//
// Two routing strategies live here:
//
// 1. `sparse_routing_perm` — sort-by-winner permutation. Used with a
//    per-non-empty-bucket loop. Replaces the original n_caps-deep
//    index_add chain with a single cat + inverse-permute, so the
//    autograd graph carries O(1) intermediates per cap-keyed component.
//    Compute does n_buckets small matmuls (one per non-empty bucket).
//
// 2. `sparse_grouped_routing` + `apply_grouped_projection` — the
//    grouped/batched dispatch path. Builds pad/unpad permutations so
//    the entire cap-keyed projection collapses to ONE 3-D batched
//    matmul (n_caps, max_n_k, d_in) @ (n_caps, d_in, d_out). Reduces
//    kernel-launch count by ~n_caps× on Metal/CUDA (where dispatch
//    overhead dominates at small per-cap matmul size).

use candle_core::{Device, Result, Tensor};

/// Given a per-token argmax winner cap, build the sort-by-winner
/// permutation and contiguous bucket offsets needed by the sparse
/// cap-keyed forward.
///
/// Returns `(perm_t, inv_perm_t, bucket_offsets)`:
/// - `perm_t[j]` = original token index that lands at sorted position j
/// - `inv_perm_t[t]` = sorted position of original token t
/// - `bucket_offsets[k]..bucket_offsets[k+1]` is cap k's contiguous span
///    in sorted order; the vector has length `n_caps + 1`
pub fn sparse_routing_perm(
    winners: &[u32],
    n_caps: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Vec<u32>)> {
    let total = winners.len();
    let cap_max = if n_caps == 0 { 0 } else { (n_caps - 1) as u32 };

    // Pair (winner, original_index) and stable-sort by winner so
    // within-bucket order preserves token sequence order.
    let mut indexed: Vec<(u32, u32)> = winners
        .iter()
        .enumerate()
        .map(|(t, &w)| (w.min(cap_max), t as u32))
        .collect();
    indexed.sort_by_key(|&(w, _)| w);

    let perm: Vec<u32> = indexed.iter().map(|&(_, t)| t).collect();

    // inv_perm[original_idx] = sorted_position
    let mut inv_perm = vec![0u32; total];
    for (sorted_pos, &original_idx) in perm.iter().enumerate() {
        inv_perm[original_idx as usize] = sorted_pos as u32;
    }

    // bucket_offsets[k+1] = count of tokens with winner == k (prefix-sum after)
    let mut bucket_offsets = vec![0u32; n_caps + 1];
    for &(w, _) in &indexed {
        bucket_offsets[w as usize + 1] += 1;
    }
    for k in 1..=n_caps {
        bucket_offsets[k] += bucket_offsets[k - 1];
    }

    let perm_t = Tensor::from_vec(perm, (total,), device)?;
    let inv_perm_t = Tensor::from_vec(inv_perm, (total,), device)?;

    Ok((perm_t, inv_perm_t, bucket_offsets))
}

/// Bounded grouped routing state — batched dispatch with worst-case
/// memory guarantees. Splits each cap's tokens into:
///   - a "fits" portion of at most `bound` rows that flow through ONE
///     batched matmul of shape `(n_caps, bound, d_in)`
///   - an "overflow" portion (only when a cap's bucket exceeds `bound`)
///     that falls back to a per-bucket loop, preserving correctness
///     without unbounded padding
///
/// `bound` is sized so the padded tensor stays ~tens of MB regardless
/// of how skewed the cap firing distribution is.
#[derive(Clone)]
pub struct BoundedGroupedRouting {
    pub perm_t: Tensor,
    pub inv_perm_t: Tensor,
    pub bucket_offsets: Vec<u32>,
    pub bound: usize,
    pub n_caps: usize,
    pub total: usize,

    /// Padded gather indices into `xs_with_zero` (xs_sorted prefixed by
    /// a zero sentinel row). Length `n_caps * bound`. `0` reads the
    /// sentinel; `i+1` reads sorted-position `i`.
    pub pad_perm_t: Tensor,
    /// Tells where each fit-slot lands in sorted order. Length
    /// `n_caps * bound`; entries past a cap's fit_count point to the
    /// sentinel slot and are masked out via `combine_perm`.
    /// For each sorted position `p` covered by a fit, this lookup is
    /// folded into `combine_perm_t` directly.
    pub n_fits: usize,

    /// Sorted positions of tokens that overflowed (n_k > bound per cap).
    /// Empty if no overflow.
    pub overflow_indices_t: Option<Tensor>,
    /// Per-cap bucket offsets within the overflow batch. Length n_caps+1.
    pub overflow_bucket_offsets: Vec<u32>,
    pub n_overflow: usize,

    /// For each sorted position, where to read its projected value:
    ///   - `i in 0..n_fits`: read padded-flat[fit_layout_t[i]]
    ///   - `n_fits..n_fits+n_overflow`: read overflow result row
    /// Used as a single `index_select` after concatenating
    /// `(fits_in_padded_order, overflow_result)`.
    pub combine_perm_t: Tensor,
    /// Maps the `n_fits` fit-slot order back to indices in the
    /// padded-flat layout (`k * bound + i_in_pad`). Length n_fits.
    pub fit_layout_t: Tensor,
}

/// Routing state for the grouped/batched cap-keyed projection. Built
/// once per forward and reused across each cap-keyed component
/// (attention QKV+O, MoE gate+value+out, output projection) that
/// shares the same per-token cap winner.
pub struct SparseGroupedRouting {
    /// Sort-by-winner permutation. Length `total`.
    pub perm_t: Tensor,
    /// Inverse of `perm_t` — un-permutes from sorted back to original.
    pub inv_perm_t: Tensor,
    /// Maps each padded position `(k, i)` (flattened as `k*max_n_k+i`) to
    /// `sorted_index + 1` for the real tokens or `0` for the zero-row
    /// sentinel used to pad short buckets. Length `n_caps * max_n_k`.
    pub pad_perm_t: Tensor,
    /// Maps each sorted position back to its slot in the padded layout.
    /// Length `total`.
    pub unpad_perm_t: Tensor,
    pub bucket_offsets: Vec<u32>,
    pub max_n_k: usize,
    pub total: usize,
    pub n_caps: usize,
}

/// Compute a per-batch `bound` for bounded grouped routing. Returns a
/// padded-tensor size that stays bounded regardless of cap-firing
/// imbalance. Tokens that exceed `bound` per cap fall back to a
/// per-bucket overflow path.
///
/// `bound` directly drives both peak memory and kernel-launch count:
/// larger bound → fewer overflow-loop matmuls (more batched) → more
/// padded-tensor memory. Default is `2× uniform mean` with a `max(8)`
/// floor so the batched path stays worthwhile when n_caps is large.
/// Tunable via `AWARE_CN_BOUND_MULT` (multiplier on uniform mean).
fn choose_bound(total: usize, n_caps: usize) -> usize {
    if n_caps == 0 {
        return 0;
    }
    let uniform = (total + n_caps - 1) / n_caps; // ceil(total / n_caps)
    let mult: usize = std::env::var("AWARE_CN_BOUND_MULT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    (uniform * mult).max(8)
}

/// Build bounded grouped routing state. The padded tensor used by the
/// batched matmul is `(n_caps, bound, d)` — bounded memory regardless
/// of cap distribution. Buckets whose size exceeds `bound` route their
/// surplus tokens through an overflow per-bucket loop.
pub fn bounded_grouped_routing(
    winners: &[u32],
    n_caps: usize,
    device: &Device,
) -> Result<BoundedGroupedRouting> {
    let total = winners.len();
    let cap_max = if n_caps == 0 { 0 } else { (n_caps - 1) as u32 };

    // Sort-by-winner (stable preserves within-cap token order).
    let mut indexed: Vec<(u32, u32)> = winners
        .iter()
        .enumerate()
        .map(|(t, &w)| (w.min(cap_max), t as u32))
        .collect();
    indexed.sort_by_key(|&(w, _)| w);
    let perm: Vec<u32> = indexed.iter().map(|&(_, t)| t).collect();
    let mut inv_perm = vec![0u32; total];
    for (sorted_pos, &original_idx) in perm.iter().enumerate() {
        inv_perm[original_idx as usize] = sorted_pos as u32;
    }
    let mut bucket_offsets = vec![0u32; n_caps + 1];
    for &(w, _) in &indexed {
        bucket_offsets[w as usize + 1] += 1;
    }
    for k in 1..=n_caps {
        bucket_offsets[k] += bucket_offsets[k - 1];
    }

    let bound = choose_bound(total, n_caps).max(1);

    // Pad layout: for each cap k, slots [k*bound .. k*bound + fit_n[k])
    // hold real sorted indices (+1 for the zero-sentinel prefix);
    // slots [k*bound + fit_n[k] .. (k+1)*bound) are sentinel reads.
    //
    // Overflow: cap k's tokens past slot `bound` (i.e. sorted indices
    // bucket_offsets[k] + bound .. bucket_offsets[k+1]) go through the
    // per-bucket fallback. We track them densely in `overflow_indices`.
    let mut pad_perm: Vec<u32> = vec![0; n_caps * bound];
    let mut fit_layout: Vec<u32> = Vec::with_capacity(total.min(n_caps * bound));
    let mut overflow_indices: Vec<u32> = Vec::new();
    let mut overflow_bucket_offsets = vec![0u32; n_caps + 1];

    // combine_perm[sorted_pos] tells where to read the projected value
    // from after the batched matmul + overflow loop. Indices `0..n_fits`
    // are taken from the fits buffer (in sorted-cap, sorted-pad order);
    // indices `n_fits..n_fits+n_overflow` are taken from the overflow
    // buffer (in cap order).
    let mut combine_perm = vec![0u32; total];

    let mut fit_cursor: u32 = 0;
    let mut overflow_cursor: u32 = 0;

    for k in 0..n_caps {
        let bucket_start = bucket_offsets[k] as usize;
        let n_k = (bucket_offsets[k + 1] - bucket_offsets[k]) as usize;
        let fit_n = n_k.min(bound);

        // Fits: padded slot index = k*bound + i_in_pad, fit slot in
        // densely-packed fits buffer increments by fit_cursor.
        for i in 0..fit_n {
            pad_perm[k * bound + i] = (bucket_start + i + 1) as u32;
            fit_layout.push((k * bound + i) as u32);
            combine_perm[bucket_start + i] = fit_cursor;
            fit_cursor += 1;
        }
        // Overflow: stays in cap order so the per-bucket loop sees a
        // contiguous slice of sorted indices.
        let ov_n = n_k - fit_n;
        for i in 0..ov_n {
            let sorted_pos = bucket_start + fit_n + i;
            overflow_indices.push(sorted_pos as u32);
        }
        overflow_bucket_offsets[k + 1] = overflow_bucket_offsets[k] + ov_n as u32;
        // The overflow tokens' combine_perm entries point to the
        // overflow buffer (offset by final n_fits, set below).
        for i in 0..ov_n {
            let sorted_pos = bucket_start + fit_n + i;
            combine_perm[sorted_pos] = overflow_cursor;
            overflow_cursor += 1;
        }
    }
    let n_fits = fit_cursor as usize;
    let n_overflow = overflow_cursor as usize;
    debug_assert_eq!(n_fits + n_overflow, total);
    // Now shift overflow entries in combine_perm by n_fits so they
    // address the concatenated [fits, overflow] buffer.
    for k in 0..n_caps {
        let bucket_start = bucket_offsets[k] as usize;
        let n_k = (bucket_offsets[k + 1] - bucket_offsets[k]) as usize;
        let fit_n = n_k.min(bound);
        for i in fit_n..n_k {
            combine_perm[bucket_start + i] += n_fits as u32;
        }
    }

    let perm_t = Tensor::from_vec(perm, (total,), device)?;
    let inv_perm_t = Tensor::from_vec(inv_perm, (total,), device)?;
    let pad_perm_t = Tensor::from_vec(pad_perm, (n_caps * bound,), device)?;
    let fit_layout_t = if n_fits == 0 {
        Tensor::zeros((0,), candle_core::DType::U32, device)?
    } else {
        Tensor::from_vec(fit_layout, (n_fits,), device)?
    };
    let overflow_indices_t = if n_overflow == 0 {
        None
    } else {
        Some(Tensor::from_vec(
            overflow_indices,
            (n_overflow,),
            device,
        )?)
    };
    let combine_perm_t = Tensor::from_vec(combine_perm, (total,), device)?;

    Ok(BoundedGroupedRouting {
        perm_t,
        inv_perm_t,
        bucket_offsets,
        bound,
        n_caps,
        total,
        pad_perm_t,
        n_fits,
        overflow_indices_t,
        overflow_bucket_offsets,
        n_overflow,
        combine_perm_t,
        fit_layout_t,
    })
}

/// Apply a per-cap projection `weights` (shape `(n_caps, d_in, d_out)`)
/// to `xs_sorted` (shape `(total, d_in)`) via bounded grouped dispatch.
/// Returns the result in sorted order, shape `(total, d_out)`.
///
/// Path A — batched: ONE matmul of `(n_caps, bound, d_in) @ (n_caps,
/// d_in, d_out)`. Covers every cap whose bucket fits within `bound`.
/// Path B — overflow per-bucket loop: only runs when at least one cap
/// exceeds `bound`. Handles the surplus rows via narrow + matmul, then
/// the results are concatenated and routed back into sorted positions
/// by a single `combine_perm` `index_select`.
pub fn apply_bounded_grouped_projection(
    xs_sorted: &Tensor,
    weights: &Tensor,
    routing: &BoundedGroupedRouting,
) -> Result<Tensor> {
    let (xs_total, d_in) = xs_sorted.dims2()?;
    let weights_dims = weights.dims3()?;
    let (n_caps_w, d_in_w, d_out) = weights_dims;
    debug_assert_eq!(xs_total, routing.total);
    debug_assert_eq!(n_caps_w, routing.n_caps);
    debug_assert_eq!(d_in_w, d_in);

    let device = xs_sorted.device();
    let dtype = xs_sorted.dtype();
    let zero_row = Tensor::zeros((1, d_in), dtype, device)?;
    let xs_with_zero = Tensor::cat(&[&zero_row, xs_sorted], 0)?;

    // ── Path A: batched fits ──
    let padded_flat = xs_with_zero.index_select(&routing.pad_perm_t, 0)?;
    let padded_3d = padded_flat.reshape((routing.n_caps, routing.bound, d_in))?;
    let padded_result_3d = padded_3d.matmul(weights)?;
    let padded_result_flat =
        padded_result_3d.reshape((routing.n_caps * routing.bound, d_out))?;
    // Extract just the fit slots into a dense (n_fits, d_out) buffer.
    let fits_buffer = if routing.n_fits == 0 {
        Tensor::zeros((0, d_out), dtype, device)?
    } else {
        padded_result_flat.index_select(&routing.fit_layout_t, 0)?
    };

    // ── Path B: overflow per-bucket loop ──
    let combined = if let Some(ov_indices_t) = routing.overflow_indices_t.as_ref() {
        let overflow_xs = xs_sorted.index_select(ov_indices_t, 0)?;
        let mut pieces: Vec<Tensor> = Vec::new();
        for k in 0..routing.n_caps {
            let ov_start = routing.overflow_bucket_offsets[k] as usize;
            let ov_n_k = (routing.overflow_bucket_offsets[k + 1]
                - routing.overflow_bucket_offsets[k]) as usize;
            if ov_n_k == 0 {
                continue;
            }
            let xs_k = overflow_xs.narrow(0, ov_start, ov_n_k)?;
            let w_k = weights.narrow(0, k, 1)?.squeeze(0)?;
            pieces.push(xs_k.matmul(&w_k)?);
        }
        let overflow_buffer = if pieces.is_empty() {
            Tensor::zeros((0, d_out), dtype, device)?
        } else {
            let refs: Vec<&Tensor> = pieces.iter().collect();
            Tensor::cat(&refs, 0)?
        };
        Tensor::cat(&[&fits_buffer, &overflow_buffer], 0)?
    } else {
        fits_buffer
    };

    // ── Combine: route every sorted position to its slot in `combined` ──
    combined.index_select(&routing.combine_perm_t, 0)
}

/// Build the full routing state needed for grouped cap-keyed dispatch.
pub fn sparse_grouped_routing(
    winners: &[u32],
    n_caps: usize,
    device: &Device,
) -> Result<SparseGroupedRouting> {
    let total = winners.len();
    let cap_max = if n_caps == 0 { 0 } else { (n_caps - 1) as u32 };

    let mut indexed: Vec<(u32, u32)> = winners
        .iter()
        .enumerate()
        .map(|(t, &w)| (w.min(cap_max), t as u32))
        .collect();
    indexed.sort_by_key(|&(w, _)| w);

    let perm: Vec<u32> = indexed.iter().map(|&(_, t)| t).collect();
    let mut inv_perm = vec![0u32; total];
    for (sorted_pos, &original_idx) in perm.iter().enumerate() {
        inv_perm[original_idx as usize] = sorted_pos as u32;
    }

    let mut bucket_offsets = vec![0u32; n_caps + 1];
    for &(w, _) in &indexed {
        bucket_offsets[w as usize + 1] += 1;
    }
    for k in 1..=n_caps {
        bucket_offsets[k] += bucket_offsets[k - 1];
    }

    let max_n_k = bucket_offsets
        .windows(2)
        .map(|w| (w[1] - w[0]) as usize)
        .max()
        .unwrap_or(0)
        .max(1); // ensure non-zero so padded tensor has a valid shape

    let mut pad_perm: Vec<u32> = vec![0; n_caps * max_n_k];
    let mut unpad_perm: Vec<u32> = vec![0; total];
    for k in 0..n_caps {
        let start = bucket_offsets[k] as usize;
        let n_k = (bucket_offsets[k + 1] - bucket_offsets[k]) as usize;
        for i in 0..n_k {
            pad_perm[k * max_n_k + i] = (start + i + 1) as u32;
            unpad_perm[start + i] = (k * max_n_k + i) as u32;
        }
    }

    Ok(SparseGroupedRouting {
        perm_t: Tensor::from_vec(perm, (total,), device)?,
        inv_perm_t: Tensor::from_vec(inv_perm, (total,), device)?,
        pad_perm_t: Tensor::from_vec(pad_perm, (n_caps * max_n_k,), device)?,
        unpad_perm_t: Tensor::from_vec(unpad_perm, (total,), device)?,
        bucket_offsets,
        max_n_k,
        total,
        n_caps,
    })
}

/// Apply a per-cap projection `weights` (shape `(n_caps, d_in, d_out)`)
/// to `xs_sorted` (shape `(total, d_in)`) using a precomputed grouped
/// routing state. Returns the projected tokens in sorted order, shape
/// `(total, d_out)`.
///
/// All work happens in **one** batched matmul plus two gathers — no
/// per-cap loop, so kernel launches on Metal/CUDA collapse from O(n_caps)
/// to O(1) per cap-keyed component.
pub fn apply_grouped_projection(
    xs_sorted: &Tensor,
    weights: &Tensor,
    routing: &SparseGroupedRouting,
) -> Result<Tensor> {
    let (xs_total, d_in) = xs_sorted.dims2()?;
    let weights_dims = weights.dims3()?;
    let (n_caps_w, d_in_w, d_out) = weights_dims;
    debug_assert_eq!(xs_total, routing.total);
    debug_assert_eq!(n_caps_w, routing.n_caps);
    debug_assert_eq!(d_in_w, d_in);

    // Prepend a zero row so pad_perm index 0 reads as the padding sentinel.
    let device = xs_sorted.device();
    let dtype = xs_sorted.dtype();
    let zero_row = Tensor::zeros((1, d_in), dtype, device)?;
    let xs_with_zero = Tensor::cat(&[&zero_row, xs_sorted], 0)?;

    // ONE gather builds the padded 3-D tensor.
    let padded_flat = xs_with_zero.index_select(&routing.pad_perm_t, 0)?;
    let padded_3d = padded_flat.reshape((routing.n_caps, routing.max_n_k, d_in))?;

    // ONE batched matmul: all n_caps projections in a single kernel.
    let result_3d = padded_3d.matmul(weights)?;
    let result_flat = result_3d.reshape((routing.n_caps * routing.max_n_k, d_out))?;

    // ONE gather pulls the real (un-padded) rows back into sorted order.
    result_flat.index_select(&routing.unpad_perm_t, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perm_groups_winners_contiguously() {
        let winners = vec![2u32, 0, 1, 0, 2, 1, 2];
        let device = Device::Cpu;
        let (perm_t, inv_perm_t, offsets) = sparse_routing_perm(&winners, 3, &device).unwrap();
        let perm: Vec<u32> = perm_t.to_vec1().unwrap();
        let inv: Vec<u32> = inv_perm_t.to_vec1().unwrap();

        // offsets: bucket 0 has 2, bucket 1 has 2, bucket 2 has 3
        assert_eq!(offsets, vec![0, 2, 4, 7]);

        // perm in sorted-by-winner order, stable within bucket
        // winner 0: tokens 1, 3; winner 1: tokens 2, 5; winner 2: tokens 0, 4, 6
        assert_eq!(perm, vec![1, 3, 2, 5, 0, 4, 6]);

        // inv_perm round-trip: perm[inv[t]] == t for all t
        for (t, &p) in inv.iter().enumerate() {
            assert_eq!(perm[p as usize], t as u32);
        }
    }

    #[test]
    fn handles_out_of_range_winner() {
        // winner == n_caps must clamp to n_caps-1 (defensive)
        let winners = vec![5u32, 0];
        let device = Device::Cpu;
        let (_, _, offsets) = sparse_routing_perm(&winners, 3, &device).unwrap();
        // 5 clamps to 2; bucket 0 has 1, buckets 1 has 0, bucket 2 has 1
        assert_eq!(offsets, vec![0, 1, 1, 2]);
    }

    #[test]
    fn empty_winners_yields_zero_offsets() {
        let winners: Vec<u32> = vec![];
        let device = Device::Cpu;
        let (_, _, offsets) = sparse_routing_perm(&winners, 4, &device).unwrap();
        assert_eq!(offsets, vec![0, 0, 0, 0, 0]);
    }

    /// Reference implementation matching the cap-keyed sparse paths in
    /// attention.rs/moe.rs/output.rs (per-bucket loop with sort+cat).
    fn reference_per_bucket(
        winners: &[u32],
        n_caps: usize,
        xs_flat: &Tensor,
        weights: &Tensor,
    ) -> Tensor {
        let total = winners.len();
        let d_out = weights.dims()[2];
        let device = xs_flat.device();
        let dtype = xs_flat.dtype();

        let (perm_t, inv_perm_t, offsets) =
            sparse_routing_perm(winners, n_caps, device).unwrap();
        let xs_sorted = xs_flat.index_select(&perm_t, 0).unwrap();
        let mut pieces = vec![];
        for k in 0..n_caps {
            let start = offsets[k] as usize;
            let n_k = (offsets[k + 1] - offsets[k]) as usize;
            if n_k == 0 {
                continue;
            }
            let xs_k = xs_sorted.narrow(0, start, n_k).unwrap();
            let w_k = weights.narrow(0, k, 1).unwrap().squeeze(0).unwrap();
            pieces.push(xs_k.matmul(&w_k).unwrap());
        }
        let result_sorted = if pieces.is_empty() {
            Tensor::zeros((total, d_out), dtype, device).unwrap()
        } else {
            let refs: Vec<&Tensor> = pieces.iter().collect();
            Tensor::cat(&refs, 0).unwrap()
        };
        result_sorted.index_select(&inv_perm_t, 0).unwrap()
    }

    fn assert_tensors_close(a: &Tensor, b: &Tensor, tol: f32, label: &str) {
        let av: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
        let bv: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(av.len(), bv.len(), "{}: size mismatch", label);
        let mut max_diff = 0f32;
        for (x, y) in av.iter().zip(bv.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(
            max_diff < tol,
            "{}: max diff {} exceeds tol {}",
            label,
            max_diff,
            tol
        );
    }

    #[test]
    fn bounded_grouped_matches_per_bucket_balanced() {
        // Balanced cap distribution — fits entirely within `bound`,
        // overflow path is empty.
        let device = Device::Cpu;
        let n_caps = 8;
        let total = 64;
        let d_in = 4;
        let d_out = 3;

        let winners: Vec<u32> = (0..total).map(|t| t % n_caps as u32).collect();
        let xs = Tensor::randn(0f32, 1.0, (total as usize, d_in), &device).unwrap();
        let weights = Tensor::randn(0f32, 1.0, (n_caps as usize, d_in, d_out), &device).unwrap();

        let routing = bounded_grouped_routing(&winners, n_caps as usize, &device).unwrap();
        assert_eq!(routing.n_overflow, 0, "balanced should have no overflow");

        let ref_out = reference_per_bucket(&winners, n_caps as usize, &xs, &weights);

        let inv_perm = &routing.inv_perm_t;
        let xs_sorted = xs.index_select(&routing.perm_t, 0).unwrap();
        let bounded_sorted =
            apply_bounded_grouped_projection(&xs_sorted, &weights, &routing).unwrap();
        let bounded_out = bounded_sorted.index_select(inv_perm, 0).unwrap();

        assert_tensors_close(&ref_out, &bounded_out, 1e-5, "balanced bounded vs per-bucket");
    }

    #[test]
    fn bounded_grouped_matches_per_bucket_skewed() {
        // Heavily skewed distribution — one cap captures most tokens,
        // forcing the overflow path to fire.
        let device = Device::Cpu;
        let n_caps = 6;
        let total = 200;
        let d_in = 5;
        let d_out = 7;

        // 80% of tokens go to cap 0; rest spread across others
        let winners: Vec<u32> = (0..total)
            .map(|t| if t < 160 { 0u32 } else { 1 + ((t - 160) % 5) as u32 })
            .collect();

        let xs = Tensor::randn(0f32, 1.0, (total as usize, d_in), &device).unwrap();
        let weights = Tensor::randn(0f32, 1.0, (n_caps as usize, d_in, d_out), &device).unwrap();

        let routing = bounded_grouped_routing(&winners, n_caps as usize, &device).unwrap();
        assert!(routing.n_overflow > 0, "skewed should trigger overflow");
        // sanity: bound capped at ceil(200/6)*4 = 136, max bucket has 160
        assert!(routing.bound < 160, "bound should be less than max bucket size");

        let ref_out = reference_per_bucket(&winners, n_caps as usize, &xs, &weights);

        let xs_sorted = xs.index_select(&routing.perm_t, 0).unwrap();
        let bounded_sorted =
            apply_bounded_grouped_projection(&xs_sorted, &weights, &routing).unwrap();
        let bounded_out = bounded_sorted.index_select(&routing.inv_perm_t, 0).unwrap();

        assert_tensors_close(&ref_out, &bounded_out, 1e-5, "skewed bounded vs per-bucket");
    }

    #[test]
    fn grouped_matches_per_bucket_loop() {
        // Parity test: grouped batched matmul must produce identical
        // results to the per-bucket loop within float tolerance.
        use candle_core::DType;
        let device = Device::Cpu;
        let n_caps = 5;
        let total = 23;
        let d_in = 8;
        let d_out = 4;

        // Random-ish winners
        let winners: Vec<u32> = (0..total).map(|t| (t as u32 * 7 + 3) % n_caps).collect();

        // Random inputs
        let xs_flat = Tensor::randn(0f32, 1.0, (total as usize, d_in), &device).unwrap();
        let weights = Tensor::randn(0f32, 1.0, (n_caps as usize, d_in, d_out), &device).unwrap();

        // Reference: per-bucket loop using existing sparse_routing_perm
        let (perm_t, inv_perm_t, offsets) =
            sparse_routing_perm(&winners, n_caps as usize, &device).unwrap();
        let xs_sorted = xs_flat.index_select(&perm_t, 0).unwrap();
        let mut pieces = vec![];
        for k in 0..(n_caps as usize) {
            let start = offsets[k] as usize;
            let n_k = (offsets[k + 1] - offsets[k]) as usize;
            if n_k == 0 {
                continue;
            }
            let xs_k = xs_sorted.narrow(0, start, n_k).unwrap();
            let w_k = weights.narrow(0, k, 1).unwrap().squeeze(0).unwrap();
            pieces.push(xs_k.matmul(&w_k).unwrap());
        }
        let refs: Vec<&Tensor> = pieces.iter().collect();
        let ref_sorted = Tensor::cat(&refs, 0).unwrap();
        let ref_flat = ref_sorted.index_select(&inv_perm_t, 0).unwrap();

        // Grouped: pad + batched matmul + unpad
        let routing = sparse_grouped_routing(&winners, n_caps as usize, &device).unwrap();
        let xs_sorted2 = xs_flat.index_select(&routing.perm_t, 0).unwrap();
        let grouped_sorted = apply_grouped_projection(&xs_sorted2, &weights, &routing).unwrap();
        let grouped_flat = grouped_sorted.index_select(&routing.inv_perm_t, 0).unwrap();

        let ref_vec: Vec<f32> = ref_flat
            .reshape((total as usize * d_out,))
            .unwrap()
            .to_vec1()
            .unwrap();
        let grp_vec: Vec<f32> = grouped_flat
            .reshape((total as usize * d_out,))
            .unwrap()
            .to_vec1()
            .unwrap();

        assert_eq!(ref_vec.len(), grp_vec.len());
        for (a, b) in ref_vec.iter().zip(grp_vec.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "grouped vs per-bucket mismatch: {} vs {}",
                a,
                b
            );
        }
        let _ = DType::F32; // silence unused-import lint on some toolchains
    }
}

/// Derive the sparse routing structure from cap activations.
///
/// This is the only place the routing decision leaves the device: the
/// argmax result must reach the CPU to build the permutation. Callers
/// that route several projections by the same `cap_acts` should call
/// this once and share the result. Recomputing it per projection returns
/// an identical structure - the argmax is deterministic over the same
/// tensor - at the cost of one device synchronisation each time.
pub fn routing_from_cap_acts(
    cap_acts: &Tensor,
    n_caps: usize,
    device: &Device,
) -> Result<BoundedGroupedRouting> {
    let dims = cap_acts.dims();
    let total: usize = dims[..dims.len() - 1].iter().product();
    let cap_flat = cap_acts.reshape((total, n_caps))?;
    let winners_t = cap_flat.argmax(candle_core::D::Minus1)?;
    let winners: Vec<u32> = winners_t.to_vec1::<u32>()?;
    bounded_grouped_routing(&winners, n_caps, device)
}
