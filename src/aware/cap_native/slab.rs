//! Committed base / provisional delta split for cap-keyed weights.
//!
//! # Why
//!
//! Continual learning here is not "train more carefully", it is "decide,
//! per cap, whether new learning becomes knowledge". That decision is
//! only expressible if learning is *separable* from knowledge. So a
//! cap-keyed tensor can be split into:
//!
//! - **base** — committed knowledge, frozen for the duration of a cycle
//! - **delta** — provisional learning, the only thing gradients touch
//!
//! with effective weight `base + delta`. After a cycle, each cap's delta
//! is audited and either folded into its base (commit) or discarded
//! (rollback).
//!
//! # Shape of the implementation
//!
//! Deliberately *not* a wrapper type that owns the weight. Each
//! component keeps the `Var` it already had — still under the same
//! varmap name, so optimizers, checkpointing, growth and shrink all keep
//! working untouched — and gains one `Option<Tensor>` beside it:
//!
//! - `base == None`  → ordinary training. The `Var` is the weight.
//!   Bit-identical and allocation-identical to the previous behaviour.
//! - `base == Some(w)` → a cycle is running. The `Var` now holds the
//!   *delta*, and the forward pass uses `base + delta`.
//!
//! Splitting is exact: the delta starts at zero, and adding zero is
//! exact in floating point rather than merely close. A continual-learning
//! run therefore reproduces its parent checkpoint bit-for-bit before the
//! first gradient step — the acceptance test this design rests on.

use candle_core::{DType, Result as CResult, Tensor, Var};

/// Begin a cycle: the current weight becomes the frozen base, and the
/// `Var` is zeroed to become the provisional delta.
///
/// Returns the base for the caller to store. The `Var` keeps its varmap
/// registration, so nothing else in the stack needs to know.
pub fn split(w: &Var) -> CResult<Tensor> {
    // A deep copy, not a clone. `Var::set` writes in place, and a cloned
    // tensor shares the Var's storage — so a shallow copy here would be
    // silently zeroed along with the delta on the very next line, and
    // every base in the model would be lost without any error.
    let base = w.as_tensor().copy()?;
    let zeros = Tensor::zeros(base.dims(), base.dtype(), base.device())?;
    w.set(&zeros)?;
    Ok(base)
}

/// The weight the forward pass should use.
///
/// When unsplit this is a *view* sharing the `Var`'s storage — free, and
/// correct for a forward pass, which consumes it immediately. Do not hold
/// the result across anything that mutates the `Var` (`set`, `split`,
/// `commit_cap`, `rollback_cap`): in-place writes would be visible
/// through it. Call `.copy()` if you need a snapshot.
pub fn effective(base: &Option<Tensor>, w: &Var) -> CResult<Tensor> {
    match base {
        None => Ok(w.as_tensor().clone()),
        Some(b) => b + w.as_tensor(),
    }
}

/// Fold cap `k`'s delta into its base and zero that delta row: the cap's
/// provisional learning becomes committed knowledge. Every other cap is
/// left byte-identical, which is what makes audit decisions independent
/// per cap.
pub fn commit_cap(base: &mut Option<Tensor>, w: &Var, k: usize) -> CResult<()> {
    let Some(b) = base.as_ref() else { return Ok(()) };
    let merged_row = (b.narrow(0, k, 1)? + w.as_tensor().narrow(0, k, 1)?)?;
    *base = Some(splice_row(b, k, &merged_row)?);
    zero_row(w, k)
}

/// Discard cap `k`'s provisional learning; its base is untouched, so the
/// cap returns to exactly its pre-cycle state.
pub fn rollback_cap(w: &Var, k: usize) -> CResult<()> {
    zero_row(w, k)
}

/// Largest absolute value in cap `k`'s delta — how much this cycle wants
/// to change that cap. Exactly zero means the cap never fired.
pub fn delta_magnitude(w: &Var, k: usize) -> CResult<f32> {
    w.as_tensor()
        .narrow(0, k, 1)?
        .abs()?
        .max_all()?
        .to_dtype(DType::F32)?
        .to_scalar::<f32>()
}

fn zero_row(w: &Var, k: usize) -> CResult<()> {
    let row = w.as_tensor().narrow(0, k, 1)?;
    let zeros = Tensor::zeros(row.dims(), row.dtype(), row.device())?;
    let updated = splice_row(w.as_tensor(), k, &zeros)?;
    w.set(&updated)
}

/// Replace row `k` of `t`, leaving every other row untouched.
fn splice_row(t: &Tensor, k: usize, row: &Tensor) -> CResult<Tensor> {
    let n = t.dim(0)?;
    let pieces: Vec<Tensor> = if n == 1 {
        vec![row.clone()]
    } else if k == 0 {
        vec![row.clone(), t.narrow(0, 1, n - 1)?]
    } else if k == n - 1 {
        vec![t.narrow(0, 0, k)?, row.clone()]
    } else {
        vec![
            t.narrow(0, 0, k)?,
            row.clone(),
            t.narrow(0, k + 1, n - k - 1)?,
        ]
    };
    let refs: Vec<&Tensor> = pieces.iter().collect();
    Tensor::cat(&refs, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Splitting must not change what the model computes. Zero added to
    /// the base is exact, not approximate — the invariant the whole
    /// design rests on.
    #[test]
    fn split_preserves_effective_weight_exactly() -> CResult<()> {
        let dev = Device::Cpu;
        let init = Tensor::randn(0f32, 1f32, (5, 4, 6), &dev)?;
        let w = Var::from_tensor(&init)?;

        // Snapshot: `effective` aliases the Var when unsplit, and the
        // split below mutates it in place.
        let before = effective(&None, &w)?.copy()?;
        let base = Some(split(&w)?);
        let after = effective(&base, &w)?;

        let diff = (before - after)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(diff, 0.0, "split changed the effective weight by {diff}");
        // The Var is now a pure delta.
        assert_eq!(w.as_tensor().abs()?.max_all()?.to_scalar::<f32>()?, 0.0);
        Ok(())
    }

    /// Commit affects one cap only; rollback restores a cap exactly.
    #[test]
    fn commit_and_rollback_are_per_cap_and_exact() -> CResult<()> {
        let dev = Device::Cpu;
        let init = Tensor::randn(0f32, 1f32, (4, 3, 3), &dev)?;
        let w = Var::from_tensor(&init)?;
        let mut base = Some(split(&w)?);

        // A cycle writes into every cap's delta.
        let learned = Tensor::randn(0f32, 1f32, (4, 3, 3), &dev)?;
        w.set(&learned)?;

        commit_cap(&mut base, &w, 1)?;
        let b = base.as_ref().unwrap();
        for k in 0..4 {
            let want = if k == 1 {
                (init.narrow(0, k, 1)? + learned.narrow(0, k, 1)?)?
            } else {
                init.narrow(0, k, 1)?
            };
            let diff = (b.narrow(0, k, 1)? - want)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(diff < 1e-6, "cap {k} base wrong after commit: {diff}");
        }
        assert_eq!(delta_magnitude(&w, 1)?, 0.0, "committed delta not cleared");
        assert!(delta_magnitude(&w, 2)? > 0.0, "other caps disturbed");

        rollback_cap(&w, 2)?;
        let eff = effective(&base, &w)?;
        let diff = (eff.narrow(0, 2, 1)? - init.narrow(0, 2, 1)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-6, "rollback did not restore cap 2: {diff}");
        Ok(())
    }

    /// Unsplit slabs behave exactly as before: the Var is the weight.
    #[test]
    fn unsplit_is_passthrough() -> CResult<()> {
        let dev = Device::Cpu;
        let init = Tensor::randn(0f32, 1f32, (3, 2, 2), &dev)?;
        let w = Var::from_tensor(&init)?;
        let eff = effective(&None, &w)?;
        let diff = (eff - init)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(diff, 0.0);
        Ok(())
    }
}
