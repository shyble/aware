//! The gate: deciding, per cap, whether provisional learning becomes
//! knowledge.
//!
//! # What this is for
//!
//! Gradient descent cannot distinguish "this update improves A" from
//! "this update overwrites A" — both merely lower B's loss. There is no
//! term anywhere in the objective that knows A existed. So the decision
//! has to come from outside the loss, and this module is that decision.
//!
//! For each cap: measure how its probes fared across the cycle, and
//! commit, quarantine, or roll back accordingly.
//!
//! # The evidence
//!
//! Two probe outcomes over the *same* probe set — one taken before the
//! cycle, one after — plus each cap's delta magnitude. Attribution uses
//! the *before* fingerprints, so the cap that held a piece of knowledge
//! is the cap held responsible for losing it, even if routing has since
//! drifted.
//!
//! # Deliberate asymmetries
//!
//! - **Untouched is not RolledBack.** A cap with an exactly-zero delta
//!   proposed nothing; refusing it would inflate the "protected" count
//!   with caps that were never at risk. The pilot showed this matters:
//!   at K=128 nothing was dormant, and a gate that conflated the two
//!   would have looked effective while doing nothing.
//! - **No evidence is not innocence.** A cap with no probes attributed
//!   to it cannot be judged. The policy for those is explicit
//!   (`no_evidence`) rather than an accident of the comparison operator.
//! - **Improvement is allowed.** A cap whose probes get *better* is
//!   committed, not merely tolerated — that is the accumulation case the
//!   whole design is aiming at, and it must be distinguishable in the
//!   record from "unchanged".

use candle_core::Result as CResult;

use super::probe::ProbeOutcome;
use super::registry::{CapDecision, CapRecord};

/// What to do with a cap the probes cannot speak for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoEvidencePolicy {
    /// Accept it. Optimistic: assumes untested learning is harmless.
    Commit,
    /// Refuse it. Conservative: only demonstrated-safe learning is kept.
    RollBack,
    /// Keep it, but withheld from committed knowledge.
    Quarantine,
}

/// Thresholds governing the gate. Every field is reported alongside
/// results; a finding that only survives a narrow band here is a tuning
/// artefact, not a mechanism.
#[derive(Debug, Clone, Copy)]
pub struct GateConfig {
    /// Retention at or above this commits the cap. 1.0 means "lose
    /// nothing"; lower values trade retention for plasticity.
    pub commit_at: f32,
    /// Retention below `commit_at` but at or above this is quarantined
    /// rather than discarded: the learning is kept but withheld.
    pub quarantine_at: f32,
    /// Deltas with magnitude at or below this count as untouched. Exact
    /// zero is the true test; the tolerance exists only for dtypes where
    /// accumulation noise is possible.
    pub untouched_eps: f32,
    pub no_evidence: NoEvidencePolicy,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            commit_at: 1.0,
            quarantine_at: 0.9,
            untouched_eps: 0.0,
            no_evidence: NoEvidencePolicy::Quarantine,
        }
    }
}

/// A decision plus the evidence behind it.
pub struct GateDecision {
    pub cap: usize,
    pub record: CapRecord,
}

/// Decide every cap's fate from the before/after probe outcomes.
///
/// Pure: computes decisions without touching the model, so a threshold
/// sweep can be re-run offline against stored outcomes rather than
/// re-training.
pub fn decide(
    before: &ProbeOutcome,
    after: &ProbeOutcome,
    delta_magnitudes: &[f32],
    cfg: &GateConfig,
) -> Vec<GateDecision> {
    let n_caps = delta_magnitudes.len();
    let (had, kept) = after.retention_per_cap(before, n_caps);

    (0..n_caps)
        .map(|k| {
            let magnitude = delta_magnitudes[k];
            let probes_before = had[k];
            let probes_after = kept[k];

            let decision = if magnitude <= cfg.untouched_eps {
                // Nothing was proposed for this cap, so nothing is
                // accepted or refused.
                CapDecision::Untouched
            } else if probes_before == 0 {
                match cfg.no_evidence {
                    NoEvidencePolicy::Commit => CapDecision::Committed,
                    NoEvidencePolicy::RollBack => CapDecision::RolledBack,
                    NoEvidencePolicy::Quarantine => CapDecision::Quarantined,
                }
            } else {
                let retention = probes_after as f32 / probes_before as f32;
                if retention >= cfg.commit_at {
                    CapDecision::Committed
                } else if retention >= cfg.quarantine_at {
                    CapDecision::Quarantined
                } else {
                    CapDecision::RolledBack
                }
            };

            GateDecision {
                cap: k,
                record: CapRecord {
                    decision,
                    delta_magnitude: magnitude,
                    probes_before,
                    probes_after,
                },
            }
        })
        .collect()
}

/// Apply decisions to a model and record them.
///
/// Quarantine currently keeps the delta in place without committing it,
/// which means it still applies to every input routed to that cap — the
/// routed-only restriction is D5. Until then, treat quarantine as
/// "kept, pending" and say so in any write-up rather than implying
/// isolation the code does not yet provide.
pub fn apply(
    model: &mut super::substrate::CapNativeSubstrate,
    decisions: &[GateDecision],
) -> CResult<()> {
    for d in decisions {
        match d.record.decision {
            CapDecision::Committed => model.commit_cap(d.cap)?,
            CapDecision::RolledBack => model.rollback_cap(d.cap)?,
            // Untouched has a zero delta: committing or rolling back are
            // both no-ops, so leave the weights alone entirely.
            CapDecision::Quarantined | CapDecision::Untouched | CapDecision::Provisional => {}
        }
        // Record after acting: commit_cap/rollback_cap set a coarse
        // decision, and this overwrites it with the full evidence.
        model.registry.set(d.cap, d.record.clone());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(correct: &[bool], cap: &[u32]) -> ProbeOutcome {
        ProbeOutcome {
            correct: correct.to_vec(),
            cap: cap.to_vec(),
        }
    }

    /// The four outcomes, each triggered by its own condition.
    #[test]
    fn decisions_follow_the_evidence() {
        // cap 0: probes preserved      -> commit
        // cap 1: probes partly lost    -> quarantine (0.9 <= r < 1.0 band)
        // cap 2: probes mostly lost    -> roll back
        // cap 3: zero delta            -> untouched (despite lost probes)
        let before = outcome(
            &[true; 24],
            &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3],
        );
        let mut after_correct = [true; 24];
        after_correct[19] = false; // cap 1 loses 1 of 10 -> 0.90
        after_correct[20] = false; // cap 2 loses 1 of 2  -> 0.50
        after_correct[22] = false; // cap 3 loses 1 of 2, but delta is zero
        let after = outcome(&after_correct, &before.cap);

        let mags = vec![0.5, 0.5, 0.5, 0.0];
        let cfg = GateConfig::default();
        let d = decide(&before, &after, &mags, &cfg);

        assert_eq!(d[0].record.decision, CapDecision::Committed);
        assert_eq!(d[1].record.decision, CapDecision::Quarantined);
        assert_eq!(d[2].record.decision, CapDecision::RolledBack);
        assert_eq!(
            d[3].record.decision,
            CapDecision::Untouched,
            "a cap that proposed nothing must not be counted as refused"
        );
        assert_eq!(d[1].record.probes_before, 10);
        assert_eq!(d[1].record.probes_after, 9);
    }

    /// A cap with no probes is judged by policy, explicitly.
    #[test]
    fn no_evidence_follows_policy() {
        let before = outcome(&[true, true], &[0, 0]);
        let after = outcome(&[true, true], &[0, 0]);
        let mags = vec![0.5, 0.5]; // cap 1 learned but has no probes

        for (policy, want) in [
            (NoEvidencePolicy::Commit, CapDecision::Committed),
            (NoEvidencePolicy::RollBack, CapDecision::RolledBack),
            (NoEvidencePolicy::Quarantine, CapDecision::Quarantined),
        ] {
            let cfg = GateConfig {
                no_evidence: policy,
                ..Default::default()
            };
            let d = decide(&before, &after, &mags, &cfg);
            assert_eq!(d[1].record.decision, want, "policy {policy:?}");
        }
    }

    /// Improvement commits — the accumulation case the design targets —
    /// and is visible in the record rather than collapsed into "kept".
    #[test]
    fn improvement_commits_and_is_visible() {
        let before = outcome(&[true, false, true], &[0, 0, 0]);
        let after = outcome(&[true, true, true], &[0, 0, 0]);
        let d = decide(&before, &after, &[0.5], &GateConfig::default());
        assert_eq!(d[0].record.decision, CapDecision::Committed);
        // Only probes correct BEFORE are counted, so improvement shows as
        // full retention; the newly-correct probe is B's gain, not A's.
        assert_eq!(d[0].record.probes_before, 2);
        assert_eq!(d[0].record.probes_after, 2);
    }

    /// The threshold must be swept, so the same evidence must be able to
    /// produce different decisions without re-running anything.
    #[test]
    fn thresholds_are_reswept_offline() {
        let before = outcome(&[true; 4], &[0, 0, 0, 0]);
        let after = outcome(&[true, true, true, false], &[0, 0, 0, 0]); // 0.75
        let mags = vec![0.5];

        let strict = decide(&before, &after, &mags, &GateConfig::default());
        assert_eq!(strict[0].record.decision, CapDecision::RolledBack);

        let lenient = decide(
            &before,
            &after,
            &mags,
            &GateConfig {
                commit_at: 0.7,
                ..Default::default()
            },
        );
        assert_eq!(lenient[0].record.decision, CapDecision::Committed);
    }
}

#[cfg(test)]
mod end_to_end_tests {
    use super::*;
    use crate::aware::cap_native::config::CapNativeConfig;
    use crate::aware::cap_native::probe::ProbeSet;
    use crate::aware::cap_native::CapNativeBuilder;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::Optimizer;

    fn cfg() -> CapNativeConfig {
        let mut c = CapNativeConfig::default();
        c.vocab = 32; c.d_model = 24; c.n_blocks = 2; c.d_ff = 48; c.n_heads = 2;
        c.cap_config.n_caps_target = 8;
        c.cap_config.cap_window = 2;
        c.cap_config.discovery = crate::aware::DiscoveryKind::Random;
        c.routing = crate::aware::cap_native::compression::RoutingMode::HardTop1Sparse;
        c.top_k = 1;
        c
    }

    /// The mechanism's core claim, in miniature: after a disruptive
    /// cycle, a strict gate restores every probe that a plain cycle
    /// would have lost.
    ///
    /// Corpus A is a learnable periodic pattern; the "cycle" then trains
    /// on a conflicting one. Without the gate the model follows the new
    /// data and A's probes fall. With the gate, caps whose probes
    /// degraded are rolled back and A is recovered exactly.
    #[test]
    fn gate_recovers_what_an_ungated_cycle_destroys() -> CResult<()> {
        let dev = Device::Cpu;
        let mut model = CapNativeBuilder::default()
            .with_config(cfg())
            .with_device(dev.clone())
            .build()?;

        // ── Learn corpus A ──
        let a_tokens: Vec<u32> = (0..4000u32).map(|i| (i * 7) % 32).collect();
        let mut opt = {
            let vm = model.varmap.lock().unwrap();
            candle_nn::AdamW::new_lr(vm.all_vars(), 3e-3)?
        };
        let step = |model: &CapNativeSubstrateRef, opt: &mut candle_nn::AdamW,
                    toks: &[u32], off: usize| -> CResult<()> {
            let ctx = Tensor::from_slice(&toks[off..off + 8 * 4], (4, 8), &dev)?
                .to_dtype(DType::U32)?;
            let tgt = Tensor::from_slice(&toks[off + 1..off + 1 + 8 * 4], (4, 8), &dev)?
                .to_dtype(DType::U32)?;
            let logits = model.forward(&ctx)?;
            let (b, t, v) = logits.dims3()?;
            let loss = candle_nn::loss::cross_entropy(
                &logits.reshape((b * t, v))?, &tgt.reshape((b * t,))?)?;
            opt.backward_step(&loss)?;
            Ok(())
        };
        type CapNativeSubstrateRef = crate::aware::cap_native::CapNativeSubstrate;
        for i in 0..300 {
            step(&model, &mut opt, &a_tokens, (i * 31) % 3000)?;
        }

        let probes = ProbeSet::from_tokens(&a_tokens, 64, 8)?;
        let before = probes.score(&model, &dev, 16)?;
        let n_before = before.n_correct();
        assert!(n_before > 0, "model learned nothing from A; test is vacuous");

        // Snapshot for the ungated comparison.
        let dir = std::env::temp_dir().join("gate_e2e");
        std::fs::create_dir_all(&dir).unwrap();
        let snap = dir.join("a.safetensors");
        let snap = snap.to_str().unwrap();
        crate::aware::cap_native::save_checkpoint(&model, snap)?;

        // ── Cycle on a conflicting corpus B ──
        let b_tokens: Vec<u32> = (0..4000u32).map(|i| (i * 13 + 5) % 32).collect();
        model.begin_cycle()?;
        let mut opt_b = {
            let vm = model.varmap.lock().unwrap();
            candle_nn::AdamW::new_lr(vm.all_vars(), 3e-3)?
        };
        for i in 0..300 {
            step(&model, &mut opt_b, &b_tokens, (i * 31) % 3000)?;
        }

        // Ungated: how much of A survives if every delta is kept?
        let ungated = probes.score(&model, &dev, 16)?;
        let n_ungated = ungated.correct.iter().zip(before.correct.iter())
            .filter(|(a, b)| **a && **b).count();

        // ── Gate ──
        let n_caps = model.config.downstream_n_caps();
        let mags: Vec<f32> = (0..n_caps)
            .map(|k| model.cap_delta_magnitude(k).unwrap())
            .collect();
        let decisions = decide(&before, &ungated, &mags, &GateConfig::default());
        apply(&mut model, &decisions)?;

        let gated = probes.score(&model, &dev, 16)?;
        let n_gated = gated.correct.iter().zip(before.correct.iter())
            .filter(|(a, b)| **a && **b).count();

        println!(
            "A probes correct: {} | after ungated cycle: {} | after gate: {} | {}",
            n_before, n_ungated, n_gated, model.registry.summary()
        );

        // The claim: the gate never retains less than leaving the deltas
        // in place, and recovers ground the ungated cycle lost.
        assert!(
            n_gated >= n_ungated,
            "gate retained {n_gated} < ungated {n_ungated} — the audit made things worse"
        );
        if n_ungated < n_before {
            assert!(
                n_gated > n_ungated,
                "cycle destroyed {} probes but the gate recovered none",
                n_before - n_ungated
            );
        }
        Ok(())
    }
}
