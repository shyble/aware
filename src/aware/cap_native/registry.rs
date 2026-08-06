//! Per-cap decision registry: what happened to each cap's provisional
//! learning, and why.
//!
//! The commit/rollback operations in `slab` change weights. This records
//! the *decisions* — which cap was committed, which rolled back, which
//! quarantined, and the evidence behind each. Three reasons that record
//! has to exist rather than being implied by the weights:
//!
//! 1. **Rollback is invisible afterwards.** A rolled-back cap is
//!    byte-identical to one that never fired. Without a record, "the
//!    mechanism protected 40 caps" and "40 caps saw no data" are
//!    indistinguishable — and they mean opposite things.
//! 2. **The decision is the result.** The experiment's headline is not
//!    only the retention number but the commit/rollback distribution: a
//!    gate that commits everything is doing nothing, and a gate that
//!    rolls back everything has bought non-interference by refusing to
//!    learn. Both look like "it ran fine" from the weights alone.
//! 3. **Thresholds must be reportable.** The score behind each decision
//!    is kept so a threshold sweep can be re-derived offline without
//!    re-running training.
//!
//! The registry persists in the checkpoint sidecar, so decisions survive
//! save/load along with cap identity.

use serde::{Deserialize, Serialize};

/// What was decided about one cap's provisional learning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapDecision {
    /// No cycle has run, or the cap's delta is still open.
    Provisional,
    /// Delta folded into the base: this learning is now knowledge.
    Committed,
    /// Delta discarded; the cap is exactly as it was before the cycle.
    RolledBack,
    /// Delta kept but withheld from committed knowledge — applied only
    /// to inputs routed through this cycle's own caps.
    Quarantined,
    /// The cap received no gradient at all this cycle (delta exactly
    /// zero). Distinct from RolledBack: nothing was proposed, so nothing
    /// was refused.
    Untouched,
}

/// One cap's record for one cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapRecord {
    pub decision: CapDecision,
    /// Largest absolute delta value across the cap's slabs — how much
    /// this cycle wanted to change it. Exactly 0.0 means untouched.
    pub delta_magnitude: f32,
    /// Probe items attributed to this cap that were correct before the
    /// cycle and after it. The audit's evidence, retained so a
    /// threshold sweep can be redone offline.
    pub probes_before: u32,
    pub probes_after: u32,
}

impl Default for CapRecord {
    fn default() -> Self {
        Self {
            decision: CapDecision::Provisional,
            delta_magnitude: 0.0,
            probes_before: 0,
            probes_after: 0,
        }
    }
}

impl CapRecord {
    /// Fraction of this cap's probes that survived the cycle. `None`
    /// when the cap had no probes attributed to it — an important case,
    /// because a cap with no evidence cannot be judged and the policy
    /// for it must be explicit rather than accidental.
    pub fn retention(&self) -> Option<f32> {
        if self.probes_before == 0 {
            None
        } else {
            Some(self.probes_after as f32 / self.probes_before as f32)
        }
    }
}

/// Decisions for every cap over one cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitRegistry {
    pub cycle: u32,
    pub records: Vec<CapRecord>,
}

impl CommitRegistry {
    pub fn new(n_caps: usize) -> Self {
        Self {
            cycle: 0,
            records: vec![CapRecord::default(); n_caps],
        }
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Start a new cycle: bump the counter and reset every record.
    pub fn begin_cycle(&mut self) {
        self.cycle += 1;
        for r in &mut self.records {
            *r = CapRecord::default();
        }
    }

    pub fn set(&mut self, k: usize, record: CapRecord) {
        if k < self.records.len() {
            self.records[k] = record;
        }
    }

    pub fn decision(&self, k: usize) -> CapDecision {
        self.records
            .get(k)
            .map(|r| r.decision)
            .unwrap_or(CapDecision::Provisional)
    }

    pub fn count(&self, d: CapDecision) -> usize {
        self.records.iter().filter(|r| r.decision == d).count()
    }

    /// Caps whose deltas are live in the forward pass: committed folds
    /// into the base, quarantined stays in the delta and still applies
    /// to its own routed inputs.
    pub fn quarantined(&self) -> Vec<usize> {
        self.records
            .iter()
            .enumerate()
            .filter(|(_, r)| r.decision == CapDecision::Quarantined)
            .map(|(k, _)| k)
            .collect()
    }

    /// One-line summary for run logs and reports.
    pub fn summary(&self) -> String {
        format!(
            "cycle {}: {} committed, {} rolled back, {} quarantined, {} untouched (of {})",
            self.cycle,
            self.count(CapDecision::Committed),
            self.count(CapDecision::RolledBack),
            self.count(CapDecision::Quarantined),
            self.count(CapDecision::Untouched),
            self.records.len()
        )
    }

    /// Overall probe retention across every cap that had evidence.
    /// Weighted by probe count, so caps carrying more of A's knowledge
    /// count for more.
    pub fn overall_retention(&self) -> Option<f32> {
        let before: u32 = self.records.iter().map(|r| r.probes_before).sum();
        let after: u32 = self.records.iter().map(|r| r.probes_after).sum();
        if before == 0 {
            None
        } else {
            Some(after as f32 / before as f32)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_and_summary_reflect_decisions() {
        let mut reg = CommitRegistry::new(4);
        reg.begin_cycle();
        reg.set(0, CapRecord { decision: CapDecision::Committed, delta_magnitude: 0.4, probes_before: 10, probes_after: 10 });
        reg.set(1, CapRecord { decision: CapDecision::RolledBack, delta_magnitude: 0.9, probes_before: 8, probes_after: 3 });
        reg.set(2, CapRecord { decision: CapDecision::Quarantined, delta_magnitude: 0.5, probes_before: 6, probes_after: 4 });
        reg.set(3, CapRecord { decision: CapDecision::Untouched, ..Default::default() });

        assert_eq!(reg.count(CapDecision::Committed), 1);
        assert_eq!(reg.count(CapDecision::RolledBack), 1);
        assert_eq!(reg.quarantined(), vec![2]);
        assert_eq!(reg.cycle, 1);
        assert!(reg.summary().contains("1 committed"));
    }

    /// A cap with no probes cannot be judged; that must surface as None
    /// rather than as a misleading 0% or 100%.
    #[test]
    fn caps_without_evidence_report_none() {
        let r = CapRecord::default();
        assert!(r.retention().is_none());
        let r2 = CapRecord { probes_before: 4, probes_after: 3, ..Default::default() };
        assert_eq!(r2.retention(), Some(0.75));
    }

    #[test]
    fn overall_retention_is_probe_weighted() {
        let mut reg = CommitRegistry::new(2);
        // A cap carrying 90 probes should dominate one carrying 10.
        reg.set(0, CapRecord { probes_before: 90, probes_after: 90, ..Default::default() });
        reg.set(1, CapRecord { probes_before: 10, probes_after: 0, ..Default::default() });
        assert_eq!(reg.overall_retention(), Some(0.9));
    }

    #[test]
    fn begin_cycle_resets_records_and_bumps_counter() {
        let mut reg = CommitRegistry::new(2);
        reg.set(0, CapRecord { decision: CapDecision::Committed, probes_before: 5, probes_after: 5, ..Default::default() });
        reg.begin_cycle();
        assert_eq!(reg.cycle, 1);
        assert_eq!(reg.decision(0), CapDecision::Provisional);
        assert_eq!(reg.records[0].probes_before, 0);
    }
}
