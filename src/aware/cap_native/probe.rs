//! Probe sets: what a corpus must still know, and which caps know it.
//!
//! # Why not perplexity
//!
//! Aggregate perplexity can stay flat while specific capability is gone,
//! or degrade while the model still works. It answers "how surprised is
//! the model overall", not "what does it still know". For continual
//! learning the second question is the one that matters, and it needs a
//! per-item measure.
//!
//! A probe set is a fixed sample of held-out positions from corpus A.
//! Before a cycle we record which the model predicts correctly — call
//! that set `R0`. After the cycle we re-test only those. The surviving
//! fraction is retention: a capability measure, fully automatic, no
//! generation scoring and no human judgement.
//!
//! # Why it is cap-shaped
//!
//! Each probe also carries a **fingerprint**: the cap that fires when
//! the model processes it. That turns retention from an aggregate number
//! into an attribution — for every cap we know how many of its probes
//! survived, which is precisely the evidence the audit needs to decide
//! whether that cap's provisional learning should become knowledge.
//!
//! A dense model can produce the retention number. It cannot produce the
//! attribution, because there is no unit to attribute to.
//!
//! # Determinism
//!
//! The probe set is drawn once from a seeded feeder and then *stored as
//! token ids*, not regenerated. Re-deriving it later from a feeder would
//! silently depend on batch order, seq_len and RNG state; storing it
//! means before/after comparisons are over literally the same positions.

use candle_core::{DType, Device, Result as CResult, Tensor};
use serde::{Deserialize, Serialize};

use super::substrate::CapNativeSubstrate;

/// A fixed set of held-out sequences with their expected next tokens.
#[derive(Clone, Serialize, Deserialize)]
pub struct ProbeSet {
    /// Flattened `[n_probes, seq_len]` context token ids.
    pub contexts: Vec<u32>,
    /// Expected next token for each probe's final position.
    pub targets: Vec<u32>,
    pub seq_len: usize,
}

/// Which probes the model answered correctly, and which cap fired for
/// each — the evidence a later cycle is judged against.
#[derive(Clone, Serialize, Deserialize)]
pub struct ProbeOutcome {
    /// Per probe: was the argmax prediction the expected token?
    pub correct: Vec<bool>,
    /// Per probe: the routing cap that fired at the scored position.
    pub cap: Vec<u32>,
}

impl ProbeSet {
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// Draw a probe set from a token stream. Positions are strided
    /// rather than random so coverage is spread across the corpus and
    /// the set is reproducible from `(path, n_probes, seq_len)` alone.
    pub fn from_tokens(tokens: &[u32], n_probes: usize, seq_len: usize) -> CResult<Self> {
        let need = seq_len + 1;
        if tokens.len() < need + 1 {
            return Err(candle_core::Error::Msg(
                "probe corpus shorter than one probe".into(),
            ));
        }
        let usable = tokens.len() - need;
        let n = n_probes.min(usable);
        let stride = (usable / n).max(1);

        let mut contexts = Vec::with_capacity(n * seq_len);
        let mut targets = Vec::with_capacity(n);
        for i in 0..n {
            let start = i * stride;
            contexts.extend_from_slice(&tokens[start..start + seq_len]);
            targets.push(tokens[start + seq_len]);
        }
        Ok(Self {
            contexts,
            targets,
            seq_len,
        })
    }

    /// Read a `.bin` token cache and draw a probe set from it.
    pub fn from_bin(path: &str, n_probes: usize, seq_len: usize) -> CResult<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| candle_core::Error::Msg(format!("probe corpus {path}: {e}")))?;
        let tokens: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Self::from_tokens(&tokens, n_probes, seq_len)
    }

    /// Score the probe set: which items the model gets right, and which
    /// cap fires for each.
    ///
    /// Batched, and every intermediate stays on the device — one host
    /// transfer per batch for the two small result vectors, never per
    /// item. Only the final position of each context is scored, which is
    /// the position whose prediction the target defines.
    pub fn score(
        &self,
        model: &CapNativeSubstrate,
        device: &Device,
        batch_size: usize,
    ) -> CResult<ProbeOutcome> {
        let n = self.len();
        let mut correct = Vec::with_capacity(n);
        let mut cap = Vec::with_capacity(n);

        for start in (0..n).step_by(batch_size) {
            let b = batch_size.min(n - start);
            let ctx = Tensor::from_slice(
                &self.contexts[start * self.seq_len..(start + b) * self.seq_len],
                (b, self.seq_len),
                device,
            )?
            .to_dtype(DType::U32)?;

            let logits = model.forward(&ctx)?; // (b, seq_len, vocab)
            let last = logits.narrow(1, self.seq_len - 1, 1)?.squeeze(1)?;
            let pred: Vec<u32> = last.argmax(candle_core::D::Minus1)?.to_vec1()?;

            // Routing cap at the scored position. In hierarchical models
            // this is layer 1 — the layer that actually keys the
            // downstream slabs, so it is the one an audit can act on.
            let (l0, l1) = model.cap_winners(&ctx)?;
            let winners = l1.unwrap_or(l0);
            let w_last: Vec<u32> = winners
                .narrow(1, self.seq_len - 1, 1)?
                .squeeze(1)?
                .to_vec1()?;

            for i in 0..b {
                correct.push(pred[i] == self.targets[start + i]);
                cap.push(w_last[i]);
            }
        }
        Ok(ProbeOutcome { correct, cap })
    }
}

impl ProbeOutcome {
    pub fn n_correct(&self) -> usize {
        self.correct.iter().filter(|c| **c).count()
    }

    /// Per-cap counts of probes that were correct in `self`.
    pub fn correct_per_cap(&self, n_caps: usize) -> Vec<u32> {
        let mut out = vec![0u32; n_caps];
        for (ok, &k) in self.correct.iter().zip(self.cap.iter()) {
            if *ok && (k as usize) < n_caps {
                out[k as usize] += 1;
            }
        }
        out
    }

    /// Retention of `self` relative to an earlier outcome: of the probes
    /// correct *before*, how many are still correct, counted per cap.
    ///
    /// Attribution uses the BEFORE fingerprint deliberately. A probe's
    /// cap can change during a cycle (the pilot measured 10–20% routing
    /// drift), and the question the audit asks is "did the cap that used
    /// to answer this still answer it" — so the cap that held the
    /// knowledge is the one held responsible.
    pub fn retention_per_cap(&self, before: &ProbeOutcome, n_caps: usize) -> (Vec<u32>, Vec<u32>) {
        let mut had = vec![0u32; n_caps];
        let mut kept = vec![0u32; n_caps];
        for i in 0..before.correct.len().min(self.correct.len()) {
            if !before.correct[i] {
                continue;
            }
            let k = before.cap[i] as usize;
            if k >= n_caps {
                continue;
            }
            had[k] += 1;
            if self.correct[i] {
                kept[k] += 1;
            }
        }
        (had, kept)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aware::cap_native::config::CapNativeConfig;
    use crate::aware::cap_native::CapNativeBuilder;

    #[test]
    fn probes_are_strided_and_well_formed() -> CResult<()> {
        let tokens: Vec<u32> = (0..1000u32).collect();
        let ps = ProbeSet::from_tokens(&tokens, 10, 8)?;
        assert_eq!(ps.len(), 10);
        assert_eq!(ps.contexts.len(), 80);
        // First probe: context 0..8, target 8.
        assert_eq!(&ps.contexts[0..8], &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(ps.targets[0], 8);
        // Strided, so the second probe starts well past the first.
        assert!(ps.contexts[8] > 8);
        Ok(())
    }

    /// Scoring must be deterministic and must attribute every probe to a
    /// cap — the audit has no evidence otherwise.
    #[test]
    fn scoring_is_deterministic_and_attributed() -> CResult<()> {
        let dev = Device::Cpu;
        let mut cfg = CapNativeConfig::default();
        cfg.vocab = 32; cfg.d_model = 16; cfg.n_blocks = 1; cfg.d_ff = 32; cfg.n_heads = 2;
        cfg.cap_config.n_caps_target = 6;
        cfg.cap_config.cap_window = 2;
        cfg.cap_config.discovery = crate::aware::DiscoveryKind::Random;
        cfg.routing = crate::aware::cap_native::compression::RoutingMode::HardTop1Sparse;
        cfg.top_k = 1;
        let model = CapNativeBuilder::default()
            .with_config(cfg)
            .with_device(dev.clone())
            .build()?;

        let tokens: Vec<u32> = (0..800u32).map(|i| i % 32).collect();
        let ps = ProbeSet::from_tokens(&tokens, 16, 8)?;

        let a = ps.score(&model, &dev, 4)?;
        let b = ps.score(&model, &dev, 4)?;
        assert_eq!(a.correct, b.correct, "scoring is not deterministic");
        assert_eq!(a.cap, b.cap, "fingerprints are not deterministic");
        assert_eq!(a.cap.len(), ps.len(), "every probe must have a cap");
        assert!(a.cap.iter().all(|&k| (k as usize) < 6));
        Ok(())
    }

    /// Retention counts only probes that were correct BEFORE, and
    /// attributes them to the cap that held them at that time.
    #[test]
    fn retention_uses_before_fingerprints() {
        let before = ProbeOutcome {
            correct: vec![true, true, false, true],
            cap: vec![0, 1, 1, 0],
        };
        // Probe 3 was lost; probe 1's cap drifted but it is still
        // credited to cap 1, which is where the knowledge lived.
        let after = ProbeOutcome {
            correct: vec![true, true, true, false],
            cap: vec![0, 2, 1, 0],
        };
        let (had, kept) = after.retention_per_cap(&before, 3);
        assert_eq!(had, vec![2, 1, 0]);
        assert_eq!(kept, vec![1, 1, 0]);
    }
}
