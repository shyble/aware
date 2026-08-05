# Continual learning through stable cap identity

Design and tracking document. Branch `continual-learning`.
Started 2026-08-06.

## The question

Both reported architectures claim identifiability as their contribution,
and both admit in their limitations that no experiment demonstrates a
capability a conventional transformer structurally *cannot* provide.
Audit, replacement and growth are implemented and motivate the design;
nothing measures them.

This is that experiment. The question is narrow and falsifiable:

> After training on corpus A and then corpus B, does what the model
> retains from A depend on **which caps** B disturbed?

A dense transformer can be measured for forgetting. It cannot be measured
for *this*, because there is no addressable unit to attribute forgetting
to. That asymmetry is the whole contribution.

## The prediction

Cap identity gives a quantity computable **before** the second training
run: the overlap between the caps A fires and the caps B fires. The claim:

> Retention of an item from A is predictable from whether that item's caps
> were disturbed while training on B.

Falsifiable three ways. It fails if retention is uncorrelated with cap
overlap; if cap-native forgets no differently than a dense baseline; or
if cap identity is not stable enough for "the same cap" to mean anything
across a retraining cycle. The third is the real risk — see Risks.

## Independent variable: domain distance

Not one retraining scenario but a graded ladder. The architecture should
respond differently along it, and that response is the result.

| # | distance | B corpus | expectation |
|---|---|---|---|
| 1 | same distribution | held-out WikiText slice | same caps fire; audit inert; no forgetting |
| 2 | continuation | later WikiText articles | high overlap; slight drift |
| 3 | near domain | different register (Simple English) | partial overlap; some recruitment |
| 4 | far domain | TinyStories | low overlap; audit recruits dormant caps |
| 5 | foreign | source code | near-disjoint firing; heavy recruitment |

Distances 1, 2 and 4 need no new data. 3 and 5 need fetch scripts.

## Measurement

Perplexity is the wrong instrument. It can stay flat while specific
capability is gone, or degrade while the model still works. What is needed
is **what survived**, per item.

**Probe-set retention.** Fix ~5000 held-out sequences from A.

1. Before training on B, record which items the model predicts correctly
   (top-1 next token). Call this set `R0`.
2. After training on B, re-test only `R0`. The surviving fraction is the
   retention rate.

Automatic, no generation scoring, no human judgement — and unlike
perplexity it is a capability measure.

**Per-item cap fingerprint.** Each probe item also has the set of caps
that fire when processing it. That turns retention from a number into an
attribution: for every item we know whether its caps fired during B's
training, whether audit reseeded any of them, and whether it survived.

Three tiers, in order of harshness:

- top-1 next-token accuracy on the probe set (cheapest, already computable)
- exact k-token completion (the byte-exact test)
- perplexity on A (secondary, sanity only)

## Baselines

- **Dense transformer**, same budget, same corpora. Gives the retention
  rate but no attribution — that contrast is the point.
- **Cap-native, audit disabled.** Isolates whether audit *does* anything,
  or whether retention is just a consequence of sparse routing.

## Compute

Measured per-seed costs on the CPU machine:

| config | h/seed |
|---|---|
| cap-native hierarchical, WikiText 5000 steps | 5.5 |
| cap-native hierarchical, TinyStories 3000 steps | 2.7 |
| dense 853K, WikiText 5000 steps | 0.7 |

A two-cycle condition is A + B, so ~11 h for cap-native on WikiText.
Five distances x 3 seeds x 11 h = **165 h** for the main grid, plus ~22 h
for dense baselines. Roughly **8 days** single-machine.

Reductions worth considering before committing: run cycle A **once** and
snapshot it, since every distance shares the same A — that alone cuts the
grid to ~90 h. Use the 4060 for the hierarchical config, which runs at
0.446 s/step there against 3.98 on CPU.

## Phases

- [ ] **P0 — Probe harness.** Probe-set construction, `R0` capture,
      retention scoring, cap-fingerprint recording. Reusable across every
      later experiment. ~1 day.
- [ ] **P1 — Identity stability.** Before any continual-learning claim:
      does a cap respond to the same region of input space after a
      retraining cycle? If not, the rest is meaningless. Cheapest possible
      falsification, run it first.
- [ ] **P2 — Two-cycle protocol.** Snapshot after A; train B; measure
      retention. One distance only (far domain, the clearest signal).
- [ ] **P3 — Distance ladder.** Extend to all five distances, 3 seeds.
- [ ] **P4 — Attribution.** Correlate per-item retention against cap
      disturbance. This is the paper's central figure.
- [ ] **P5 — Baselines.** Dense, and audit-disabled cap-native.

Stop after P1 if identity does not survive. Stop after P2 if retention is
indistinguishable from dense.

## Risks

**Cap identity may not be stable enough.** Keys are frozen after
discovery, which is the mechanism that should preserve identity — but the
*projections* keyed by those caps train freely. A cap could keep its key
and still change what it contributes. P1 exists to find this out before
anything is built on top of it.

**Audit is wired into discovery, not into a retraining protocol.**
Nothing currently runs train-A → audit → train-B. That plumbing is part
of P2.

**Overlap may be trivially high or trivially low.** If every corpus fires
essentially the same caps, there is no variance to correlate against; if
foreign corpora fire disjoint caps by construction, the result is
uninteresting. Measure overlap on existing checkpoints *before* P2 — it
is nearly free and determines whether the design has headroom.

**The result may be that caps forget like everything else.** That is a
publishable negative result, and it is the honest outcome to plan for.

## Open decisions

1. **Which architecture?** Cap-native hierarchical is the stronger and
   faster configuration, and is what the second line reports. The
   cap-augmented transformer is cheaper and simpler to reason about.
   Leaning cap-native.
2. **Audit-and-replace only, or the biology mechanisms too?** Replay,
   pattern separation and engram cells exist in the other architecture
   line and would need re-implementation here. Leaning narrow: the
   overlap-predicts-forgetting result stands alone, and the biology
   mechanisms become the obvious follow-up if it holds.
3. **Does generation appear at all?** If so, qualitative and small, and
   labelled illustrative — small models at a 512-token vocabulary cannot
   support a quantitative generation claim.

## Notes

- Set `AWARE_BENCH_OUTPUT_DIR` on every run. The default `data/bench/`
  overwrites, and results have been lost that way.
- Record `device` and `backend_features` (already in `report.json`) so
  cross-machine comparisons stay attributable.
