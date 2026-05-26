# Cap-Native Paper - Planning Notes

Working document for the cap-native research line. Not the paper itself;
just the experiment plan and decisions.

## Where we are

- Code complete: `src/aware/cap_native/` (single-discovery variant)
- Bench harness: `examples/cap_native_run_benchmark.rs`
- Disk-streaming feeder verified consistent with paper #1's pipeline
  (kmeans_w3 seed=42 reproduced: 13.72 vs paper #1 mean 13.71 ± 0.33)
- Memory note: cap-native at d=128 with soft routing hits ~5 GB peak
  through per-cap forward+backward loops (inherent, not a leak).
  **Sparse routing is mandatory for d=128 + n_caps=330** runs.
- Pre-set configs in `scripts/run_multi_seed.sh`:
  - d=128 baselines (paper #1 protocol): `pure_transformer`, `kmeans_w3`
  - d=128 cap-native: `cap_native_full`, `cap_native_indexed`,
    `cap_native_w3`, `cap_native_sparse`, `cap_native_topk4`
  - d=64 small-scale pilots (kept for memory/sanity): superseded as
    headline path but still useful for fast smoke tests

## Scale decision (this paper)

Match paper #1's scale exactly so the **only varying axis** is "where do
caps live":

| Knob | Value | Reason |
|---|---|---|
| d_model | 128 | paper #1 winning scale |
| n_caps_target | 330 | paper #1 winning scale |
| n_caps_budget | 1024 | paper #1 winning scale |
| n_blocks | 4 (baseline) / 2 (cap-native to bound params) | see below |
| cap_window | sweep {1..8}, default 3 | paper #1 §6.3 winner |
| corpus | TinyStories small (~3.5 M tokens) | paper #1 |
| steps | 5000 | paper #1 |
| seeds | {7, 42, 123} | paper #1 |

### n_blocks compromise

Cap-native blocks carry per-cap projection stacks. Memory & compute cost
scales as `n_blocks × n_caps × d_model²`. At n_caps=330, d=128:

- n_blocks=4 (paper #1 match): ~200+ M params, possibly tight on memory
- n_blocks=2: ~100 M params, tractable

**Decision: start with n_blocks=2 for cap-native variants**, then a
follow-up run at n_blocks=4 if Phase A indicates headroom. Document the
n_blocks gap explicitly in the paper; baselines stay at n_blocks=4
matching paper #1.

### Routing decision

| Routing | Compute scaling | Memory | Feasible at n_caps=330? |
|---|---|---|---|
| SoftTopK (full softmax) | linear in n_caps | per-cap activations retained | NO — ~32 h/seed |
| HardTop1Sparse | O(1) in n_caps | one stack/token | YES — ~3-4 h/seed |
| TopK with K=4 | O(K) in n_caps | K stacks/token | maybe — ~4-5 h/seed |

**Sparse routing is the headline path.** Soft routing is studied
separately at small n_caps (e.g., n_caps=32) as a methodology ablation,
not at headline scale.

## Paper thesis

The cap primitive scales from input augmentation (paper #1) to a
**discovered hierarchical substrate**. We test two architectural
variants:

1. **Single-discovery cap-native**: one CapLayer at layer 0,
   per-cap parameter stacks at every downstream block.
2. **Hierarchical cap-native**: two stacked CapLayers (layer 0 over
   token windows, layer 1 over h_0), per-cap stacks at blocks 2..N
   routed by layer 1's cap activations.

The hierarchical variant matches the legacy AWARE intuition of
caps -> higher-level caps -> downstream computation, with each layer's
cap basis discovered (not random).

## Architecture variants

### Variant 1: Single-discovery cap-native (Path X)

```
emb -> Layer 0 CapLayer -> h_0
                        -> cap_acts_0 (routing signal for all downstream)
h_0 -> block 1..N (cap-keyed using cap_acts_0)
     -> final cap-keyed norm + output
```

- One discovery (KMeans on token-window samples) at layer 0
- All downstream cap-keyed components routed by single cap_acts_0
- Already implemented; ready to run

### Variant 2: Hierarchical cap-native (Path Y addition)

```
emb -> Layer 0 CapLayer (window W_0, n_caps_0) -> h_0, cap_acts_0
h_0 -> Layer 1 CapLayer (window W_1, n_caps_1) -> h_1, cap_acts_1
h_1 -> block 2..N (cap-keyed using cap_acts_1)
     -> final cap-keyed norm + output (routed by cap_acts_1)
```

- Two discoveries: layer 0 on token windows, layer 1 on h_0
- Bootstrap order: layer 0 bootstrap -> freeze keys_0 -> forward to
  produce h_0 samples -> layer 1 bootstrap on h_0 -> freeze keys_1
- Per-cap stacks at blocks 2..N sized (n_caps_1, d, ...) and routed
  by cap_acts_1
- NOT yet implemented; engineering work below

### Design decisions for hierarchical variant

| Decision | Default | Alternatives |
|---|---|---|
| Layer 1 window W_1 | 1 (per-position over h_0) | 2, 3 (n-grams of contextualized reps) |
| Layer 1 n_caps | n_caps_1 = 128 (smaller than layer 0's 330) | 64, 256 |
| Routing for blocks 2..N | cap_acts_1 only | concat(cap_acts_0, cap_acts_1); per-block choice |
| Layer 0 / layer 1 cap kind | both Discovered (frozen after bootstrap) | Hybrid; gradient |
| Bootstrap dependency order | layer 0 -> layer 1 (sequential) | parallel re-bootstrap during training |

## Engineering work needed (for hierarchical variant)

- Build-time sequencing: layer 0 bootstrap completes before layer 1
  bootstrap can collect h_0 samples
- `CapLayer` over h_0: the primitive supports arbitrary input dim;
  needs wiring with d_model input (not d_emb)
- `CapNativeBlock` routing logic: use cap_acts_1 instead of cap_acts_0,
  or some combination
- Bootstrap sample collection for h_0 (forward layer 0 on a batch
  of bootstrap tokens, collect h_0 activations)
- Optimizer setup: per-cap stack shapes derived from n_caps_1, not n_caps_0
- Optional: cap_acts_0 still available as auxiliary routing signal
- Estimated effort: 100-200 LOC + tests, 1-2 days

## Baselines (reuse from paper #1)

No re-runs needed — disk-streaming feeder confirmed consistent. Cite
paper #1's published numbers directly:

| Config | val_ppl (mean ± std, 3 seeds) | params |
|---|---|---|
| `pure_transformer` (d=128, n_blocks=4) | 28.00 ± 0.11 | 853 K |
| `kmeans_w3` cap-input (d=128, n_blocks=4) | 13.71 ± 0.33 | 895 K |

These are the comparison anchors for every cap-native variant.

## Experiment plan (Phases A-G)

### Phase A: Single-seed sanity at d=128 (first real cap-native runs)

Single-seed sanity check at the headline scale.

| Config | Notes |
|---|---|
| `cap_native_sparse` (d=128, n_caps=330, n_blocks=2, w=3, kmeans, top_k=1) | first real headline-config run |
| `cap_native_hier_sparse` (d=128, n_caps_0=330, n_caps_1=128, n_blocks=2, top_k=1) | hierarchical variant (after implementation) |

Decision gate: do both train cleanly at d=128 / n_caps=330? If yes,
proceed to multi-seed. If memory pressure or instability appears,
revisit n_blocks or n_caps_target.

### Phase B: Routing comparison (3 seeds, d=128)

At n_caps=330, **soft routing is infeasible** (~32 h/seed). Routing
comparison runs at a tractable smaller n_caps (e.g., 32) to characterize
the soft-vs-sparse signal, then headline runs use sparse only at full
n_caps=330.

| Config | n_caps | Routing | Cost/seed |
|---|---|---|---|
| `cap_native_full_small` | 32 | SoftTopK | ~3 h |
| `cap_native_topk4_small` | 32 | top-K=4 | ~1.5 h |
| `cap_native_sparse_small` | 32 | HardTop1 | ~50 min |
| `cap_native_sparse_full` | 330 | HardTop1 | ~3-4 h |

### Phase C: Cap-window sweep (3 seeds, d=128, sparse, n_caps=330)

Mirrors paper #1 §6.3 — does the w ∈ {2,3,4} activation regime persist
in cap-native?

| Config | Window |
|---|---|
| `cap_native_sparse_w1` | 1 |
| `cap_native_sparse_w2` | 2 |
| `cap_native_sparse_w3` | 3 |
| `cap_native_sparse_w4` | 4 |
| `cap_native_sparse_w5` | 5 |
| `cap_native_sparse_w8` | 8 |

### Phase D: Discovery sweep (single seed, d=128, sparse, n_caps=330)

Key ablation: does layer-0 discovery matter MORE in cap-native than in
paper #1's §6.4 (where it didn't, at window=1)?

| Config | Layer 0 discovery |
|---|---|
| `cap_native_kmeans` | KMeans on token windows (default) |
| `cap_native_random` | Random unit vectors (FrozenRandom) |
| `cap_native_nodiscovery` | Xavier init, gradient-train layer-0 keys |
| `cap_native_hybrid` | KMeans init + gradient-train |

### Phase E: Single vs hierarchical discovery (3 seeds, d=128) — HEADLINE ABLATION

Direct A/B test of the architectural claim.

| Config | Architecture |
|---|---|
| `cap_native_single` (best from Phase B/C/D, sparse, n_caps=330, w=3) | one CapLayer at layer 0 |
| `cap_native_hier` (matched scale + n_caps_1=128) | two CapLayers (layer 0 + layer 1 on h_0) |

This is the headline experiment of the paper. Reported against paper #1's
13.71 ± 0.33 (kmeans_w3) and 28.00 ± 0.11 (pure_transformer).

### Phase F: Indexed-mask ablation (single seed, d=128)

| Config | Cap-indexed attention mask |
|---|---|
| `cap_native_sparse` | off |
| `cap_native_indexed` | on |

### Phase G: Hierarchical variant ablations (single seed, d=128)

Only run if Phase E shows hierarchical promising. Otherwise skip.

| Config | Variation |
|---|---|
| `cap_native_hier_w1_1` | layer 1 window = 1 (per-position) |
| `cap_native_hier_w1_2` | layer 1 window = 2 |
| `cap_native_hier_w1_3` | layer 1 window = 3 |
| `cap_native_hier_n64` | n_caps_1 = 64 |
| `cap_native_hier_n256` | n_caps_1 = 256 |
| `cap_native_hier_concat` | routing uses concat(cap_acts_0, cap_acts_1) |

## Compute estimate (d=128, sparse routing where applicable)

| Phase | Approx cost |
|---|---|
| Phase A | ~6-8 hours (2 configs × 1 seed at d=128) |
| Phase B | ~15-20 hours (4 variants × 3 seeds, mixed routing) |
| Phase C | ~60-80 hours (6 windows × 3 seeds, sparse) |
| Phase D | ~12-16 hours (4 strategies × 1 seed) |
| Phase E | ~20 hours (2 variants × 3 seeds, sparse) |
| Phase F | ~6-8 hours (2 configs × 1 seed) |
| Phase G | ~18-24 hours (6 variants × 1 seed) — contingent |
| **Total** | **~140-180 hours CPU** (~6-8 days continuous, more if parallel) |

Trade-offs:
- Phase C is the biggest cost; could trim to 3 windows {w=2, w=3, w=4} if
  Phase A/B confirms paper #1's window regime persists, saving ~30 h
- If Phase E shows hierarchical is clearly better or worse, Phase G can
  be skipped (~20 h saved)

## First practical move

Smoke test cap-native at the headline scale to verify training works:

```bash
cd ~/limnr/aware
./scripts/run_multi_seed.sh cap_native_sparse 42
```

Reveals: does it train at d=128 / n_caps=330 / n_blocks=2 / sparse? Does
loss decrease? Rough perplexity vs paper #1 baselines (28.00 pure /
13.71 cap-input)?

If positive, proceed to Phase B/C runs in parallel-able batches.

## Open questions / decisions to make later

- Title for the paper
- Whether to position single-discovery as the headline if hierarchical
  underperforms, or whether to position hierarchical as the headline
  even if single-discovery is competitive
- Should the param-mismatch story be foregrounded? cap-native at
  n_caps=330 n_blocks=2 is ~100 M params vs paper #1's 895 K — a
  ~100× gap. Honest framing: "capacity-scaled architectural variant"
- Should we include any of the cap-memory / cap-pair Phase B' work
  in this paper, or keep separate?

## d=64 pilot data (superseded as headline path)

Earlier work at d=64 / n_caps=32 produced:
- `pure_transformer_d64`: 35.80 (seed=42 only)
- `kmeans_w3_d64`: 33.91 ± 2.18 (seeds=42, 123)
- `cap_native_sparse_d64_small`: 10.54 (seed=42 only)

Variance was too high at d=64 (std 2.18 for kmeans_w3 vs paper #1's
0.33 at d=128) to support headline claims. Demoted to pilot data;
useful for fast-iteration smoke testing only. d=128 is the canonical
scale for paper #2.

## What's NOT in scope

- **Layer-0 cap growth during training** (GrowableAdamW exercises): future work
- **Per-block independent discovery**: defeats cap identity, defer
- **Multi-head re-implementations of CapMemory/CapPair** (paper #1
  Phase B' follow-up): separate paper
- **Continual learning**: separate paper
- **3+ stacked discovered layers**: future work; paper #2 caps at 2

## Status

- [x] Run smoke test at d=64 (small) — done; sparse routing verified
- [x] Verify disk-streaming feeder consistency vs paper #1 (kmeans_w3 d=128 seed=42)
- [ ] Implement hierarchical variant (Variant 2) in src/aware/cap_native/
- [ ] Phase A: Sanity at d=128 (cap_native_sparse seed=42 — first real headline run)
- [ ] Phase A: Sanity at d=128 (cap_native_hier_sparse seed=42, after impl)
- [ ] Phase B: Routing comparison (4 configs × 3 seeds)
- [ ] Phase C: Cap-window sweep (6 windows × 3 seeds)
- [ ] Phase D: Discovery sweep (4 strategies × 1 seed)
- [ ] Phase E: Single vs hierarchical discovery (2 variants × 3 seeds) — HEADLINE
- [ ] Phase F: Indexed-mask ablation (2 configs × 1 seed)
- [ ] Phase G: Hierarchical variant ablations (6 variants × 1 seed) — contingent
- [ ] Analysis + comparison to paper #1
- [ ] Paper draft
