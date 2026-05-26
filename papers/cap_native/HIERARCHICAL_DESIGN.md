# Hierarchical Cap-Native Design

Working design for Phase E's hierarchical variant (Path Y in PLAN.md).
Two stacked discovered CapLayers; the second discovered over h_0
activations rather than token embeddings.

## Architecture

```
emb         -> Layer 0 CapLayer (window W_0, n_caps_0)        -> h_0
                                                              -> cap_acts_0  (B, S, n_caps_0)
h_0         -> Layer 1 CapLayer (window W_1, n_caps_1)        -> h_1
                                                              -> cap_acts_1  (B, S, n_caps_1)
h_1         -> block 1..N   (cap-keyed using cap_acts_1)      -> h_N
            -> final_norm   (cap-keyed using cap_acts_1)
            -> output       (cap-keyed using cap_acts_1)
            -> logits
```

Routing signal handoff is the critical detail: every cap-keyed
component (norm/MHA/MoE/output) downstream of layer 1 routes by
`cap_acts_1`. Their `n_caps` dim is sized to `n_caps_1` (smaller than
`n_caps_0` by default — see Open Questions).

## Bootstrap sequencing

KMeans discovery for layer 1 requires forward passes through layer 0
to obtain h_0 samples. Build order:

1. Construct layer 0 CapLayer from token-bootstrap samples (existing
   `CapLayer::new` path; uses `cap_window=W_0`).
2. Forward layer 0 on the bootstrap-sample token batch to produce h_0.
3. Construct layer 1 CapLayer using h_0 as the sample tensor; this
   discovers cap centroids in the d_model-dimensional activation space.
4. Build downstream cap-keyed components sized to `n_caps_1`.

Layer 1's `d_in_per_window` is `d_model * W_1` (not `d_emb * W_1`).
With `W_1 = 1` the cap_input is just h_0; with `W_1 > 1` we'd window
contextualised activations.

## Config additions

`CapNativeConfig` gets two new optional fields:

```rust
pub struct CapNativeConfig {
    // … existing fields …
    pub hierarchical: bool,                       // default false
    pub cap_layer_1: Option<CapLayer1Config>,     // None when hierarchical=false
}

pub struct CapLayer1Config {
    pub n_caps_target: usize,        // default 128 (smaller than layer 0's 330)
    pub n_caps_budget: usize,        // default 512
    pub cap_window: usize,           // default 1
    pub discovery: DiscoveryKind,    // default KMeans
}
```

Env vars exposed by the runner:

```
AWARE_CN_HIERARCHICAL=true
AWARE_CN_L1_N_CAPS=128
AWARE_CN_L1_WINDOW=1
AWARE_CN_L1_DISCOVERY=kmeans
```

## Forward changes in `CapNativeSubstrate`

Add an optional `cap_layer_1: Option<CapLayer>` field. Forward becomes:

```rust
let emb = self.embeddings.forward(token_ids)?;
let h_after_layer0 = self.cap_layer.forward(&emb)?;
let cap_acts_0 = self.cap_layer.cap_activations(&emb)?;

let (h, cap_acts) = if let Some(layer1) = &self.cap_layer_1 {
    let h_1 = layer1.forward(&h_after_layer0)?;
    let cap_acts_1 = layer1.cap_activations(&h_after_layer0)?;
    (h_1, cap_acts_1)
} else {
    (h_after_layer0, cap_acts_0)
};

let mut h = h;
for block in &self.blocks {
    h = block.forward(&h, &cap_acts)?;
}
let h = self.final_norm.forward(&h, &cap_acts)?;
self.output.forward(&h, &cap_acts)
```

`cap_acts_0` is unused in the routing signal of single-discovery
configs (the existing path retains it as the routing source). In
hierarchical mode it is *only* used by layer 1's input — the blocks
downstream see `cap_acts_1`. Optional extension: concat both for
richer routing (deferred; see Open Questions).

## Builder changes

`CapNativeBuilder::build()` runs the bootstrap in two phases when
hierarchical:

1. Discover layer 0 (unchanged path).
2. Forward layer 0 on the bootstrap sample tokens (no_grad / detach to
   avoid graph retention).
3. Hand the resulting `(N_bootstrap, S, d_model)` tensor (flattened to
   `(N_bootstrap * S, d_model)`) to layer 1's `CapLayer::new` as the
   `sample` for KMeans.
4. Build downstream cap-keyed components sized to `n_caps_1`.

The blocks/norm/output all need to be built with `n_caps_1` instead of
`n_caps_0`. This is a single substitution in the builder loop —
`CapNativeBlock::new` and friends already take `n_caps` as a parameter.

## Engineering checklist

- [ ] `CapLayer1Config` + `hierarchical` field in `config.rs`
- [ ] Add `cap_layer_1: Option<CapLayer>` field to `CapNativeSubstrate`
- [ ] Refactor builder: discover layer 0 → forward → discover layer 1 → build blocks at `n_caps_1`
- [ ] Update `forward()` to thread layer 1 cap_acts to downstream
- [ ] Update `n_params()` and `cap_stats()` to include layer 1
- [ ] Two unit tests:
  - `build_with_hierarchical_kmeans` — substrate builds, params nonzero
  - `forward_hierarchical_runs` — produces correctly-shaped logits
- [ ] `cap_native_run_benchmark.rs`: parse `AWARE_CN_HIERARCHICAL` and
      `AWARE_CN_L1_*` env vars; thread into `CapNativeConfig`
- [ ] Smoke test: hierarchical at d=64, n_caps_0=32, n_caps_1=16,
      verify training trajectory is sane
- [ ] Phase E: hierarchical at d=128, n_caps_0=330, n_caps_1=128 vs
      single-discovery (this run's config)

Estimated effort: ~200-300 LOC core + ~100 LOC bench wiring + tests.
1-2 days of focused work.

## Memory analysis (advance)

At d=128, n_caps_1=128, the cap-keyed components downstream of layer 1
shrink relative to single-discovery's n_caps=330:

| Component | Single n=330 | Hier n_caps_1=128 |
|---|---|---|
| `w_qkv` per block | 330 × 128 × 384 = 16M | 128 × 128 × 384 = 6M (2.6× less) |
| `w_moe` per block (gate+value+out) | 65M | 25M |
| Per block | ~82M | ~32M |
| 4 blocks | 328M | 128M |
| Plus layer 0 + layer 1 + standard parts | ~5M | ~22M (extra layer 1) |
| **Total** | **~333M** | **~150M** |

Hierarchical at n_caps_1=128 is ~2.2× smaller than single-discovery at
n_caps=330. Memory: ~3 GB peak vs ~13 GB. Compute per step: similar
ratio reduction.

## Open questions / decisions deferred

- **Concat routing**: feed `concat(cap_acts_0, cap_acts_1)` to downstream
  blocks instead of just `cap_acts_1`. Doubles routing dim; n_caps for
  cap-keyed components needs to be `n_caps_0 + n_caps_1`. Phase G ablation.
- **Layer 1 window > 1**: `W_1 = 2` or `3` windows over contextualised
  representations. Phase G ablation.
- **Trainable layer 1 keys**: gradient-train layer 1 after KMeans
  bootstrap. Phase G ablation.
- **Bootstrap sample size for layer 1**: how many h_0 samples does
  KMeans need? Default 2000 token positions × seq_len. May need more
  if h_0 distribution is highly non-uniform.

## What's NOT in this design

- Layer 2+: paper #2 caps the hierarchy at 2 layers (per PLAN.md).
- Cap growth during training: future work.
- Per-block independent discovery: defeats cap identity (per PLAN.md).
- Soft routing in hierarchical: same memory concern as single-discovery
  soft at n_caps=330; deferred or run only at small n_caps_1.
