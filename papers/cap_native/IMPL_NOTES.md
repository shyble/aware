# Hierarchical Cap-Native — Implementation Notes

Design sketch for adding a second discovered CapLayer (layer 1) downstream
of layer 0, before the cap-keyed transformer blocks. Not yet implemented;
this document captures decisions to make before writing code.

## Architecture target

```
emb (B, S, d_emb)
  |
Layer 0 CapLayer
  - window=W_0, fires on token embeddings
  - keys_0 discovered (KMeans on windowed embedding samples)
  - cap_acts_0 -> (B, S, n_caps_0)
  - W_proj_0 -> h_0 (B, S, d_model)
  |
Layer 1 CapLayer
  - window=W_1 (typically 1), fires on h_0
  - keys_1 discovered (KMeans on h_0 samples, captured after layer 0 build)
  - cap_acts_1 -> (B, S, n_caps_1)
  - W_proj_1 -> h_1 (B, S, d_model)
  |
Cap-keyed blocks 2..N
  - Routed by cap_acts_1
  - Per-cap stacks sized (n_caps_1, d, ...)
  |
Final cap-keyed norm + output projection
  - Sized (n_caps_1, ...), routed by cap_acts_1
  |
logits
```

## Key design decisions

### 1. Bootstrap sequencing (build-time)

Layer 1's KMeans requires h_0 samples, which require a built layer 0.
Substrate build must therefore be sequential:

```
1. Build embedding
2. Build layer 0 CapLayer (KMeans on token windows -> keys_0)
3. Forward layer 0 on bootstrap sample tokens -> collect h_0
4. Build layer 1 CapLayer (KMeans on h_0 -> keys_1)
5. Build cap-keyed blocks (sized n_caps_1)
6. Build cap-keyed output (sized n_caps_1)
7. Register all gradient-trainable vars in optimizer
```

This must happen at construction time, not training time, because all
downstream stack shapes depend on n_caps_1.

### 2. Routing source for downstream blocks

Three options; recommended is C.

| Option | Blocks use | Pros | Cons |
|---|---|---|---|
| A | cap_acts_0 only | Layer 1 only contextualizes h, not routing | Layer 1 is partially wasted |
| B | both, concat | Richer routing signal | Per-cap stacks become (n_0+n_1, d, d), larger |
| C | cap_acts_1 only | Cleanest hierarchical replacement | Layer 0 is only an input encoder |

Going with **C** for paper #2's headline experiment. Variant B is a Phase G
ablation if interesting.

### 3. Layer 1 window W_1

W_1 = 1 (per-position over h_0) is the default. Layer 0 already
contextualized across W_0 tokens; layer 1 just clusters the resulting
per-position representations. Phase G can sweep W_1 in {1, 2, 3}.

### 4. Layer 1 cap kind

Default: `Discovered` (KMeans + freeze). Matches layer 0's default.
Phase D's discovery sweep applies to layer 1 as well (could expand
to a 4x4 grid: each layer's discovery kind). For paper #2 scope,
both layers use the same kind in any given run.

### 5. n_caps_1 sizing

Default: n_caps_1 = n_caps_0 / 2 or n_caps_0 / 4 (abstraction reduces
granularity). At d=64 with n_caps_0=64, try n_caps_1=32 first.

## API changes needed

### `CapNativeConfig` (config.rs)

Add fields:
```rust
pub struct CapNativeConfig {
    // ... existing fields ...

    /// Enable hierarchical mode: two CapLayers in sequence.
    pub hierarchical: bool,

    /// Layer 1 configuration (only used if hierarchical=true).
    /// `n_caps` of layer 1 determines the size of all downstream
    /// cap-keyed stacks (per-cap QKV/O, MoE experts, etc.).
    pub cap_config_layer1: Option<CapConfig>,
}
```

Sensible default for non-hierarchical mode: `hierarchical = false,
cap_config_layer1 = None`. Behavior unchanged from current cap-native.

### `CapNativeSubstrate::build()` (substrate.rs)

Add a branch:
```
if hierarchical {
    1. Build embedding
    2. Build layer 0 CapLayer (existing logic)
    3. With torch::no_grad(): forward bootstrap_sample_tokens through
       embedding + layer 0 -> collect h_0
    4. Build layer 1 CapLayer with `bootstrap_sample = h_0`
       (note: h_0's d_in is d_model, layer 0's was d_emb)
    5. Build cap-keyed blocks sized n_caps_1
    6. Build cap-keyed output sized n_caps_1
}
```

The `CapLayer::new()` already accepts any `d_in`. The bootstrap step is
the new part: forwarding a sample batch through partially-built layers.

### `CapNativeSubstrate::forward()` (substrate.rs)

Two new lines if hierarchical:
```rust
let (h_0, cap_acts_0) = self.cap_layer_0.forward_with_acts(&embeds)?;
let (h_routing, cap_acts) = if let Some(cap_layer_1) = &self.cap_layer_1 {
    cap_layer_1.forward_with_acts(&h_0)?  // produces h_1, cap_acts_1
} else {
    (h_0, cap_acts_0)  // single-discovery path
};
// blocks 2..N now use h_routing as h and cap_acts as routing
```

### `CapNativeBlock` (block.rs)

No changes — block.forward already takes cap_acts as a parameter.
The substrate just passes cap_acts_1 instead of cap_acts_0 when hierarchical.

### `CapKeyedRmsNorm`, `CapKeyedMha`, `CapMoeMlp`, `CapKeyedOutput`

No code changes — they take n_caps from their construction config. The
substrate just constructs them with n_caps = n_caps_1 in hierarchical mode.

## Tricky bits to handle

1. **Bootstrap-time sample size**: layer 1's KMeans needs enough h_0
   samples. If bootstrap_sample_tokens = 2000 tokens (paper #1 default),
   that gives 2000 h_0 vectors after forward — sufficient.

2. **Memory at bootstrap time**: forwarding 2000 tokens through layer 0
   to collect h_0 needs a single inference pass with no_grad. Cheap.

3. **`GrowableAdamW` registration**: each cap-keyed component already
   registers its variables. Hierarchical doesn't change this; just more
   variables (layer 1's keys_1 and W_proj_1).

4. **Audit lifecycle**: layer 1 caps could be audited too (same as
   layer 0). For paper #2, both layers' caps are bootstrap+freeze
   (no live audit). Add audit later if needed.

5. **Layer 1 window > 1**: if W_1 > 1, the windowed input is computed
   over h_0 (not tokens). `CapLayer.build_windowed_input` works for
   any input dim, so this just works.

6. **Discovery sweep across both layers**: Phase D as scoped today only
   varies layer 0. To extend to layer 1, either:
   - Vary only layer 1 (with layer 0 fixed at KMeans)
   - Vary both jointly with `--both` option (4x4 = 16 cells, expensive)

   Defer the 4x4 grid; for paper #2 set both to KMeans.

## What this implementation does NOT add

- Growable n_caps during training (paper #3+)
- Per-layer audit (paper #3+ with continual learning)
- 3+ stacked discovered layers (depth > 2)
- Mix of discovery strategies across layers (paper-defined nuance)

## Implementation order

1. Add `hierarchical: bool` + `cap_config_layer1: Option<CapConfig>` to
   `CapNativeConfig`
2. Add `cap_layer_1: Option<CapLayer>` field to `CapNativeSubstrate`
3. Modify `CapNativeBuilder::build()` to handle hierarchical branch
4. Modify `CapNativeSubstrate::forward()` to use cap_acts_1 if present
5. Add bootstrap collection helper: `collect_h0_samples(layer_0, embed,
   tokens) -> Tensor`
6. Add a `cap_native_hier_d64` preset in `scripts/run_multi_seed.sh`
7. Smoke test the hierarchical variant; iterate on bootstrap order
8. Once stable, slot into Phase A baselines

Estimated 1-2 days of careful work + ~150 LOC + tests.

## Open questions to resolve before coding

- Default value for n_caps_1 (proposal: n_caps_0 / 2)
- Whether to bootstrap layer 1 on h_0 or on raw embeddings (proposal: h_0)
- Whether layer 1's W_proj_1 output dim matches layer 0's d_model (yes,
  for downstream block compatibility)
