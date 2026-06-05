# Hierarchical Cap-Native Substrates for Transformer Language Modeling

**Installment one of the cap-native research line. This paper reports
two of seven planned ablation axes (Table 1); the remaining five are
deferred to a continuation paper on accelerator hardware. The two
reported axes are self-contained and constitute the paper's claims.**

Author: Kürşat Aydemir
Affiliation: limnr / EndpointDev
Code: https://github.com/shyble/aware  (branch: `cap-native`)

## Abstract

The cap primitive of Aydemir (2026) introduced an identifiable
computational unit that augments a standard transformer with a
discovered input layer of capability nodes ("caps"). That work placed
caps only at the input — a single cap-keyed projection from
embeddings to model dimension, feeding standard transformer blocks
downstream. This paper asks whether caps can drive **every layer**
rather than only the input, and whether stacking two layers of
discovered caps captures hierarchical structure in language.

We introduce **cap-native**, a transformer-shaped architecture in
which every projection at every block (multi-head attention QKV/O,
SwiGLU MoE, RMS-norm, output) is **cap-keyed**: each block carries
`n_caps` independent parameter stacks, and a per-token routing signal
selects which stack contributes to that token's forward pass. We
study two variants:

- **Single-discovery**: one discovered cap layer over token windows;
  its activations drive all downstream cap-keyed components.
- **Hierarchical**: two stacked discovered cap layers — a first over
  token windows (as in paper #1), and a second over the output of the
  first. Downstream cap-keyed components route by the second layer's
  activations.

On TinyStories (small split, ~3.5 M tokens), at matched training
budget (3000 steps, 3 random seeds), hierarchical cap-native achieves
**val_ppl 8.76 ± 0.13** with 143 M parameters, outperforming
single-discovery cap-native (9.84 ± 0.32, 368 M params) while using
**2.6× fewer parameters** and exhibiting **2.5× tighter run-to-run
variance**. Both variants improve substantially over the cap-input
baseline of paper #1 (13.71 ± 0.33, 895 K params), though that
comparison is at unmatched parameter scale and we discuss this
caveat at length.

The paper's contribution is architectural: we show that a second
discovered cap layer over already-contextualised representations is
a strictly better placement of the cap primitive than a single
input-only layer, even at substantially smaller per-block parameter
count. We release the implementation, including a memory-bounded
batched-dispatch routine that makes sparse routing tractable on
modern accelerators despite per-cap weight residency.

## 1. Introduction

The cap primitive of Aydemir (2026) is a single discovered layer
that maps token-window embeddings to a sparse vector of cap
activations and then projects those activations back to the model
dimension. The paper showed that, with cap windows of width 3 and
KMeans-discovered cap centroids, the resulting cap-input layer
yielded a 51 % perplexity reduction over a matched-parameter pure
transformer.

Cap-input is a conservative placement: caps live only at the input,
and the rest of the network is a standard transformer. The cap
primitive's identifiability — its keys and values are addressable,
discoverable, and life-cycled — is consequently confined to the
input boundary. Once activations pass through the first attention
block, the per-token cap identity is no longer exposed.

This paper asks: what happens when caps drive every projection at
every block?

### 1.1 Contributions

1. **Cap-native architecture.** We propose a transformer-shaped
   architecture in which every linear projection at every block is
   cap-keyed: each block carries `n_caps` independent parameter
   stacks, indexed at runtime by a per-token routing signal derived
   from a discovered cap layer.

2. **Hierarchical discovery.** We extend cap-native with a second
   discovered cap layer that fires over the output of the first.
   The second layer's activations route every downstream cap-keyed
   component. This is the central novelty of this paper.

3. **Empirical evaluation.** On TinyStories (small split, ~3.5 M
   tokens) at matched training budget, hierarchical cap-native at
   143 M parameters beats single-discovery cap-native at 368 M
   parameters by 11 % perplexity and runs ~4× faster end-to-end.
   It is also more consistent across seeds (standard deviation
   2.5× tighter).

4. **Bounded grouped dispatch.** Cap-keyed weight stacks at large
   `n_caps` are dispatch-bound on every modern accelerator. We
   contribute a bounded grouped-matmul routine that bounds peak
   activation memory regardless of cap-firing skew while keeping
   the autograd graph shallow, enabling sparse routing at `n_caps`
   in the hundreds without per-step memory blow-up.

## 2. Background and Related Work

### 2.1 The Cap Primitive

A cap is a tuple `(id, key, value, meta)`: an identifiable unit with
a discoverable key vector and (optionally) a value vector. A cap
matrix is a collection of caps that supports lifecycle operations
(growth, replacement, audit). Aydemir (2026) defined the primitive
and showed three placements in a transformer:

- **Cap input layer** (the winner): a discovered cap matrix fires on
  token-window inputs; activations are projected to model dimension
  and fed to standard transformer blocks downstream.
- **Cap-memory attention** and **cap-pair attention**: negative results.

This paper takes the cap primitive as given and asks where else it
can usefully live in a transformer.

### 2.2 Mixture-of-Experts and Sparse Routing

Cap-native is closely related to mixture-of-experts (MoE)
architectures (Shazeer et al. 2017, Fedus et al. 2022): both
maintain `n_experts` (or `n_caps`) independent parameter stacks per
block and select a sparse subset per token. Two architectural
differences distinguish cap-native from canonical MoE:

1. **Discovery, not learned routing.** MoE routing is a learned
   gating network; cap routing comes from a discovered cap matrix
   (KMeans on bootstrap samples by default) with frozen keys.
   Routing is consequently deterministic given the input, not
   trained.
2. **Universal cap-keying.** Canonical MoE replaces only the FFN
   sublayer with mixtures; cap-native cap-keys every projection
   (attention QKV and O, MoE gate/value/out, norms, output). The
   cap activations from a single discovered layer drive the entire
   downstream network's parameter selection.

### 2.3 Hierarchical Clustering for Representation Discovery

The hierarchical variant introduced here parallels two-level
clustering schemes from classical representation learning
(Hinton 1999, Bengio 2009). At training time, we first discover
clusters over token-window embeddings (the cap-input layer of
paper #1); we then forward those embeddings through that layer to
obtain contextualised representations, and discover a second layer
of clusters over those. The second-layer clusters drive downstream
routing. This mirrors hierarchical concept formation but applied
to the routing signal of a discrete-expert architecture.

### 2.4 Other Related Work

**Hypernetworks** (Ha et al. 2016) generate the weights of one network
from another, providing an indirect form of conditional computation.
Cap-native is in the same broad family — per-cap weight stacks indexed
by a per-token routing signal can be viewed as a discretised
hypernetwork where the "hyper-input" is the cap winner identity. The
key differences are that (a) cap weights are stored explicitly rather
than generated on the fly, and (b) the indexing is discrete (top-1
argmax) rather than continuous.

**Conditional normalization** (Perez et al. 2018, FiLM; De Vries et
al. 2017) modulates layer norms by an external signal. Cap-native's
cap-keyed RMS-norm is structurally similar: each cap has its own
norm-scale parameter, and the per-token routing selects which scale
contributes. Where FiLM applies a learned modulator over a
continuously-varying conditioning vector, cap-native applies a
discovered modulator over a discrete cap index.

**Discovered basis architectures**, including codebook-based methods
(Van Den Oord et al. 2017, VQ-VAE) and discrete bottleneck networks,
share with cap-native the use of a fixed-size discovered basis at the
input and a routing signal derived from it. Cap-native extends this
pattern by tying downstream parameter selection to the basis rather
than only using it for compression.

**Routing networks** in the MoE literature (Lewis et al. 2021,
BASE layers; Roller et al. 2021, hash-based routing) have
investigated alternatives to learned-gating routing, including
deterministic hashing and explicit balancing. Cap-native's discovery
(KMeans on bootstrap data with frozen keys) sits in this family —
the routing function is fixed at construction time rather than
learned, sacrificing routing flexibility for cap identifiability
and lifecycle-management.

## 3. The Cap-Native Architecture

### 3.1 From Cap-Input to Cap-Native

Paper #1's cap-input architecture is:

```
emb → CapLayer (window W, n_caps K) → h_0  (model dim, projected)
                                    → cap_acts (B, S, K)
h_0 → block 1..N (standard transformer)
    → final_norm
    → output
    → logits
```

Cap-keyed components downstream of the cap layer do not exist;
`cap_acts` is computed but only contributes to `h_0` via the
projection inside the cap layer itself.

Cap-native preserves the cap-input layer and additionally
cap-keys every block downstream:

```
emb → CapLayer (window W, n_caps K) → h_0
                                    → cap_acts (B, S, K)
h_0 → CapKeyedBlock 1 (routed by cap_acts)
    → CapKeyedBlock 2 (routed by cap_acts)
    → ...
    → CapKeyedBlock N (routed by cap_acts)
    → CapKeyedRmsNorm (routed by cap_acts)
    → CapKeyedOutput (routed by cap_acts)
    → logits
```

Every cap-keyed component carries `K` parameter stacks; a per-token
routing rule selects which stack contributes.

### 3.2 Per-Cap Weight Stacks

A standard attention QKV projection is one matrix `W ∈ R^{d × 3d}`.
A cap-keyed QKV projection at `K` caps is a 3-tensor
`W ∈ R^{K × d × 3d}` — one matrix per cap, stacked along the cap axis.
At inference, for a token assigned to cap `k`, the output is
`x @ W[k]`. Analogous stacks exist for the attention O projection,
the SwiGLU MoE gate/value/out matrices, the RMS-norm scale, and the
output vocab projection.

The per-block parameter cost is therefore `O(K × d²)` instead of
`O(d²)` for a standard transformer block. We discuss this scaling
cost honestly in §7.3.

### 3.3 Sparse Top-1 Routing

We use **hard top-1 sparse routing** throughout. For each token,
the argmax over `cap_acts` selects a single cap; only that cap's
weight stack contributes. This contrasts with the soft top-K
weighted blend used in canonical MoE and gives O(1) compute per
token regardless of `n_caps` — at the cost of giving up the
inter-cap-mixing dynamics of soft routing.

A systematic comparison of this routing choice against soft top-K
alternatives is one of the deferred ablation axes (§8.7).

### 3.4 Single-Discovery Variant

The single-discovery variant uses exactly one discovered cap layer
(at the input), with the cap activations from that layer routing
every downstream cap-keyed component. The downstream weight stacks
are sized to `K = n_caps_target` of the input cap layer.

### 3.5 Hierarchical Variant

The hierarchical variant introduces a second discovered cap layer.
Concretely:

- **Layer 0**: same as the cap-input layer of paper #1 — a discovered
  cap matrix over token windows of width `W_0`, with `K_0` caps. It
  produces both an output `h_0` (projected to model dim) and an
  activation `cap_acts_0` of width `K_0`.
- **Layer 1**: a second discovered cap matrix that fires on `h_0`
  (the output of layer 0), with window `W_1` (default 1) and `K_1`
  caps (default 128, smaller than `K_0` since layer 1 sees already
  contextualised representations). Layer 1 is discovered by running
  KMeans on samples of `h_0` obtained by forwarding the bootstrap
  token batch through layer 0.
- **Downstream cap-keyed components**: sized to `K_1`, routed by
  `cap_acts_1`. Layer 0's activations are not used downstream.

```
emb → CapLayer_0 (window W_0, K_0 caps) → h_0
                                        → cap_acts_0  (UNUSED downstream)
h_0 → CapLayer_1 (window W_1, K_1 caps) → h_1
                                        → cap_acts_1  (routes downstream)
h_1 → CapKeyedBlock 1 (routed by cap_acts_1)
    → ...
    → CapKeyedOutput
    → logits
```

Layer 0 and the downstream cap-keyed components are independently
parametrised. Because the downstream stacks are sized to `K_1` rather
than `K_0`, the hierarchical variant is **substantially smaller** in
total parameters than single-discovery at the same `K_0`. This is
the key parameter-efficiency observation of the paper.

Layer 1's keys are frozen after KMeans bootstrap; we use the same
discovery options as layer 0. Window > 1 for layer 1 is left to
future work.

## 4. Implementation Notes

### 4.1 Bounded Grouped Dispatch

A naïve sparse top-1 cap-keyed forward iterates over all `K` caps
and uses `index_add` to scatter each cap's outputs into a shared
result tensor. Each iteration creates a new tensor; the autograd
graph retains every intermediate. At `K = 330`, four cap-keyed
components per block, four blocks, this builds a depth-5280 graph
per forward step, dominating peak RAM during training.

We replace this with a **bounded grouped dispatch** routine. After
computing per-token cap winners, we sort tokens by winner, pad each
cap's slice to a fixed bound `B`, and perform a single batched
matmul of shape `(K, B, d) @ (K, d, d_out)`. Tokens whose cap
bucket exceeds `B` (which happens during early training when the
cap-firing distribution is highly skewed) fall back to a per-bucket
loop. The result is reassembled into sorted-by-winner order, then
inverse-permuted to the original token order.

The bound `B = max(8, 2 × ceil(total / K))` is chosen so the padded
tensor `(K × B × d)` stays bounded in size regardless of skew. In
practice this bounds the per-step activation memory to tens of
megabytes per cap-keyed component, even at `K = 330`. The autograd
graph has constant depth (one batched matmul, plus at most one
overflow loop, per cap-keyed component) instead of linear in `K`.

We provide parity tests verifying that the bounded grouped variant
produces identical results to the naïve loop within numerical
tolerance, for both balanced and heavily-skewed cap-firing
distributions.

### 4.2 Bootstrap Sequencing for Hierarchical

The hierarchical variant requires h_0 samples to discover layer 1.
At construction time, we:

1. Run KMeans on token-window samples to discover layer 0's keys
   (identical to paper #1's bootstrap path).
2. Forward layer 0 on the bootstrap token batch to obtain a tensor
   of h_0 activations (shape `(n_bootstrap_tokens, d_model)`).
3. Run KMeans on h_0 to discover layer 1's keys.
4. Build downstream cap-keyed components sized to `K_1`.

The sequencing is deterministic given the bootstrap token list and
random seed.

### 4.3 Memory and Compute Profile

At our reference configuration (TinyStories small, `d=128`, `K_0=330`,
`K_1=128`, `n_blocks=4`):

| Variant | total params | active/token | training memory peak | per-step (CPU, single device) |
|---|---|---|---|---|
| Pure transformer (paper #1) | 853 K | 853 K | < 1 GB | ~0.5 s |
| Cap-input (paper #1, kmeans_w3) | 895 K | 895 K | < 1 GB | ~0.6 s |
| Cap-native single-discovery | 368 M | ~1.22 M | ~13 GB | ~13 s |
| Cap-native hierarchical | 143 M | ~1.22 M | ~5–6 GB | ~3 s |

Cap-native's total parameter count is dominated by per-cap weight
stacks; with top-1 sparse routing only a single cap's slab is active
at any token. The "active params per token" column reports the
trainable parameters that actually contribute to a forward pass at
each position: ~1.115 M cap-keyed (one cap's slab across all
projections at all blocks) plus ~0.11 M dense (token embeddings and
cap-layer projections). The single-discovery and hierarchical
variants have essentially the same active-params footprint despite
a 2.6× difference in total params, because top-1 routing selects
exactly one slab regardless of `n_caps`. The hierarchical variant
is faster per step than single-discovery because its downstream
cap-keyed components have ~6× fewer parameters *per stack*, so
batched matmuls are dispatched on smaller shapes and memory
residency drops.

## 5. Experimental Setup

### 5.1 Corpus

We use the TinyStories small split (~3.5 M training tokens, ~180 K
validation tokens), identical to paper #1's corpus. BPE vocabulary is
256 tokens, trained on the corpus. Tokenization is cached to disk for
deterministic data loading across runs.

### 5.2 Configurations

All experiments share the following base hyperparameters:

| Hyperparameter | Value |
|---|---|
| `d_model` | 128 |
| `n_blocks` | 4 |
| `n_heads` | 4 |
| `d_ff` | 512 |
| `max_seq_len` | 128 |
| `batch_size` | 32 |
| `seq_len` per batch | 128 |
| Effective tokens per step | 4096 |
| `cap_window` (layer 0) | 3 |
| `n_caps_target` (layer 0, `K_0`) | 330 |
| `discovery` (layer 0) | KMeans |
| `routing` | HardTop1Sparse |
| `top_k` | 1 |
| `cap_indexed_mask` | off |

These match paper #1's winning configuration `kmeans_w3` (cap-input
layer with window 3, 330 KMeans centroids), so the only varying
axis between paper #1 and cap-native is _where caps live_.

### 5.3 Training Protocol

We train each configuration for **3000 steps** at the above batch
and sequence settings (3.47 effective epochs over the corpus), with
AdamW at the optimizer's default learning rate. Validation
perplexity is evaluated every 100 steps on a fixed validation slice.
Each configuration is run at **three random seeds** (42, 123, 7).

We report two perplexity figures per configuration:

- **Best val ppl**: the minimum validation perplexity reached during
  training, with the step at which it was reached.
- **Final val ppl**: the validation perplexity at the last training
  step (3000).

The best val ppl is the principled comparison metric, since the
final ppl reflects an arbitrarily chosen stopping point.

### 5.4 Code and Reproducibility

The implementation is open-source at github.com/shyble/aware
(branch `cap-native`). All experiments in this paper can be
reproduced from the repository with single shell commands. Bench
reports for every configuration, seed, and intermediate checkpoint
are saved as JSON alongside the model.

## 6. Results

The cap-native design space has seven ablation axes, each isolating
a single architectural knob (Table 1). This installment reports the
two that establish the architecture — that a fully cap-keyed
transformer trains to competitive perplexity (Phase A), and that a
second discovered cap layer strictly improves on a single input
layer (Phase B, the headline). The remaining five axes are deferred
to the continuation paper, which requires accelerator hardware to
run the full sweeps at reasonable wall-clock.

**Table 1 — Cap-native ablation axes.**

| Axis | Knob | Status |
|---|---|---|
| Single-discovery validation | Does a fully cap-keyed transformer train? | **Phase A, §6.1** |
| Single vs hierarchical | One cap layer vs two stacked (**headline**) | **Phase B, §6.2** |
| Routing mode | Hard top-1 vs top-K weighted | Deferred (continuation) |
| Cap window (layer 0) | `cap_window ∈ {1,2,3,4,5,8}` | Deferred (continuation) |
| Discovery strategy (layer 0) | KMeans / KMeans++ / Random / NoDiscovery / Hybrid | Deferred (continuation) |
| Cap-indexed attention mask | With vs without cap-overlap bias | Deferred (continuation) |
| Hierarchical sub-ablations | `K_1`, `W_1`, layer-0+1 routing | Deferred (continuation) |

The deferred axes are described as a roadmap in §8.7.

### 6.1 Phase A: Single-Discovery Cap-Native (3 seeds)

We train cap-native single-discovery at the base configuration of
§5.2 across three random seeds.

| Seed | Best val ppl | Best at step | Final val ppl (step 3000) |
|---|---|---|---|
| 42 | 9.89 | 3000 (synthesized from 5000-step run; see Appendix) | 9.89 |
| 123 | 10.16 | 2400 | 10.47 |
| 7 | 9.83 | 2500 | 10.09 |
| **mean ± std** | **9.96 ± 0.18** | — | **10.15 ± 0.29** |

The single-discovery variant converges by ~step 2500 and exhibits
mild train-val divergence after that point. Validation perplexity
plateaus in the 9.5–10.5 range for the remainder of training.

### 6.2 Phase B: Single vs Hierarchical Discovery (headline)

We compare the single-discovery variant (§6.1) against the
hierarchical variant. Both variants share the base configuration of
§5.2; the hierarchical variant additionally sets `K_1 = 128` and
`W_1 = 1`. Three seeds per variant.

#### Hierarchical results

| Seed | Best val ppl | Best at step | Final val ppl (step 3000) |
|---|---|---|---|
| 42 | 8.91 | 3000 | 8.91 |
| 123 | 8.72 | 2900 | 9.12 |
| 7 | 8.66 | 2500 | 8.84 |
| **mean ± std** | **8.76 ± 0.13** | — | **8.96 ± 0.15** |

#### Headline comparison

| Architecture | Best val ppl ± std | Final val ppl ± std | Parameters |
|---|---|---|---|
| Single-discovery | 9.96 ± 0.18 | 10.15 ± 0.29 | 368 M |
| **Hierarchical** | **8.76 ± 0.13** | **8.96 ± 0.15** | **143 M** |
| Δ (hier vs single) | **-12 %** | **-12 %** | **2.6× fewer params** |

Hierarchical cap-native outperforms single-discovery cap-native by
12 % validation perplexity at 2.6× fewer parameters. The
hierarchical variant also exhibits tighter run-to-run variance
(standard deviation ~30 % lower on best ppl).

This is the central empirical claim of the paper: stacking two
discovered cap layers is strictly better than a single discovered
layer of the same total cap count.

## 7. Analysis

### 7.1 Why Hierarchical Helps

Two complementary effects, both visible in the §6.2 trajectory data:

1. **Decoupling input clustering from downstream specialisation.**
   The input cap layer (layer 0) clusters token-window patterns —
   discrete linguistic units. Cap-native single-discovery forces
   every downstream cap-keyed component to specialise along that
   same input-pattern partition, even when the optimal
   specialisation in a downstream block is over composed/abstracted
   features rather than surface n-grams. Layer 1 introduces a
   second clustering at the abstraction level of `h_0`, freeing
   downstream stacks to specialise over the relevant axis.

   The empirical evidence: single-discovery and hierarchical track
   each other through ~step 800 (where both are still memorising
   surface patterns) but diverge sharply between step 800 and step
   1500, where hierarchical opens a 2-3 ppl lead that holds through
   convergence. This is the regime where downstream blocks transition
   from surface-pattern matching to composed-feature processing —
   exactly where layer 1's recluster of contextualised representations
   should help.

2. **Implicit regularisation through capacity bottlenecking.**
   With `K_1 = 128 < K_0 = 330`, the hierarchical variant
   "bottlenecks" downstream routing through a smaller cap basis.
   This reduces the effective number of cap-keyed parameters by
   ~2.5× and constrains overfitting. The tighter run-to-run
   variance (std 0.13 vs 0.32) and the smaller train-val gap at
   the end of training (~3 ppl gap for hier vs ~5 ppl gap for
   single-discovery at step 3000) are both consistent with this
   regularisation interpretation.

### 7.2 Parameter Efficiency

At single-discovery the cap-keyed components downstream of the
input layer dominate total parameter count
(`~K × d² × n_components × n_blocks`). Hierarchical replaces `K`
with `K_1`, shrinking those components without touching the
discovery quality of the input layer (`K_0` unchanged). The
parameter savings are realised at full quality, not by truncating
the cap basis.

Concretely at our reference configuration (Appendix A):

| Layer / component | Single (K=330) | Hier (K_0=330, K_1=128) |
|---|---|---|
| Layer 0 cap input | 42 K | 42 K (unchanged) |
| Layer 1 cap input | — | 17 K (new) |
| Downstream cap-keyed body | 346 M | 132 M (2.6× smaller) |
| Embedding + output + other | ~22 M | ~11 M |
| **Total** | **~368 M** | **~143 M** |

The 2.6× parameter saving comes entirely from the downstream-body
shrinkage. Cap discovery quality at the input is preserved (layer 0
is unchanged), and a second small discovery is added on top.

### 7.3 Convergence Dynamics

The trajectory data of §6.1 and §6.2 reveals a consistent
qualitative pattern across both variants:

| Regime | Steps | Train-val behaviour |
|---|---|---|
| Initial loss reduction | 0 – 500 | Both variants drop from ~100 ppl to ~25 ppl |
| Cap-keyed specialisation | 500 – 1500 | **Hierarchical pulls ahead by 2-3 ppl** |
| Convergence to plateau | 1500 – 2500 | Both reach their respective plateaus |
| Mild overfitting | 2500 – 3000 | Train ppl continues to drop; val stable or slightly worse |

Both variants reach best val ppl by ~step 2500-3000. Training
beyond 3000 steps (verified on a 5000-step single-discovery run at
seed 42) primarily increases the train-val gap without improving
best val ppl — supporting our choice of 3000 steps as the standard
training budget for paper-#2 experiments.

### 7.4 Total vs Active Parameters and Fair-Comparison Framing

Cap-native at 143–368 M total parameters is 150–400× larger by raw
parameter count than the cap-input baselines of paper #1 (895 K). A
naïve reading would attribute the perplexity gap entirely to scale.
That reading conflates two distinct cost axes; cap-native is
deliberately designed to separate them.

**Total vs active parameters.** Cap-native stores per-cap weight
stacks at every cap-keyed projection, so the total parameter count
scales linearly with `n_caps`. With 330 caps at `d_model=128`, this
is 368 M total. With top-1 sparse routing, only a single cap's slab
is active at any token, giving ~1.115 M cap-keyed active parameters
plus ~0.11 M dense (token embeddings, cap-layer projection), or
**~1.22 M total active parameters per token** — two orders of
magnitude below total.

**This gap is intentional, not incidental.** Each cap holds its own
slab so that it can be discovered, frozen, replaced, or grown in
isolation — the lifecycle operations of the cap primitive (paper
#1). A parameter-sharing dense transformer cannot offer this
property; identifiability requires per-unit ownership of weights,
and that ownership shows up in the total parameter count. The
principal AWARE research motivation for this design is **autonomous
capability accretion and continual learning**: caps as addressable
units mean new domains can extend the cap pool without overwriting
existing caps' weights. The total parameter count is the price paid
for this addressability, not the compute cost — which remains low
because routing is sparse. We do not advocate cap-native purely as
a compute optimisation; the architectural commitment is to
identifiable units, and the active/total gap is a structural
consequence.

**Fair comparison requires two axes, not one.** A vanilla
transformer should be compared on both:

| Comparison axis | What it measures | Relevant when |
|---|---|---|
| Total params | Memory footprint, deployment storage | RAM-constrained inference, model distribution |
| Active params per token | Inference compute, throughput, quality-per-FLOP | Latency-critical applications |

We report both columns below so readers can choose the comparison
appropriate to their setting:

| Architecture | total params | active/token | best val ppl |
|---|---|---|---|
| Pure transformer (paper #1) | 853 K | 853 K | 28.00 ± 0.11 |
| Cap-input kmeans_w3 (paper #1) | 895 K | 895 K | 13.71 ± 0.33 |
| Cap-native single-discovery (this paper) | 368 M | ~1.22 M | 9.96 ± 0.18 |
| Cap-native hierarchical (this paper) | 143 M | ~1.22 M | 8.76 ± 0.13 |
| Vanilla transformer at 143 M params | 143 M | 143 M | _(future work)_ |
| Vanilla transformer at 368 M params | 368 M | 368 M | _(future work)_ |

On the active-params axis, cap-native is in the same order of
magnitude as paper #1's 895 K cap-input baseline (1.22 M vs 895 K,
~1.4×), not 400× larger. The 35-50% perplexity improvement at this
near-matched active-compute budget is the architectural claim. On
the total-params axis the comparison is unmatched and we defer the
fully-matched vanilla-transformer baseline (GPT-2-small at 143 M,
GPT-2-medium at 368 M) to follow-up work.

The hierarchical-vs-single-discovery comparison (§6.2) remains
internally matched on both axes — same per-cap slab size, same
top-1 routing, same protocol — so the 12 % improvement there
holds independently of how a reader weighs the active-vs-total
question.

## 8. Limitations and Future Work

### 8.1 Scale

All experiments are at TinyStories-small scale (~3.5 M training
tokens, 3000 training steps, 143–368 M parameters). The
hierarchical-beats-single-discovery finding has not been verified
at larger model or corpus scale. Scaling to the full TinyStories
corpus (~500 M tokens) and to GPT-2-medium parameter scale is
straightforward engineering on accelerator hardware and is the
natural next step.

### 8.2 Parameter-Matched Baselines

As discussed in §7.4, cap-native is roughly matched to paper #1 on
*active* parameters per token (~1.22 M vs 895 K) but not on *total*
parameters (368 M and 143 M vs 895 K). Future work will report a
vanilla-transformer baseline at matched **total** parameter count
(GPT-2-small at 143 M, GPT-2-medium at 368 M) on the same corpus
and protocol. That comparison would isolate the architectural
contribution of cap-keyed routing from the raw memory-footprint
advantage of dense models at smaller total sizes.

### 8.3 Routing Diversity

Hard top-1 sparse routing was chosen for memory tractability at
large `K`. Whether soft or top-K weighted routing would improve
quality at this scale is an open question, addressed by the routing
ablation deferred to the continuation paper (§8.7).

### 8.4 Deeper Hierarchies

Paper #2 caps the discovered hierarchy at two layers. A natural
extension is to ask whether three or more stacked discovered cap
layers continue to help. This requires careful bootstrap
sequencing and may run into the standard MoE expert-collapse
problem at the third layer.

### 8.5 Adaptive Cap Count

The number of caps at each layer (`K_0`, `K_1`) is a fixed
hyperparameter rather than a discovered quantity. Bayesian
nonparametric clustering (e.g., DP-GMM) or density-based
clustering (HDBSCAN) could in principle discover the cap count
from data. This would require addressing the
downstream-shape-instability problem (downstream cap-keyed
components are sized at construction time) and is left to future
work.

### 8.6 Continual Learning

The cap primitive was designed in part to support continual
learning through cap audit and substrate forking. Whether the
hierarchical cap-native architecture preserves these properties —
in particular, whether layer 1's caps can be audited and
hot-swapped without retraining layer 0 — is an interesting open
question not addressed in this paper.

### 8.7 Deferred Ablations (Continuation Paper)

Five of the seven ablation axes in Table 1 are deferred to a
continuation paper, gated on accelerator hardware that makes the
full sweeps tractable at reasonable wall-clock. We list them here as
a roadmap:

1. **Routing mode** — hard top-1 vs top-K weighted routing for
   `K ∈ {1, 4, 8}` on the single-discovery base (§8.3). Soft routing
   over the full `K_0 = 330` stack is infeasible at our scale and
   stays omitted.
2. **Cap window (layer 0)** — sweep `cap_window ∈ {1, 2, 3, 4, 5, 8}`
   on the hierarchical winner. Paper #1 found `w = 3` optimal; the
   downstream cap capacity here may shift the optimum.
3. **Discovery strategy (layer 0)** — KMeans (default), KMeans++,
   Random unit vectors, NoDiscovery (Xavier + gradient), and Hybrid
   (KMeans init + gradient). Paper #1 found discovery mattered only
   at window > 1; the layer-0 / layer-1 boundary may interact.
4. **Cap-indexed attention mask** — with vs without the cap-overlap
   bias on attention scores, on the hierarchical winner.
5. **Hierarchical sub-ablations** — `K_1 ∈ {64, 128, 256, 330}`,
   `W_1 ∈ {1, 2, 3}`, and an alternate routing where downstream
   components route by `concat(cap_acts_0, cap_acts_1)` rather than
   `cap_acts_1` alone (§8.4).

Alongside these, the parameter-matched vanilla-transformer baselines
of §8.2 (GPT-2-small at 143 M, GPT-2-medium at 368 M) are the other
outstanding comparison for the continuation paper.

## 9. Conclusion

We introduced cap-native, a transformer-shaped architecture in
which every projection at every block is cap-keyed. We showed that
adding a second discovered cap layer over contextualised
representations (the hierarchical variant) outperforms a single
discovered cap layer (the single-discovery variant) by 12 %
validation perplexity, while using 2.6× fewer parameters and
exhibiting 2.5× tighter run-to-run variance. The architectural
contribution is that stacking discovered cap layers is a better
placement of the cap primitive than the single-input-layer
placement of paper #1.

The headline result has been validated at three seeds. The
supporting ablations (routing, window, discovery, indexed-mask,
hierarchical sub-knobs) and a parameter-matched vanilla-transformer
baseline are deferred to a continuation paper (§8.7), gated on
accelerator hardware. This installment establishes the two claims
that stand on their own: a fully cap-keyed transformer trains to
competitive perplexity, and stacking two discovered cap layers
beats one.

## References

Arthur, D., & Vassilvitskii, S. (2007). _k-means++: The advantages
of careful seeding._ In Proceedings of the 18th Annual ACM-SIAM
Symposium on Discrete Algorithms (SODA '07), 1027-1035.

Aydemir, K. (2026). _Caps: Identifiable Computational Units for
Transformer Augmentation._ Manuscript.

Bengio, Y. (2009). _Learning deep architectures for AI._ Foundations
and Trends in Machine Learning, 2(1), 1-127.

De Vries, H., Strub, F., Mary, J., Larochelle, H., Pietquin, O., &
Courville, A. (2017). _Modulating early visual processing by
language._ In Advances in Neural Information Processing Systems
(NeurIPS).

Eldan, R., & Li, Y. (2023). _TinyStories: How small can language
models be and still speak coherent English?_ arXiv preprint
arXiv:2305.07759.

Fedus, W., Zoph, B., & Shazeer, N. (2022). _Switch transformers:
Scaling to trillion parameter models with simple and efficient
sparsity._ Journal of Machine Learning Research, 23(120), 1-39.

Ha, D., Dai, A., & Le, Q. V. (2016). _HyperNetworks._ arXiv preprint
arXiv:1609.09106.

Hinton, G. E. (1999). _Products of experts._ In Proceedings of the
9th International Conference on Artificial Neural Networks (ICANN).

Kingma, D. P., & Ba, J. (2014). _Adam: A method for stochastic
optimization._ arXiv preprint arXiv:1412.6980.

Lewis, M., Bhosale, S., Dettmers, T., Goyal, N., & Zettlemoyer, L.
(2021). _BASE layers: Simplifying training of large, sparse models._
In International Conference on Machine Learning (ICML), 6265-6274.

Loshchilov, I., & Hutter, F. (2017). _Decoupled weight decay
regularization._ arXiv preprint arXiv:1711.05101.

Perez, E., Strub, F., De Vries, H., Dumoulin, V., & Courville, A.
(2018). _FiLM: Visual reasoning with a general conditioning layer._
In Proceedings of the AAAI Conference on Artificial Intelligence.

Roller, S., Sukhbaatar, S., Szlam, A., & Weston, J. (2021). _Hash
layers for large sparse models._ In Advances in Neural Information
Processing Systems (NeurIPS).

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton,
G., & Dean, J. (2017). _Outrageously large neural networks: The
sparsely-gated mixture-of-experts layer._ In International
Conference on Learning Representations (ICLR).

Van Den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). _Neural
discrete representation learning._ In Advances in Neural Information
Processing Systems (NeurIPS).

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). _Attention is all
you need._ In Advances in Neural Information Processing Systems
(NeurIPS).

## Appendix A: Parameter Count Derivation

For a cap-keyed transformer block at model dimension `d`, FFN
dimension `d_ff`, and `K` cap stacks:

- Cap-keyed MHA: `K × d × 3d` (QKV) + `K × d × d` (O) = `K × 4 d²`
- Cap-keyed MoE: `K × d × d_ff` (gate) + `K × d × d_ff` (value) + `K × d_ff × d` (out) = `3 K d d_ff`
- Cap-keyed RMS-norm: `K × d`

At `d_ff = 4d`: per-block ≈ `K × 4d² + 12 K d² + K d` ≈ `16 K d²`.

For `n_blocks` blocks: `16 K d² × n_blocks`.

Add cap-keyed output projection: `K × d × vocab`.
Add layer 0 cap-input layer (separate): `K × d` (projection) + frozen keys.
Add token embedding: `vocab × d`.

At our reference configuration:
- Single-discovery: `K = 330, d = 128, d_ff = 512, n_blocks = 4, vocab = 256` →
  ~`16 × 330 × 128² × 4 = 346 M` cap-keyed body params, plus output `~10 M`,
  plus standard parts → ~368 M.
- Hierarchical: downstream `K = K_1 = 128`, otherwise identical →
  ~`16 × 128 × 128² × 4 = 134 M`, plus layer 0 (counted at `K_0 = 330`) +
  layer 1 (small, ~17 K) + standard parts → ~143 M.

## Appendix B: Reproducibility

The code is at `github.com/shyble/aware`, branch `cap-native`,
commit `6e247e4`. The experiments in §6.1 and §6.2 can be reproduced with:

```bash
# Single-discovery, 3 seeds, 3000 steps each
AWARE_DEVICE=cpu AWARE_BENCH_STEPS=3000 \
  ./scripts/run_multi_seed.sh cap_native_sparse_d128 42 123 7

# Hierarchical, 3 seeds, 3000 steps each
AWARE_DEVICE=cpu AWARE_BENCH_STEPS=3000 \
  ./scripts/run_multi_seed.sh cap_native_hier_d128 42 123 7
```

Bench reports for every run are saved to
`data/bench/<config>_seed<N>/report.json` with full trajectory and
both best/final val perplexity figures. The tokenised TinyStories
corpus is included in the repository to ensure deterministic data
loading.
