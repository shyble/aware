# GPU dispatch optimisation for cap-native

Design note, Aug 2026. Written after measuring the 368M single-discovery
config at **30.4 s/step on an RTX 4060** versus **15.1 s/step on an M1 Pro
CPU** — the GPU was twice as slow.

That measurement is deliberately not reported anywhere: it describes our
reference implementation on one memory-constrained card with an
unoptimised backend, not a property of the architecture. MoE systems
(Switch Transformer, Megablocks) run sparse routing on accelerators
efficiently, so sparse dispatch is clearly GPU-friendly given the right
kernels. Ours are written for correctness.

## Root cause

`attention.rs:165`, and the same pattern in `moe.rs`, `output.rs`, `norm.rs`:

```rust
let winners_t = cap_flat.argmax(D::Minus1)?;
let winners: Vec<u32> = winners_t.to_vec1::<u32>()?;   // GPU -> CPU sync
let routing = bounded_grouped_routing(&winners, ...)?;  // sort in Rust, on CPU
```

Every cap-keyed component pulls routing off the device, sorts 4096 tokens
single-threaded, and pushes permutation tensors back. That is a full
pipeline stall, and it happens roughly 8x per block (attention QKV,
attention output, three MoE projections, norms, output head) x 4 blocks =
~32 round-trips per forward, doubled through backward. On CPU the same
code costs nothing, because there is no transfer.

## The decisive fact

`substrate.rs:46` computes `cap_acts` **once** and passes the same tensor
to every block. So every one of those ~32 routing computations consumes an
identical input and returns an identical result.

## Measured outcome (Aug 2026, RTX 4060)

Changes 1-2 are implemented and verified.

| config | baseline | with change | |
|---|---|---|---|
| single-discovery 368M, 330 caps | 30.4 s/step | **19.6 s/step** | 35% faster |
| hierarchical 143M, 128 caps | 0.453 s/step | 0.446 s/step | ~unchanged |

Both bit-identical to baseline on the same GPU and seed: single-discovery
163.69 = 163.69, hierarchical 166.52 = 166.52.

**The saving is routing *construction*, not sync latency.** The first
prediction here was that GPU->CPU round-trips dominated; that was wrong.
`bounded_grouped_routing` builds bucket offsets and a `pad_perm` tensor of
size `n_caps x bound`, so its cost scales with cap count. At 330 caps that
work is ~2.6x heavier than at 128, and it ran 32 times per forward. Hence
the large gain on single-discovery and none on hierarchical.

**Still slower than CPU for single-discovery**: 19.6 s/step on the 4060
versus 15.1 s/step on an M1 Pro. What remains is the dispatch itself -
gather/scatter plus a padded matmul over 330 buckets - and VRAM pressure
at ~5.9 GB static on an 8 GB card. Those are changes 3-5 below.

Note the config dependence: hierarchical (143M, 128 caps) runs at
0.446 s/step on the same GPU, 9x faster than CPU. The problem is specific
to many caps and a large model on a small card, not to cap-native.

## Proposed changes, ordered by safety

### 1. Hoist routing to one call per block — DONE, verified bit-identical
Compute `bounded_grouped_routing` once per block, thread it to the
components. Same input tensor, same deterministic `argmax`, same output.
Cannot change results. Helps CPU too.

### 2. Cache routing across all blocks — DONE, verified bit-identical
Since `cap_acts` is shared substrate-wide, one routing structure serves
every block. Reduces ~32 round-trips to 1. Same argument as (1).

### 3. On-device sort — NEEDS VERIFICATION
Replace `to_vec1` + Rust sort with tensor ops (`argsort`) so routing never
leaves the device. The Rust sort is *stable*, preserving token order within
each cap's bucket; a GPU argsort may not be.

Argument for equivalence: each token's output depends only on its own row
and its assigned weight slab, and the inverse permutation restores order,
so within-bucket ordering should not matter. **That is an argument, not a
proof.** Verify empirically before trusting it.

### 4. Block-sparse kernels — CHANGES NUMERICS
The Megablocks approach: pad buckets to tile boundaries, one block-sparse
matmul. Different tiling means a different floating-point reduction order,
so results will differ in the last digits. Statistically equivalent, not
bit-identical. This is the real fix if cap-native ever needs to scale on
accelerators.

### 5. Larger card
5.9 GB static state on an 8 GB card leaves the allocator no room. A
16-24 GB card removes memory pressure as a confound and should be measured
before concluding anything about GPU behaviour.

## Verification protocol — required, not optional

For changes (1) and (2), the claim is bit-identical output. Test it:

```bash
# baseline exists: data/bench_wikitext_capnative/cap_native_hier_d128_seed42
# after the change, re-run the same config and seed, then diff final ppl
```

Expect an exact match. If it differs at all, the refactor moved something
we did not intend and the change is not what it claims to be. For (3) and
(4), expect small differences and compare distributions across seeds
instead.

## Timing

Do not do this while experiments are running. Modifying the dispatch path
mid-campaign makes it impossible to tell whether a seed differed because of
the seed or the code. Finish any in-flight campaign first.

## How to describe this

Not as a result. As a future-work note:

> The reference implementation dispatches per-cap projections through
> grouped matmuls with a per-bucket overflow path, which is not competitive
> with the block-sparse kernels used by production mixture-of-experts
> systems. Efficient accelerator dispatch for cap-keyed projections is left
> to future work.
