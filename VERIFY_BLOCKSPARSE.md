# Block-sparse dispatch — status and verification protocol

Branch `block-sparse`, off `gpu-dispatch` (which already contains the
routing hoist: 30.4 → 19.6 s/step on the 368 M single-discovery config).

## Status: WRITTEN, NOT COMPILED

Authored while stage C was mid-run on the CPU machine, so no build was
attempted locally — a release build would have competed for cores with a
run that had ~20 h left. **Expect compile errors on first build.** One was
already caught by reading the candle source (`broadcast_eq` does not
exist; the one-hot now uses `scatter_add`), but tensor code that has never
been type-checked usually has more.

## What it changes

`src/aware/cap_native/blocksparse.rs` — a second dispatch path that builds
routing indices **on the device** instead of on the host.

The existing path (`sparse_routing.rs`) does, per forward:

```rust
let winners: Vec<u32> = winners_t.to_vec1::<u32>()?;   // GPU -> CPU stall
indexed.sort_by_key(...);                              // single-threaded
let mut pad_perm = vec![0; n_caps * bound];            // grows with n_caps
```

The new path replaces all of that with a cumulative sum over a one-hot of
the winners: column `k` of the running sum counts how many tokens so far
chose cap `k`, so reading each token's own column gives its position
inside its bucket. Indices never leave the device. The only host transfer
is one scalar — the largest bucket size — which decides the pass count.

**Not Megablocks.** True block-sparse matmul needs custom kernels candle
does not expose; compute is still a dense batched matmul over padded
capacity blocks. What moved is index construction, not matmul tiling.

**No tokens are dropped.** Capacity-based MoE dispatch normally discards
tokens past capacity, which would change what the model computes. Here
pass `p` handles bucket positions `[p*C, (p+1)*C)` and passes repeat until
every token is covered; skew costs passes, not accuracy.

## Opt-in

Off by default, so the default path and every published number are
untouched:

```bash
AWARE_CN_BLOCKSPARSE=1     # enable
AWARE_CN_BS_CAP_MULT=1     # capacity multiplier over the uniform mean
```

## Verification protocol

Not optional. Two questions, in order.

### 1. Does it produce the same numbers?

Expected: yes. Each token's output is `x_t @ W_{cap(t)}`; the reduction for
one output element runs over `d_in` regardless of block grouping, and zero
padding contributes nothing to other rows. That is an argument, not a
proof — kernel tiling may reassociate — so measure it.

```bash
# same config, same seed, 100 steps, one flag apart
AWARE_BENCH_STEPS=100 AWARE_BENCH_OUTPUT_DIR=/tmp/bs_off \
  ./scripts/run_multi_seed.sh cap_native_sparse_d128 42
AWARE_CN_BLOCKSPARSE=1 AWARE_BENCH_STEPS=100 AWARE_BENCH_OUTPUT_DIR=/tmp/bs_on \
  ./scripts/run_multi_seed.sh cap_native_sparse_d128 42
```

Compare `final_val_perplexity`. Identical → the path is a pure
optimisation. Differing in the last digits → plausible reassociation;
compare across seeds instead and say so in any write-up. Differing
materially → a bug, not a numerics artefact.

### 2. Is it faster?

Only meaningful on CUDA. The interesting config is **single-discovery
(330 caps)**, where index construction scales with `n_caps * bound`.
Hierarchical (128 caps) already runs at 0.446 s/step on a 4060 and has
little to gain — worth measuring, but do not expect movement.

Reference points on the RTX 4060:

| config | CPU path | routing hoist | block-sparse |
|---|---|---|---|
| single-discovery 368 M, 330 caps | 30.4 s/step | 19.6 s/step | ? |
| hierarchical 143 M, 128 caps | 0.453 s/step | 0.446 s/step | ? |

## Do not merge before

Any reported result depends on it. Nothing currently reported uses this
path, the hierarchical configuration barely benefits from it, and changing
the dispatch path mid-campaign makes a seed difference impossible to
attribute.
