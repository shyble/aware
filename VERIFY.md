# gpu-dispatch: verification protocol

This branch changes how sparse cap routing is dispatched. Two of the
planned changes claim to be **bit-identical** to `cap-native`; that claim
must be tested, not assumed.

## Baseline (already measured, on cap-native)

WikiText-103, 5000 steps, corpus-specific BPE:

| config | seed | final val ppl |
|---|---|---|
| cap_native_hier_d128 | 42 | 14.58 |
| cap_native_hier_d128 | 123 | 14.33 |
| cap_native_hier_d128 | 7 | 14.62 |

## Test after each change

```bash
AWARE_BENCH_CORPUS=data/wikitext103/wikitext_train.txt \
AWARE_BENCH_VAL_CORPUS=data/wikitext103/wikitext_val.txt \
AWARE_BENCH_BPE_DIR=data/bpe_wikitext103 \
AWARE_BENCH_OUTPUT_DIR=data/bench_gpudispatch \
AWARE_BENCH_STEPS=5000 \
  ./scripts/run_multi_seed.sh cap_native_hier_d128 42
```

Compare against 14.58.

## What counts as passing

- **Changes 1-2 (hoist / cache routing):** must match **14.58 exactly**.
  Any difference means the refactor altered something unintended - stop and
  find out what.
- **Change 3 (on-device sort):** expected identical, but stability of the
  GPU sort is unproven. A small difference is a finding, not a pass.
- **Change 4 (block-sparse kernels):** will differ in the last digits by
  design. Compare across three seeds and check the distribution overlaps.

A cheaper smoke check for iteration: 100 steps and compare the step-100
perplexity, which is deterministic for the same seed and catches gross
breakage in minutes rather than hours.

## Results (Aug 2026)

Both changes verified on an RTX 4060, comparing branches on the *same*
device so the code is the only variable:

| config | cap-native | gpu-dispatch | verdict |
|---|---|---|---|
| hier, seed 42, step 100 | ppl 166.52 / 45.3s | ppl 166.52 / 44.6s | identical, no speedup |
| single-disc, seed 42, step 100 | ppl 163.69 / 3038.9s | ppl 163.69 / 1959.4s | identical, 35% faster |

Bit-identical claim: **confirmed** on both configs.

Note the first attempt compared gpu-dispatch-on-GPU against
cap-native-on-CPU, which changes two variables at once and cannot settle
the question. Always hold the device fixed.
