# Scripts & Examples — by architecture line

Navigation index mapping every benchmark script, example binary, and
`run_multi_seed.sh` preset to the architecture it exercises. **Nothing is
moved** — this is an index only, because the shell scripts are path-coupled
(see notes at the end).

Two architecture lines share this repo:

- **Cap-augmented transformer** — a discovered cap layer between embeddings
  and an otherwise standard transformer stack, plus two cap-based attention
  placements evaluated against it.
- **Cap-native** — every learned projection at every block keyed by a cap,
  with per-token top-1 routing. Single-discovery and hierarchical variants.

---

## Shared infrastructure (used by both lines)

| File | Role |
|---|---|
| `scripts/fetch_tinystories.py` | Download/split TinyStories → `data/tinystories/{tinystories_train,tinystories_val}.txt` + `stats.json`. `--out-dir` overridable. Feeds both benchmark binaries. |
| `scripts/fetch_wikitext.py` | Stream WikiText-103 → `data/wikitext103/`. Writes LF endings explicitly; the corpus feeds a BPE, so a stray carriage return changes the token stream. |
| `scripts/aggregate_benchmarks.py` | Walk `<bench_dir>/*/report.json` → `benchmark_results.{csv,md}`, (`--plot`) `benchmark_trajectories.png`. Takes `bench_dir` as `argv[1]`. Called by every runner. |
| `scripts/run_multi_seed.sh` | Single preset dispatcher for both lines. Runs one config across N seeds (default `42 123 7`), switching the example binary to `cap_native_run_benchmark` for `cap_native_*` configs, then aggregates. |
| `scripts/run_backend_calibration.sh` | Trains one config on a second backend and compares against the first. Exists because results are spread across a CPU laptop and a CUDA box, and comparing perplexities across backends needs the equivalence measured rather than assumed. |

**Always set `AWARE_BENCH_OUTPUT_DIR`** for runs worth keeping. It defaults
to `data/bench/`, so unset runs overwrite each other — earlier results have
been lost that way.

---

## Cap-augmented transformer

**Example binary:** `examples/run_benchmark.rs` → `target/release/examples/run_benchmark`

**Scripts:**
| File | Role |
|---|---|
| `scripts/run_benchmarks.sh` | Baselines, attention sweep, window sweep, discovery sweep. |
| `scripts/run_phase_I.sh` | Multi-seed confirmation: `pure_transformer`, `kmeans_w1`, `kmeans_w4`, `kmeans_w8` × 3 seeds. Wraps `run_multi_seed.sh`. |
| `scripts/run_dense_sweep.sh` | Dense transformers at 3.3M / 10.8M / 25.4M for the size-to-perplexity curve. GPU-friendly; ~203h on CPU. |
| `scripts/make_figures.py` | Result figures → `papers/caps_primitive/figures/` (gitignored). Data embedded inline; run from repo root. |

**`run_multi_seed.sh` presets:**
`pure_transformer`, `kmeans_w1`, `kmeans_w2` *(WikiText optimum)*,
`kmeans_w3` *(TinyStories optimum)*, `kmeans_w4`, `kmeans_w5`, `kmeans_w8`,
`capmem`, `cappair`, plus the `*_multihead*` variants that vary head count
and cap-matrix discovery independently.

*Dense scaling:* `dense_3m`, `dense_10m`, `dense_30m`

---

## Cap-native

**Example binary:** `examples/cap_native_run_benchmark.rs` → `target/release/examples/cap_native_run_benchmark`

**Scripts:** driven mostly through `run_multi_seed.sh` presets, plus
`scripts/run_stage_c.sh` for the single-discovery WikiText campaign and
`scripts/run_blocksparse_ab.sh` for the device-routed dispatch comparison.

**`run_multi_seed.sh` presets:**

*Core:* `cap_native_full`, `cap_native_indexed`, `cap_native_w3`, `cap_native_sparse`, `cap_native_topk4`

*Headline d=128:* `cap_native_sparse_d128`, `cap_native_sparse_d128_n64`,
`cap_native_hier_d128` *(the reported hierarchical config)*,
`cap_native_hier_d128_{n64,n256,n330}` (layer-1 cap count),
`cap_native_hier_d128_{w2,w3}` (layer-1 window),
`cap_native_hier_d128_kmeanspp`, `cap_native_hier_d128_indexed`,
`cap_native_hier_d128_{top4,top8}` (routing),
`cap_native_hier_d128_{cw1,cw5}` (layer-0 window)

Most of these ablation axes are implemented but unmeasured — the presets
exist so a run is one command away, not because results exist.

*d=64 ablations:* `cap_native_full_d64`, `cap_native_indexed_d64`, `cap_native_w3_d64`,
`cap_native_full_d64_small`, `cap_native_sparse_d64_small`

*Total-parameter-matched dense baselines* (run on `run_benchmark`):
`pure_transformer_d64`, `kmeans_w3_d64`,
`pure_transformer_match_hier` (d=896, n_blocks=16, ~154M),
`pure_transformer_match_single` (d=1024, n_blocks=24, ~302M).
These are undertrained by construction at current corpus sizes — roughly
0.08 tokens per parameter against a compute-optimal ~20 — so they would
measure data starvation rather than architecture. Kept for a larger corpus.

---

## Dispatch experiments

`kernels/grouped_gemm.cu` plus `src/aware/cap_native/{blocksparse,grouped}.rs`
hold work on cap dispatch efficiency. Both paths are opt-in
(`AWARE_CN_BLOCKSPARSE`, `AWARE_CN_GROUPED_GEMM`) and off by default, so
existing results are unaffected. `VERIFY_BLOCKSPARSE.md` records what is
measured and what is still unverified.

---

## Concept layer

**Example binary:** `examples/train_with_concepts.rs` (standalone; not wired
into any runner). Design notes live in `.local/concept_layer/` (gitignored).

---

## Path-coupling notes (read before moving any file)

- The `*.sh` scripts find the repo root with `cd "$(dirname "$0")/.."`, which
  assumes they sit exactly one level below repo root (`scripts/`). Moving them deeper
  requires changing `/..` to match the new depth — and the failure is **silent** (wrong
  paths, not an error).
- `run_phase_I.sh` calls `./scripts/run_multi_seed.sh` and `./scripts/aggregate_benchmarks.py`
  by literal path; `run_multi_seed.sh` and `run_benchmarks.sh` call
  `./scripts/aggregate_benchmarks.py` by literal path. Moving any of these requires
  updating the others.
- `aggregate_benchmarks.py`, `fetch_tinystories.py` and `fetch_wikitext.py` are
  relocatable (args/flags, no caller-path coupling).
- The `.rs` examples resolve `data/...` paths at runtime from the invoking shell's cwd,
  not from the source location — moving the source doesn't change those. But the example
  *name* (filename stem) is what `--example` and `target/release/examples/<name>` use, so
  do not rename them.
