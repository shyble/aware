# Scripts & Examples — by Paper

Navigation index mapping every benchmark script, example binary, and
`run_multi_seed.sh` preset to the paper it serves. **Nothing is moved** — this is
an index only, because the shell scripts are path-coupled (see notes at the end).

Papers live under `papers/`:
- **Paper #1** — *Caps: Identifiable Computational Units for Transformer Augmentation* (submitted) — `papers/caps_primitive/`
- **Paper #2** — *Hierarchical Cap-Native Substrates for Language Modeling* (installment one) — `papers/cap_native/`
- **Paper #3** — concept paper (planning only, not active) — no paper dir yet

---

## Shared infrastructure (used across papers — do NOT file under one paper)

| File | Role |
|---|---|
| `scripts/fetch_tinystories.py` | Download/split TinyStories → `data/tinystories/{tinystories_train,tinystories_val}.txt` + `stats.json`. `--out-dir` overridable. Feeds both benchmark binaries. |
| `scripts/aggregate_benchmarks.py` | Walk `data/bench/*/report.json` → `benchmark_results.{csv,md}`, (`--plot`) `benchmark_trajectories.png`. Takes `bench_dir` as `argv[1]`. Called by every runner. |
| `scripts/run_multi_seed.sh` | Single preset dispatcher for **Paper #1 and Paper #2**. Runs one config across N seeds (default `42 123 7`), switching the example binary to `cap_native_run_benchmark` for `cap_native_*` configs, then aggregates. |

---

## Paper #1 — Caps (transformer augmentation)

**Example binary:** `examples/run_benchmark.rs` → `target/release/examples/run_benchmark`

**Scripts:**
| File | Role |
|---|---|
| `scripts/run_benchmarks.sh` | Phases A–D (paper §6.1–6.4): baselines, attention sweep, window sweep, discovery sweep. |
| `scripts/run_phase_I.sh` | Phase I multi-seed confirmation: `pure_transformer`, `kmeans_w1`, `kmeans_w4`, `kmeans_w8` × 3 seeds. Wraps `run_multi_seed.sh`. |
| `scripts/make_figures.py` | 5 publication figures → `papers/caps_primitive/figures/`. Data embedded inline; run from repo root. |

**`run_multi_seed.sh` presets (run on `run_benchmark`):**
`pure_transformer`, `kmeans_w1`, `kmeans_w2`, `kmeans_w3` *(winner)*, `kmeans_w4`,
`kmeans_w5`, `kmeans_w8`, `capmem`, `cappair`

---

## Paper #2 — Hierarchical Cap-Native Substrates

**Example binary:** `examples/cap_native_run_benchmark.rs` → `target/release/examples/cap_native_run_benchmark`

**Scripts:** none of its own — driven entirely through `run_multi_seed.sh` presets.
GPU campaign runbook + driver live in `.local/cap-native-2/` (gitignored).

**`run_multi_seed.sh` presets (run on `cap_native_run_benchmark` unless noted):**

*Core:* `cap_native_full`, `cap_native_indexed`, `cap_native_w3`, `cap_native_sparse`, `cap_native_topk4`

*Headline d=128:* `cap_native_sparse_d128`, `cap_native_sparse_d128_n64`,
`cap_native_hier_d128` *(Phase E winner)*, `cap_native_hier_d128_{n64,n256,n330}` (layer-1 n_caps, Phase G),
`cap_native_hier_d128_{w2,w3}` (layer-1 window, Phase G), `cap_native_hier_d128_kmeanspp` (Phase D),
`cap_native_hier_d128_indexed` (Phase F), `cap_native_hier_d128_{top4,top8}` (Phase B),
`cap_native_hier_d128_{cw1,cw5}` (Phase C, layer-0 window)

*d=64 ablations:* `cap_native_full_d64`, `cap_native_indexed_d64`, `cap_native_w3_d64`,
`cap_native_full_d64_small`, `cap_native_sparse_d64_small`

*Param-matched baselines (run on `run_benchmark`; Paper #2 §7.4 fair-comparison):*
`pure_transformer_d64`, `kmeans_w3_d64`,
`pure_transformer_match_hier` (d=896, n_blocks=16, ~154M — matches `cap_native_hier_d128` ~143M),
`pure_transformer_match_single` (d=1024, n_blocks=24, ~302M — matches `cap_native_sparse_d128` ~368M)

---

## Paper #3 — Concept paper (planning only, not active)

**Example binary:** `examples/train_with_concepts.rs` (standalone; not wired into any runner)

**Scripts / presets:** none. Plan + design notes live in `.local/concept_layer/` (gitignored).

---

## Path-coupling notes (read before moving any file)

- The three `*.sh` scripts find the repo root with `cd "$(dirname "$0")/.."`, which
  assumes they sit exactly one level below repo root (`scripts/`). Moving them deeper
  requires changing `/..` to match the new depth — and the failure is **silent** (wrong
  paths, not an error).
- `run_phase_I.sh` calls `./scripts/run_multi_seed.sh` and `./scripts/aggregate_benchmarks.py`
  by literal path; `run_multi_seed.sh` and `run_benchmarks.sh` call
  `./scripts/aggregate_benchmarks.py` by literal path. Moving any of these requires
  updating the others.
- `aggregate_benchmarks.py` and `fetch_tinystories.py` are relocatable (args/flags, no
  caller-path coupling).
- The `.rs` examples resolve `data/...` paths at runtime from the invoking shell's cwd,
  not from the source location — moving the source doesn't change those. But the example
  *name* (filename stem) is what `--example` and `target/release/examples/<name>` use, so
  do not rename them.
