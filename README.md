# AWARE

**Architectural Primitives for Identifiable, Growable AI**

AWARE introduces **caps** (capability nodes): identifiable, lifecycle-
managed computational units. Each cap carries a stable identifier across
training cycles, supports lifecycle operations (discovery, freezing,
replacement, growth), and exposes its learned content for inspection.
The cap primitive is general; the current implementation integrates it
into transformer-style language models.

The paper introducing caps: [`papers/caps_primitive.md`](papers/caps_primitive.md).

Headline result: cap-input projection with 3-token windows achieves
**val perplexity 13.71 ± 0.33** on TinyStories, a **51% perplexity
reduction** over a same-scale pure-transformer baseline (28.00 ± 0.11),
multi-seed validated across 3 seeds.

---

## Quick start

### Build

```bash
# CPU (default)
cargo build --release

# Apple Silicon GPU
cargo build --release --features metal

# Linux/Windows with CUDA
cargo build --release --features cuda
```

### Reproduce the headline result (kmeans_w3, roughly an hour on modern CPU)

```bash
./scripts/run_multi_seed.sh kmeans_w3
```

This runs three seeds (42, 123, 7) and writes per-run reports to
`data/bench/kmeans_w3_seed{42,123,7}/report.json`. The aggregator
script then summarizes:

```bash
python3 ./scripts/aggregate_benchmarks.py data/bench
```

### Single-run with custom config

```bash
AWARE_BENCH_ID=my_custom_run \
AWARE_BENCH_CORPUS=data/tinystories_small/tinystories_train.txt \
AWARE_BENCH_VAL_CORPUS=data/tinystories_small/tinystories_val.txt \
AWARE_BENCH_STEPS=5000 \
AWARE_BENCH_CAP_DISCOVERY=kmeans \
AWARE_BENCH_CAP_WINDOW=3 \
AWARE_BENCH_SEED=42 \
./target/release/examples/run_benchmark
```

---

## Architecture summary

Three coexisting cap-based architectures share a single set of
primitives (CapMatrix, Discovery, CapLayer):

| Architecture | File | Status |
|---|---|---|
| **Substrate** (cap-augmented transformer) | `src/aware/substrate.rs` | Main result of this codebase |
| **CapNativeSubstrate** (cap-keyed deep) | `src/aware/cap_native/substrate.rs` | Future work; code ready, experiments pending |
| **ConceptLayer** (frozen-identity concepts) | `src/aware/concept_layer.rs` | Future work; code ready |

---

## License

Dual-licensed under either of

- **Apache License 2.0** ([LICENSE-APACHE](LICENSE-APACHE) or
  http://www.apache.org/licenses/LICENSE-2.0)
- **MIT License** ([LICENSE-MIT](LICENSE-MIT) or
  https://opensource.org/licenses/MIT)

at your option. This is the standard Rust ecosystem dual license; you
may use either license when redistributing or modifying this code.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache
License 2.0, shall be dual-licensed as above, without any additional
terms or conditions.
