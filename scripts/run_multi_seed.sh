#!/usr/bin/env bash
# Multi-seed runner: run a single config across N seeds to quantify
# run-to-run variance and confirm a finding is robust.
#
# Usage:
#   ./scripts/run_multi_seed.sh <config_id> [seeds]
#
# Pre-set configs (just pass the config name):
#   pure_transformer    - standard attention, no cap layer (baseline)
#   kmeans_w1           - cap layer (KMeans, window=1) + standard attention
#   kmeans_w2           - cap layer (KMeans, window=2) + standard attention
#   kmeans_w3           - cap layer (KMeans, window=3) + standard attention  (paper winner)
#   kmeans_w4           - cap layer (KMeans, window=4) + standard attention
#   kmeans_w5           - cap layer (KMeans, window=5) + standard attention
#   kmeans_w8           - cap layer (KMeans, window=8) + standard attention
#   capmem              - cap-memory attention (no input cap layer)
#   cappair             - cap-pair attention (no input cap layer)
#
# Default seeds: 42 123 7. Override by passing additional args.
#
# Outputs: data/bench/<config_id>_seed<N>/report.json per seed.
# Aggregator picks them up automatically.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG_ID="${1:-}"
shift || true
DEFAULT_SEEDS=(42 123 7)
SEEDS=("${@:-${DEFAULT_SEEDS[@]}}")

if [ -z "$CONFIG_ID" ]; then
    cat <<USAGE
Usage: $0 <config_id> [seeds...]
Configs: pure_transformer | kmeans_w1 | kmeans_w2 | kmeans_w3 | kmeans_w4 | kmeans_w5 | kmeans_w8 | capmem | cappair
Default seeds: 42 123 7
USAGE
    exit 1
fi

CORPUS="${AWARE_BENCH_CORPUS:-data/tinystories_small/tinystories_train.txt}"
DEFAULT_VAL="${CORPUS/_train/_val}"
[ -f "$DEFAULT_VAL" ] || DEFAULT_VAL=""
VAL_CORPUS="${AWARE_BENCH_VAL_CORPUS:-$DEFAULT_VAL}"
STEPS="${AWARE_BENCH_STEPS:-5000}"
OUTPUT_DIR="${AWARE_BENCH_OUTPUT_DIR:-data/bench}"
# EXAMPLE may be overridden per-config (cap-native uses a different binary).
# Default to run_benchmark; cap_native_* configs override below.
EXAMPLE="./target/release/examples/run_benchmark"
AGGREGATOR="./scripts/aggregate_benchmarks.py"

# The build check happens AFTER the case statement now (so EXAMPLE may have
# been overridden to cap_native_run_benchmark for cap_native_* configs).

# Resolve the config to its env-var args
case "$CONFIG_ID" in
    pure_transformer)
        ARGS=(AWARE_BENCH_ATTENTION=standard AWARE_BENCH_INCLUDE_CAP_LAYER=false)
        ;;
    kmeans_w1)
        ARGS=(AWARE_BENCH_ATTENTION=standard AWARE_BENCH_INCLUDE_CAP_LAYER=true
              AWARE_BENCH_CAP_DISCOVERY=kmeans AWARE_BENCH_CAP_WINDOW=1)
        ;;
    kmeans_w2)
        ARGS=(AWARE_BENCH_ATTENTION=standard AWARE_BENCH_INCLUDE_CAP_LAYER=true
              AWARE_BENCH_CAP_DISCOVERY=kmeans AWARE_BENCH_CAP_WINDOW=2)
        ;;
    kmeans_w3)
        ARGS=(AWARE_BENCH_ATTENTION=standard AWARE_BENCH_INCLUDE_CAP_LAYER=true
              AWARE_BENCH_CAP_DISCOVERY=kmeans AWARE_BENCH_CAP_WINDOW=3)
        ;;
    kmeans_w4)
        ARGS=(AWARE_BENCH_ATTENTION=standard AWARE_BENCH_INCLUDE_CAP_LAYER=true
              AWARE_BENCH_CAP_DISCOVERY=kmeans AWARE_BENCH_CAP_WINDOW=4)
        ;;
    kmeans_w5)
        ARGS=(AWARE_BENCH_ATTENTION=standard AWARE_BENCH_INCLUDE_CAP_LAYER=true
              AWARE_BENCH_CAP_DISCOVERY=kmeans AWARE_BENCH_CAP_WINDOW=5)
        ;;
    kmeans_w8)
        ARGS=(AWARE_BENCH_ATTENTION=standard AWARE_BENCH_INCLUDE_CAP_LAYER=true
              AWARE_BENCH_CAP_DISCOVERY=kmeans AWARE_BENCH_CAP_WINDOW=8)
        ;;
    capmem)
        ARGS=(AWARE_BENCH_ATTENTION=cap_memory AWARE_BENCH_CAP_SOURCE=local
              AWARE_BENCH_INCLUDE_CAP_LAYER=false)
        ;;
    cappair)
        ARGS=(AWARE_BENCH_ATTENTION=cap_pair AWARE_BENCH_CAP_SOURCE=local
              AWARE_BENCH_INCLUDE_CAP_LAYER=false)
        ;;

    # ── d=64 baselines (matched to cap-native d=64 ablations) ──
    pure_transformer_d64)
        ARGS=(AWARE_BENCH_D_MODEL=64 AWARE_BENCH_N_BLOCKS=4 AWARE_BENCH_D_FF=256
              AWARE_BENCH_ATTENTION=standard AWARE_BENCH_INCLUDE_CAP_LAYER=false)
        ;;
    kmeans_w3_d64)
        ARGS=(AWARE_BENCH_D_MODEL=64 AWARE_BENCH_N_BLOCKS=4 AWARE_BENCH_D_FF=256
              AWARE_BENCH_CAP_N_TARGET=64
              AWARE_BENCH_ATTENTION=standard AWARE_BENCH_INCLUDE_CAP_LAYER=true
              AWARE_BENCH_CAP_DISCOVERY=kmeans AWARE_BENCH_CAP_WINDOW=3)
        ;;

    # ── Cap-native architecture configs ──
    # All use the cap_native_run_benchmark example (different EXAMPLE binary).
    cap_native_full)
        EXAMPLE="./target/release/examples/cap_native_run_benchmark"
        ARGS=(AWARE_CN_TOP_K=0 AWARE_CN_CAP_WINDOW=4
              AWARE_CN_PER_BLOCK_CAP_EVALS=true
              AWARE_CN_CAP_INDEXED_MASK=false
              AWARE_CN_ROUTING=soft)
        ;;
    cap_native_indexed)
        EXAMPLE="./target/release/examples/cap_native_run_benchmark"
        ARGS=(AWARE_CN_TOP_K=0 AWARE_CN_CAP_WINDOW=4
              AWARE_CN_PER_BLOCK_CAP_EVALS=true
              AWARE_CN_CAP_INDEXED_MASK=true
              AWARE_CN_ROUTING=soft)
        ;;
    cap_native_w3)
        EXAMPLE="./target/release/examples/cap_native_run_benchmark"
        ARGS=(AWARE_CN_TOP_K=0 AWARE_CN_CAP_WINDOW=3
              AWARE_CN_PER_BLOCK_CAP_EVALS=true
              AWARE_CN_ROUTING=soft)
        ;;
    cap_native_sparse)
        EXAMPLE="./target/release/examples/cap_native_run_benchmark"
        ARGS=(AWARE_CN_TOP_K=1 AWARE_CN_CAP_WINDOW=4
              AWARE_CN_PER_BLOCK_CAP_EVALS=true
              AWARE_CN_ROUTING=sparse)
        ;;
    cap_native_topk4)
        EXAMPLE="./target/release/examples/cap_native_run_benchmark"
        ARGS=(AWARE_CN_TOP_K=4 AWARE_CN_ROUTING=soft
              AWARE_BENCH_CAP_WINDOW=4
              AWARE_BENCH_CAP_DISCOVERY=kmeans)
        ;;

    # ── Stage-1 d=64 presets (cap-native small-scale ablations) ──
    # Compute-tractable configs for soft-routing ablations.
    cap_native_full_d64)
        EXAMPLE="./target/release/examples/cap_native_run_benchmark"
        ARGS=(AWARE_BENCH_D_MODEL=64 AWARE_BENCH_N_BLOCKS=4 AWARE_BENCH_D_FF=256
              AWARE_BENCH_CAP_N_TARGET=64 AWARE_BENCH_CAP_WINDOW=4
              AWARE_BENCH_CAP_DISCOVERY=kmeans
              AWARE_CN_TOP_K=0 AWARE_CN_ROUTING=soft
              AWARE_CN_CAP_INDEXED_MASK=false)
        ;;
    cap_native_indexed_d64)
        EXAMPLE="./target/release/examples/cap_native_run_benchmark"
        ARGS=(AWARE_BENCH_D_MODEL=64 AWARE_BENCH_N_BLOCKS=4 AWARE_BENCH_D_FF=256
              AWARE_BENCH_CAP_N_TARGET=64 AWARE_BENCH_CAP_WINDOW=4
              AWARE_BENCH_CAP_DISCOVERY=kmeans
              AWARE_CN_TOP_K=0 AWARE_CN_ROUTING=soft
              AWARE_CN_CAP_INDEXED_MASK=true)
        ;;
    cap_native_w3_d64)
        EXAMPLE="./target/release/examples/cap_native_run_benchmark"
        ARGS=(AWARE_BENCH_D_MODEL=64 AWARE_BENCH_N_BLOCKS=4 AWARE_BENCH_D_FF=256
              AWARE_BENCH_CAP_N_TARGET=64 AWARE_BENCH_CAP_WINDOW=3
              AWARE_BENCH_CAP_DISCOVERY=kmeans
              AWARE_CN_TOP_K=0 AWARE_CN_ROUTING=soft)
        ;;
    # Smallest cap-native preset (blocks=2, n_caps=32, soft routing) for smoke testing.
    # Soft routing here is intentionally expensive; use *_sparse_* variant for low memory.
    cap_native_full_d64_small)
        EXAMPLE="./target/release/examples/cap_native_run_benchmark"
        ARGS=(AWARE_BENCH_D_MODEL=64 AWARE_BENCH_N_BLOCKS=2 AWARE_BENCH_D_FF=256
              AWARE_BENCH_CAP_N_TARGET=32 AWARE_BENCH_CAP_WINDOW=4
              AWARE_BENCH_CAP_DISCOVERY=kmeans
              AWARE_CN_TOP_K=0 AWARE_CN_ROUTING=soft)
        ;;
    # Sparse-routing counterpart for low-memory smoke tests (HardTop1; only the
    # winning cap's stack is computed per token).
    cap_native_sparse_d64_small)
        EXAMPLE="./target/release/examples/cap_native_run_benchmark"
        ARGS=(AWARE_BENCH_D_MODEL=64 AWARE_BENCH_N_BLOCKS=2 AWARE_BENCH_D_FF=256
              AWARE_BENCH_CAP_N_TARGET=32 AWARE_BENCH_CAP_WINDOW=4
              AWARE_BENCH_CAP_DISCOVERY=kmeans
              AWARE_CN_TOP_K=1 AWARE_CN_ROUTING=sparse)
        ;;

    # ── Stage-2 d=128 sparse presets (cap-native headline runs) ──
    cap_native_sparse_d128)
        EXAMPLE="./target/release/examples/cap_native_run_benchmark"
        ARGS=(AWARE_BENCH_D_MODEL=128 AWARE_BENCH_N_BLOCKS=4 AWARE_BENCH_D_FF=512
              AWARE_BENCH_CAP_N_TARGET=330 AWARE_BENCH_CAP_WINDOW=3
              AWARE_BENCH_CAP_DISCOVERY=kmeans
              AWARE_CN_TOP_K=1 AWARE_CN_ROUTING=sparse)
        ;;
    # Memory-tractable variant: paper #1 block structure (d=128, n_blocks=4)
    # but n_caps reduced from 330 to 64 — per-cap weight stacks at every
    # block force ~5.6 GB floor at n_caps=330. n_caps=64 brings model
    # ~80 M params (~1.4 GB floor + ~3-4 GB peak).
    cap_native_sparse_d128_n64)
        EXAMPLE="./target/release/examples/cap_native_run_benchmark"
        ARGS=(AWARE_BENCH_D_MODEL=128 AWARE_BENCH_N_BLOCKS=4 AWARE_BENCH_D_FF=512
              AWARE_BENCH_CAP_N_TARGET=64 AWARE_BENCH_CAP_WINDOW=3
              AWARE_BENCH_CAP_DISCOVERY=kmeans
              AWARE_CN_TOP_K=1 AWARE_CN_ROUTING=sparse)
        ;;
    # Phase E: hierarchical variant. Layer 0 over token windows (W_0=3,
    # n_caps=330 matched to paper #1); layer 1 discovered over h_0
    # (W_1=1, n_caps_1=128). Downstream cap-keyed components sized to
    # n_caps_1 — much smaller than single-discovery at n=330. Expected
    # ~150M params, ~6-7 GB peak.
    cap_native_hier_d128)
        EXAMPLE="./target/release/examples/cap_native_run_benchmark"
        ARGS=(AWARE_BENCH_D_MODEL=128 AWARE_BENCH_N_BLOCKS=4 AWARE_BENCH_D_FF=512
              AWARE_BENCH_CAP_N_TARGET=330 AWARE_BENCH_CAP_WINDOW=3
              AWARE_BENCH_CAP_DISCOVERY=kmeans
              AWARE_CN_TOP_K=1 AWARE_CN_ROUTING=sparse
              AWARE_CN_HIERARCHICAL=true
              AWARE_CN_L1_N_CAPS=128 AWARE_CN_L1_WINDOW=1
              AWARE_CN_L1_DISCOVERY=kmeans)
        ;;
    *)
        echo "Unknown config: $CONFIG_ID"
        echo "Configs (cap-augmented transformer): pure_transformer kmeans_w1 kmeans_w2 kmeans_w3 kmeans_w4 kmeans_w5 kmeans_w8 capmem cappair"
        echo "Configs (cap-native): cap_native_full cap_native_indexed cap_native_w3 cap_native_sparse cap_native_topk4"
        exit 1
        ;;
esac

# Build the right example binary (default run_benchmark, or cap_native_run_benchmark
# if a cap_native_* config overrode EXAMPLE above).
EXAMPLE_NAME=$(basename "$EXAMPLE")
# AWARE_FEATURES selects the candle backend at build time: candle (CPU,
# default), metal, or cuda. The example binary is shared across
# backends, so the active feature flag determines which device the
# runtime will see. Match this with AWARE_DEVICE at runtime.
AWARE_FEATURES="${AWARE_FEATURES:-candle}"
if [ ! -x "$EXAMPLE" ]; then
    echo "[bench] building $EXAMPLE_NAME (features=$AWARE_FEATURES)…"
    cargo build --release --features "$AWARE_FEATURES" --example "$EXAMPLE_NAME"
fi

echo "[multi-seed] config=$CONFIG_ID  seeds=${SEEDS[*]}  steps=$STEPS"

for seed in "${SEEDS[@]}"; do
    id="${CONFIG_ID}_seed${seed}"
    echo
    echo "════════ $id ════════"
    AWARE_BENCH_ID="$id" \
    AWARE_BENCH_CORPUS="$CORPUS" \
    AWARE_BENCH_VAL_CORPUS="$VAL_CORPUS" \
    AWARE_BENCH_OUTPUT_DIR="$OUTPUT_DIR" \
    AWARE_BENCH_STEPS="$STEPS" \
    AWARE_BENCH_SEED="$seed" \
    env "${ARGS[@]}" "$EXAMPLE" || echo "[multi-seed] $id FAILED" >&2
done

echo
echo "[multi-seed] all seeds done for $CONFIG_ID"
if [ -x "$AGGREGATOR" ]; then
    python3 "$AGGREGATOR" "$OUTPUT_DIR" || true
fi

# Summary: average val_ppl across seeds for this config
echo
echo "[multi-seed] summary for $CONFIG_ID:"
python3 - <<PYEOF
import json, statistics
from pathlib import Path

bench = Path("$OUTPUT_DIR")
config = "$CONFIG_ID"
ppls = []
runs = []
for p in sorted(bench.glob(f"{config}_seed*/report.json")):
    with open(p) as f:
        r = json.load(f)
    ppl = r.get("final_val_perplexity")
    if ppl is not None:
        ppls.append(ppl)
        runs.append((r["seed"], ppl))

if not ppls:
    print(f"  No reports for {config}")
else:
    print(f"  Runs: {len(ppls)}")
    for s, p in runs:
        print(f"    seed={s:>5}  val_ppl={p:.2f}")
    mean = statistics.mean(ppls)
    sd = statistics.stdev(ppls) if len(ppls) >= 2 else 0.0
    print(f"  mean = {mean:.2f}  std = {sd:.2f}  ({len(ppls)} seeds)")
PYEOF
