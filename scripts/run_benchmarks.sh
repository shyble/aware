#!/usr/bin/env bash
# Benchmark sweep - Phases A through D (matches paper sections 6.1-6.4).
#
# Usage:
#   ./scripts/run_benchmarks.sh                # run all phases
#   ./scripts/run_benchmarks.sh A              # only Phase A
#   ./scripts/run_benchmarks.sh A B C          # specific phases
#
# Output: data/bench/<run_id>/report.json per run + substrate.safetensors.
# Aggregate at the end via scripts/aggregate_benchmarks.py (or equivalent).

set -euo pipefail
cd "$(dirname "$0")/.."

CORPUS="${AWARE_BENCH_CORPUS:-data/tinystories_small/tinystories_train.txt}"
# Default: derive val path from train path by swapping _train -> _val.
# Override with AWARE_BENCH_VAL_CORPUS if you want a custom held-out file.
DEFAULT_VAL="${CORPUS/_train/_val}"
if [ ! -f "$DEFAULT_VAL" ] || [ "$DEFAULT_VAL" = "$CORPUS" ]; then
    DEFAULT_VAL=""
fi
VAL_CORPUS="${AWARE_BENCH_VAL_CORPUS:-$DEFAULT_VAL}"
STEPS="${AWARE_BENCH_STEPS:-5000}"
SEED="${AWARE_BENCH_SEED:-42}"
OUTPUT_DIR="${AWARE_BENCH_OUTPUT_DIR:-data/bench}"
EXAMPLE="./target/release/examples/run_benchmark"
AGGREGATOR="./scripts/aggregate_benchmarks.py"

if [ -n "$VAL_CORPUS" ]; then
    echo "[bench] val corpus: $VAL_CORPUS"
else
    echo "[bench] WARNING: no val corpus found; runner will slice train"
fi

if [ $# -eq 0 ]; then
    PHASES=(A B C D)
else
    PHASES=("$@")
fi

ensure_build() {
    if [ ! -x "$EXAMPLE" ]; then
        echo "[bench] building example…"
        cargo build --release --features candle --example run_benchmark
    fi
}

run_one() {
    local id="$1"; shift
    echo
    echo "════════ $id ════════"
    AWARE_BENCH_ID="$id" \
    AWARE_BENCH_CORPUS="$CORPUS" \
    AWARE_BENCH_VAL_CORPUS="$VAL_CORPUS" \
    AWARE_BENCH_OUTPUT_DIR="$OUTPUT_DIR" \
    AWARE_BENCH_STEPS="$STEPS" \
    AWARE_BENCH_SEED="$SEED" \
    "$@" "$EXAMPLE" || { echo "[bench] $id FAILED" >&2; }
}

# ─────────────────────────────── Phase A: Baselines ───────────────────────────────
phase_A() {
    # A1: pure transformer (no cap layer, standard attention)
    run_one A1_pure_transformer env \
        AWARE_BENCH_ATTENTION=standard \
        AWARE_BENCH_INCLUDE_CAP_LAYER=false

    # A2: NoDiscovery cap layer at input
    run_one A2_nodiscovery_caps env \
        AWARE_BENCH_ATTENTION=standard \
        AWARE_BENCH_INCLUDE_CAP_LAYER=true \
        AWARE_BENCH_CAP_DISCOVERY=nodiscovery

    # A3: KMeans-discovered cap layer
    run_one A3_kmeans_caps env \
        AWARE_BENCH_ATTENTION=standard \
        AWARE_BENCH_INCLUDE_CAP_LAYER=true \
        AWARE_BENCH_CAP_DISCOVERY=kmeans

    # A4: frozen-random caps (reservoir style)
    run_one A4_random_frozen_caps env \
        AWARE_BENCH_ATTENTION=standard \
        AWARE_BENCH_INCLUDE_CAP_LAYER=true \
        AWARE_BENCH_CAP_DISCOVERY=random
}

# ─────────────────────────────── Phase B: Attention sweep ─────────────────────────
phase_B() {
    # B1 ≡ A2 (skip)
    # B2: cap-memory + per-block cap matrices
    run_one B2_capmem_local env \
        AWARE_BENCH_ATTENTION=cap_memory \
        AWARE_BENCH_CAP_SOURCE=local \
        AWARE_BENCH_INCLUDE_CAP_LAYER=false

    # B3: cap-memory + shared global cap matrix
    run_one B3_capmem_shared env \
        AWARE_BENCH_ATTENTION=cap_memory \
        AWARE_BENCH_CAP_SOURCE=shared \
        AWARE_BENCH_INCLUDE_CAP_LAYER=false

    # B4: cap-memory + input cap layer (Config 4)
    run_one B4_capmem_with_input_caps env \
        AWARE_BENCH_ATTENTION=cap_memory \
        AWARE_BENCH_CAP_SOURCE=local \
        AWARE_BENCH_INCLUDE_CAP_LAYER=true \
        AWARE_BENCH_CAP_DISCOVERY=nodiscovery

    # B5: cap-pair attention + per-block matrices
    run_one B5_cappair_local env \
        AWARE_BENCH_ATTENTION=cap_pair \
        AWARE_BENCH_CAP_SOURCE=local \
        AWARE_BENCH_INCLUDE_CAP_LAYER=false

    # B6: cap-pair + input cap layer
    run_one B6_cappair_with_input_caps env \
        AWARE_BENCH_ATTENTION=cap_pair \
        AWARE_BENCH_CAP_SOURCE=local \
        AWARE_BENCH_INCLUDE_CAP_LAYER=true \
        AWARE_BENCH_CAP_DISCOVERY=nodiscovery
}

# ─────────────────────────────── Phase C: Window sweep ────────────────────────────
phase_C() {
    # Hold attention + discovery; vary cap window.
    for win in 1 4 8; do
        run_one "C_window_${win}" env \
            AWARE_BENCH_ATTENTION=standard \
            AWARE_BENCH_INCLUDE_CAP_LAYER=true \
            AWARE_BENCH_CAP_DISCOVERY=kmeans \
            AWARE_BENCH_CAP_WINDOW="$win"
    done
}

# ─────────────────────────────── Phase D: Discovery sweep ─────────────────────────
phase_D() {
    # Hold attention=standard, include_cap_layer=true; vary discovery.
    for disc in kmeans random nodiscovery hybrid; do
        run_one "D_disc_${disc}" env \
            AWARE_BENCH_ATTENTION=standard \
            AWARE_BENCH_INCLUDE_CAP_LAYER=true \
            AWARE_BENCH_CAP_DISCOVERY="$disc"
    done
}

ensure_build

for p in "${PHASES[@]}"; do
    case "$p" in
        A) phase_A ;;
        B) phase_B ;;
        C) phase_C ;;
        D) phase_D ;;
        all) phase_A; phase_B; phase_C; phase_D ;;
        *) echo "[bench] unknown phase: $p"; exit 1 ;;
    esac
done

echo
echo "[bench] all done. reports under $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR" 2>/dev/null || true

# Aggregate results into CSV + markdown table.
if [ -x "$AGGREGATOR" ]; then
    echo
    echo "[bench] aggregating…"
    python3 "$AGGREGATOR" "$OUTPUT_DIR" || echo "(aggregation failed, but runs are intact)"
fi
