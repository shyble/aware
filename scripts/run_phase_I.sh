#!/usr/bin/env bash
# Phase I: multi-seed runs.
# 4 configs × 3 seeds = 12 runs. ~10 hours wall-clock.
#
# Outputs to data/bench/<config>_seed<N>/ in the same format as Phase A-D.
# After each config completes, the per-config summary is printed.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "[phase-I] starting multi-seed sweep"
echo "[phase-I] $(date)"
echo

./scripts/run_multi_seed.sh pure_transformer
echo
./scripts/run_multi_seed.sh kmeans_w1
echo
./scripts/run_multi_seed.sh kmeans_w4
echo
./scripts/run_multi_seed.sh kmeans_w8
echo

echo "[phase-I] all multi-seed runs done at $(date)"
echo "[phase-I] aggregating final results…"
python3 ./scripts/aggregate_benchmarks.py data/bench
echo "[phase-I] done"
