#!/usr/bin/env bash
# Dense scaling sweep on WikiText-103 (paper #2, stage B).
#
# Trains plain transformers at 3.3M / 10.8M / 25.4M parameters so the
# size->perplexity curve can be fitted and the dense size matching
# cap-native's quality read off. Runs on whatever backend the build
# selects: set AWARE_FEATURES/AWARE_DEVICE to cuda on the Windows box.
#
# Cost is dominated by the largest model and differs enormously by device:
#   CPU (M1 Pro)  ~203h for 9 runs  - impractical
#   RTX 4060      ~4-8h  for 9 runs - dense matmul is the GPU's best case
#
# Prerequisites (see scripts/WINDOWS_SETUP.md for the Windows path):
#   - data/wikitext103/{wikitext_train,wikitext_val}.txt   (fetch_wikitext.py)
#   - data/bpe_wikitext103/bpe.bin                          (tracked in git)
#
# Usage:
#   AWARE_FEATURES=cuda AWARE_DEVICE=cuda ./scripts/run_dense_sweep.sh
#   AWARE_FEATURES=cuda AWARE_DEVICE=cuda SEEDS="42" ./scripts/run_dense_sweep.sh
set -uo pipefail
cd "$(dirname "$0")/.."

export AWARE_BENCH_CORPUS="${AWARE_BENCH_CORPUS:-data/wikitext103/wikitext_train.txt}"
export AWARE_BENCH_VAL_CORPUS="${AWARE_BENCH_VAL_CORPUS:-data/wikitext103/wikitext_val.txt}"
export AWARE_BENCH_BPE_DIR="${AWARE_BENCH_BPE_DIR:-data/bpe_wikitext103}"
export AWARE_BENCH_OUTPUT_DIR="${AWARE_BENCH_OUTPUT_DIR:-data/bench_dense_sweep_wt}"
export AWARE_BENCH_STEPS="${AWARE_BENCH_STEPS:-10000}"
SEEDS="${SEEDS:-42 123 7}"

for f in "$AWARE_BENCH_CORPUS" "$AWARE_BENCH_VAL_CORPUS" "$AWARE_BENCH_BPE_DIR/bpe.bin"; do
    [ -e "$f" ] || { echo "ERROR: missing $f - see scripts/WINDOWS_SETUP.md" >&2; exit 1; }
done

echo "[sweep] device=${AWARE_DEVICE:-<build default>}  steps=$AWARE_BENCH_STEPS  seeds=$SEEDS"
echo "[sweep] output=$AWARE_BENCH_OUTPUT_DIR"

for cfg in dense_3m dense_10m dense_30m; do
    echo
    echo "════════════════ $cfg ════════════════"
    # shellcheck disable=SC2086
    ./scripts/run_multi_seed.sh "$cfg" $SEEDS || echo "[sweep] $cfg FAILED" >&2
done

echo
echo "════════════════ DENSE SIZE -> PERPLEXITY ════════════════"
python3 - <<'PYEOF'
import json, glob, os, statistics as st
out = os.environ.get("AWARE_BENCH_OUTPUT_DIR", "data/bench_dense_sweep_wt")
pts = []
for cfg, params in (("dense_3m", 3279104), ("dense_10m", 10818432), ("dense_30m", 25436672)):
    v = [json.load(open(p))["final_val_perplexity"]
         for p in sorted(glob.glob(f"{out}/{cfg}_seed*/report.json"))]
    if v:
        m = st.mean(v)
        s = st.stdev(v) if len(v) > 1 else 0.0
        pts.append((params, m))
        print(f"  {cfg:10s} {params:>10,} params   {m:6.2f} +/- {s:4.2f}  ({len(v)} seeds)")
print("\n  Reference points on the same corpus:")
print("    853,120 dense (measured)          51.42")
print("    cap-input W=3, 895,360            30.37")
print("    cap-native hierarchical, 143M     14.45   <- iso-quality target")
if len(pts) >= 2:
    print("\n  Fit log(ppl) vs log(params) across the sweep to read off the dense")
    print("  size reaching 14.45; if the curve does not reach it, report the bound.")
PYEOF
