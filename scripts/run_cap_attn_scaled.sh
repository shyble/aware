#!/usr/bin/env bash
# Does cap-affinity scoring survive at scale, once its parameters are
# spent the way a transformer spends them?
#
# The matched-parameter control settled the previous claim badly: at
# ~4.95M, standard attention reached 22.24 while full-rank cap_pair
# reached 27.87. But those models are not the same SHAPE -- std_big is
# d=256 / 6 blocks, while cap_pair was d=128 / 4 blocks that put its
# parameters into a 330x330 affinity matrix per head.
#
# The rank sweep says that matrix is a poor investment: rank 16 -> 28.53,
# rank 32 -> 28.39, full -> 27.87. Full rank costs 4.8x the parameters
# for 0.66 ppl.
#
# So this spends the budget the way std_big does -- depth and width --
# and keeps only the cheap part of cap affinity.
#
#   scaled_rank32   d=256, 6 blocks, cap_pair rank 32   ~4.66M
#   (reference)     d=256, 6 blocks, standard            4.94M -> 22.24
#
# Near or below 22.24 -> cap-affinity scoring holds at scale.
# Near 27           -> the crossover is real, and the honest claim is
#                      parameter-efficiency at small scale, not a
#                      replacement for dot-product attention.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${CS_OUT:-data/bench_cap_scaled}"
STEPS="${CS_STEPS:-5000}"
ARMS="${CS_ARMS:-scaled_rank32}"
CN=./target/release/examples/run_benchmark

[ -x "$CN" ] || { echo "ERROR: build run_benchmark first" >&2; exit 1; }
mkdir -p "$OUT"

for arm in $ARMS; do
    case "$arm" in
        scaled_rank32) RANK=32 ;;
        scaled_rank64) RANK=64 ;;
        *) echo "ERROR: unknown arm $arm" >&2; continue ;;
    esac
    echo
    echo "════════ $arm — d=256, 6 blocks, cap_pair rank $RANK ════════"
    env AWARE_BENCH_CORPUS=data/wikitext103/wikitext_train.txt \
        AWARE_BENCH_VAL_CORPUS=data/wikitext103/wikitext_val.txt \
        AWARE_BENCH_BPE_DIR=data/bpe_wikitext103 \
        AWARE_BENCH_D_MODEL=256 AWARE_BENCH_N_BLOCKS=6 AWARE_BENCH_N_HEADS=4 \
        AWARE_BENCH_D_FF=1024 AWARE_BENCH_SEED=42 AWARE_BENCH_STEPS="$STEPS" \
        AWARE_BENCH_INCLUDE_CAP_LAYER=true \
        AWARE_BENCH_CAP_DISCOVERY=kmeans AWARE_BENCH_CAP_WINDOW=3 \
        AWARE_BENCH_CAP_N_TARGET=330 \
        AWARE_BENCH_ATTENTION=cap_pair AWARE_BENCH_CAP_SOURCE=shared \
        AWARE_BENCH_CAP_ATTN_DISCOVERY=kmeans \
        AWARE_CAP_PAIR_RANK=$RANK \
        AWARE_BENCH_ID="$arm" AWARE_BENCH_OUTPUT_DIR="$OUT" \
        "$CN" > "$OUT/$arm.log" 2>&1
    rc=$?
    grep -E 'params|done\.' "$OUT/$arm.log" | tail -2 || true
    [ $rc -eq 0 ] || { echo "ERROR: arm $arm failed (rc=$rc):" >&2; tail -20 "$OUT/$arm.log" >&2; }
done

echo
echo "═════════════ SCALED, SHAPE-MATCHED COMPARISON ═════════════"
OUT="$OUT" python3 - <<'PYEOF'
import json, os, glob
out = os.environ["OUT"]
print(f"  {'model':<38}{'params':>10}{'val ppl':>10}")
print(f"  {'cap + standard attn (d128, 4blk)':<38}{895360:>10}{30.55:>10.2f}")
print(f"  {'cap + cap_pair rank16 (d128, 4blk)':<38}{1026432:>10}{28.53:>10.2f}")
print(f"  {'cap + cap_pair full (d128, 4blk)':<38}{4958592:>10}{27.87:>10.2f}")
print(f"  {'cap + standard attn (d256, 6blk)':<38}{4937472:>10}{22.24:>10.2f}   <- to beat")
for f in sorted(glob.glob(f"{out}/*/report.json")):
    d = json.load(open(f))
    print(f"  {'cap + ' + d['run_id'] + ' (d256, 6blk)':<38}"
          f"{d.get('params', 0):>10}{d.get('final_val_perplexity', float('nan')):>10.2f}")
PYEOF
