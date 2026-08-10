#!/usr/bin/env bash
# Is cap-affinity scoring better, or did it just have more capacity?
#
# cap layer + cap_pair[shared] reached 27.87 against cap layer + standard
# attention at 30.55 -- but with 4.96M parameters against 895K, and ~7x
# the attention compute, since A_pair costs n_caps^2 per head per token.
# Neither advantage was controlled, so the result cannot be claimed.
#
# Two ways to close it, from opposite directions:
#
#   std_big    standard attention scaled to 4.94M (d=256, 6 blocks,
#              d_ff=1024) -- matches full-rank cap_pair's parameters to
#              within 0.4%. If it reaches 27.87, capacity explains
#              everything and cap-affinity buys nothing.
#
#   rank16/32  cap_pair with A factored as U V^T. Never materialises A,
#              so parameters AND compute drop to 2*n_caps*r. Rank 32 is
#              1.29M; rank 16 lands near the 895K baseline. If the gain
#              survives, cap-affinity wins at matched cost -- which is
#              the claim worth having.
#
# The factorisation also shows what cap-pair IS: (cap.U)(cap.V)^T is
# query-key attention with q,k computed from cap activations rather than
# from x. Standard attention is the same form with an identity feature
# map at rank d_head. Rank is the axis between them, and these arms
# sample it.
#
# References (identical corpus, tokenizer, seed, steps):
#   pure transformer                 51.22    853,120
#   cap layer, NO attention          36.13    173,568
#   cap layer + standard attention   30.55    895,360
#   cap layer + cap_pair [shared]    27.87  4,958,592
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${CM_OUT:-data/bench_cap_matched}"
STEPS="${CM_STEPS:-5000}"
ARMS="${CM_ARMS:-rank16 rank32 std_big}"
CN=./target/release/examples/run_benchmark

[ -x "$CN" ] || { echo "ERROR: build run_benchmark first" >&2; exit 1; }
mkdir -p "$OUT"

for arm in $ARMS; do
    case "$arm" in
        rank16) D=128; NB=4; FF=512; ATTN=cap_pair; RANK=16 ;;
        rank32) D=128; NB=4; FF=512; ATTN=cap_pair; RANK=32 ;;
        std_big) D=256; NB=6; FF=1024; ATTN=standard; RANK=0 ;;
        *) echo "ERROR: unknown arm $arm" >&2; continue ;;
    esac
    echo
    echo "════════ $arm (d=$D, blocks=$NB, attn=$ATTN, rank=$RANK) ════════"
    env AWARE_BENCH_CORPUS=data/wikitext103/wikitext_train.txt \
        AWARE_BENCH_VAL_CORPUS=data/wikitext103/wikitext_val.txt \
        AWARE_BENCH_BPE_DIR=data/bpe_wikitext103 \
        AWARE_BENCH_D_MODEL=$D AWARE_BENCH_N_BLOCKS=$NB AWARE_BENCH_N_HEADS=4 \
        AWARE_BENCH_D_FF=$FF AWARE_BENCH_SEED=42 AWARE_BENCH_STEPS="$STEPS" \
        AWARE_BENCH_INCLUDE_CAP_LAYER=true \
        AWARE_BENCH_CAP_DISCOVERY=kmeans AWARE_BENCH_CAP_WINDOW=3 \
        AWARE_BENCH_CAP_N_TARGET=330 \
        AWARE_BENCH_ATTENTION=$ATTN AWARE_BENCH_CAP_SOURCE=shared \
        AWARE_BENCH_CAP_ATTN_DISCOVERY=kmeans \
        AWARE_CAP_PAIR_RANK=$RANK \
        AWARE_BENCH_ID="$arm" AWARE_BENCH_OUTPUT_DIR="$OUT" \
        "$CN" > "$OUT/$arm.log" 2>&1
    rc=$?
    grep -E 'params|done\.' "$OUT/$arm.log" | tail -2 || true
    [ $rc -eq 0 ] || { echo "ERROR: arm $arm failed (rc=$rc):" >&2; tail -20 "$OUT/$arm.log" >&2; }
done

echo
echo "═══════════════ MATCHED-COST COMPARISON ═══════════════"
OUT="$OUT" python3 - <<'PYEOF'
import json, os, glob
out = os.environ["OUT"]
rows = [("pure transformer", 853120, 51.22),
        ("cap layer, no attention", 173568, 36.13),
        ("cap layer + standard attn", 895360, 30.55),
        ("cap layer + cap_pair full", 4958592, 27.87)]
for f in sorted(glob.glob(f"{out}/*/report.json")):
    d = json.load(open(f))
    rows.append((d["run_id"], d.get("params", 0), d.get("final_val_perplexity", float("nan"))))
print(f"  {'model':<30}{'params':>10}{'val ppl':>10}")
for n, p, v in rows:
    print(f"  {n:<30}{p:>10}{v:>10.2f}")
print()
print("  std_big vs 27.87 : if it reaches it, capacity explained the gain.")
print("  rank16/32        : if they hold near 27.87 at ~1M params, cap-affinity")
print("                     scoring beats dot-product at matched cost.")
PYEOF
