#!/usr/bin/env bash
# Seed replicates for the matched-parameter comparison.
#
# At d=256/6 blocks and 4,937,472 trainable parameters BOTH, seed 42 gave
# cap-affinity 21.41 against dot-product 22.24 -- a 0.83 ppl margin. This
# harness has shown a 1.47 ppl spread across five seeds at a comparable
# config (kmeans_w3: 29.87 / 30.03 / 30.06 / 30.55 / 31.34), so one seed
# per arm cannot separate a real 0.83 from a lucky draw.
#
# Three seeds per arm. 42 already exists; this adds 123 and 7.
#
# Read it as: leads on all three, or means separated by more than the
# within-arm spread -> real. Interleaved -> noise.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${CS_OUT:-data/bench_cap_seeds}"
STEPS="${CS_STEPS:-5000}"
SEEDS="${CS_SEEDS:-123 7}"
ARMS="${CS_ARMS:-std cappair}"
CN=./target/release/examples/run_benchmark

[ -x "$CN" ] || { echo "ERROR: build run_benchmark first" >&2; exit 1; }
mkdir -p "$OUT"

for seed in $SEEDS; do
  for arm in $ARMS; do
    case "$arm" in
        std)     ATTN=standard; RANK=0 ;;
        cappair) ATTN=cap_pair; RANK=32 ;;
        *) echo "ERROR: unknown arm $arm" >&2; continue ;;
    esac
    id="${arm}_seed${seed}"
    echo
    echo "════════ $id (d=256, 6 blocks, attn=$ATTN rank=$RANK) ════════"
    env AWARE_BENCH_CORPUS=data/wikitext103/wikitext_train.txt \
        AWARE_BENCH_VAL_CORPUS=data/wikitext103/wikitext_val.txt \
        AWARE_BENCH_BPE_DIR=data/bpe_wikitext103 \
        AWARE_BENCH_D_MODEL=256 AWARE_BENCH_N_BLOCKS=6 AWARE_BENCH_N_HEADS=4 \
        AWARE_BENCH_D_FF=1024 AWARE_BENCH_SEED="$seed" AWARE_BENCH_STEPS="$STEPS" \
        AWARE_BENCH_INCLUDE_CAP_LAYER=true \
        AWARE_BENCH_CAP_DISCOVERY=kmeans AWARE_BENCH_CAP_WINDOW=3 \
        AWARE_BENCH_CAP_N_TARGET=330 \
        AWARE_BENCH_ATTENTION=$ATTN AWARE_BENCH_CAP_SOURCE=shared \
        AWARE_BENCH_CAP_ATTN_DISCOVERY=kmeans \
        AWARE_CAP_PAIR_RANK=$RANK \
        AWARE_BENCH_ID="$id" AWARE_BENCH_OUTPUT_DIR="$OUT" \
        "$CN" > "$OUT/$id.log" 2>&1
    rc=$?
    grep -E 'params|done\.' "$OUT/$id.log" | tail -2 || true
    [ $rc -eq 0 ] || { echo "ERROR: $id failed (rc=$rc):" >&2; tail -20 "$OUT/$id.log" >&2; }
  done
done

echo
echo "════════════════ SEED REPLICATES ════════════════"
OUT="$OUT" python3 - <<'PYEOF'
import json, os, glob, statistics as st
out = os.environ["OUT"]
res = {"std": {42: 22.24}, "cappair": {42: 21.41}}   # seed 42 from the earlier run
for f in glob.glob(f"{out}/*/report.json"):
    d = json.load(open(f))
    rid = d["run_id"]
    arm = "std" if rid.startswith("std") else "cappair"
    seed = int(rid.split("seed")[1])
    res[arm][seed] = d.get("final_val_perplexity", float("nan"))

print(f"  {'arm':<10}{'seed 42':>10}{'seed 123':>10}{'seed 7':>10}{'mean':>9}{'spread':>9}")
means = {}
for arm in ("std", "cappair"):
    vals = [res[arm].get(s) for s in (42, 123, 7)]
    got = [v for v in vals if v is not None]
    cells = "".join(f"{v:>10.2f}" if v is not None else f"{'—':>10}" for v in vals)
    if got:
        means[arm] = st.mean(got)
        print(f"  {arm:<10}{cells}{st.mean(got):>9.2f}{max(got)-min(got):>9.2f}")
    else:
        print(f"  {arm:<10}{cells}{'—':>9}{'—':>9}")

if len(means) == 2:
    d = means["std"] - means["cappair"]
    print(f"\n  cap-affinity is {d:+.2f} ppl vs dot-product on the mean.")
    print("  Compare that against each arm's own spread above: a margin")
    print("  smaller than the spread is not yet a result.")
PYEOF
