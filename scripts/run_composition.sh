#!/usr/bin/env bash
# Does either architecture learn to use distance, given a longer budget?
#
# At 5000 steps neither does. Replacing all 123 context tokens between
# distance 5 and 127 with random tokens costs 1.4% perplexity for standard
# attention and 0.2% for cap-affinity -- the content is unused, not merely
# its order. Every architecture comparison so far has therefore compared
# two local models, and nothing has tested composition.
#
# Two changes:
#   4x the steps, so distance has a chance to become useful
#   RoPE on cap-pair's factored q,k, so cap-affinity CAN use it
#
# The second matters because cap-pair scores depended only on which caps
# fired, never on where. Its attention was permutation-invariant over the
# context. That cost nothing while nothing used distance; running the
# composition test without fixing it would have measured a model that
# structurally could not pass.
#
# Baselines at 5000 steps, same shape and parameter count (4,937,472):
#   standard 22.24 · cap-affinity 21.41  (3 seeds: 21.96 vs 21.19 mean)
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${CO_OUT:-data/bench_composition}"
STEPS="${CO_STEPS:-20000}"
SEED="${CO_SEED:-42}"
ARMS="${CO_ARMS:-cappair std}"
CN=./target/release/examples/run_benchmark

[ -x "$CN" ] || { echo "ERROR: build run_benchmark first" >&2; exit 1; }
mkdir -p "$OUT"

for arm in $ARMS; do
    case "$arm" in
        std)     ATTN=standard; RANK=0 ;;
        cappair) ATTN=cap_pair; RANK=32 ;;
        *) echo "ERROR: unknown arm $arm" >&2; continue ;;
    esac
    id="${arm}_${STEPS}steps"
    echo
    echo "════════ $id ════════"
    env AWARE_BENCH_CORPUS=data/wikitext103/wikitext_train.txt \
        AWARE_BENCH_VAL_CORPUS=data/wikitext103/wikitext_val.txt \
        AWARE_BENCH_BPE_DIR=data/bpe_wikitext103 \
        AWARE_BENCH_D_MODEL=256 AWARE_BENCH_N_BLOCKS=6 AWARE_BENCH_N_HEADS=4 \
        AWARE_BENCH_D_FF=1024 AWARE_BENCH_SEED="$SEED" AWARE_BENCH_STEPS="$STEPS" \
        AWARE_BENCH_EVAL_EVERY=500 \
        AWARE_BENCH_INCLUDE_CAP_LAYER=true \
        AWARE_BENCH_CAP_DISCOVERY=kmeans AWARE_BENCH_CAP_WINDOW=3 \
        AWARE_BENCH_CAP_N_TARGET=330 \
        AWARE_BENCH_ATTENTION=$ATTN AWARE_BENCH_CAP_SOURCE=shared AWARE_BENCH_CAP_N=512 \
        AWARE_BENCH_CAP_ATTN_DISCOVERY=kmeans AWARE_CAP_PAIR_RANK=$RANK \
        AWARE_BENCH_ID="$id" AWARE_BENCH_OUTPUT_DIR="$OUT" \
        "$CN" > "$OUT/$id.log" 2>&1
    rc=$?
    grep -E 'params|done\.' "$OUT/$id.log" | tail -2 || true
    [ $rc -eq 0 ] || { echo "ERROR: $id failed (rc=$rc):" >&2; tail -20 "$OUT/$id.log" >&2; }
done

# ── Probe both, both modes ──
echo
echo "════════ COMPOSITIONAL PROBE ════════"
for arm in $ARMS; do
    case "$arm" in
        std)     ATTN=standard; RANK=0 ;;
        cappair) ATTN=cap_pair; RANK=32 ;;
    esac
    ck="$OUT/${arm}_${STEPS}steps/substrate.safetensors"
    [ -f "$ck" ] || { echo "  ($arm: no checkpoint)"; continue; }
    for mode in replace shuffle; do
        echo "-- $arm / $mode --"
        env AWARE_BENCH_CORPUS=data/wikitext103/wikitext_train.txt \
            AWARE_BENCH_VAL_CORPUS=data/wikitext103/wikitext_val.txt \
            AWARE_BENCH_BPE_DIR=data/bpe_wikitext103 \
            AWARE_BENCH_D_MODEL=256 AWARE_BENCH_N_BLOCKS=6 AWARE_BENCH_N_HEADS=4 \
            AWARE_BENCH_D_FF=1024 AWARE_BENCH_SEED="$SEED" \
            AWARE_BENCH_INCLUDE_CAP_LAYER=true AWARE_BENCH_CAP_WINDOW=3 \
            AWARE_BENCH_CAP_N_TARGET=330 \
            AWARE_BENCH_ATTENTION=$ATTN AWARE_BENCH_CAP_SOURCE=shared AWARE_BENCH_CAP_N=512 \
            AWARE_CAP_PAIR_RANK=$RANK AWARE_BENCH_LOAD_WEIGHTS="$ck" \
            AWARE_BENCH_PROBE_COMPOSITION=true AWARE_BENCH_PROBE_SEQS=512 \
            AWARE_BENCH_PROBE_MODE=$mode \
            AWARE_BENCH_PROBE_BANDS="1-4,5-12,13-32,33-64,5-127" \
            AWARE_BENCH_OUTPUT_DIR="$OUT/probe" \
            "$CN" 2>&1 | grep -E '^\[probe\] (clean|  )'
    done
done
echo
echo "  Nonzero degradation beyond band 1-4 means distance is finally being"
echo "  used. Only then does 'does cap-affinity compose' become answerable."
