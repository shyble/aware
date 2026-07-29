#!/usr/bin/env bash
# Post-campaign queue for the SSCI/TNNLS papers, in value order:
#
#   A. WikiText ladder, rung 2: cap-native hierarchical x 3 seeds @ 5000
#      steps (matching the budget of rungs 0-1 already measured there).
#      Runs from the cap-native worktree with the corpus-specific BPE.
#   B. Dense scaling sweep on WikiText: dense_3m/10m/30m x 3 seeds @ 10000
#      steps -> dense size->ppl curve, read against stage A's hier number
#      for the iso-quality exchange rate. All comparisons live on WikiText.
#
# Waits for any in-flight benchmark to finish before starting.
set -uo pipefail

MAIN=/Users/kursataydemir/limnr/aware
WT=/Users/kursataydemir/limnr/aware-capnative

log() { echo; echo "════ [queue $(date '+%F %T')] $* ════"; }

# Wait for the extra-seeds runs (or anything else) to release the machine.
while pgrep -f 'examples/run_benchmark' >/dev/null; do sleep 300; done

log "stage A: WikiText ladder rung 2 - cap_native_hier_d128 x 3 seeds @ 5000 steps"
cd "$WT"
AWARE_BENCH_CORPUS="$MAIN/data/wikitext103/wikitext_train.txt" \
AWARE_BENCH_VAL_CORPUS="$MAIN/data/wikitext103/wikitext_val.txt" \
AWARE_BENCH_BPE_DIR="$MAIN/data/bpe_wikitext103" \
AWARE_BENCH_OUTPUT_DIR="$MAIN/data/bench_wikitext_capnative" \
AWARE_BENCH_STEPS=5000 \
    ./scripts/run_multi_seed.sh cap_native_hier_d128 || echo "[queue] stage A FAILED" >&2

log "stage B: dense scaling sweep @ 10000 steps (WikiText-103, primary corpus)"
# The sweep is target-independent: it draws the dense size->ppl curve; the
# iso-quality crossing is read off later against stage A's hier number.
cd "$MAIN"
for cfg in dense_3m dense_10m dense_30m; do
    AWARE_BENCH_CORPUS=data/wikitext103/wikitext_train.txt \
    AWARE_BENCH_VAL_CORPUS=data/wikitext103/wikitext_val.txt \
    AWARE_BENCH_BPE_DIR=data/bpe_wikitext103 \
    AWARE_BENCH_STEPS=10000 AWARE_BENCH_OUTPUT_DIR=data/bench_dense_sweep_wt \
        ./scripts/run_multi_seed.sh "$cfg" || echo "[queue] $cfg FAILED" >&2
done

log "stage C: cap-native single-discovery x 3 seeds @ 5000 steps (WikiText)"
# Budget-matched to stage A's hier runs -> hier-vs-single comparison on the
# primary corpus. The 368M config is slow (~19-30h/seed): deliberately LAST
# so it is a killable tail - the SSCI paper can submit without it and the
# runs then feed the journal extension.
cd "$WT"
AWARE_BENCH_CORPUS="$MAIN/data/wikitext103/wikitext_train.txt" \
AWARE_BENCH_VAL_CORPUS="$MAIN/data/wikitext103/wikitext_val.txt" \
AWARE_BENCH_BPE_DIR="$MAIN/data/bpe_wikitext103" \
AWARE_BENCH_OUTPUT_DIR="$MAIN/data/bench_wikitext_capnative" \
AWARE_BENCH_STEPS=5000 \
    ./scripts/run_multi_seed.sh cap_native_sparse_d128 || echo "[queue] stage C FAILED" >&2

log "queue complete"
