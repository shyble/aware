#!/usr/bin/env bash
# Paper #1 v2 (IEEE) experiment campaign — fills the four pending sections
# of the draft, in order of paper value:
#
#   chunk 1  WikiText-103 window sweep      w in {1,2,4,5,8} x 3 seeds  (15)
#            -> §VI-C: is the activation regime corpus-invariant?
#   chunk 2  Phase B' on TinyStories        9 configs x 3 seeds        (27)
#            -> §VI-D: fair cap-attention re-evaluation + head-count
#               controls + self-contained baseline repro
#   chunk 3  Discovery sweep on WikiText    3 configs x 3 seeds         (9)
#            -> §VI-E: does discovery strategy matter on real text?
#   chunk 4  Converged-gap long runs        2 configs x 15k steps       (2)
#            -> §VII-B: does the WikiText gap widen at convergence?
#
# Total 53 runs, ~2 days on M1 Pro CPU. Each chunk aggregates on completion,
# so partial progress is usable. Output dirs are disjoint from v1 results
# and from each other where run ids would collide.
#
# Usage:  nohup ./scripts/run_v2_campaign.sh > /tmp/v2_campaign.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

WT_ENV=(AWARE_BENCH_CORPUS=data/wikitext103/wikitext_train.txt
        AWARE_BENCH_VAL_CORPUS=data/wikitext103/wikitext_val.txt
        AWARE_BENCH_BPE_DIR=data/bpe_wikitext103)

log() { echo; echo "════ [campaign $(date '+%F %T')] $* ════"; }

log "chunk 1/4: WikiText-103 window sweep (15 runs)"
for cfg in kmeans_w1 kmeans_w2 kmeans_w4 kmeans_w5 kmeans_w8; do
    env "${WT_ENV[@]}" AWARE_BENCH_OUTPUT_DIR=data/bench_wikitext \
        ./scripts/run_multi_seed.sh "$cfg" || echo "[campaign] $cfg FAILED" >&2
done

log "chunk 2/4: Phase B' on TinyStories (27 runs)"
for cfg in pure_transformer pure_transformer_1head capmem cappair \
           capmem_multihead capmem_multihead_discovered \
           cappair_multihead cappair_multihead_matched cappair_multihead_kmeans; do
    AWARE_BENCH_OUTPUT_DIR=data/bench_bprime \
        ./scripts/run_multi_seed.sh "$cfg" || echo "[campaign] $cfg FAILED" >&2
done

log "chunk 3/4: discovery sweep on WikiText-103 (9 runs)"
for cfg in disc_random_w3 disc_nodiscovery_w3 disc_hybrid_w3; do
    env "${WT_ENV[@]}" AWARE_BENCH_OUTPUT_DIR=data/bench_wikitext \
        ./scripts/run_multi_seed.sh "$cfg" || echo "[campaign] $cfg FAILED" >&2
done

log "chunk 4/4: converged-gap long runs (2 runs, 15000 steps, seed 42)"
# Separate output dir: at default steps these ids already exist in
# data/bench_wikitext from the gate and would be overwritten.
for cfg in pure_transformer kmeans_w3; do
    env "${WT_ENV[@]}" AWARE_BENCH_STEPS=15000 \
        AWARE_BENCH_OUTPUT_DIR=data/bench_wikitext_long \
        ./scripts/run_multi_seed.sh "$cfg" 42 || echo "[campaign] $cfg FAILED" >&2
done

log "campaign complete"
