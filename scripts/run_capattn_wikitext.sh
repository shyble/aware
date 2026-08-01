#!/usr/bin/env bash
# Cap-attention on WikiText-103 (paper #1 v2, Table 3 second corpus).
#
# Phase B' established on TinyStories that cap-attention placements
# underperform standard attention, and that neither head count nor
# cap-matrix discovery rescues them. That result is currently single-corpus
# while every other result in the paper is two-corpus - so the paper's one
# negative claim rests on the synthetic corpus alone.
#
# These runs use the multi-head variants, which are the fair comparison
# against the 4-head WikiText baseline already measured (51.42 +/- 0.26).
# 2 configs x 3 seeds = 6 runs, ~5h.
#
# Deliberately a separate script: the post-campaign queue is running, and
# editing a live bash script corrupts its execution.
#
# Usage:  nohup ./scripts/run_capattn_wikitext.sh > /tmp/capattn_wt.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

export AWARE_BENCH_CORPUS=data/wikitext103/wikitext_train.txt
export AWARE_BENCH_VAL_CORPUS=data/wikitext103/wikitext_val.txt
export AWARE_BENCH_BPE_DIR=data/bpe_wikitext103
export AWARE_BENCH_OUTPUT_DIR=data/bench_wikitext

for cfg in capmem_multihead cappair_multihead; do
    echo
    echo "════════════════ $cfg (WikiText-103) ════════════════"
    ./scripts/run_multi_seed.sh "$cfg" || echo "[capattn-wt] $cfg FAILED" >&2
done

echo
echo "════════════════ SUMMARY vs WikiText baseline ════════════════"
python3 - <<'PYEOF'
import json, glob, statistics as st
rows = []
for cfg in ("pure_transformer", "kmeans_w3", "capmem_multihead", "cappair_multihead"):
    v = [json.load(open(p))["final_val_perplexity"]
         for p in sorted(glob.glob(f"data/bench_wikitext/{cfg}_seed*/report.json"))]
    if v:
        m = st.mean(v)
        s = st.stdev(v) if len(v) > 1 else 0.0
        rows.append((cfg, m, s, len(v)))
        print(f"  {cfg:22s} {m:6.2f} +/- {s:4.2f}  ({len(v)} seeds)")
base = next((m for c, m, _, _ in rows if c == "pure_transformer"), None)
if base:
    print()
    for c, m, _, _ in rows:
        if c != "pure_transformer":
            print(f"  {c:22s} {100.0*(base-m)/base:+6.1f}% vs baseline")
PYEOF
