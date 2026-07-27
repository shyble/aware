#!/usr/bin/env bash
# Extra seeds for the WikiText w2-vs-w3 distinction (paper #1 v2 §VI-C).
#
# At 3 seeds, w2 (29.17 +/- 0.31) beats w3 (30.16 +/- 0.35) at p ~= 0.02 —
# real but borderline with so few degrees of freedom. Two additional seeds
# per config (5 total each) firm up whether "w2 is the WikiText optimum"
# can be stated plainly or must stay "marginally better".
#
# Seeds 100 and 200 join the existing 42/123/7; the aggregate summary then
# reports all five.
#
# Usage:  ./scripts/run_w2_w3_extra_seeds.sh
set -uo pipefail
cd "$(dirname "$0")/.."

WT_ENV=(AWARE_BENCH_CORPUS=data/wikitext103/wikitext_train.txt
        AWARE_BENCH_VAL_CORPUS=data/wikitext103/wikitext_val.txt
        AWARE_BENCH_BPE_DIR=data/bpe_wikitext103)

for cfg in kmeans_w2 kmeans_w3; do
    env "${WT_ENV[@]}" AWARE_BENCH_OUTPUT_DIR=data/bench_wikitext \
        ./scripts/run_multi_seed.sh "$cfg" 100 200 || echo "[extra-seeds] $cfg FAILED" >&2
done

echo
echo "════ five-seed summary ════"
python3 - <<'PYEOF'
import json, glob, statistics as st
for cfg in ("kmeans_w2", "kmeans_w3"):
    ppls = []
    for p in sorted(glob.glob(f"data/bench_wikitext/{cfg}_seed*/report.json")):
        with open(p) as f:
            ppls.append(json.load(f)["final_val_perplexity"])
    if len(ppls) > 1:
        print(f"  {cfg}: {st.mean(ppls):6.2f} +/- {st.stdev(ppls):4.2f}  ({len(ppls)} seeds)")
PYEOF
