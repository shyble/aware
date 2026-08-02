#!/usr/bin/env bash
# Stage C standalone: cap-native single-discovery on WikiText-103.
#
# cap_native_sparse_d128 x 3 seeds @ 5000 steps, budget-matched to the
# hierarchical runs of stage A (14.51 +/- 0.16), giving the
# hierarchical-versus-single comparison on the primary corpus. The 368M
# config runs ~24h/seed on CPU, so this is the longest single job in the
# programme and has to start as early as possible.
#
# Kept separate from the post-campaign queue because the dense sweep
# (stage B) moved to the CUDA machine, and a running bash script cannot be
# safely edited in place.
#
# Usage:  nohup ./scripts/run_stage_c.sh > /tmp/stage_c.log 2>&1 &
set -uo pipefail

MAIN=/Users/kursataydemir/limnr/aware
WT=/Users/kursataydemir/limnr/aware-capnative

cd "$WT" || { echo "ERROR: cap-native worktree missing at $WT" >&2; exit 1; }

echo "[stage C] $(date '+%F %T') cap_native_sparse_d128 x 3 seeds @ 5000 steps"

AWARE_BENCH_CORPUS="$MAIN/data/wikitext103/wikitext_train.txt" \
AWARE_BENCH_VAL_CORPUS="$MAIN/data/wikitext103/wikitext_val.txt" \
AWARE_BENCH_BPE_DIR="$MAIN/data/bpe_wikitext103" \
AWARE_BENCH_OUTPUT_DIR="$MAIN/data/bench_wikitext_capnative" \
AWARE_BENCH_STEPS=5000 \
    ./scripts/run_multi_seed.sh cap_native_sparse_d128 || echo "[stage C] FAILED" >&2

echo
echo "════════ hierarchical vs single-discovery, WikiText-103 ════════"
cd "$MAIN"
python3 - <<'PYEOF'
import json, glob, statistics as st
for cfg, label in (("cap_native_hier_d128", "hierarchical"),
                   ("cap_native_sparse_d128", "single-discovery")):
    v = [json.load(open(p))["final_val_perplexity"]
         for p in sorted(glob.glob(f"data/bench_wikitext_capnative/{cfg}_seed*/report.json"))]
    if v:
        s = st.stdev(v) if len(v) > 1 else 0.0
        print(f"  {label:18s} {st.mean(v):6.2f} +/- {s:4.2f}  ({len(v)} seeds)")
print("  (TinyStories reference: hier 8.76, single 9.96 - hier 12% better)")
PYEOF
