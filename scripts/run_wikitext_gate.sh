#!/usr/bin/env bash
# WikiText-103 generalisation gate for paper #1 v2.
#
# Runs only the headline pair — pure_transformer vs kmeans_w3 (the paper's
# winner) — across 3 seeds. Everything else in the paper #1 matrix (window
# sweep, Phase B', discovery sweep) is conditional on this result, so it is
# worth ~5 hours to learn whether the core claim survives on real text
# before committing ~50 hours to the rest.
#
# Published TinyStories reference (paper #1 §6.3):
#   pure_transformer  28.00 +/- 0.11
#   kmeans_w3         13.71 +/- 0.33   (51% reduction)
#
# The absolute perplexities will NOT match those numbers — different corpus,
# different tokenizer. What matters is whether the *gap* survives.
#
# Usage:  ./scripts/run_wikitext_gate.sh
set -uo pipefail
cd "$(dirname "$0")/.."

export AWARE_BENCH_CORPUS="${AWARE_BENCH_CORPUS:-data/wikitext103/wikitext_train.txt}"
export AWARE_BENCH_VAL_CORPUS="${AWARE_BENCH_VAL_CORPUS:-data/wikitext103/wikitext_val.txt}"
# Corpus-specific tokenizer. The TinyStories BPE fragments encyclopaedic
# English ~16% worse, which would look like a generalisation failure.
export AWARE_BENCH_BPE_DIR="${AWARE_BENCH_BPE_DIR:-data/bpe_wikitext103}"
# Separate output dir: run ids collide with the TinyStories results otherwise.
export AWARE_BENCH_OUTPUT_DIR="${AWARE_BENCH_OUTPUT_DIR:-data/bench_wikitext}"

echo "[gate] corpus=$AWARE_BENCH_CORPUS"
echo "[gate] tokenizer=$AWARE_BENCH_BPE_DIR"
echo "[gate] output=$AWARE_BENCH_OUTPUT_DIR"
echo

for cfg in pure_transformer kmeans_w3; do
    echo "════════════════ $cfg ════════════════"
    ./scripts/run_multi_seed.sh "$cfg" || echo "[gate] $cfg FAILED" >&2
done

echo
echo "════════════════ GATE SUMMARY ════════════════"
python3 - <<'PYEOF'
import json, glob, statistics
rows = []
for cfg in ("pure_transformer", "kmeans_w3"):
    ppls = []
    for p in sorted(glob.glob(f"data/bench_wikitext/{cfg}_seed*/report.json")):
        with open(p) as f:
            r = json.load(f)
        v = r.get("final_val_perplexity")
        if v is not None:
            ppls.append(v)
    if ppls:
        m = statistics.mean(ppls)
        s = statistics.stdev(ppls) if len(ppls) > 1 else 0.0
        rows.append((cfg, m, s, len(ppls)))
        print(f"  {cfg:20s} {m:8.2f} +/- {s:5.2f}  ({len(ppls)} seeds)")
    else:
        print(f"  {cfg:20s} no reports")

if len(rows) == 2:
    base, cap = rows[0][1], rows[1][1]
    print()
    print(f"  cap-input vs baseline: {100.0*(base-cap)/base:.1f}% perplexity REDUCTION")
    print(f"  (TinyStories reference: 51.0% reduction)")
PYEOF
