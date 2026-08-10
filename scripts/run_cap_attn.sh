#!/usr/bin/env bash
# Cap layers + cap attention: the fully cap-structured model.
#
# Never run before. Every previous cap-pair arm used
# include_cap_layer=false, cap_window=1, discovery=nodiscovery -- it was
# scoring against caps that carried almost nothing, which is the exact
# configuration the input-layer ablation shows is worst (33.61 for
# nodiscovery vs 30.55 for KMeans window 3). Cap-pair's flat results say
# little about cap-indexed attention and a lot about the caps it was
# given.
#
# Two ways to give it real caps:
#   shared  attention scores with the INPUT cap layer's own caps -- one
#           cap vocabulary for the whole model, which is the unified
#           structure worth wanting
#   local   attention keeps its own cap matrix, discovered by KMeans
#           rather than Xavier-initialised
#
# References on the identical corpus, tokenizer, seed and budget:
#   pure transformer                 51.22    853,120
#   cap layer only, NO attention     36.13    173,568   (measured today)
#   cap layer + standard attention   30.55    895,360
#
# The prize is the ~4.9 ppl that attention buys and hierarchical
# composition could not reach. Landing near 30.55 means a model with no
# dot-product anywhere is competitive; landing near 36 means cap-indexed
# scoring cannot supply it and the honest answer is that attention stays.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${CA_OUT:-data/bench_cap_attn}"
STEPS="${CA_STEPS:-5000}"
ARMS="${CA_ARMS:-shared local}"
CN=./target/release/examples/run_benchmark

[ -x "$CN" ] || { echo "ERROR: build run_benchmark first" >&2; exit 1; }
mkdir -p "$OUT"

for arm in $ARMS; do
    echo
    echo "════════ cap layer (w3, kmeans) + cap_pair [$arm] ════════"
    env AWARE_BENCH_CORPUS=data/wikitext103/wikitext_train.txt \
        AWARE_BENCH_VAL_CORPUS=data/wikitext103/wikitext_val.txt \
        AWARE_BENCH_BPE_DIR=data/bpe_wikitext103 \
        AWARE_BENCH_D_MODEL=128 AWARE_BENCH_N_BLOCKS=4 AWARE_BENCH_N_HEADS=4 \
        AWARE_BENCH_D_FF=512 AWARE_BENCH_SEED=42 AWARE_BENCH_STEPS="$STEPS" \
        AWARE_BENCH_INCLUDE_CAP_LAYER=true \
        AWARE_BENCH_CAP_DISCOVERY=kmeans \
        AWARE_BENCH_CAP_WINDOW=3 \
        AWARE_BENCH_CAP_N_TARGET=330 \
        AWARE_BENCH_ATTENTION=cap_pair \
        AWARE_BENCH_CAP_SOURCE="$arm" \
        AWARE_BENCH_CAP_N=330 \
        AWARE_BENCH_CAP_ATTN_DISCOVERY=kmeans \
        AWARE_BENCH_ID="cappair_${arm}_caplayer" \
        AWARE_BENCH_OUTPUT_DIR="$OUT" \
        "$CN" > "$OUT/$arm.log" 2>&1
    rc=$?
    grep -E 'params|final val_ppl|done\.' "$OUT/$arm.log" | tail -3 || true
    [ $rc -eq 0 ] || { echo "ERROR: arm $arm failed (rc=$rc):" >&2; tail -20 "$OUT/$arm.log" >&2; }
done

echo
echo "═════════════════ CAP LAYER + CAP ATTENTION ═════════════════"
OUT="$OUT" python3 - <<'PYEOF'
import json, os, glob
out = os.environ["OUT"]
print(f"  {'model':<34}{'params':>10}{'val ppl':>10}")
print(f"  {'pure transformer':<34}{853120:>10}{51.22:>10.2f}")
print(f"  {'cap layer only (no attention)':<34}{173568:>10}{36.13:>10.2f}")
print(f"  {'cap layer + STANDARD attention':<34}{895360:>10}{30.55:>10.2f}")
print("  " + "-" * 54)
for f in sorted(glob.glob(f"{out}/*/report.json")):
    d = json.load(open(f))
    src = d.get("cap_source", "?")
    print(f"  {'cap layer + cap_pair [' + src + ']':<34}"
          f"{d.get('params', 0):>10}{d.get('final_val_perplexity', float('nan')):>10.2f}")
print()
print("  Near 30.55 -> a model with no dot-product anywhere is competitive,")
print("  and the fully cap-structured architecture is real.")
print("  Near 36    -> cap-indexed scoring cannot supply what attention does;")
print("  the honest conclusion is that caps own the representation and")
print("  attention stays for the mixing.")
PYEOF
