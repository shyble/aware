#!/usr/bin/env bash
# Does hierarchical composition substitute for attention?
#
# Removes the transformer entirely and stacks the cap-input layer, whose
# single use at the input is the strongest measured result in this
# project (51.22 -> 30.55 for 42,240 extra trainable parameters, frozen
# keys). Range comes from composing local windows, not from attending.
#
# Depth ladder, each arm adding one layer and extending the receptive
# field. Contiguous windows compose ADDITIVELY (w0+w1-1), not
# multiplicatively -- the arm labels say the true field.
#
# Baselines on the identical corpus, tokenizer, seed, steps and budget:
#   pure transformer              51.22   853,120 params
#   cap layer + 4 transformer     30.55   895,360 params
#
# Every arm here is a fraction of that size, so the comparison is not
# parameter-matched and must not be reported as one. The question is
# whether DEPTH buys anything: if arm 2 beats arm 1 substantially,
# composition is carrying weight and cap-conditioned variable offsets
# (step 3) are worth building. If depth is flat, hierarchy does not
# substitute for attention and the direction is closed.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${CS_OUT:-data/bench_capstack}"
STEPS="${CS_STEPS:-5000}"
ARMS="${CS_ARMS:-3 3,9 3,9,27 3,9,27,81}"
CN=./target/release/examples/cap_stack_run_benchmark

[ -x "$CN" ] || { echo "ERROR: build cap_stack_run_benchmark first" >&2; exit 1; }
mkdir -p "$OUT"

for w in $ARMS; do
    id="w$(echo "$w" | tr ',' '_')"
    echo
    echo "════════ cap stack: windows $w ════════"
    env AWARE_BENCH_CORPUS=data/wikitext103/wikitext_train.txt \
        AWARE_BENCH_VAL_CORPUS=data/wikitext103/wikitext_val.txt \
        AWARE_BENCH_BPE_DIR=data/bpe_wikitext103 \
        AWARE_BENCH_D_MODEL=128 AWARE_BENCH_CAP_N_TARGET=330 \
        AWARE_BENCH_SEED=42 AWARE_BENCH_STEPS="$STEPS" \
        AWARE_STACK_WINDOWS="$w" \
        AWARE_BENCH_ID="$id" AWARE_BENCH_OUTPUT_DIR="$OUT" \
        "$CN" > "$OUT/$id.log" 2>&1
    rc=$?
    grep -E 'receptive field|trainable params|done\.' "$OUT/$id.log" || true
    [ $rc -eq 0 ] || { echo "ERROR: arm $w failed (rc=$rc):" >&2; tail -15 "$OUT/$id.log" >&2; }
done

echo
echo "═══════════════════ CAP STACK SUMMARY ═══════════════════"
OUT="$OUT" python3 - <<'PYEOF'
import json, os, glob
out = os.environ["OUT"]
print(f"  {'windows':<16}{'RF':>5}{'params':>10}{'val ppl':>10}")
print(f"  {'pure transformer':<16}{'-':>5}{853120:>10}{51.22:>10.2f}   <- no caps")
print(f"  {'cap + 4 blocks':<16}{'-':>5}{895360:>10}{30.55:>10.2f}   <- caps + attention")
print("  " + "-" * 41)
for f in sorted(glob.glob(f"{out}/w*/report.json")):
    d = json.load(open(f))
    w = ",".join(str(x) for x in d["windows"])
    print(f"  {w:<16}{d['receptive_field']:>5}{d['params']:>10}{d['final_val_perplexity']:>10.2f}")
print()
print("  Depth is the question, not the transformer comparison: these arms")
print("  are a fraction of the baseline's size. If each added layer keeps")
print("  buying perplexity, composition works and step 3 (cap-conditioned")
print("  variable offsets, from the PMI table) is worth building.")
PYEOF
