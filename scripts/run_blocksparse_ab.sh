#!/usr/bin/env bash
# A/B the device-routed dispatch against the default host-routed path.
#
# Same config, same seed, same step count -- one environment variable
# apart. Answers two questions in order:
#
#   1. Does it compute the same thing?  Compare final_val_perplexity.
#      Each token's output is x_t @ W_cap(t) either way, so the expected
#      answer is "identical". A small difference means the matmul
#      reassociated; a large one means a bug.
#   2. Is it faster?  Compare wall_clock_seconds. Only meaningful on
#      CUDA -- on CPU there is no host round trip to remove, so expect
#      no gain there and possibly a small loss.
#
# The interesting config is single-discovery (330 caps): host-side index
# construction scales with n_caps * bound, which is what made the 4060
# slower than the M1 CPU. Hierarchical (128 caps) already runs at
# 0.446 s/step and has little to reclaim.
#
# Usage:
#   ./scripts/run_blocksparse_ab.sh                      # default cfg+seed
#   CFG=cap_native_hier_d128 SEED=7 ./scripts/run_blocksparse_ab.sh
#   STEPS=200 ./scripts/run_blocksparse_ab.sh
set -uo pipefail
cd "$(dirname "$0")/.."

CFG="${CFG:-cap_native_sparse_d128}"
SEED="${SEED:-42}"
STEPS="${STEPS:-100}"
OUT="${OUT:-/tmp/bs_ab}"

: "${AWARE_BENCH_CORPUS:?set AWARE_BENCH_CORPUS (this branch has no corpus committed)}"
: "${AWARE_BENCH_VAL_CORPUS:?set AWARE_BENCH_VAL_CORPUS}"
: "${AWARE_BENCH_BPE_DIR:?set AWARE_BENCH_BPE_DIR}"

export AWARE_BENCH_STEPS="$STEPS"
rm -rf "$OUT"; mkdir -p "$OUT"

echo "[ab] cfg=$CFG seed=$SEED steps=$STEPS device=${AWARE_DEVICE:-<build default>}"

echo; echo "──────── A: default host-routed dispatch ────────"
AWARE_BENCH_OUTPUT_DIR="$OUT/off" \
    ./scripts/run_multi_seed.sh "$CFG" "$SEED" || echo "[ab] A FAILED" >&2

echo; echo "──────── B: device-routed dispatch ────────"
AWARE_CN_BLOCKSPARSE=1 AWARE_BENCH_OUTPUT_DIR="$OUT/on" \
    ./scripts/run_multi_seed.sh "$CFG" "$SEED" || echo "[ab] B FAILED" >&2

echo; echo "════════ RESULT ════════"
OUT="$OUT" CFG="$CFG" SEED="$SEED" python3 - <<'PYEOF'
import json, os, glob, sys

out, cfg, seed = os.environ["OUT"], os.environ["CFG"], os.environ["SEED"]

def load(side):
    hits = glob.glob(f"{out}/{side}/{cfg}_seed{seed}/report.json")
    if not hits:
        return None
    return json.load(open(hits[0]))

a, b = load("off"), load("on")
if not a or not b:
    sys.exit("  one side produced no report - check the run logs above")

pa, pb = a["final_val_perplexity"], b["final_val_perplexity"]
wa, wb = a.get("wall_clock_seconds", 0), b.get("wall_clock_seconds", 0)

print(f"  {'':<12}{'host-routed':>14}{'device-routed':>16}")
print(f"  {'val ppl':<12}{pa:>14.6f}{pb:>16.6f}")
print(f"  {'seconds':<12}{wa:>14.1f}{wb:>16.1f}")
print(f"  {'s/step':<12}{wa/a['steps']:>14.3f}{wb/b['steps']:>16.3f}")

print()
if pa == pb:
    print("  CORRECTNESS: identical -- pure optimisation, safe to keep.")
elif abs(pa - pb) < 0.01:
    print(f"  CORRECTNESS: differs by {abs(pa-pb):.2e} -- consistent with")
    print("    floating-point reassociation. Compare across seeds before")
    print("    using this path for any reported number.")
else:
    print(f"  CORRECTNESS: differs by {abs(pa-pb):.4f} -- too large for")
    print("    reassociation. Treat as a BUG, not a numerics artefact.")

if wa > 0 and wb > 0:
    print(f"  SPEED: {'%.1f%% faster' % (100*(wa-wb)/wa) if wb < wa else '%.1f%% slower' % (100*(wb-wa)/wa)}")
PYEOF
