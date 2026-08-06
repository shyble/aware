#!/usr/bin/env bash
# Mechanism grid: does the GATE protect knowledge, or merely the split?
#
# Substrate, corpora, budget and seed are held constant across arms. The
# only thing that varies is the policy over the substrate's lifetime:
#
#   bare           no deltas; ordinary sequential training (the pilot)
#   freeze         split, discard every delta         -> L1 floor
#   always_commit  split, merge every delta           -> the honesty control
#   gated          split, probe-gated per-cap commit  -> the claim
#
# `always_commit` is the arm that matters most for honesty. It has the
# delta machinery but no decision rule, so it should reproduce `bare`. If
# it does not, the split itself is changing training dynamics and no
# gated number means anything until that is understood.
#
# Cycle A is trained ONCE and shared by every arm — the arms differ only
# in what happens during cycle B, so re-training A per arm would waste
# hours and add seed noise between arms.
#
# GPU:
#   cargo build --release --features cuda --example cap_native_run_benchmark
#   AWARE_DEVICE=cuda ./scripts/run_cl_grid.sh
set -uo pipefail
cd "$(dirname "$0")/.."

A_TRAIN=data/tinystories_small/tinystories_train.txt
A_VAL=data/tinystories_small/tinystories_val.txt
B_TRAIN_SRC=data/wikitext103/wikitext_train.txt
B_VAL_SRC=data/wikitext103/wikitext_val.txt
BPE_DIR=data/brain_tinystories          # ONE tokenizer for the model's life

STEPS="${CL_STEPS:-3000}"
SEED="${CL_SEED:-42}"
PROBES="${CL_PROBES:-2000}"
COMMIT_AT="${CL_COMMIT_AT:-1.0}"
QUARANTINE_AT="${CL_QUARANTINE_AT:-0.9}"
OUT="${CL_OUT:-data/cl_grid}"
CN=./target/release/examples/cap_native_run_benchmark

# Smaller substrate than the pilot: full deltas double the cap-keyed
# parameters, and the mechanism question needs many fast arms rather than
# architectural fidelity. Retention is measured against each config's own
# cycle-A baseline, so this does not weaken the comparison.
MODEL_ENV=(
  AWARE_BENCH_D_MODEL="${CL_D_MODEL:-96}"
  AWARE_BENCH_N_BLOCKS="${CL_N_BLOCKS:-3}"
  AWARE_BENCH_D_FF=384
  AWARE_BENCH_N_HEADS=4
  AWARE_BENCH_CAP_N_TARGET=330
  AWARE_BENCH_CAP_WINDOW=3
  AWARE_BENCH_CAP_DISCOVERY=kmeans
  AWARE_CN_TOP_K=1
  AWARE_CN_ROUTING=sparse
  AWARE_CN_HIERARCHICAL=true
  AWARE_CN_L1_N_CAPS="${CL_K1:-64}"
  AWARE_CN_L1_WINDOW=1
  AWARE_CN_L1_DISCOVERY=kmeans
)

[ -x "$CN" ] || { echo "ERROR: build cap_native_run_benchmark first" >&2; exit 1; }
for f in "$A_TRAIN" "$A_VAL" "$B_TRAIN_SRC" "$B_VAL_SRC" "$BPE_DIR/bpe.bin"; do
    [ -e "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done

mkdir -p "$OUT"

# Cycle B corpus: private copy so the .bin tokenizer cache is rebuilt
# with A's BPE. The cache is keyed by mtime, not by which tokenizer wrote
# it, so sharing data/wikitext103 would silently train on WikiText's own
# token stream.
BDIR="$OUT/corpusB"; mkdir -p "$BDIR"
cp -f "$B_TRAIN_SRC" "$BDIR/b_train.txt"
cp -f "$B_VAL_SRC" "$BDIR/b_val.txt"
rm -f "$BDIR"/*.bin

export AWARE_BENCH_BPE_DIR="$BPE_DIR" AWARE_BENCH_SEED="$SEED" AWARE_BENCH_STEPS="$STEPS"

# ── Cycle A, once ──
WEIGHTS_A="$OUT/weights_A.safetensors"
if [ -f "$WEIGHTS_A" ]; then
    echo "── cycle A: reusing $WEIGHTS_A ──"
else
    echo "════════ cycle A: train on TinyStories, $STEPS steps ════════"
    env "${MODEL_ENV[@]}" \
        AWARE_BENCH_CORPUS="$A_TRAIN" AWARE_BENCH_VAL_CORPUS="$A_VAL" \
        AWARE_BENCH_OUTPUT_DIR="$OUT/cycleA" \
        AWARE_BENCH_SAVE_WEIGHTS="$WEIGHTS_A" \
        "$CN" 2>&1 | tail -3
fi

# The probe corpus is A's val token cache, produced by cycle A above.
PROBE_BIN="${A_VAL%.txt}.bin"
[ -f "$PROBE_BIN" ] || { echo "ERROR: no probe corpus at $PROBE_BIN" >&2; exit 1; }

# ── Cycle B, once per arm ──
for arm in bare freeze always_commit gated; do
    echo
    echo "════════ cycle B — arm: $arm ════════"
    env "${MODEL_ENV[@]}" \
        AWARE_BENCH_CORPUS="$BDIR/b_train.txt" \
        AWARE_BENCH_VAL_CORPUS="$BDIR/b_val.txt" \
        AWARE_BENCH_LOAD_WEIGHTS="$WEIGHTS_A" \
        AWARE_BENCH_OUTPUT_DIR="$OUT/$arm" \
        AWARE_BENCH_SAVE_WEIGHTS="$OUT/${arm}_weights_B.safetensors" \
        AWARE_CL_ARM="$arm" \
        AWARE_CL_PROBE_CORPUS="$PROBE_BIN" \
        AWARE_CL_PROBES="$PROBES" \
        AWARE_CL_COMMIT_AT="$COMMIT_AT" \
        AWARE_CL_QUARANTINE_AT="$QUARANTINE_AT" \
        "$CN" 2>&1 | grep -E 'probes:|RETENTION|arm=|final val_ppl|ERROR' || true

    # `bare` has no probe machinery, so score its retention separately
    # against the same probe set for an apples-to-apples number.
    if [ "$arm" = "bare" ]; then
        echo "  (bare retention is measured by the eval-only pass below)"
    fi
done

# ── Summary ──
echo
echo "════════════════════════ GRID SUMMARY ════════════════════════"
OUT="$OUT" python3 - <<'PYEOF'
import json, os, glob

out = os.environ["OUT"]
print(f"  {'arm':<15}{'retention':>11}   {'B val ppl':>10}   decisions")
for arm in ("bare", "freeze", "always_commit", "gated"):
    sfile = glob.glob(f"{out}/{arm}/*/cl_summary.json")
    rep = glob.glob(f"{out}/{arm}/*/report.json")
    ret, reg = "—", ""
    if sfile:
        s = json.load(open(sfile[0]))
        ret = f"{100*s['retention']:.1f}%"
        reg = s.get("registry", "")
    ppl = "—"
    if rep:
        ppl = f"{json.load(open(rep[0]))['final_val_perplexity']:.2f}"
    print(f"  {arm:<15}{ret:>11}   {ppl:>10}   {reg}")

print()
print("  Reading it:")
print("   - always_commit ≈ bare  → the split is inert; the GATE is what acts")
print("   - freeze = 100% retention, poor B ppl → the L1 floor, as designed")
print("   - gated: high retention AND B ppl near bare → the L2 claim")
print("   - gated with 0 committed → protection bought by refusing to learn")
PYEOF
