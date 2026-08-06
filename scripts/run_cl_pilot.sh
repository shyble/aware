#!/usr/bin/env bash
# Continual-learning pilot: two-cycle A→B across the identity ladder.
#
#   dense (pure transformer)  →  cap-augmented (kmeans_w3)  →  cap-native hier
#
# For each rung: train on A (TinyStories), checkpoint, continue training
# on B (WikiText through A's tokenizer), then re-evaluate the B-model on
# A's validation set. Cap rungs also dump per-token winning caps before
# and after B, and cap-native's checkpoints get per-cap slab comparison.
#
# GPU: build with --features cuda and export AWARE_DEVICE=cuda first.
# Every run's report.json records the device — verify it says cuda.
#
#   cargo build --release --features cuda --example run_benchmark \
#                                          --example cap_native_run_benchmark
#   AWARE_DEVICE=cuda ./scripts/run_cl_pilot.sh
#
# Runs everything under data/cl_pilot (NEVER data/bench).
set -uo pipefail
cd "$(dirname "$0")/.."

# ── Fixed pilot knobs (mirror .local/continual-learning/pilot_config.sh) ──
A_TRAIN=data/tinystories_small/tinystories_train.txt
A_VAL=data/tinystories_small/tinystories_val.txt
B_TRAIN_SRC=data/wikitext103/wikitext_train.txt
B_VAL_SRC=data/wikitext103/wikitext_val.txt
BPE_DIR=data/brain_tinystories          # ONE tokenizer for the model's life
STEPS="${CL_STEPS:-3000}"
SEED="${CL_SEED:-42}"
EVAL_BATCHES="${CL_EVAL_BATCHES:-64}"
OUT="${CL_OUT:-data/cl_pilot}"

BENCH=./target/release/examples/run_benchmark
CN=./target/release/examples/cap_native_run_benchmark

for f in "$A_TRAIN" "$A_VAL" "$B_TRAIN_SRC" "$B_VAL_SRC" "$BPE_DIR/bpe.bin"; do
    [ -e "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done
[ -x "$BENCH" ] && [ -x "$CN" ] || { echo "ERROR: build both example binaries first" >&2; exit 1; }

mkdir -p "$OUT/analysis"

# ── Cycle-B corpus: PRIVATE COPY, tokenized fresh with A's BPE ──
# The .bin tokenization cache next to a corpus is keyed by mtime only,
# not by which tokenizer produced it. data/wikitext103/*.bin already
# exists from the WikiText BPE; reusing it here would silently train
# cycle B on the wrong token stream. A private copy gets a fresh cache.
BDIR="$OUT/corpusB"
mkdir -p "$BDIR"
cp -f "$B_TRAIN_SRC" "$BDIR/b_train.txt"
cp -f "$B_VAL_SRC" "$BDIR/b_val.txt"
rm -f "$BDIR"/*.bin
B_TRAIN="$BDIR/b_train.txt"
B_VAL="$BDIR/b_val.txt"

export AWARE_BENCH_BPE_DIR="$BPE_DIR"
export AWARE_BENCH_SEED="$SEED"
export AWARE_BENCH_STEPS="$STEPS"
export AWARE_BENCH_EVAL_BATCHES="$EVAL_BATCHES"

ppl_of() { python3 -c "import json;print(json.load(open('$1'))['final_val_perplexity'])" 2>/dev/null || echo "?"; }
report_in() { ls "$1"/*/report.json 2>/dev/null | head -1; }

# run <binary> <outdir> <extra env as VAR=VAL ...>
run() {
    local bin="$1" outdir="$2"; shift 2
    mkdir -p "$outdir"
    env AWARE_BENCH_OUTPUT_DIR="$outdir" "$@" "$bin" 2>&1 | tail -2
}

ladder_rung() {
    # $1 name, $2 binary, $3.. config env (shared by every phase of the rung)
    local name="$1" bin="$2"; shift 2
    local root="$OUT/$name"
    echo
    echo "════════════════════════ $name ════════════════════════"

    echo "── cycle A: train $STEPS steps on TinyStories, save ──"
    run "$bin" "$root/cycleA" "$@" \
        AWARE_BENCH_CORPUS="$A_TRAIN" AWARE_BENCH_VAL_CORPUS="$A_VAL" \
        AWARE_BENCH_SAVE_WEIGHTS="$root/weights_A.safetensors"

    echo "── eval A-model on A-val (+ winners before) ──"
    run "$bin" "$root/evalA" "$@" \
        AWARE_BENCH_CORPUS="$A_TRAIN" AWARE_BENCH_VAL_CORPUS="$A_VAL" \
        AWARE_BENCH_EVAL_ONLY=1 \
        AWARE_BENCH_LOAD_WEIGHTS="$root/weights_A.safetensors" \
        AWARE_CN_DUMP_WINNERS="$root/winners_before"

    echo "── cycle B: load A, train $STEPS steps on WikiText(A-tokenizer), save ──"
    run "$bin" "$root/cycleB" "$@" \
        AWARE_BENCH_CORPUS="$B_TRAIN" AWARE_BENCH_VAL_CORPUS="$B_VAL" \
        AWARE_BENCH_LOAD_WEIGHTS="$root/weights_A.safetensors" \
        AWARE_BENCH_SAVE_WEIGHTS="$root/weights_B.safetensors"

    echo "── eval B-model on A-val (+ winners after) ──"
    run "$bin" "$root/evalB" "$@" \
        AWARE_BENCH_CORPUS="$A_TRAIN" AWARE_BENCH_VAL_CORPUS="$A_VAL" \
        AWARE_BENCH_EVAL_ONLY=1 \
        AWARE_BENCH_LOAD_WEIGHTS="$root/weights_B.safetensors" \
        AWARE_CN_DUMP_WINNERS="$root/winners_after"
}

# ── Rung 1: dense (no identity) ──
ladder_rung dense "$BENCH" \
    AWARE_BENCH_ATTENTION=standard AWARE_BENCH_INCLUDE_CAP_LAYER=false

# ── Rung 2: cap-augmented (identity at the input only) ──
ladder_rung cap_augmented "$BENCH" \
    AWARE_BENCH_ATTENTION=standard AWARE_BENCH_INCLUDE_CAP_LAYER=true \
    AWARE_BENCH_CAP_DISCOVERY=kmeans AWARE_BENCH_CAP_N_TARGET=330 \
    AWARE_BENCH_CAP_WINDOW=3

# ── Rung 3: cap-native hierarchical (total identity) ──
ladder_rung cap_native "$CN" \
    AWARE_BENCH_D_MODEL=128 AWARE_BENCH_N_BLOCKS=4 AWARE_BENCH_D_FF=512 \
    AWARE_BENCH_CAP_N_TARGET=330 AWARE_BENCH_CAP_WINDOW=3 \
    AWARE_BENCH_CAP_DISCOVERY=kmeans \
    AWARE_CN_TOP_K=1 AWARE_CN_ROUTING=sparse AWARE_CN_HIERARCHICAL=true \
    AWARE_CN_L1_N_CAPS=128 AWARE_CN_L1_WINDOW=1 AWARE_CN_L1_DISCOVERY=kmeans

# ── Mechanism analysis (cap rungs) ──
echo
echo "════════════════════════ analysis ════════════════════════"
for name in cap_augmented cap_native; do
    root="$OUT/$name"
    echo "── $name: slabs A vs B ──"
    python3 scripts/cl_analysis.py slabs \
        "$root/weights_A.safetensors" "$root/weights_B.safetensors" \
        | tee "$OUT/analysis/${name}_slabs.txt" | tail -8
    for layer in l0 l1; do
        if [ -f "$root/winners_before.$layer.bin" ]; then
            echo "── $name: firing stability ($layer) ──"
            python3 scripts/cl_analysis.py winners \
                "$root/winners_before.$layer.bin" "$root/winners_after.$layer.bin" \
                | tee "$OUT/analysis/${name}_winners_$layer.txt" | head -3
        fi
    done
done

# ── Summary: retention on A, per rung ──
echo
echo "════════════════════════ RETENTION SUMMARY ════════════════════════"
echo "ppl on A-val, before → after training on B (lower change = better retention)"
for name in dense cap_augmented cap_native; do
    root="$OUT/$name"
    before=$(ppl_of "$(report_in "$root/evalA")")
    after=$(ppl_of "$(report_in "$root/evalB")")
    printf "  %-14s %s → %s\n" "$name" "$before" "$after"
done
echo
echo "Decision table: .local/continual-learning/experiment_plan.md §3"
echo "Verify every report.json says the device you expected."
