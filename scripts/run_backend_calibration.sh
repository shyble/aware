#!/usr/bin/env bash
# Backend calibration: pure_transformer on WikiText-103, CUDA.
#
# Why this run exists. Paper #1 v2's WikiText results were produced on two
# machines: the kmeans window sweep, the discovery sweep and the baseline on
# an M1 Pro (candle CPU), the multi-head cap-attention variants on an RTX
# 4060 (CUDA). That split falls straight through Table 4 - every
# cap-attention number is CUDA, the baseline it is compared against is CPU -
# and the margin at stake for cap-pair is 0.44 perplexity, which is inside
# what a change in floating-point reduction order could plausibly produce.
#
# Re-running the baseline on the CUDA box makes that column same-backend and
# produces the one measurement neither machine has: the same configuration
# and seeds on both devices. If the two agree the paper can state that
# outright, which is stronger than never having asked.
#
# The TinyStories arm of the same table was always internally clean (every
# config there ran on CPU), so this is the only gap.
#
# IMPORTANT: writes to data/bench_backend_calib, NOT data/bench_wikitext.
# The CPU baseline in bench_wikitext is the comparison target - overwriting
# it would destroy the very thing this run is calibrating against.
#
# Prerequisites (see scripts/WINDOWS_SETUP.md):
#   python scripts/fetch_wikitext.py --max-train-words 10000000
#   cargo build --release --features cuda --example run_benchmark
#
# Usage (Git Bash, from the repo root):
#   AWARE_FEATURES=cuda AWARE_DEVICE=cuda ./scripts/run_backend_calibration.sh
set -uo pipefail
cd "$(dirname "$0")/.."

export AWARE_BENCH_CORPUS="${AWARE_BENCH_CORPUS:-data/wikitext103/wikitext_train.txt}"
export AWARE_BENCH_VAL_CORPUS="${AWARE_BENCH_VAL_CORPUS:-data/wikitext103/wikitext_val.txt}"
export AWARE_BENCH_BPE_DIR="${AWARE_BENCH_BPE_DIR:-data/bpe_wikitext103}"
export AWARE_BENCH_OUTPUT_DIR="${AWARE_BENCH_OUTPUT_DIR:-data/bench_backend_calib}"
export AWARE_BENCH_STEPS="${AWARE_BENCH_STEPS:-5000}"
SEEDS="${SEEDS:-42 123 7}"

for f in "$AWARE_BENCH_CORPUS" "$AWARE_BENCH_VAL_CORPUS" "$AWARE_BENCH_BPE_DIR/bpe.bin"; do
    [ -e "$f" ] || { echo "ERROR: missing $f - see scripts/WINDOWS_SETUP.md" >&2; exit 1; }
done
[ -x ./target/release/examples/run_benchmark ] || {
    echo "ERROR: build first:" >&2
    echo "  cargo build --release --features cuda --example run_benchmark" >&2
    exit 1
}

# Guard the CPU baseline explicitly rather than trusting the default above.
case "$AWARE_BENCH_OUTPUT_DIR" in
    *bench_wikitext) echo "ERROR: refusing to overwrite the CPU baseline in $AWARE_BENCH_OUTPUT_DIR" >&2; exit 1 ;;
esac

# The corpus must be byte-identical to the CPU machine's or the comparison
# measures tokenisation, not the backend. This is the check that caught the
# CRLF corruption when the Windows box was first set up.
if [ -f data/wikitext103.sha256 ]; then
    (cd data/wikitext103 && sha256sum -c ../wikitext103.sha256) || {
        echo "ERROR: corpus differs from the reference slice - results would not be comparable" >&2
        exit 1
    }
fi

echo "[calib] device=${AWARE_DEVICE:-<build default>}  steps=$AWARE_BENCH_STEPS  seeds=$SEEDS"
echo "[calib] output=$AWARE_BENCH_OUTPUT_DIR"

# shellcheck disable=SC2086
./scripts/run_multi_seed.sh pure_transformer $SEEDS || echo "[calib] FAILED" >&2

echo
echo "════════════════ CPU vs CUDA, pure_transformer, WikiText-103 ════════════════"
python3 - <<'PYEOF'
import json, glob, os, statistics as st

def load(d):
    out = {}
    for p in sorted(glob.glob(f"{d}/pure_transformer_seed*/report.json")):
        j = json.load(open(p))
        out[j["seed"]] = (j["final_val_perplexity"], j.get("device", "?"))
    return out

cpu = load("data/bench_wikitext")
gpu = load(os.environ.get("AWARE_BENCH_OUTPUT_DIR", "data/bench_backend_calib"))
shared = sorted(set(cpu) & set(gpu))
if not shared:
    raise SystemExit("  no overlapping seeds yet")

print(f"  {'seed':>6} {'CPU':>9} {'CUDA':>9} {'delta':>8}")
for s in shared:
    print(f"  {s:>6} {cpu[s][0]:>9.2f} {gpu[s][0]:>9.2f} {gpu[s][0]-cpu[s][0]:>+8.2f}")

c = [cpu[s][0] for s in shared]
g = [gpu[s][0] for s in shared]
sd = st.stdev(c) if len(c) > 1 else 0.0
diff = st.mean(g) - st.mean(c)
print(f"\n  CPU  {st.mean(c):6.2f} +/- {sd:4.2f}")
print(f"  CUDA {st.mean(g):6.2f} +/- {st.stdev(g) if len(g)>1 else 0.0:4.2f}")
print(f"  mean difference {diff:+.2f} ppl")

# The question this run was launched to answer: is the backend difference
# small next to the cap-pair margin it would otherwise confound?
print(f"\n  cap-pair sits 0.44 ppl below the CPU baseline.")
if abs(diff) < sd:
    print(f"  Backend difference ({abs(diff):.2f}) is within baseline seed noise ({sd:.2f}).")
    print("  -> backends agree; Table 4's WikiText column is safe to report as-is,")
    print("     and the paper can state that backend agreement was verified.")
else:
    print(f"  Backend difference ({abs(diff):.2f}) EXCEEDS baseline seed noise ({sd:.2f}).")
    print("  -> use this CUDA baseline for Table 4's WikiText column so the")
    print("     comparison is same-backend, and report the CPU baseline elsewhere.")
PYEOF
