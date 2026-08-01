# Windows + NVIDIA GPU setup

For running benchmarks on a CUDA machine. Written for the dense scaling
sweep (`run_dense_sweep.sh`), but the build steps apply to any run.

Why bother: dense transformers are the GPU's best case. The sweep's nine
runs take ~203 h on an M1 Pro CPU and roughly 4-8 h on an RTX 4060.
Cap-native models, by contrast, are sparse and dispatch-bound — they gain
far less from a GPU, which is why they can stay on the CPU machine.

## 0. One-time install

1. **NVIDIA driver** — recent Game Ready or Studio driver.
2. **CUDA Toolkit 12.x** — confirm with `nvcc --version`.
3. **Visual Studio Build Tools 2022**, "Desktop development with C++"
   workload. This supplies `cl.exe`, which `nvcc` needs as its host
   compiler. Builds fail confusingly without it.
4. **Rust** via rustup, default `x86_64-pc-windows-msvc` toolchain.
5. **Git for Windows** — gives you Git Bash, which runs these scripts
   unmodified. Run builds from a *Developer PowerShell for VS 2022* (or
   x64 Native Tools prompt) so `cl.exe` is on `PATH`.
6. **Python 3** with `pip install datasets` (for fetching the corpus).

## 1. Repo and data

```bash
git clone <repo-url> aware
cd aware
git checkout phase-b-prime          # dense presets live here

# Corpus is not in git (52 MB); the tokenizer is.
python scripts/fetch_wikitext.py --max-train-words 10000000

# Verify the slice is byte-identical to the one used on the other machine
cd data/wikitext103 && sha256sum -c ../wikitext103.sha256 && cd ../..
```

Both files must print `OK`. If they do not, the upstream dataset changed —
copy `data/wikitext103/` from the Mac instead. The tokenizer
(`data/bpe_wikitext103/bpe.bin`) comes from the clone, so tokenisation is
identical either way.

## 2. Build

From the VS developer shell:

```bash
cargo build --release --features cuda --example run_benchmark
```

If this compiles, CUDA is wired correctly. Failures here are almost always
a missing `CUDA_PATH`, `nvcc` not on `PATH`, or `cl.exe` not on `PATH`.

## 3. Smoke probe first (~5 min)

Never start a long run on an unprobed backend. Open `nvidia-smi -l 1` in a
second window and run 100 steps of the largest config:

```bash
AWARE_FEATURES=cuda AWARE_DEVICE=cuda AWARE_BENCH_STEPS=100 \
  AWARE_BENCH_OUTPUT_DIR=/tmp/probe SEEDS=42 ./scripts/run_dense_sweep.sh
```

Check three things: it reports `device: cuda` (not cpu — a silent CPU
fallback is the failure mode this probe exists to catch), VRAM stays well
under 8 GB, and the loss decreases. `dense_30m` at batch 32 should sit
around 2-3 GB.

## 4. The sweep

```bash
AWARE_FEATURES=cuda AWARE_DEVICE=cuda \
  nohup ./scripts/run_dense_sweep.sh > /tmp/dense_sweep.log 2>&1 &
```

Nine runs, ~4-8 h. Results land in `data/bench_dense_sweep_wt/<id>/report.json`
and the script prints the size-to-perplexity table at the end.

## 5. Getting results back

The `report.json` files are small. Copy `data/bench_dense_sweep_wt/` back to
the Mac (or commit them to a scratch branch) so the curve can be fitted
alongside the cap-native numbers.

## Notes

- `AWARE_FEATURES` selects the build backend; `AWARE_DEVICE` selects the
  runtime device. Set both to `cuda`.
- If a run OOMs, lower `AWARE_BENCH_BATCH_SIZE` — but record it, since the
  sweep's comparability depends on a constant batch across sizes.
- Cap-native configs are a different story: the 368M single-discovery model
  needs ~5.9 GB of static state before activations, so its fit on an 8 GB
  card is uncertain. Probe it separately before attempting.
