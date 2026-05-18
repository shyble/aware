#!/usr/bin/env python3
"""fetch_tinystories.py - download a TinyStories subset for AWARE training.

TinyStories is the corpus from Eldan & Li (2023, "TinyStories: How Small
Can Language Models Be and Still Speak Coherent English?"). It's a
synthetic dataset of short stories using vocabulary that 3-4-year-olds
understand. Despite tiny vocabulary, the stories have coherent narrative
structure, characters, and simple causality - exactly the properties
that emerge with capable language modeling.

Source: HuggingFace `roneneldan/TinyStories`, ~2.7B tokens total.
We pull a configurable subset and concatenate into a single text file
suitable for AWARE's BPE tokenizer + the run_benchmark example.

Output: data/tinystories/tinystories_train.txt (one story per "paragraph"
                                                 separated by blank lines)
        data/tinystories/tinystories_val.txt   (10% held-out)
        data/tinystories/stats.json

Default: 100K stories ≈ ~22M tokens (good Phase-1 baseline budget;
~30 GPU-hours per Eldan & Li for a 28M-param model).

Usage:
    python3 scripts/fetch_tinystories.py
    python3 scripts/fetch_tinystories.py --n-stories 200000   # ~45M tokens
    python3 scripts/fetch_tinystories.py --n-stories 1000000  # full ≈ 220M
    python3 scripts/fetch_tinystories.py --out-dir data/ts_small --n-stories 10000
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data/tinystories",
                    help="output directory")
    ap.add_argument("--n-stories", type=int, default=100_000,
                    help="number of stories to pull from the train split")
    ap.add_argument("--val-ratio", type=float, default=0.05,
                    help="fraction of stories held out for validation")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for the held-out split")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[tinystories] loading dataset (streamed, will pull {args.n_stories} stories)...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    # Stream the requested number of stories. iter is fast - the streaming
    # mode downloads on-demand without materializing the full corpus.
    stories = []
    for i, row in enumerate(ds):
        if i >= args.n_stories:
            break
        text = row.get("text", "").strip()
        if text:
            stories.append(text)
        if (i + 1) % 10_000 == 0:
            print(f"  pulled {i+1:>8} / {args.n_stories}")

    print(f"[tinystories] pulled {len(stories)} stories.")

    # Deterministic train/val split
    import random
    rng = random.Random(args.seed)
    indices = list(range(len(stories)))
    rng.shuffle(indices)
    n_val = max(1, int(len(stories) * args.val_ratio))
    val_idx = set(indices[:n_val])

    train_stories = [stories[i] for i in range(len(stories)) if i not in val_idx]
    val_stories   = [stories[i] for i in range(len(stories)) if i in val_idx]

    train_path = out / "tinystories_train.txt"
    val_path   = out / "tinystories_val.txt"

    # Separate stories by blank lines (preserve per-story structure
    # so a future training driver can use sentence/document boundaries
    # if desired).
    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(train_stories) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(val_stories) + "\n")

    train_bytes = train_path.stat().st_size
    val_bytes   = val_path.stat().st_size

    # Rough token estimate: ~4 chars/token for English BPE-128
    est_train_tokens = train_bytes // 4
    est_val_tokens   = val_bytes // 4

    stats = {
        "n_stories": len(stories),
        "n_train_stories": len(train_stories),
        "n_val_stories": len(val_stories),
        "train_bytes": train_bytes,
        "val_bytes": val_bytes,
        "estimated_train_tokens": est_train_tokens,
        "estimated_val_tokens": est_val_tokens,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
    }
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print()
    print(f"[tinystories] wrote {train_path} ({train_bytes:>11,} bytes, ~{est_train_tokens:>10,} tokens)")
    print(f"[tinystories] wrote {val_path}   ({val_bytes:>11,} bytes, ~{est_val_tokens:>10,} tokens)")
    print(f"[tinystories] stats: {out / 'stats.json'}")
    print()
    print("Next:")
    print(f"  AWARE_BENCH_CORPUS={train_path} \\")
    print(f"  AWARE_BENCH_VAL_CORPUS={val_path} \\")
    print(f"  AWARE_BENCH_CAP_DISCOVERY=kmeans AWARE_BENCH_CAP_WINDOW=3 \\")
    print(f"  cargo run --release --example run_benchmark")


if __name__ == "__main__":
    main()
