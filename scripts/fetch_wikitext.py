#!/usr/bin/env python3
"""fetch_wikitext.py - download WikiText for AWARE generalisation experiments.

WikiText (Merity et al. 2016) is the standard word-level language-modelling
benchmark, drawn from verified-good Wikipedia articles. It is the second
corpus for paper #1: TinyStories is synthetic, small-vocabulary, child-level
English, so a result that only holds there is uncomparable to the wider
literature. WikiText is real encyclopaedic prose with a much richer
vocabulary, which is exactly the point of testing on it.

Two variants:
  wikitext-103-raw-v1  ~103M train tokens - THE standard benchmark. At the
                       paper's budget (5000 steps x 32 x 128 = 20.5M tokens)
                       this is ~0.2 epochs, so every token is fresh.
  wikitext-2-raw-v1    ~2M train tokens - same budget means ~10 epochs over
                       the same text, so repetition and overfitting dominate.

We default to 103 for that reason, streamed and truncated to a token budget
so the download stays modest.

IMPORTANT: train a corpus-specific BPE for these runs. The TinyStories
tokenizer fragments encyclopaedic English badly, which would show up as an
architecture result rather than a tokenizer mismatch:

    AWARE_BENCH_BPE_DIR=data/bpe_wikitext103 \\
    AWARE_BENCH_CORPUS=data/wikitext103/wikitext_train.txt \\
    AWARE_BENCH_VAL_CORPUS=data/wikitext103/wikitext_val.txt \\
    ./scripts/run_multi_seed.sh kmeans_w3

Output: <out-dir>/wikitext_train.txt
        <out-dir>/wikitext_val.txt
        <out-dir>/stats.json

Usage:
    python3 scripts/fetch_wikitext.py
    python3 scripts/fetch_wikitext.py --variant wikitext-2-raw-v1 --out-dir data/wikitext2
    python3 scripts/fetch_wikitext.py --max-train-words 40000000
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", default="data/wikitext103", help="output directory")
    ap.add_argument(
        "--variant",
        default="wikitext-103-raw-v1",
        choices=["wikitext-103-raw-v1", "wikitext-2-raw-v1"],
        help="which WikiText config to pull",
    )
    ap.add_argument(
        "--max-train-words",
        type=int,
        default=30_000_000,
        help="stop after roughly this many whitespace words of train text. "
        "The paper's budget consumes ~20.5M BPE tokens; 30M words leaves "
        "headroom without downloading the whole 500MB corpus. 0 = no limit.",
    )
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # The dataset moved to a namespaced repo; newer huggingface_hub rejects
    # the bare name ("repo id must be namespace/name"), while older datasets
    # releases only know the canonical alias. Try both.
    repo_ids = ["Salesforce/wikitext", "wikitext"]

    def open_split(split):
        last = None
        for repo in repo_ids:
            try:
                return load_dataset(repo, args.variant, split=split, streaming=True)
            except Exception as e:  # noqa: BLE001 - fall through to the next id
                last = e
        raise SystemExit(
            f"ERROR: could not load {args.variant} from any of {repo_ids}: {last}"
        )

    def pull(split, cap_words):
        """Stream one split, keeping real article text and dropping the
        section headings WikiText marks with '=' rules."""
        ds = open_split(split)
        lines, n_words = [], 0
        for row in ds:
            text = row.get("text", "")
            stripped = text.strip()
            if not stripped or stripped.startswith("="):
                continue
            lines.append(stripped)
            n_words += stripped.count(" ") + 1
            if cap_words and n_words >= cap_words:
                break
            if len(lines) % 50_000 == 0:
                print(f"    {split}: {n_words:>12,} words")
        return lines, n_words

    print(f"[wikitext] variant={args.variant} (streaming)")
    train_lines, train_words = pull("train", args.max_train_words)
    print(f"  train: {len(train_lines):,} paragraphs, ~{train_words:,} words")
    # The official validation split is the honest holdout - never slice train.
    val_lines, val_words = pull("validation", 0)
    print(f"  val:   {len(val_lines):,} paragraphs, ~{val_words:,} words")

    if not train_lines or not val_lines:
        print("ERROR: empty split - check the variant name", file=sys.stderr)
        sys.exit(1)

    train_path = out / "wikitext_train.txt"
    val_path = out / "wikitext_val.txt"
    # Blank-line separated paragraphs, matching the TinyStories layout so the
    # same benchmark runner consumes either corpus unchanged.
    train_path.write_text("\n\n".join(train_lines), encoding="utf-8")
    val_path.write_text("\n\n".join(val_lines), encoding="utf-8")

    stats = {
        "variant": args.variant,
        "train_paragraphs": len(train_lines),
        "val_paragraphs": len(val_lines),
        "train_words_approx": train_words,
        "val_words_approx": val_words,
        "train_bytes": train_path.stat().st_size,
        "val_bytes": val_path.stat().st_size,
        "max_train_words": args.max_train_words,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"[wikitext] wrote {train_path} ({stats['train_bytes']:,} bytes)")
    print(f"[wikitext] wrote {val_path} ({stats['val_bytes']:,} bytes)")
    print("[wikitext] remember AWARE_BENCH_BPE_DIR for a corpus-specific tokenizer.")


if __name__ == "__main__":
    main()
