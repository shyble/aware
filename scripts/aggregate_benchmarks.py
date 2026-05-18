#!/usr/bin/env python3
"""aggregate_benchmarks.py - walk data/bench/*/report.json into a comparison table.

Usage:
    python3 scripts/aggregate_benchmarks.py [bench_dir]

Default bench_dir = data/bench/

Outputs:
    <bench_dir>/benchmark_results.csv        all fields, sorted by val_ppl
    <bench_dir>/benchmark_results.md         readable comparison table

Optional (requires matplotlib): pass --plot to overlay loss curves.
"""
import csv
import json
import sys
from pathlib import Path


PRIORITY_FIELDS = [
    "run_id",
    "final_val_perplexity",
    "final_val_loss",
    "final_train_loss",
    "params",
    "attention",
    "discovery",
    "include_cap_layer",
    "cap_source",
    "cap_window",
    "n_caps_target",
    "d_model",
    "n_blocks",
    "n_heads",
    "d_ff",
    "steps",
    "seed",
    "wall_clock_seconds",
]

MD_COLS = [
    "run_id",
    "final_val_perplexity",
    "final_val_loss",
    "params",
    "attention",
    "discovery",
    "cap_source",
    "cap_window",
    "wall_clock_seconds",
]


def load_runs(bench_dir: Path) -> list[dict]:
    runs = []
    for report_path in sorted(bench_dir.glob("*/report.json")):
        try:
            with open(report_path) as f:
                run = json.load(f)
                run["_path"] = str(report_path)
                runs.append(run)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  skip {report_path}: {e}", file=sys.stderr)
    return runs


def write_csv(runs: list[dict], path: Path) -> None:
    if not runs:
        return
    # union of fields
    all_keys = sorted({k for r in runs for k in r.keys() if not k.startswith("_") and k != "trajectory"})
    ordered = [f for f in PRIORITY_FIELDS if f in all_keys] + [f for f in all_keys if f not in PRIORITY_FIELDS]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow(r)


def write_markdown(runs: list[dict], path: Path) -> None:
    if not runs:
        with open(path, "w") as f:
            f.write("# Benchmark Results\n\n(no runs found)\n")
        return

    with open(path, "w") as f:
        f.write("# Benchmark Results\n\n")
        f.write(f"Aggregated from `{len(runs)}` runs. Sorted by `final_val_perplexity` (lower = better).\n\n")
        f.write("| " + " | ".join(MD_COLS) + " |\n")
        f.write("|" + "|".join(["---"] * len(MD_COLS)) + "|\n")
        for r in runs:
            row = []
            for c in MD_COLS:
                v = r.get(c, "-")
                if c == "final_val_perplexity" and isinstance(v, (int, float)):
                    row.append(f"{v:.2f}")
                elif c == "final_val_loss" and isinstance(v, (int, float)):
                    row.append(f"{v:.4f}")
                elif c == "wall_clock_seconds" and isinstance(v, (int, float)):
                    row.append(f"{v:.1f}")
                elif c == "params" and isinstance(v, int):
                    row.append(f"{v:,}")
                else:
                    row.append(str(v))
            f.write("| " + " | ".join(row) + " |\n")

        # Best-3 summary
        f.write("\n## Top 3 Configurations\n\n")
        for i, r in enumerate(runs[:3], 1):
            ppl = r.get("final_val_perplexity", float("inf"))
            f.write(f"{i}. **{r.get('run_id', '?')}** - val_ppl `{ppl:.2f}` - "
                    f"attention=`{r.get('attention')}`, discovery=`{r.get('discovery')}`, "
                    f"window=`{r.get('cap_window')}`, source=`{r.get('cap_source')}`\n")

        # Notes on trajectories (if present)
        with_trajectory = [r for r in runs if r.get("trajectory")]
        if with_trajectory:
            f.write(f"\n## Trajectories\n\n{len(with_trajectory)} run(s) include step-by-step trajectories ")
            f.write("in their `report.json`. Use `--plot` flag to visualize.\n")


def maybe_plot(runs: list[dict], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping plot)", file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in runs:
        traj = r.get("trajectory", [])
        if not traj:
            continue
        steps = [p["step"] for p in traj]
        ppls = [p["val_perplexity"] for p in traj]
        ax.plot(steps, ppls, label=r.get("run_id", "?"), alpha=0.7)
    ax.set_xlabel("step")
    ax.set_ylabel("val perplexity")
    ax.set_yscale("log")
    ax.set_title("Benchmark: val perplexity over training")
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path)
    print(f"  plot: {output_path}", file=sys.stderr)


def main():
    args = sys.argv[1:]
    do_plot = False
    if "--plot" in args:
        do_plot = True
        args.remove("--plot")
    bench_dir = Path(args[0] if args else "data/bench")

    if not bench_dir.exists():
        print(f"ERROR: {bench_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    runs = load_runs(bench_dir)
    if not runs:
        print(f"No report.json files found under {bench_dir}/*/", file=sys.stderr)
        sys.exit(1)

    # Sort by val perplexity, then loss, missing -> infinity
    runs.sort(key=lambda r: (
        r.get("final_val_perplexity", float("inf")),
        r.get("final_val_loss", float("inf")),
    ))

    csv_path = bench_dir / "benchmark_results.csv"
    md_path = bench_dir / "benchmark_results.md"
    write_csv(runs, csv_path)
    write_markdown(runs, md_path)

    if do_plot:
        plot_path = bench_dir / "benchmark_trajectories.png"
        maybe_plot(runs, plot_path)

    print(f"Aggregated {len(runs)} runs.")
    print(f"  CSV:      {csv_path}")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    main()
