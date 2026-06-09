"""Compare benchmark methods and write aggregate CSV/PNG outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from open_pc_lgo.benchmarks import BENCHMARKS
from open_pc_lgo.experiments.run_benchmark import METHODS, run_experiment, summarize, write_csv


def compare_methods(
    *,
    benchmark_name: str,
    methods: list[str],
    n_instances: int,
    budget: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        rows = run_experiment(
            benchmark_name=benchmark_name,
            method=method,
            n_instances=n_instances,
            budget=budget,
            seed=seed,
        )
        all_rows.extend(rows)
        summary = summarize(rows)
        summary_rows.append({"benchmark": benchmark_name, "method": method, "budget": budget, **summary})
    return all_rows, summary_rows


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["benchmark", "method", "budget", "accepted_rate", "mean_gap", "mean_runtime"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_figure(summary_rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = [str(row["method"]) for row in summary_rows]
    accepted = [float(row["accepted_rate"]) for row in summary_rows]
    gaps = [float(row["mean_gap"]) for row in summary_rows]

    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
    axes[0].bar(x, accepted, color="#3d7f5f")
    axes[0].set_title("Acceptance Rate")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xticks(x, methods, rotation=20, ha="right")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, gaps, color="#b44e52")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Mean Gap to Oracle")
    axes[1].set_xticks(x, methods, rotation=20, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), default="conic_active_set")
    parser.add_argument("--methods", default="unguided,random_guided,expert_guided,oracle")
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    invalid = [method for method in methods if method not in METHODS]
    if invalid:
        raise SystemExit(f"unknown methods: {invalid}; available: {METHODS}")
    all_rows, summary_rows = compare_methods(
        benchmark_name=args.benchmark,
        methods=methods,
        n_instances=args.instances,
        budget=args.budget,
        seed=args.seed,
    )
    stem = f"{args.benchmark}_compare_seed{args.seed}"
    detail_csv = args.output_dir / f"{stem}_details.csv"
    summary_csv = args.output_dir / f"{stem}_summary.csv"
    png_path = args.output_dir / f"{stem}.png"
    write_csv(all_rows, detail_csv)
    write_summary_csv(summary_rows, summary_csv)
    write_comparison_figure(summary_rows, png_path)
    print(f"wrote {detail_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
