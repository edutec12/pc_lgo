"""Run one OPEN-pc_lgo benchmark and write CSV/PNG summaries."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from open_pc_lgo.audit import metrics_from_solution
from open_pc_lgo.benchmarks import (
    BENCHMARKS,
    CapacityExpansionToyBenchmark,
    ConicActiveSetBenchmark,
    EVChargingBenchmark,
    RastriginWarmStartBenchmark,
    make_benchmark,
)
from open_pc_lgo.core import Context, Guidance, SolverResult


METHODS = ("unguided", "expert_guided", "random_guided", "oracle")


def guidance_for_method(
    problem: Any,
    theta: Context,
    method: str,
    oracle: SolverResult,
) -> Guidance:
    """Create guidance for a named benchmark method."""

    if method == "unguided":
        return Guidance(strategy_name="unguided")
    if method == "expert_guided":
        return replace(oracle.guidance, strategy_name="expert_guided")
    if method != "random_guided":
        raise KeyError(f"unknown method {method!r}")

    if isinstance(problem, RastriginWarmStartBenchmark):
        lower, upper = problem.bounds_scalar
        warm_start = problem.rng.uniform(lower, upper, size=problem.dimension)
        return Guidance(warm_start=warm_start, strategy_name="random_guided")
    if isinstance(problem, ConicActiveSetBenchmark):
        scores = problem.rng.random(problem.n_constraints)
        return Guidance(active_constraint_scores=scores, strategy_name="random_guided")
    if isinstance(problem, EVChargingBenchmark):
        warm_start = np.zeros(problem.n_evs * problem.horizon, dtype=float)
        return Guidance(warm_start=warm_start, strategy_name="random_guided")
    if isinstance(problem, CapacityExpansionToyBenchmark):
        scores = problem.rng.random(problem.n_assets)
        return Guidance(active_constraint_scores=scores, strategy_name="random_guided")
    return Guidance(strategy_name="random_guided")


def run_experiment(
    *,
    benchmark_name: str,
    method: str,
    n_instances: int,
    budget: int,
    seed: int,
) -> list[dict[str, float | bool | str | int]]:
    """Run a benchmark method over independent seeded contexts."""

    if method not in METHODS:
        raise KeyError(f"method must be one of {METHODS}")
    rows: list[dict[str, float | bool | str | int]] = []
    for instance in range(int(n_instances)):
        problem = make_benchmark(benchmark_name, seed=seed + instance)
        theta = problem.sample_context()
        oracle = problem.expert_solve(theta)
        if method == "oracle":
            result = oracle
            result.guidance = replace(result.guidance, strategy_name="oracle")
        else:
            guidance = guidance_for_method(problem, theta, method, oracle)
            result = problem.budgeted_solve(theta, guidance, budget=budget)
        metrics = metrics_from_solution(problem, theta, result, oracle=oracle)
        metrics.update(
            {
                "benchmark": problem.name,
                "method": method,
                "instance": instance,
                "budget": int(budget),
                "seed": int(seed + instance),
            }
        )
        rows.append(metrics)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark",
        "method",
        "instance",
        "budget",
        "seed",
        "strategy_name",
        "objective",
        "gap_to_oracle",
        "max_violation",
        "mean_violation",
        "runtime",
        "accepted",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_figures(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    instances = [int(row["instance"]) for row in rows]
    objectives = [float(row["objective"]) for row in rows]
    gaps = [float(row["gap_to_oracle"]) for row in rows]
    accepted = [1.0 if bool(row["accepted"]) else 0.0 for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    axes[0].plot(instances, objectives, marker="o", linewidth=1.2)
    axes[0].set_title("Objective")
    axes[0].set_xlabel("Instance")
    axes[0].grid(alpha=0.25)

    axes[1].plot(instances, gaps, marker="o", color="#b44e52", linewidth=1.2)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Gap to Oracle")
    axes[1].set_xlabel("Instance")
    axes[1].grid(alpha=0.25)

    axes[2].bar(["accepted", "rejected"], [sum(accepted), len(accepted) - sum(accepted)], color=["#3d7f5f", "#c28b3c"])
    axes[2].set_title("Audit Outcomes")
    axes[2].set_ylim(0, max(1, len(accepted)))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    materialized = list(rows)
    if not materialized:
        return {"accepted_rate": float("nan"), "mean_gap": float("nan"), "mean_runtime": float("nan")}
    accepted = np.asarray([bool(row["accepted"]) for row in materialized], dtype=float)
    gaps = np.asarray([float(row["gap_to_oracle"]) for row in materialized], dtype=float)
    runtimes = np.asarray([float(row["runtime"]) for row in materialized], dtype=float)
    return {
        "accepted_rate": float(np.mean(accepted)),
        "mean_gap": float(np.nanmean(gaps)),
        "mean_runtime": float(np.mean(runtimes)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), default="rastrigin_warm_start")
    parser.add_argument("--method", choices=METHODS, default="expert_guided")
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = run_experiment(
        benchmark_name=args.benchmark,
        method=args.method,
        n_instances=args.instances,
        budget=args.budget,
        seed=args.seed,
    )
    stem = f"{args.benchmark}_{args.method}_seed{args.seed}"
    csv_path = args.output_dir / f"{stem}.csv"
    png_path = args.output_dir / f"{stem}.png"
    write_csv(rows, csv_path)
    write_figures(rows, png_path)
    summary = summarize(rows)
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    print(
        "accepted_rate={accepted_rate:.3f} mean_gap={mean_gap:.6g} mean_runtime={mean_runtime:.4f}s".format(
            **summary
        )
    )


if __name__ == "__main__":
    main()
