import numpy as np

from open_pc_lgo.audit import full_constraint_audit, metrics_from_solution, projection_clean_acceptance
from open_pc_lgo.benchmarks import RastriginWarmStartBenchmark
from open_pc_lgo.core import Guidance, SolverResult


def test_full_constraint_audit_reports_positive_violation_metrics():
    problem = RastriginWarmStartBenchmark(seed=41)
    theta = problem.sample_context()
    x = np.full(problem.dimension, 10.0)

    audit = full_constraint_audit(problem, x, theta)

    assert not audit.accepted
    assert audit.max_violation > 0.0
    assert audit.mean_violation > 0.0


def test_projection_clean_acceptance_rejects_hidden_correction():
    problem = RastriginWarmStartBenchmark(seed=42)
    theta = problem.sample_context()
    raw_x = np.full(problem.dimension, 10.0)
    corrected_x = np.zeros(problem.dimension)

    audit = projection_clean_acceptance(problem, corrected_x, theta, raw_x=raw_x)

    assert not audit.accepted
    assert audit.correction_norm > 0.0
    assert audit.info["rejection_reason"] == "hidden_projection_correction"


def test_metrics_from_solution_uses_oracle_gap_and_strategy_name():
    problem = RastriginWarmStartBenchmark(seed=43)
    theta = problem.sample_context()
    oracle = problem.expert_solve(theta)
    result = SolverResult(
        x=oracle.x.copy(),
        objective=oracle.objective,
        violations=oracle.violations.copy(),
        runtime=0.0,
        guidance=Guidance(strategy_name="unit_test"),
    )

    metrics = metrics_from_solution(problem, theta, result, oracle=oracle)

    assert metrics["accepted"]
    assert metrics["strategy_name"] == "unit_test"
    assert abs(float(metrics["gap_to_oracle"])) < 1e-9
