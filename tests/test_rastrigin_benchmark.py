import numpy as np

from open_pc_lgo.audit import full_constraint_audit, metrics_from_solution
from open_pc_lgo.benchmarks import RastriginWarmStartBenchmark
from open_pc_lgo.core import Guidance


def test_rastrigin_expert_and_guided_budgeted_solve_are_auditable():
    problem = RastriginWarmStartBenchmark(seed=3)
    theta = problem.sample_context()

    oracle = problem.expert_solve(theta)
    oracle_audit = full_constraint_audit(problem, oracle.x, theta, tolerance=1e-8)

    assert oracle.x.shape == (problem.dimension,)
    assert oracle_audit.accepted
    assert oracle.guidance.warm_start is not None

    guided = problem.budgeted_solve(
        theta,
        Guidance(warm_start=oracle.guidance.warm_start, strategy_name="test_guided"),
        budget=20,
    )
    metrics = metrics_from_solution(problem, theta, guided, oracle=oracle)

    assert np.isfinite(guided.objective)
    assert metrics["accepted"]
    assert "gap_to_oracle" in metrics


def test_rastrigin_feature_vector_is_reproducible_shape():
    problem = RastriginWarmStartBenchmark(dimension=3, seed=4)
    theta = problem.sample_context()

    features = problem.feature_vector(theta)

    assert features.shape == (5,)
    assert np.all(np.isfinite(features))
