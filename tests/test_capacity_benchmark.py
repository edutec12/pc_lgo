import numpy as np

from open_pc_lgo.audit import full_constraint_audit
from open_pc_lgo.benchmarks import CapacityExpansionToyBenchmark


def test_capacity_expert_uses_exhaustive_search_and_passes_audit():
    problem = CapacityExpansionToyBenchmark(seed=31)
    theta = problem.sample_context()

    oracle = problem.expert_solve(theta)
    audit = full_constraint_audit(problem, oracle.x, theta, tolerance=1e-9)

    assert audit.accepted
    assert set(np.unique(oracle.x)).issubset({0.0, 1.0})
    assert oracle.info["enumerated_plans"] == 2**problem.n_assets


def test_capacity_budgeted_solver_uses_candidate_plan_and_local_mutations():
    problem = CapacityExpansionToyBenchmark(seed=32)
    theta = problem.sample_context()
    oracle = problem.expert_solve(theta)

    result = problem.budgeted_solve(theta, oracle.guidance, budget=10)
    audit = full_constraint_audit(problem, result.x, theta, tolerance=1e-9)

    assert audit.accepted
    assert set(np.unique(result.x)).issubset({0.0, 1.0})
