import numpy as np

from open_pc_lgo.audit import full_constraint_audit
from open_pc_lgo.benchmarks import EVChargingBenchmark


def test_ev_expert_satisfies_departure_and_grid_constraints():
    problem = EVChargingBenchmark(seed=21)
    theta = problem.sample_context()

    oracle = problem.expert_solve(theta)
    audit = full_constraint_audit(problem, oracle.x, theta, tolerance=1e-7)

    assert oracle.x.shape == (problem.n_evs * problem.horizon,)
    assert audit.accepted
    assert np.max(np.maximum(oracle.violations, 0.0)) <= 1e-7


def test_ev_budgeted_solver_can_use_expert_warm_start():
    problem = EVChargingBenchmark(seed=22)
    theta = problem.sample_context()
    oracle = problem.expert_solve(theta)

    result = problem.budgeted_solve(theta, oracle.guidance, budget=8)
    audit = full_constraint_audit(problem, result.x, theta, tolerance=1e-7)

    assert audit.accepted
    assert result.objective <= oracle.objective + 1e-8
