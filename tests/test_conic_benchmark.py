import numpy as np

from open_pc_lgo.audit import full_constraint_audit
from open_pc_lgo.benchmarks import ConicActiveSetBenchmark
from open_pc_lgo.core import Guidance


def test_conic_expert_identifies_active_scores_and_passes_audit():
    problem = ConicActiveSetBenchmark(seed=11)
    theta = problem.sample_context()

    oracle = problem.expert_solve(theta)
    audit = full_constraint_audit(problem, oracle.x, theta, tolerance=1e-5)

    assert oracle.guidance.active_constraint_scores is not None
    assert oracle.guidance.active_constraint_scores.shape == (problem.n_constraints,)
    assert oracle.guidance.candidate_mask is not None
    assert int(np.sum(oracle.guidance.candidate_mask)) == problem.top_k
    assert audit.accepted


def test_conic_budgeted_solver_uses_top_k_guidance_and_audits_all_constraints():
    problem = ConicActiveSetBenchmark(seed=12)
    theta = problem.sample_context()
    scores = np.linspace(0.0, 1.0, problem.n_constraints)
    guidance = Guidance(active_constraint_scores=scores, strategy_name="test_scores")

    result = problem.budgeted_solve(theta, guidance, budget=15)
    audit = full_constraint_audit(problem, result.x, theta)

    assert len(result.info["selected_constraints"]) == problem.top_k
    assert audit.violations.shape == (problem.n_constraints,)
    assert np.isfinite(audit.objective)
