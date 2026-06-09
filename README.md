# OPEN-pc_lgo

Projection-Clean Learning-Guided Optimization benchmarks.

This package is not a universal optimizer. It is a small, solver-free benchmark suite for studying a specific protocol:

1. Learning proposes guidance.
2. Physics softens the search landscape with penalties.
3. A budgeted optimizer makes the final decision.
4. A projection-clean audit accepts or rejects the raw result.

The learning component proposes warm starts, candidate masks, active-constraint scores, or strategy choices. It does not get credit for a final solution unless the budgeted optimizer produces a candidate that passes the full audit. The optimizer remains responsible for final decisions, and the audit prevents hidden correction or post-hoc projection from being counted as learning success.

No Pascal connection, proprietary optimizer, or external solver is used.

## Package Layout

- `open_pc_lgo.core`: `BenchmarkProblem`, `Guidance`, and `SolverResult`.
- `open_pc_lgo.physics`: soft penalties for conic, reserve, and EV constraints.
- `open_pc_lgo.audit`: projection-clean acceptance, full audits, and metrics.
- `open_pc_lgo.benchmarks`: four benchmark problems.
- `open_pc_lgo.models`: PyTorch behavioral cloning prior and contextual bandit selector.
- `open_pc_lgo.experiments`: CSV and PNG experiment runners.

## Benchmarks

| Benchmark | Guidance type | Optimizer behavior | Audit behavior |
| --- | --- | --- | --- |
| `rastrigin_warm_start` | Warm start | Local random pattern search from proposed start | Box constraints |
| `conic_active_set` | Active SOC scores | Search with top-k predicted constraints | Checks all SOC constraints |
| `ev_charging` | Warm charging schedule | Greedy/local charge moves with physics penalty | SOC, departure, rate, and grid checks |
| `capacity_expansion_toy` | Candidate binary plans or asset scores | Local bit mutations | Reserve adequacy and binary feasibility |

## Install

From this folder:

```bash
python3 -m pip install -e ".[dev]"
```

## Run

Run one method on one benchmark:

```bash
open-pc-lgo-run --benchmark rastrigin_warm_start --method expert_guided --instances 20 --budget 80 --seed 7 --output-dir outputs
```

Compare methods:

```bash
open-pc-lgo-compare --benchmark conic_active_set --methods unguided,random_guided,expert_guided,oracle --instances 20 --budget 80 --seed 7 --output-dir outputs
```

Both commands produce CSV summaries and static PNG figures.

## Python API

```python
from open_pc_lgo.audit import metrics_from_solution
from open_pc_lgo.benchmarks import RastriginWarmStartBenchmark
from open_pc_lgo.core import Guidance

problem = RastriginWarmStartBenchmark(seed=7)
theta = problem.sample_context()
oracle = problem.expert_solve(theta)
guidance = Guidance(warm_start=oracle.guidance.warm_start, strategy_name="demo")
result = problem.budgeted_solve(theta, guidance, budget=50)
metrics = metrics_from_solution(problem, theta, result, oracle=oracle)
print(metrics)
```

## Tests

```bash
python3 -m pytest
```

The tests cover every benchmark and the projection-clean audit module.
