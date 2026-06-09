"""Benchmark problem registry."""

from __future__ import annotations

from .capacity import CapacityExpansionToyBenchmark
from .conic import ConicActiveSetBenchmark
from .ev_charging import EVChargingBenchmark
from .rastrigin import RastriginWarmStartBenchmark

BENCHMARKS = {
    RastriginWarmStartBenchmark.name: RastriginWarmStartBenchmark,
    ConicActiveSetBenchmark.name: ConicActiveSetBenchmark,
    EVChargingBenchmark.name: EVChargingBenchmark,
    CapacityExpansionToyBenchmark.name: CapacityExpansionToyBenchmark,
}


def make_benchmark(name: str, *, seed: int | None = None):
    normalized = name.strip().lower().replace("-", "_")
    aliases = {
        "rastrigin": RastriginWarmStartBenchmark.name,
        "conic": ConicActiveSetBenchmark.name,
        "ev": EVChargingBenchmark.name,
        "ev_charging": EVChargingBenchmark.name,
        "capacity": CapacityExpansionToyBenchmark.name,
    }
    key = aliases.get(normalized, normalized)
    if key not in BENCHMARKS:
        available = ", ".join(sorted(BENCHMARKS))
        raise KeyError(f"unknown benchmark {name!r}; available: {available}")
    return BENCHMARKS[key](seed=seed)


__all__ = [
    "BENCHMARKS",
    "CapacityExpansionToyBenchmark",
    "ConicActiveSetBenchmark",
    "EVChargingBenchmark",
    "RastriginWarmStartBenchmark",
    "make_benchmark",
]
