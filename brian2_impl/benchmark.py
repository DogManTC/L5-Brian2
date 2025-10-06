"""Utilities for benchmarking Brian2 simulation performance."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

from brian2 import prefs

from .neuron_model import build_neuron
from .protocols import run_current_clamp
from .stimuli import CurrentStep

__all__ = ["BenchmarkResult", "benchmark_simulation_speed"]


ROOT = Path(__file__).resolve().parent.parent


@dataclass
class BenchmarkResult:
    """Summary of a benchmarking run."""

    neuron_count: int
    simulated_seconds: float
    wall_seconds: float

    @property
    def wall_seconds_per_sim_second(self) -> float:
        """Return wall-clock seconds required to simulate one simulated second."""

        if self.simulated_seconds == 0:
            return float("nan")
        return self.wall_seconds / self.simulated_seconds


def benchmark_simulation_speed(
    neuron_counts: Iterable[int],
    duration_ms: float,
    steps: Sequence[CurrentStep] | None = None,
    hoc_path: str | Path | None = None,
    param_path: str | Path | None = None,
) -> list[BenchmarkResult]:
    """Benchmark wall-clock performance for several neuron counts.

    Parameters
    ----------
    neuron_counts:
        Iterable of population sizes to benchmark. Each entry triggers that
        many sequential simulations, which approximates the cost of simulating
        the same number of independent neurons.
    duration_ms:
        Duration of the simulation in milliseconds.
    steps:
        Optional collection of :class:`CurrentStep` stimuli to apply.
    hoc_path, param_path:
        Optional overrides for the morphology and parameter file paths.
    """

    steps = list(steps or [])
    hoc_path = Path(hoc_path) if hoc_path is not None else ROOT / "ModelSetup.hoc"
    param_path = (
        Path(param_path) if param_path is not None else ROOT / "L5_params_exponential.hoc"
    )

    # Use the NumPy backend to avoid expensive code generation passes during benchmarks.
    previous_target = prefs.codegen.target
    prefs.codegen.target = "numpy"
    try:
        results: list[BenchmarkResult] = []
        for count in neuron_counts:
            start = perf_counter()
            for _ in range(count):
                neuron = build_neuron(hoc_path=hoc_path, param_path=param_path)
                run_current_clamp(neuron, steps, duration_ms)
            wall = perf_counter() - start
            results.append(
                BenchmarkResult(
                    neuron_count=count,
                    simulated_seconds=(duration_ms / 1000.0) * count,
                    wall_seconds=wall,
                )
            )
        return results
    finally:
        prefs.codegen.target = previous_target
