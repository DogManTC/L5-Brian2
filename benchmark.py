"""Benchmark script to measure Brian2 simulation speed for multiple neuron counts."""
from __future__ import annotations

from pathlib import Path

from brian2_impl.benchmark import benchmark_simulation_speed

ROOT = Path(__file__).resolve().parent


def main() -> None:
    counts = [1, 2, 4]
    duration_ms = 10.0
    steps: list = []

    results = benchmark_simulation_speed(
        neuron_counts=counts,
        duration_ms=duration_ms,
        steps=steps,
        hoc_path=ROOT / "ModelSetup.hoc",
        param_path=ROOT / "L5_params_exponential.hoc",
    )

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "benchmark_speed.csv"
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("neuron_count,simulated_seconds,wall_seconds,wall_per_sim_second\n")
        for entry in results:
            fh.write(
                f"{entry.neuron_count},{entry.simulated_seconds:.6f},{entry.wall_seconds:.6f},{entry.wall_seconds_per_sim_second:.6f}\n"
            )

    print("Benchmark results:")
    for entry in results:
        ratio = entry.wall_seconds_per_sim_second
        print(
            f"  {entry.neuron_count:>3} neurons: {entry.wall_seconds:.3f}s wall for "
            f"{entry.simulated_seconds:.3f}s simulated (wall/sim = {ratio:.3f})"
        )
    print(f"Detailed results saved to {output_path}")


if __name__ == "__main__":
    main()
