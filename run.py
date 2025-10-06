"""Entry point for running the Brian2 translation of the layer 5 neuron."""
from __future__ import annotations

from pathlib import Path

from brian2_impl.neuron_model import build_neuron
from brian2_impl.protocols import CurrentStep, run_current_clamp

ROOT = Path(__file__).resolve().parent


def main() -> None:
    neuron = build_neuron(
        hoc_path=ROOT / "ModelSetup.hoc",
        param_path=ROOT / "L5_params_exponential.hoc",
    )
    steps = [CurrentStep(amplitude_nA=0.3, start_ms=50.0, duration_ms=100.0, target_compartment=0)]
    result = run_current_clamp(neuron, steps, duration_ms=200.0)
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    trace_path = output_dir / "current_clamp_trace.csv"
    with trace_path.open("w", encoding="utf-8") as fh:
        fh.write("time_ms,voltage_mV,current_nA\n")
        for t, v, i in zip(result.time_ms, result.voltage_mV, result.current_nA):
            fh.write(f"{t:.6f},{v:.6f},{i:.6f}\n")
    print(f"Simulation complete. Results written to {trace_path}")


if __name__ == "__main__":
    main()
