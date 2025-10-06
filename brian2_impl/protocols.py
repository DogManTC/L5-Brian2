"""Simulation protocols to replicate the NEURON experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from brian2 import Network, StateMonitor, defaultclock, ms, mV, amp, meter
from brian2.units.constants import faraday_constant

from .neuron_model import BrianL5Neuron
from .stimuli import CurrentStep, apply_current_steps

__all__ = ["SimulationResult", "run_current_clamp"]


defaultclock.dt = 0.025 * ms


@dataclass
class SimulationResult:
    time_ms: list[float]
    voltage_mV: list[float]
    current_nA: list[float]


def run_current_clamp(
    neuron: BrianL5Neuron,
    steps: Iterable[CurrentStep],
    duration_ms: float,
) -> SimulationResult:
    apply_current_steps(neuron, steps)
    soma_idx = 0
    monitor = StateMonitor(neuron.neuron, ["v", "I_inj"], record=[soma_idx])
    net = Network(neuron.neuron, monitor)
    namespace = {"faraday_constant": faraday_constant}
    steps_info = sorted(getattr(neuron.neuron, "_prepared_steps", []), key=lambda s: s[0])
    current = 0.0
    for start, duration, density in steps_info:
        if start > current:
            net.run((start - current) * ms, namespace=namespace)
            current = start
        neuron.neuron.I_inj = density * amp / meter**2
        net.run(duration * ms, namespace=namespace)
        current += duration
    if current < duration_ms:
        neuron.neuron.I_inj = 0 * amp / meter**2
        net.run((duration_ms - current) * ms, namespace=namespace)
    area = neuron.neuron.area[soma_idx]
    return SimulationResult(
        time_ms=(monitor.t / ms).tolist(),
        voltage_mV=(monitor.v[soma_idx] / mV).tolist(),
        current_nA=((monitor.I_inj[soma_idx] * area) / amp * 1e9).tolist(),
    )
