"""Stimulus utilities for the Brian2 implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from brian2 import amp, meter, ms

from .neuron_model import BrianL5Neuron

__all__ = ["CurrentStep", "apply_current_steps"]


@dataclass
class CurrentStep:
    amplitude_nA: float
    start_ms: float
    duration_ms: float
    target_compartment: int = 0

    @property
    def amplitude(self):
        return self.amplitude_nA * 1e-9 * amp

    @property
    def start(self):
        return self.start_ms * ms

    @property
    def stop(self):
        return (self.start_ms + self.duration_ms) * ms


def apply_current_steps(neuron: BrianL5Neuron, steps: Iterable[CurrentStep]) -> None:
    I = neuron.neuron.I_inj
    I[:] = 0 * amp / meter**2
    area = neuron.neuron.area
    prepared_steps = []
    for step in steps:
        idx = step.target_compartment
        compartment_area = area[idx]
        density_value = float((step.amplitude / compartment_area) / (amp / meter**2))
        prepared_steps.append((step.start_ms, step.duration_ms, density_value))
    neuron.neuron.I_inj = 0 * amp / meter**2
    neuron.neuron._prepared_steps = prepared_steps
