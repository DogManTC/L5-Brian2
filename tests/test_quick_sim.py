"""Quick smoke test for Brian2 layer 5 neuron implementation."""
from __future__ import annotations

from brian2 import Cylinder, um

from brian2_impl.morphology import MorphologyData, SectionGeometry
from brian2_impl.neuron_model import build_neuron
from brian2_impl.protocols import run_current_clamp


def build_single_compartment() -> MorphologyData:
    soma = Cylinder(length=30 * um, diameter=30 * um, n=1)
    geometry = SectionGeometry(
        index=0,
        length_um=30.0,
        diameter_um=30.0,
        section_type="Soma",
        parent=None,
    )
    return MorphologyData(morphology=soma, sections={0: geometry})


def test_quick_simulation_runs():
    neuron = build_neuron(
        hoc_path="ModelSetup.hoc",
        param_path="L5_params_exponential.hoc",
        morphology=build_single_compartment(),
    )
    result = run_current_clamp(neuron, steps=[], duration_ms=0.1)
    assert len(result.time_ms) > 0
    assert len(result.time_ms) == len(result.voltage_mV) == len(result.current_nA)
