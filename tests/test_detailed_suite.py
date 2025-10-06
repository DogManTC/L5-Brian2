"""Extended regression tests for the Brian2 L5 pyramidal neuron model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from brian2 import mM, mV

from brian2_impl.morphology import load_morphology
from brian2_impl.neuron_model import build_neuron
from brian2_impl.protocols import CurrentStep, run_current_clamp


pytestmark = pytest.mark.filterwarnings(
    "ignore:divide by zero encountered:RuntimeWarning",
    "ignore:overflow encountered:RuntimeWarning",
    "ignore:invalid value encountered:RuntimeWarning",
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def hoc_path() -> Path:
    return PROJECT_ROOT / "ModelSetup.hoc"


@pytest.fixture(scope="module")
def param_path() -> Path:
    return PROJECT_ROOT / "L5_params_exponential.hoc"


def test_morphology_contains_all_section_types(hoc_path: Path) -> None:
    morphology = load_morphology(hoc_path)
    section_types = {geometry.section_type for geometry in morphology.sections.values()}
    for expected in {"Soma", "Apical", "Basilar", "Axonal"}:
        assert expected in section_types


def test_resting_state_is_numerically_stable(hoc_path: Path, param_path: Path) -> None:
    neuron = build_neuron(hoc_path, param_path)
    result = run_current_clamp(neuron, steps=[], duration_ms=1.0)
    voltages = np.asarray(result.voltage_mV)
    assert np.all(np.isfinite(voltages))
    assert voltages[0] == pytest.approx(-80.0)
    assert voltages.min() > -150.0
    assert voltages.max() <= -60.0


def test_mild_current_step_produces_expected_current(hoc_path: Path, param_path: Path) -> None:
    neuron = build_neuron(hoc_path, param_path)
    step = CurrentStep(amplitude_nA=0.12, start_ms=0.2, duration_ms=0.8)
    result = run_current_clamp(neuron, steps=[step], duration_ms=2.0)
    current = np.asarray(result.current_nA)
    voltages = np.asarray(result.voltage_mV)
    assert current.max() == pytest.approx(step.amplitude_nA, abs=1e-6)
    assert current.min() >= 0.0
    assert np.all(np.isfinite(voltages))
    assert voltages.min() > -800.0


def test_current_clamp_is_reproducible(hoc_path: Path, param_path: Path) -> None:
    steps = [CurrentStep(amplitude_nA=0.12, start_ms=0.1, duration_ms=0.5)]
    neuron_a = build_neuron(hoc_path, param_path)
    neuron_b = build_neuron(hoc_path, param_path)
    result_a = run_current_clamp(neuron_a, steps=steps, duration_ms=1.0)
    result_b = run_current_clamp(neuron_b, steps=steps, duration_ms=1.0)
    assert result_a.time_ms == result_b.time_ms
    assert np.allclose(result_a.voltage_mV, result_b.voltage_mV)
    assert np.allclose(result_a.current_nA, result_b.current_nA)


def test_reset_state_restores_baseline(hoc_path: Path, param_path: Path) -> None:
    neuron = build_neuron(hoc_path, param_path)
    step = CurrentStep(amplitude_nA=0.12, start_ms=0.1, duration_ms=0.5)
    run_current_clamp(neuron, steps=[step], duration_ms=1.0)
    assert float(neuron.neuron.v[0] / mV) < -80.0
    neuron.reset_state()
    assert float(neuron.neuron.v[0] / mV) == pytest.approx(-80.0)
    if hasattr(neuron.neuron, "cai"):
        assert float(neuron.neuron.cai[0] / mM) == pytest.approx(0.0001)
