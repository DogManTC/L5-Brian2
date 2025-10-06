"""Factory functions that create the Brian2 neuron."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from brian2 import SpatialNeuron, amp, farad, mM, meter, ms, mV, ohm, siemens, um

from .channels import build_membrane_equations
from .morphology import MorphologyData, load_morphology
from .parameters import SimulationParameters, load_parameter_file

__all__ = ["BrianL5Neuron", "build_neuron"]


def _cm_to_SI(cm_uF: float) -> float:
    return cm_uF * 0.01 * farad / meter**2


def _ra_to_SI(ra_ohm_cm: float) -> float:
    return ra_ohm_cm * 0.01 * ohm * meter


def _conductance_to_SI(g_s_per_cm2: float) -> float:
    return g_s_per_cm2 * 1e4 * siemens / meter**2


@dataclass
class BrianL5Neuron:
    neuron: SpatialNeuron
    morphology: MorphologyData
    parameters: SimulationParameters

    def reset_state(self) -> None:
        self.neuron.v = -80 * mV
        if hasattr(self.neuron, "cai"):
            self.neuron.cai = 0.0001 * mM
        if hasattr(self.neuron, "I_inj"):
            self.neuron.I_inj = 0 * amp / meter**2


def _section_distances(morph: MorphologyData) -> Dict[int, float]:
    distances: Dict[int, float] = {0: 0.0}
    lengths = {idx: geom.length_um for idx, geom in morph.sections.items()}
    parents = {idx: geom.parent for idx, geom in morph.sections.items()}
    remaining = set(morph.sections.keys()) - {0}
    while remaining:
        for idx in list(remaining):
            parent = parents.get(idx)
            if parent is None:
                distances[idx] = 0.0
                remaining.remove(idx)
            elif parent in distances:
                distances[idx] = distances[parent] + lengths.get(idx, 0.0)
                remaining.remove(idx)
    return distances


SECTION_TYPES = {"Soma", "Apical", "Basilar", "Axonal"}


def _load_parameters(param_path: Path) -> SimulationParameters:
    return load_parameter_file(param_path)


def build_neuron(
    hoc_path: str | Path,
    param_path: str | Path,
    morphology: MorphologyData | None = None,
) -> BrianL5Neuron:
    hoc_path = Path(hoc_path)
    param_path = Path(param_path)
    params = _load_parameters(param_path)
    morph = morphology or load_morphology(hoc_path)
    eqs = build_membrane_equations(params)

    morpho = morph.morphology
    Ri = _ra_to_SI(params.ra)
    Cm = _cm_to_SI(params.cm)

    neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=Cm, Ri=Ri, method="exponential_euler")

    _apply_parameters(neuron, morph, params)

    model = BrianL5Neuron(neuron=neuron, morphology=morph, parameters=params)
    model.reset_state()
    return model


def _apply_parameters(neuron: SpatialNeuron, morph: MorphologyData, params: SimulationParameters) -> None:
    decay_um = params.decay

    # Passive properties
    neuron.g_pas = _conductance_to_SI(params.g_pas)
    neuron.e_pas = -params.e_pas * mV
    neuron.ena = 55 * mV
    neuron.ek = -90 * mV
    neuron.eca = 140 * mV

    # Channel parameters
    g = params.g_values.values

    neuron.qt = 2.3 ** ((34.0 - 21.0) / 10.0)
    neuron.gImbar_Im = _conductance_to_SI(g.get("PARAM_gImbar", 0.0))
    neuron.gSK_bar = _conductance_to_SI(g.get("PARAM_gSK_E2bar", 0.0))
    neuron.tau_SK = 1.0
    neuron.gSKv3_bar = _conductance_to_SI(g.get("PARAM_gSKv3_1bar", 0.0))
    neuron.shift_SKv3 = g.get("shift_SKv3_1", 0.0)
    neuron.gKPst_bar = _conductance_to_SI(g.get("PARAM_gK_Pstbar", 0.0))
    neuron.shift_KPst = g.get("shift_K_Pst", 0.0)
    neuron.gKTst_bar = _conductance_to_SI(g.get("PARAM_gK_Tstbar", 0.0))
    neuron.gCaLVA_bar = _conductance_to_SI(g.get("PARAM_gCa_LVAstbar", 0.0))
    neuron.gCaHVA_bar = _conductance_to_SI(g.get("PARAM_gCa_HVAbar", 0.0))
    neuron.gNaTa_bar = _conductance_to_SI(g.get("PARAM_NaTa_t_soma", 0.0))
    neuron.gNap_bar = _conductance_to_SI(g.get("PARAM_Nap_Et2", 0.0))
    neuron.shift_NaTa = g.get("shift_NaTa_t", 0.0)
    neuron.shift_Nap = g.get("shift_Nap_Et2", 0.0)
    neuron.cai_gamma = 0.000501
    neuron.depth_cadyn = 0.1 * um
    neuron.decay_cadyn = 460 * ms
    neuron.minCai = 0.0001 * mM

    neuron.vh_Ih = g.get("vh_Ih", -90.0)
    neuron.k_Ih = g.get("k_Ih", 8.0)
    neuron.a_Ih = g.get("a_Ih", 23.0)
    neuron.b_Ih = g.get("b_Ih", 0.2) / mV
    neuron.c_Ih = g.get("c_Ih", 1.0)
    neuron.d_Ih = g.get("d_Ih", 0.08) / mV
    neuron.e_Ih = g.get("e_Ih", 0.0)
    neuron.ehcn_Ih = g.get("ehcn_Ih", -49.0)

    # Section specific adjustments
    neuron.gIhbar_Ih = _conductance_to_SI(g.get("PARAM_gIhbar", 0.0))
    neuron.gImbar_Im = _conductance_to_SI(g.get("PARAM_gImbar", 0.0))
    neuron.gNaTa_bar = _conductance_to_SI(g.get("PARAM_NaTa_t_soma", 0.0))
    neuron.gCaLVA_bar = _conductance_to_SI(g.get("PARAM_gCa_LVAstbar", 0.0))
    neuron.gCaHVA_bar = _conductance_to_SI(g.get("PARAM_gCa_HVAbar", 0.0))

