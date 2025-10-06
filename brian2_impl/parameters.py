"""Utilities for loading the parameter sets defined in the NEURON HOC files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

__all__ = ["MechanismParameters", "SimulationParameters", "load_parameter_file"]


@dataclass
class MechanismParameters:
    """Conductance and kinetic parameters for ion channels."""

    values: Dict[str, float]

    def __getitem__(self, item: str) -> float:
        return self.values[item]

    def get(self, item: str, default: float | None = None) -> float | None:
        return self.values.get(item, default)


@dataclass
class SimulationParameters:
    """Global parameters for the layer 5 pyramidal cell model."""

    ra: float  # Ohm cm
    cm: float  # uF/cm^2
    g_pas: float  # S/cm^2
    e_pas: float  # mV
    decay: float  # um
    g_values: MechanismParameters

    @property
    def as_dict(self) -> Dict[str, float]:
        return {
            "Ra": self.ra,
            "cm": self.cm,
            "g_pas": self.g_pas,
            "e_pas": self.e_pas,
            "decay": self.decay,
            **self.g_values.values,
        }


def _clean_line(line: str) -> str:
    """Strip comments and whitespace from a hoc line."""

    line = line.strip()
    if "//" in line:
        line = line.split("//", 1)[0]
    if ":" in line:
        line = line.split(":", 1)[0]
    return line.strip()


def load_parameter_file(path: str | Path) -> SimulationParameters:
    """Parse the `L5_params_exponential.hoc` (or compatible) file."""

    path = Path(path)
    contents: Dict[str, float] = {}
    for raw_line in path.read_text().splitlines():
        line = _clean_line(raw_line)
        if not line or "=" not in line:
            continue
        key, value = map(str.strip, line.split("=", 1))
        try:
            contents[key] = float(value)
        except ValueError as exc:  # pragma: no cover - helps diagnose parsing issues
            raise ValueError(f"Could not parse parameter line: {raw_line!r}") from exc

    mechanism_keys = {
        k: v
        for k, v in contents.items()
        if k.startswith("PARAM_g")
        or k.startswith("PARAM_Na")
        or k.startswith("shift_")
        or k in {"a_Ih", "b_Ih", "c_Ih", "d_Ih", "e_Ih", "k_Ih", "vh_Ih", "ehcn_Ih"}
    }
    global_params = {
        "ra": contents.get("PARAM_Ra", 150.0),
        "cm": contents.get("PARAM_cm", 1.0),
        "g_pas": contents.get("PARAM_g_pas", 1e-5),
        "e_pas": contents.get("PARAM_e_pas", -70.0),
        "decay": contents.get("PARAM_decay", 1000.0),
    }
    return SimulationParameters(
        ra=global_params["ra"],
        cm=global_params["cm"],
        g_pas=global_params["g_pas"],
        e_pas=global_params["e_pas"],
        decay=global_params["decay"],
        g_values=MechanismParameters(mechanism_keys),
    )
