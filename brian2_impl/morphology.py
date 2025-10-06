"""Parse the NEURON morphology and produce a Brian2 Morphology object."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from brian2 import Cylinder, Morphology, um

__all__ = [
    "SectionGeometry",
    "MorphologyData",
    "load_morphology",
]

SECTION_PREFIX = "filament_100000042"


@dataclass
class SectionGeometry:
    """Simplified geometric description of a NEURON section."""

    index: int
    length_um: float
    diameter_um: float
    section_type: str
    parent: int | None

    @property
    def length(self) -> float:
        return self.length_um * um

    @property
    def diameter(self) -> float:
        return self.diameter_um * um


@dataclass
class MorphologyData:
    """Container that holds the Brian2 morphology and section metadata."""

    morphology: Morphology
    sections: Dict[int, SectionGeometry]


def load_morphology(path: str | Path) -> MorphologyData:
    """Parse ``ModelSetup.hoc`` and build a :class:`~brian2.Morphology` object."""

    path = Path(path)
    hoc_text = path.read_text()
    section_points = _parse_section_points(hoc_text)
    section_types = _parse_section_types(hoc_text)
    parents = _parse_connectivity(hoc_text)

    geometries: Dict[int, SectionGeometry] = {}
    for index, points in section_points.items():
        length, diameter = _compute_length_and_diameter(points)
        geometries[index] = SectionGeometry(
            index=index,
            length_um=length,
            diameter_um=diameter,
            section_type=section_types.get(index, "Unknown"),
            parent=parents.get(index),
        )

    morphology = _build_brian2_morphology(geometries)
    return MorphologyData(morphology=morphology, sections=geometries)


Point = Tuple[float, float, float, float]


def _parse_section_points(hoc_text: str) -> Dict[int, List[Point]]:
    """Return the 3D point cloud for each section."""

    lines = iter(hoc_text.splitlines())
    sections: Dict[int, List[Point]] = {}
    current_index: int | None = None
    for line in lines:
        line = line.strip()
        if line.startswith(f"{SECTION_PREFIX}[") and "pt3dclear" in line:
            left = line.split("[")[1]
            index = int(left.split("]", 1)[0])
            current_index = index
            sections.setdefault(index, [])
            continue
        if current_index is not None and line.startswith("pt3dadd"):
            coord_text = line[line.find("(") + 1 : line.rfind(")")]
            x, y, z, diam = map(float, coord_text.split(","))
            sections[current_index].append((x, y, z, diam))
        if line == "}":
            current_index = None
    return sections


def _parse_section_types(hoc_text: str) -> Dict[int, str]:
    """Map section indices to their SectionList membership."""

    types: Dict[int, str] = {}
    for label in ("Soma", "Apical", "Basilar", "Axonal"):
        start = hoc_text.find(f"{label} = new SectionList()")
        if start == -1:
            continue
        block = hoc_text[start:]
        for line in block.splitlines()[1:]:
            stripped = line.strip()
            if not stripped:
                break
            if stripped.startswith("for i="):
                # pattern: for i=0, 6 filament_... label.append()
                range_segment = stripped.split("filament_", 1)[0]
                range_segment = range_segment.split("for i=")[1].strip()
                start_idx, end_idx = map(int, range_segment.split(","))
                for idx in range(start_idx, end_idx + 1):
                    types[idx] = label
            elif stripped.startswith(SECTION_PREFIX):
                index = int(stripped[stripped.find("[") + 1 : stripped.find("]")])
                types[index] = label
            else:
                break
    return types


def _parse_connectivity(hoc_text: str) -> Dict[int, int | None]:
    """Parse the ``connect`` statements and build the parent map."""

    parents: Dict[int, int | None] = {0: None}
    for line in hoc_text.splitlines():
        line = line.strip()
        if not line.startswith("connect"):
            continue
        # e.g. "connect filament[idx](0), filament[parent](1)"
        left = line.split(" ", 1)[1]
        first, second = left.split(",")
        child_idx = int(first[first.find("[") + 1 : first.find("]")])
        parent_idx = int(second[second.find("[") + 1 : second.find("]")])
        parents[child_idx] = parent_idx
    return parents


def _compute_length_and_diameter(points: Iterable[Point]) -> Tuple[float, float]:
    pts = list(points)
    if len(pts) < 2:
        return 1.0, pts[0][3] if pts else 1.0
    length = 0.0
    diam_sum = 0.0
    for (x1, y1, z1, d1), (x2, y2, z2, d2) in zip(pts[:-1], pts[1:]):
        seg_len = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        length += seg_len
        diam_sum += 0.5 * (d1 + d2)
    avg_diam = diam_sum / max(len(pts) - 1, 1)
    return length, avg_diam


def _build_brian2_morphology(sections: Dict[int, SectionGeometry]) -> Morphology:
    """Construct a Brian2 morphology tree from the parsed geometry."""

    # Determine root (section with no parent)
    root_index = min((idx for idx, geom in sections.items() if geom.parent is None), default=0)
    root_geom = sections[root_index]
    morph = Cylinder(length=root_geom.length, diameter=root_geom.diameter, n=1)

    children_map: Dict[int, List[int]] = {}
    for idx, geom in sections.items():
        if geom.parent is None:
            continue
        children_map.setdefault(geom.parent, []).append(idx)

    def attach(parent_cyl: Morphology, parent_index: int) -> None:
        for child_index in children_map.get(parent_index, []):
            child_geom = sections[child_index]
            attr_name = f"sec_{child_index}"
            child_cyl = Cylinder(length=child_geom.length, diameter=child_geom.diameter, n=1)
            setattr(parent_cyl, attr_name, child_cyl)
            attach(child_cyl, child_index)

    attach(morph, root_index)
    return morph
