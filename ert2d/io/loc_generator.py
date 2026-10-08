"""Generate straight-line ``.loc`` files for a batch of profiles from a
single origin, azimuth and line spacing - for surveys laid out by tape/
compass in the field (no real survey coordinates), rather than measured with
a GPS/total station.

Profile 0 (the first item, in whatever order it is given) starts exactly at
``origin`` and runs ``azimuth_deg`` (compass bearing, 0°=North/+Y,
90°=East/+X, clockwise) for its own electrode count/spacing (read from its
``Survey``). Each later profile is offset ``line_spacing`` further along the
perpendicular to that direction, on ``side`` ("right"/"left" of the travel
direction) - the same construction used for the EGRIKOY survey earlier in
this project, generalised to an arbitrary azimuth and any number of lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _direction_vectors(azimuth_deg: float, side: str):
    """``(along, perp)`` unit vectors: ``along`` is the compass bearing
    ``azimuth_deg``; ``perp`` points to ``side`` ("right"/"left") of it -
    e.g. facing due North (0°), "right" points due East."""
    az = np.radians(azimuth_deg)
    along = np.array([np.sin(az), np.cos(az)])      # (x, y): N=(0,1), E=(1,0)
    right = np.array([along[1], -along[0]])         # rotate -90° (clockwise)
    perp = right if side == "right" else -right
    return along, perp


def generate_positions(n_electrodes: int, spacing: float, *, origin,
                       azimuth_deg: float, line_index: int = 0,
                       line_spacing: float = 0.0, side: str = "right") -> np.ndarray:
    """Electrode positions ``(n_electrodes, 2)`` for one profile that is
    ``line_index`` lines away from ``origin`` (``line_index * line_spacing``
    perpendicular to the profile direction, on ``side``)."""
    along, perp = _direction_vectors(azimuth_deg, side)
    start = np.asarray(origin, float) + line_index * line_spacing * perp
    k = np.arange(n_electrodes, dtype=float)
    return start[None, :] + k[:, None] * spacing * along[None, :]


def write_loc(path, positions: np.ndarray) -> None:
    """Write ``positions`` as ``index<TAB>X<TAB>Y`` rows, one per electrode -
    the per-electrode ``.loc`` format (see ``ert2d.io.loc``'s module docstring
    for the minimal 2-line alternative it also accepts)."""
    lines = [f"{i + 1}\t{x:.4f}\t{y:.4f}" for i, (x, y) in enumerate(positions)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class LocGenSpec:
    origin_x: float = 0.0
    origin_y: float = 0.0
    azimuth_deg: float = 90.0            # compass bearing, 0=N, 90=E, clockwise
    line_spacing: float = 0.0            # perpendicular distance between lines (m)
    side: str = "right"                  # "right" | "left" of the travel direction


def generate_loc_files(items, spec: LocGenSpec) -> list[Path]:
    """Write one ``.loc`` next to each item's ``.dat`` (same stem), in the
    order given - ``items[0]`` is the reference profile, starting exactly at
    ``(spec.origin_x, spec.origin_y)``. Items with no loaded ``Survey`` are
    skipped (their line_index in the stack is still counted, so removing an
    unloaded file from the selection doesn't shift the rest). Returns the
    paths written."""
    from ert2d.io.loc import loc_path_for

    written = []
    for i, it in enumerate(items):
        if it.survey is None:
            continue
        pos = generate_positions(
            it.survey.n_electrodes, it.survey.spacing,
            origin=(spec.origin_x, spec.origin_y), azimuth_deg=spec.azimuth_deg,
            line_index=i, line_spacing=spec.line_spacing, side=spec.side)
        out = loc_path_for(it.path)
        write_loc(out, pos)
        written.append(out)
    return written
