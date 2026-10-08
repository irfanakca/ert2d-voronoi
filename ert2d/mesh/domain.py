"""Inversion domain geometry.

Port of the ``ops == 1`` branch of ``meshgen.m``: a rectangular domain that
hugs the electrode line at the surface and extends ``wing`` electrode spacings
left/right and ``depth_factor * z_max`` downward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ert2d.io.dat_reader import Survey


@dataclass
class Domain:
    boundary: np.ndarray       # (K, 2) polygon vertices, open ring (no repeated last point)
    surface_x: np.ndarray      # x of the forced surface nodes (electrode line, densified)
    surface_z: np.ndarray
    electrodes: np.ndarray     # (n_elec, 2)
    z_max: float
    spacing: float
    wing: float                # absolute left/right extension

    @property
    def x_bounds(self) -> tuple[float, float]:
        return float(self.boundary[:, 0].min()), float(self.boundary[:, 0].max())

    @property
    def z_bounds(self) -> tuple[float, float]:
        return float(self.boundary[:, 1].min()), float(self.boundary[:, 1].max())


def build_domain(
    survey: Survey,
    *,
    surface_div: int = 2,
    wing: float = 5.0,
    depth_factor: float = 4.0,
) -> Domain:
    """Build the inversion domain polygon.

    Parameters mirror the hard-coded constants in ``meshgen.m``:
    ``surface_div=4`` -> surface nodes every ``spacing/4``;
    ``wing=3``        -> side extension ``3 * spacing``;
    ``depth_factor=4``-> bottom at ``-4 * z_max``.
    """
    ela = float(survey.spacing)
    xe = np.asarray(survey.electrode_x, float)
    ze = np.asarray(survey.electrode_z, float)
    ek = wing * ela
    zmax = float(survey.z_max)

    # densified surface sampling, electrodes guaranteed to be nodes
    step = ela / surface_div
    dense = np.arange(xe[0], xe[-1] + step * 0.5, step)
    surface_x = np.unique(np.concatenate([dense, xe]))
    if np.any(ze):
        surface_z = np.interp(surface_x, xe, ze)
    else:
        surface_z = np.zeros_like(surface_x)

    x_left = xe[0] - ek
    x_right = xe[-1] + ek
    z_bot = -depth_factor * zmax

    top = np.column_stack([surface_x, surface_z])
    boundary = np.vstack([
        [x_left, surface_z[0]],
        top,
        [x_right, surface_z[-1]],
        [x_right, z_bot],
        [x_left, z_bot],
    ])
    # drop a duplicated first surface node if it coincides with the left corner
    boundary = _dedupe_consecutive(boundary)

    return Domain(
        boundary=boundary,
        surface_x=surface_x,
        surface_z=surface_z,
        electrodes=np.column_stack([xe, ze]),
        z_max=zmax,
        spacing=ela,
        wing=ek,
    )


def _dedupe_consecutive(pts: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    keep = [pts[0]]
    for p in pts[1:]:
        if np.hypot(*(p - keep[-1])) > tol:
            keep.append(p)
    return np.asarray(keep)
