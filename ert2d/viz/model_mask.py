"""Blank the lower-left / lower-right corners of a model section.

The pseudo-section's data footprint is a trapezoid (triangle for a dipole-dipole
style layout): its flanks run in from the end electrodes as depth grows, so
the bottom corners of the rectangular model space are never crossed by any
data. The inversion still assigns those cells a value (via smoothing), and the
current flow lines make the *sensitive* zone a bit wider than the data
footprint - so the corners are cut along a flank that is the data flank pulled
outward by ``widen`` (a fraction, 0..1):

    inset(d) = (1 - widen) * inset_data(d)       # d = depth, inset from the end

``widen = 0`` follows the data flank exactly, ``widen = 1`` cuts nothing.
There is no universal literature value (it depends on array, noise and the
regularisation); typical practice is to show the pseudo-section trapezoid
or slightly more, so the default is a modest 0.25.

Two boundary shapes:

* ``"triangle"`` - straight flank (matches the shape of the data footprint);
* ``"ellipse"``  - a quarter-ellipse that leaves the surface vertically and
  levels out at the bottom (a smoother, more generous cut).

The result is a polygon in (x, z) data coordinates (z <= 0 downward) used as a
clip path, so it works identically for Voronoi, triangle and rectangular cells.
"""

from __future__ import annotations

import numpy as np

SHAPES = ("triangle", "ellipse")


def data_flank_slopes(x_mid, pseudo_depth) -> tuple[float, float, float, float]:
    """Fit the left/right edge of the data footprint against depth.

    Returns ``(a_left, s_left, a_right, s_right)`` such that the footprint at
    depth ``d`` (positive) spans ``a_left + s_left*d`` to ``a_right - s_right*d``.
    """
    x = np.asarray(x_mid, float)
    d = np.abs(np.asarray(pseudo_depth, float))
    ok = np.isfinite(x) & np.isfinite(d)
    x, d = x[ok], d[ok]
    levels = np.unique(np.round(d, 6))
    if levels.size < 2:                          # no depth structure: assume 45 deg
        return float(x.min()), 1.0, float(x.max()), 1.0
    lo = np.array([x[np.round(d, 6) == L].min() for L in levels])
    hi = np.array([x[np.round(d, 6) == L].max() for L in levels])
    s_l, a_l = np.polyfit(levels, lo, 1)
    s_r, a_r = np.polyfit(levels, hi, 1)
    s_r = -s_r
    return float(a_l), max(float(s_l), 0.0), float(a_r), max(float(s_r), 0.0)


def model_footprint(x_mid, pseudo_depth, electrodes, z_max: float, *,
                    widen: float = 0.25, shape: str = "triangle",
                    n: int = 80) -> np.ndarray:
    """Polygon ``(m, 2)`` of the part of the model space to keep visible."""
    widen = float(np.clip(widen, 0.0, 1.0))
    ex = np.asarray(electrodes, float)
    x0, x1 = float(ex.min()), float(ex.max())
    D = abs(float(z_max))
    a_l, s_l, a_r, s_r = data_flank_slopes(x_mid, pseudo_depth)

    d = np.linspace(0.0, D, n)
    in_l = np.maximum(a_l + s_l * d - x0, 0.0)           # data inset from left end
    in_r = np.maximum(x1 - (a_r - s_r * d), 0.0)
    if shape == "ellipse":
        u = 1.0 - np.sqrt(np.clip(1.0 - (d / D) ** 2, 0.0, 1.0)) if D > 0 else d
        in_l = (1.0 - widen) * in_l[-1] * u
        in_r = (1.0 - widen) * in_r[-1] * u
    else:
        in_l = (1.0 - widen) * in_l
        in_r = (1.0 - widen) * in_r

    xl, xr = x0 + in_l, x1 - in_r
    cross = xl > xr                                       # flanks met: nothing left
    mid = 0.5 * (xl + xr)
    xl, xr = np.where(cross, mid, xl), np.where(cross, mid, xr)

    left = np.column_stack([xl, -d])
    right = np.column_stack([xr, -d])[::-1]
    return np.vstack([left, right])
