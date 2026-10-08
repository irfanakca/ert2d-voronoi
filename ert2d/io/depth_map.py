"""Combine several inverted, georeferenced 2D lines into a depth-slice map.

Each line contributes a regular x-z resistivity grid (:func:`ert2d.io.results.
to_regular_grid`), whose local along-line x is projected to real-world
(easting, northing) with its ``.loc`` :class:`~ert2d.io.loc.LineLocation`
(straight-line assumption, same as the Map View). Every line's depth column is
then resampled onto one shared depth axis, giving - at each shared depth - a
scatter of (easting, northing, rho) points from every line. Rendering is left
to the GUI, which contours these points the same way ``pseudosection`` does
(masked Delaunay triangulation - no fill outside the data's convex hull).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ert2d.io.results import to_regular_grid


@dataclass
class DepthMap:
    depths: np.ndarray                     # (K,) positive depth below surface (m)
    points: list                           # length K; each (E, N, rho) arrays
    bounds: tuple                          # (e0, e1, n0, n1) - fixed map extent
    line_traces: list = field(default_factory=list)   # [(name, (2,2) start/end)]

    @property
    def is_global(self) -> bool:
        """True when the ``.loc`` files hold real-world (projected) coordinates
        rather than small local offsets - then the axes are Easting/Northing."""
        return min(abs(self.bounds[0]), abs(self.bounds[2])) >= 1000.0


def _line_columns(item, depths: np.ndarray):
    """Resample one line's regular x-z grid onto the shared ``depths`` axis and
    project its along-line x to real-world (E, N). Returns ``(xy, RO)`` with
    ``xy`` shape ``(nx, 2)`` and ``RO`` shape ``(len(depths), nx)``."""
    xi, zi, RO = to_regular_grid(item.result, survey=item.survey)
    dep = -zi                              # zi: -z_max..0 ascending -> dep descending
    order = np.argsort(dep)
    dep_s, RO_s = dep[order], RO[order, :]  # ascending in depth

    out = np.full((depths.size, xi.size), np.nan)
    for i in range(xi.size):
        col = RO_s[:, i]
        ok = np.isfinite(col)
        if ok.sum() < 2:
            continue
        out[:, i] = np.interp(depths, dep_s[ok], col[ok], left=np.nan, right=np.nan)

    xy = item.location.positions_along(xi)
    return xy, out


def build_depth_map(items, *, n_levels: int = 20, depths=None,
                    pad_frac: float = 0.08) -> DepthMap:
    """Build a :class:`DepthMap` from already-inverted, georeferenced lines.

    ``items`` must all have both ``.location`` (a ``.loc`` file was found) and
    ``.result`` (a finished inversion) - callers should filter/validate before
    calling this. ``depths`` overrides the default evenly-spaced ``n_levels``
    axis with an explicit one (e.g. a fixed step, for an animation export).
    """
    missing_loc = [it.name for it in items if it.location is None]
    if missing_loc:
        raise ValueError(f"missing .loc for: {', '.join(missing_loc)}")
    missing_res = [it.name for it in items if it.result is None]
    if missing_res:
        raise ValueError(f"not inverted: {', '.join(missing_res)}")
    if len(items) < 2:
        raise ValueError("need at least two lines")

    if depths is None:
        z_max_common = max(float(it.survey.z_max) for it in items)
        depths = np.linspace(z_max_common / n_levels, z_max_common, n_levels)
    else:
        depths = np.asarray(depths, dtype=float)
        if depths.size < 2:
            raise ValueError("need at least two depth levels")

    per_item = [(it, *_line_columns(it, depths)) for it in items]

    all_xy = np.vstack([xy for _, xy, _ in per_item])
    e0, e1 = float(all_xy[:, 0].min()), float(all_xy[:, 0].max())
    n0, n1 = float(all_xy[:, 1].min()), float(all_xy[:, 1].max())
    pad_e = max((e1 - e0) * pad_frac, 1.0)
    pad_n = max((n1 - n0) * pad_frac, 1.0)
    bounds = (e0 - pad_e, e1 + pad_e, n0 - pad_n, n1 + pad_n)

    points = []
    for k in range(depths.size):
        es, ns, rs = [], [], []
        for _, xy, RO in per_item:
            ok = np.isfinite(RO[k])
            if ok.any():
                es.append(xy[ok, 0])
                ns.append(xy[ok, 1])
                rs.append(RO[k][ok])
        if es:
            points.append((np.concatenate(es), np.concatenate(ns), np.concatenate(rs)))
        else:
            points.append((np.array([]), np.array([]), np.array([])))

    line_traces = [(it.name, np.vstack([it.location.start, it.location.end]))
                   for it in items]
    return DepthMap(depths=depths, points=points, bounds=bounds,
                    line_traces=line_traces)
