"""Stack a :class:`~ert2d.io.depth_map.DepthMap`'s per-depth scatter points
into one regular 3D (x, y, z) resistivity volume - the data behind the GUI's
3D View tab (isosurfaces + orthogonal slices)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Volume:
    x: np.ndarray           # (nx,) easting or local x (m), ascending
    y: np.ndarray           # (ny,) northing or local y (m), ascending
    z: np.ndarray           # (nz,) depth below surface (m), ascending, >= 0
    rho: np.ndarray         # (nz, ny, nx) linear Ω·m, NaN outside data footprint
    bounds: tuple           # (e0, e1, n0, n1) - same as DepthMap.bounds


def build_volume(dm, *, grid_step: float = 1.0, method: str = "linear",
                 clip_to_lines: bool = True) -> Volume:
    """Grid every depth level of ``dm`` onto one shared (x, y) raster (linear
    Ω·m, no log transform - isosurface thresholds are then plain physical
    resistivity values) and stack them into a 3D array.

    Each level is gridded independently via :func:`ert2d.viz.raster.grid_points`
    (same "no extrapolation past the convex hull" behaviour as the 2D Depth
    Map tab); levels with fewer than 3 data points come back all-NaN.

    ``clip_to_lines`` (default on) additionally NaNs out every (x, y) cell
    outside the profiles' footprint (:func:`ert2d.viz.depth_map.line_footprint_path`)
    at *every* depth - the same clip the 2D Depth Map tab applies when
    drawing a slice - so the space between separate, far-apart profile lines
    stays empty instead of being smoothly bridged by the raster interpolation
    (which otherwise looks badly over-interpolated: a rectangular grid has no
    notion of "no line ran through here").
    """
    from ert2d.viz.raster import grid_points

    bounds = dm.bounds
    Xg = Yg = None
    layers = []
    for k in range(len(dm.depths)):
        E, N, rho = dm.points[k]
        if len(rho) >= 3:
            Xg, Yg, Zg = grid_points(E, N, rho, bounds, grid_step, method=method)
        elif Xg is not None:
            Zg = np.full_like(Xg, np.nan)
        else:
            Zg = None                          # resolved once a non-empty level is seen
        layers.append(Zg)

    if Xg is None:
        raise ValueError("no depth level has at least 3 data points")
    layers = [Zg if Zg is not None else np.full_like(Xg, np.nan) for Zg in layers]

    if clip_to_lines:
        from ert2d.viz.depth_map import line_footprint_path

        fp = line_footprint_path(dm.line_traces)
        if fp is not None:
            inside = fp.contains_points(np.column_stack([Xg.ravel(), Yg.ravel()]))
            inside = inside.reshape(Xg.shape)
            layers = [np.where(inside, Zg, np.nan) for Zg in layers]

    rho3d = np.stack(layers, axis=0)           # (nz, ny, nx)
    return Volume(x=Xg[0, :], y=Yg[:, 0], z=np.asarray(dm.depths, float),
                 rho=rho3d, bounds=bounds)
