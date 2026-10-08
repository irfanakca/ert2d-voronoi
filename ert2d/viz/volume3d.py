"""Build a PyVista :class:`~pyvista.ImageData` volume from an
:class:`ert2d.io.volume.Volume`, and helpers for the isosurface/orthogonal
slice meshes drawn on it by the GUI's 3D View tab.

VTK's Z axis points *up*; ``Volume.z`` is depth *below* the surface
(ascending, >= 0), so it is negated and reversed here (world Z = -depth) so
the survey surface renders near the top and depth increases downward, as
expected.
"""

from __future__ import annotations

import numpy as np

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def to_image_data(vol):
    """Resample onto a uniform depth axis if needed (``ImageData`` requires
    constant spacing) and return a ``pyvista.ImageData`` with a single
    point-data array ``"rho"`` (linear Ω·m, NaN where there is no data)."""
    import pyvista as pv

    z = np.asarray(vol.z, float)
    rho = vol.rho
    if z.size > 1 and not np.allclose(np.diff(z), z[1] - z[0], rtol=1e-3, atol=1e-6):
        zu = np.linspace(float(z.min()), float(z.max()), z.size)
        out = np.empty_like(rho)
        for j in range(rho.shape[1]):
            for i in range(rho.shape[2]):
                col = rho[:, j, i]
                ok = np.isfinite(col)
                out[:, j, i] = (np.interp(zu, z[ok], col[ok], left=np.nan, right=np.nan)
                                if ok.sum() >= 2 else np.nan)
        z, rho = zu, out

    nx, ny, nz = vol.x.size, vol.y.size, z.size
    dx = float(vol.x[1] - vol.x[0]) if nx > 1 else 1.0
    dy = float(vol.y[1] - vol.y[0]) if ny > 1 else 1.0

    z_world = -z[::-1]                         # ascending, surface near 0/top
    rho_world = rho[::-1, :, :]
    dz = float(z_world[1] - z_world[0]) if nz > 1 else 1.0

    grid = pv.ImageData(dimensions=(nx, ny, nz),
                        spacing=(dx or 1.0, dy or 1.0, dz or 1.0),
                        origin=(float(vol.x[0]), float(vol.y[0]), float(z_world[0])))
    arr = np.transpose(rho_world, (2, 1, 0))   # (nx, ny, nz), x fastest
    grid["rho"] = arr.ravel(order="F")
    grid.set_active_scalars("rho")
    return grid


def z_slice_from_scatter(dm, k: int, mask_ratio: float = 4.0):
    """Build a single depth level's slice mesh straight from its raw
    ``(E, N, rho)`` scatter, using the *exact* masked Delaunay triangulation
    the 2D Depth Map tab draws (:func:`ert2d.viz.sections._masked_triangulation`)
    - so this Z-slice matches that tab pixel-for-pixel instead of looking
    over-smoothed from a regridded raster (a regular grid has no notion of
    "no line ran through here" the way the mask-by-triangle-aspect-ratio
    triangulation does).

    Returns ``None`` if there are fewer than 3 points or every triangle was
    masked (e.g. no two points are close enough to form a small triangle)."""
    import pyvista as pv

    from ert2d.viz.sections import _masked_triangulation

    E, N, rho = dm.points[k]
    if len(rho) < 3:
        return None
    E = np.asarray(E, float); N = np.asarray(N, float); rho = np.asarray(rho, float)
    tri = _masked_triangulation(E, N, mask_ratio)
    triangles = tri.get_masked_triangles()
    if triangles.size == 0:
        return None

    z_world = np.full(E.size, -float(dm.depths[k]))
    points = np.column_stack([E, N, z_world])
    faces = np.hstack([np.full((len(triangles), 1), 3, dtype=np.int64),
                       triangles.astype(np.int64)]).ravel()
    mesh = pv.PolyData(points, faces)
    mesh["rho"] = rho
    return mesh


def axis_slice_mesh(grid, axis: str, position: float, count: int = 1):
    """One slice plane (``count == 1``) or ``count`` evenly-spaced parallel
    slices along ``axis`` ("x"/"y"/"z"), merged into a single mesh (or
    ``None`` if the slice(s) missed every finite cell)."""
    import pyvista as pv

    if count <= 1:
        origin = list(grid.center)
        origin[_AXIS_INDEX[axis]] = position
        mesh = grid.slice(normal=axis, origin=tuple(origin))
        return mesh if mesh.n_points else None

    mb = grid.slice_along_axis(n=count, axis=axis)
    polys = [mb[i] for i in range(mb.n_blocks) if mb[i] is not None and mb[i].n_points]
    return pv.merge(polys) if polys else None
