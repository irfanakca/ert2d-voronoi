"""Triangular FEM mesh generation.

Replaces the MATLAB PDE Toolbox chain ``decsg`` -> ``initmesh`` -> ``jigglemesh``
with Shewchuk's *Triangle*.  ``initmesh`` grades element size from the geometry
(fine at the densely-sampled surface, coarser at depth); we reproduce that by
seeding graded interior points and letting a quality Delaunay refinement
(``pq``) finish the job, then a few Laplacian smoothing sweeps stand in for
``jigglemesh``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import triangle as _tri

from ert2d.mesh.domain import Domain


@dataclass
class TriMesh:
    points: np.ndarray        # (2, N)  -- MATLAB ``p`` convention
    tris: np.ndarray          # (3, M)  -- MATLAB ``t(1:3, :)`` convention

    @property
    def n_nodes(self) -> int:
        return self.points.shape[1]

    @property
    def n_tris(self) -> int:
        return self.tris.shape[1]

    def centroids(self) -> np.ndarray:
        return self.points[:, self.tris].mean(axis=1).T   # (M, 2)

    def areas(self) -> np.ndarray:
        p = self.points
        a, b, c = (p[:, self.tris[i]] for i in range(3))
        return 0.5 * np.abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


def triangulate(
    domain: Domain,
    *,
    min_angle: float = 30.0,
    h_grad: float = 1.25,
    surface_h: float | None = None,
    max_area: float | None = None,
    smooth_iters: int = 40,
    x_stretch: float = 1.0,
    extra_points: np.ndarray | None = None,
) -> TriMesh:
    """``extra_points`` (K, 2): additional interior nodes (local refinement, e.g. from an adaptive loop)."""
    verts = domain.boundary
    n = len(verts)
    segments = np.column_stack([np.arange(n), (np.arange(n) + 1) % n])

    h0 = surface_h if surface_h is not None else domain.spacing / 4.0
    interior = _graded_points(domain, h_grad, h0, x_stretch)
    pts = np.vstack([verts, interior]) if len(interior) else verts
    if extra_points is not None and len(extra_points):
        pts = np.vstack([pts, np.asarray(extra_points, float).reshape(-1, 2)])

    pslg = {"vertices": pts, "segments": segments}
    opts = f"pq{min_angle:g}"
    if max_area is not None:
        opts += f"a{max_area:g}"
    out = _tri.triangulate(pslg, opts)

    P = np.ascontiguousarray(out["vertices"].T, float)      # (2, N)
    T = np.ascontiguousarray(out["triangles"].T)            # (3, M)

    boundary_mask = _boundary_node_mask(out)
    P = _laplacian_smooth(P, T, fixed=boundary_mask, iters=smooth_iters)
    return TriMesh(P, T)


# --------------------------------------------------------------------------- #
def _graded_points(domain: Domain, h_grad: float, h0: float, x_stretch: float = 1.0) -> np.ndarray:
    """Interior nodes on rows whose spacing grows geometrically with depth."""
    x0, x1 = domain.x_bounds
    z0 = float(domain.surface_z.min())
    z_bot = domain.z_bounds[0]
    h = h0                                  # surface target size

    rows = []
    z = z0 - h
    while z > z_bot + h:
        hx = h * x_stretch                  # >1: laterally elongated (layer-like) cells
        xs = np.arange(x0 + hx / 2.0, x1 - hx / 4.0, hx)
        rows.append(np.column_stack([xs, np.full(xs.size, z)]))
        h *= h_grad
        z -= h
    return np.vstack(rows) if rows else np.empty((0, 2))


def _boundary_node_mask(out: dict) -> np.ndarray:
    n = out["vertices"].shape[0]
    mask = np.zeros(n, bool)
    if "vertex_markers" in out:
        mask |= out["vertex_markers"].ravel().astype(bool)
    if "segments" in out:
        mask[np.unique(out["segments"])] = True
    return mask


def _laplacian_smooth(P: np.ndarray, T: np.ndarray, *, fixed: np.ndarray,
                      iters: int) -> np.ndarray:
    n = P.shape[1]
    # build node adjacency from triangle edges
    edges = np.hstack([T[[0, 1]], T[[1, 2]], T[[2, 0]]])
    edges = np.hstack([edges, edges[::-1]])
    order = np.argsort(edges[0], kind="stable")
    src, dst = edges[0, order], edges[1, order]
    starts = np.searchsorted(src, np.arange(n))
    ends = np.searchsorted(src, np.arange(n) + 1)

    Q = P.copy()
    move = ~fixed
    for _ in range(iters):
        newQ = Q.copy()
        for i in np.nonzero(move)[0]:
            nb = dst[starts[i]:ends[i]]
            if nb.size:
                newQ[:, i] = Q[:, nb].mean(axis=1)
        Q = newQ
    return Q
