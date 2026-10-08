"""Parameter-cell smoothness (roughness) matrix.

Row ``k`` always has ``C[k, k] = 1`` and negative off-diagonals that sum to
``-1`` (a Laplacian), so ``‖C m‖²`` measures model roughness.

Two neighbour weightings:

* **``voronoi_c.m``** (``edge_only=False``, the port default): neighbours are
  cells sharing >= 1 boundary point; weight = area-fraction × normalised
  inverse centroid-distance.
* **finite-volume** (``edge_only=True``, used for the ``triangle`` / ``rect``
  discretizations): neighbours share a boundary *segment*; weight =
  shared-edge-length / centroid-distance.  This is the standard control-volume
  Laplacian and handles anisotropic (wide-thin) blocks correctly.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from shapely.geometry import LineString, Polygon
from shapely.strtree import STRtree

_TINY = np.finfo(float).eps


def grid_laplacian(n_rows: int, n_cols: int) -> sp.csr_matrix:
    """5-point graph Laplacian on a structured ``n_rows × n_cols`` block grid,
    used as the roughness operator for the ``rect`` model (cf. ``elris.m``'s
    ``C = delsq(parc)``).

    Cell ``(row, col)`` -> index ``col * n_rows + row`` (column-major, matching
    ``rect_parameterization``).  ``-1`` per existing orthogonal neighbour and
    ``diag = number of neighbours`` so **every row sums to zero** - a Neumann
    Laplacian that penalises only the differences between adjacent blocks
    (``mᵀ L m = Σ_edges (m_i - m_j)²``), not the absolute model level.  (MATLAB
    ``delsq`` fixes ``diag = 4``, a Dirichlet form that only stays stable there
    because the model update is multiplicative and ``λ`` tiny.)
    """
    n = n_rows * n_cols
    rows, cols, data = [], [], []
    for c in range(n_cols):
        for r in range(n_rows):
            k = c * n_rows + r
            nb = 0
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cc, rr = c + dc, r + dr
                if 0 <= cc < n_cols and 0 <= rr < n_rows:
                    rows.append(k); cols.append(cc * n_rows + rr); data.append(-1.0)
                    nb += 1
            rows.append(k); cols.append(k); data.append(float(nb))
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def voronoi_fv_smoothness(cell_polys: list[Polygon], seeds: np.ndarray) -> sp.csr_matrix:
    """Two-point-flux finite-volume Laplacian on a Voronoi tessellation.

    Neighbours share an edge; the weight of a link is ``L_ij / |s_i - s_j|`` with
    ``L_ij`` the shared edge length and ``s`` the generator (seed) points.  For a
    Voronoi diagram the segment between two generators is perpendicular to their
    common edge, which makes this a consistent (and, for equal spacing, isotropic)
    discretization of the Laplacian.  Rows are normalised like
    :func:`voronoi_smoothness` (diagonal 1, off-diagonals sum to -1).
    """
    n = len(cell_polys)
    seeds = np.asarray(seeds, float)
    valid = [i for i, p in enumerate(cell_polys) if (p is not None and not p.is_empty)]
    polys = [cell_polys[i] for i in valid]
    tree = STRtree(polys)
    neigh: list[list[tuple[int, float]]] = [[] for _ in polys]
    for a in range(len(polys)):
        for b in tree.query(polys[a]):
            b = int(b)
            if b <= a:
                continue
            L = polys[a].intersection(polys[b]).length
            if L > 1e-9:
                d = float(np.linalg.norm(seeds[valid[a]] - seeds[valid[b]]))
                w = L / max(d, _TINY)
                neigh[a].append((b, w))
                neigh[b].append((a, w))
    rows, cols, data = [], [], []
    for k in range(len(polys)):
        rows.append(valid[k]); cols.append(valid[k]); data.append(1.0)
        if not neigh[k]:
            continue
        tot = sum(w for _, w in neigh[k])
        for j, w in neigh[k]:
            rows.append(valid[k]); cols.append(valid[j]); data.append(-w / tot)
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def fv_energy_laplacian(cell_polys: list[Polygon], points: np.ndarray | None = None,
                        area_scaled: bool = False, barrier=None, barrier_weight: float = 1.0) -> sp.csr_matrix:
    """Unnormalised symmetric finite-volume Laplacian ``L`` with link weights
    ``w_ij = L_ij / |p_i - p_j|`` (shared edge length over point distance;
    ``points`` = generators for Voronoi cells, centroids when ``None``).

    ``mᵀ L m = Σ_links w_ij (m_i - m_j)²`` is the gradient (first-order) energy
    ``∫|∇m|²`` of the cell model - dimensionless in 2D, so it does not depend on the
    cell size.  Use it with ``roughness_form="C"``.

    ``area_scaled=True`` returns ``V^{-1/2} L`` (V = cell areas); with ``"CtC"`` the
    penalty is then ``mᵀ L V⁻¹ L m ≈ ∫(∇²m)² dA``, the consistent second-order energy."""
    n = len(cell_polys)
    valid = [i for i, p in enumerate(cell_polys) if (p is not None and not p.is_empty)]
    polys = [cell_polys[i] for i in valid]
    pts = (np.array([[p.centroid.x, p.centroid.y] for p in cell_polys]) if points is None
           else np.asarray(points, float))
    tree = STRtree(polys)
    rows, cols, data = [], [], []
    diag = np.zeros(n)
    for a in range(len(polys)):
        for b in tree.query(polys[a]):
            b = int(b)
            if b <= a:
                continue
            L = polys[a].intersection(polys[b]).length
            if L > 1e-9:
                ia, ib = valid[a], valid[b]
                w = L / max(float(np.linalg.norm(pts[ia] - pts[ib])), _TINY)
                if barrier is not None and LineString([pts[ia], pts[ib]]).intersects(barrier):
                    w *= barrier_weight                      # known boundary: weaken the smoothing across it
                rows += [ia, ib]; cols += [ib, ia]; data += [-w, -w]
                diag[ia] += w; diag[ib] += w
    rows += list(range(n)); cols += list(range(n)); data += list(diag)
    Lm = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    if area_scaled:
        area = np.array([p.area if (p is not None and not p.is_empty) else 1.0 for p in cell_polys])
        Lm = sp.diags(1.0 / np.sqrt(area)) @ Lm
    return Lm


def voronoi_smoothness(cell_polys: list[Polygon],
                       edge_only: bool = False) -> sp.csr_matrix:
    n = len(cell_polys)
    valid = [i for i, p in enumerate(cell_polys) if (p is not None and not p.is_empty)]
    polys = [cell_polys[i] for i in valid]
    if not polys:
        return sp.csr_matrix((n, n))

    cent = np.array([[p.centroid.x, p.centroid.y] for p in polys])
    area = np.array([p.area for p in polys])

    tree = STRtree(polys)
    neigh: list[list[tuple[int, float]]] = [[] for _ in polys]
    for a in range(len(polys)):
        for b in tree.query(polys[a]):
            b = int(b)
            if b <= a:
                continue
            shared = polys[a].intersection(polys[b]).length
            if shared > 1e-9 or (not edge_only and polys[a].touches(polys[b])):
                neigh[a].append((b, shared))
                neigh[b].append((a, shared))

    rows, cols, data = [], [], []
    for k in range(len(polys)):
        rows.append(valid[k]); cols.append(valid[k]); data.append(1.0)
        if not neigh[k]:
            continue
        nb = np.array([j for j, _ in neigh[k]])
        dist = np.linalg.norm(cent[k] - cent[nb], axis=1)
        dist = np.where(dist > 0, dist, _TINY)

        if edge_only:
            ln = np.array([max(L, _TINY) for _, L in neigh[k]])
            wgt = ln / dist
        else:                                   # voronoi_c.m weighting
            oran = nb.size / np.sum(1.0 / dist)
            d_m = oran / dist
            o_m = area[nb] / area[nb].sum()
            wgt = o_m * d_m

        wgt = wgt / wgt.sum()                    # off-diagonals sum to -1
        for m, j in enumerate(nb):
            rows.append(valid[k]); cols.append(valid[int(j)]); data.append(-wgt[m])

    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
