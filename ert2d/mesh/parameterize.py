"""Model discretization - three ways to turn one FEM mesh into inversion
parameters:

* ``"triangle"`` - every FEM triangle is its own parameter (the classic
  unstructured scheme; regularisation becomes the triangle-neighbour Laplacian).
* ``"rect"``     - a graded rectangular block grid (columns ``spacing /
  surface_div`` wide across the electrode line, geometric rows with depth);
  each FEM triangle joins the block containing its centroid.
* ``"voronoi"``  - bounded Voronoi cells of the mesh nodes (``main_vor_v00``).

All three return a :class:`ParamMesh` the FEM / inversion code already
understands; only the element→parameter partition and the ``cell_polys`` (used
for the smoothness matrix and the model plot) differ.
"""

from __future__ import annotations

import numpy as np
from shapely import contains, points
from shapely.geometry import Polygon, box
from shapely.strtree import STRtree

from ert2d.io.dat_reader import Survey
from ert2d.mesh.domain import build_domain
from ert2d.mesh.triangulate import TriMesh, triangulate
from ert2d.mesh.voronoi_param import ParamMesh, voronoi_parameterization

KINDS = ("voronoi", "triangle", "rect")


def core_region_mask(pm: ParamMesh, survey: Survey) -> np.ndarray:
    """Boolean mask (length ``pm.n_param``) marking which parameter cells lie
    inside the "core"/interpretable region ``[electrode_x.min(),
    electrode_x.max()] x [-z_max, 0]`` (the survey's natural investigation
    depth), as opposed to the ``wing``/``depth_factor`` padding that
    :func:`build_param_mesh` ALSO parameterizes.

    That full-domain parameterization (including the low-sensitivity padding)
    is intentional and unchanged by this function - keeping the padding
    parameterized (rather than fixed/extrapolated) avoids edge artefacts at
    the transition to the core region. This mask exists purely for
    *reporting*: it lets a "how many parameters does each geometry spend on
    the zone actually resolved by the data" comparison exclude padding cells
    that every geometry carries for numerical reasons, not resolving-power
    reasons.

    Containment is tested on each cell's **polygon centroid**, not an
    area-overlap fraction with the core box. Centroid containment was chosen
    because (1) it is a single point-in-box test per cell (cheap, no
    clipping), (2) it gives an unambiguous whole-cell yes/no count that
    matches how ``params``/``n_param`` are already reported elsewhere (a
    single integer per kind, not a fractional area), and (3) it matches how
    a reader visually assigns a cell straddling the core/padding boundary to
    "the section" or "the padding" when looking at a plotted model section -
    whichever side most of the cell (and hence its centroid) sits on. An
    overlap-fraction count would be more precise for cells that straddle the
    boundary by a large fraction of their area, but those are a small
    minority (mostly the outermost core column/row and its immediate
    neighbours) and would require carrying the core-box geometry (not just a
    scalar count) through the ``.npz`` cache for every kind - unnecessary
    complexity for a manuscript-table metric.

    Uses ``pm.cell_polys`` (not ``pm.seeds``): for ``kind="voronoi"``,
    ``ParamMesh.seeds`` holds the original Voronoi seed points (unique FEM
    nodes), not the *clipped* cell polygons' centroids, which can differ
    substantially for cells clipped by the domain boundary. ``cell_polys``
    centroids are correct and consistent for all three kinds.
    """
    x_lo = float(np.min(survey.electrode_x))
    x_hi = float(np.max(survey.electrode_x))
    z_max = float(survey.z_max)

    import os
    if os.environ.get("ERT_CORE_RULE", "centroid") == "intersect":
        # a cell is a parameter if it overlaps the core box at all (no notches left along the bottom/side edges)
        from shapely.geometry import box
        core_box = box(x_lo, -z_max, x_hi, 0.0)
        keep = np.zeros(len(pm.cell_polys), bool)
        for k, poly in enumerate(pm.cell_polys):
            if poly is not None and not poly.is_empty:
                keep[k] = poly.intersection(core_box).area > 1e-6
        return keep

    cx = np.full(len(pm.cell_polys), np.nan)
    cz = np.full(len(pm.cell_polys), np.nan)
    for k, poly in enumerate(pm.cell_polys):
        if poly is None or poly.is_empty:
            continue
        c = poly.centroid
        cx[k], cz[k] = c.x, c.y

    return (cx >= x_lo) & (cx <= x_hi) & (cz >= -z_max) & (cz <= 0.0)


def prolong_padding(pm: ParamMesh, survey: Survey) -> ParamMesh:
    """Make the padding (``wing`` / ``depth_factor`` zone, or the rect frame) a
    *prolongation* of the data-region model instead of free parameters.

    Only cells whose centroid lies in the core region (electrode span x
    ``[0, z_max]``, see :func:`core_region_mask`) stay parameters.  Every FEM
    element outside the cells that remain (padded cells of the triangular /
    Voronoi schemes, the fixed frame of the rect scheme) is assigned to the
    core cell found by clamping its centroid to the core box, i.e. it takes the
    value of the nearest core cell; the Jacobian then automatically includes
    its sensitivity.  This is the treatment used by pyGIMLi/BERT (a background
    region whose values are prolonged from the parametric domain) and leaves
    the number of unknowns equal to the number of cells in the data region.
    """
    from shapely import contains, points
    from shapely.strtree import STRtree

    keep = core_region_mask(pm, survey)
    k_idx = np.nonzero(keep)[0]
    polys = [pm.cell_polys[i] for i in k_idx]
    elems = [np.asarray(pm.cell_elements[i], int) for i in k_idx]
    owned = np.zeros(pm.t2.shape[1], bool)
    for els in elems:
        owned[els] = True
    free = np.nonzero(~owned)[0]
    if free.size:
        x_lo, x_hi = float(np.min(survey.electrode_x)), float(np.max(survey.electrode_x))
        z_lo = -float(survey.z_max)
        cent = pm.p2[:, pm.t2[:, free]].mean(axis=1).T
        q = np.column_stack([np.clip(cent[:, 0], x_lo, x_hi), np.clip(cent[:, 1], z_lo, 0.0)])
        tree = STRtree(polys)
        pts = points(q[:, 0], q[:, 1])
        pc = np.array([[p.centroid.x, p.centroid.y] for p in polys])
        owner = np.empty(free.size, int)
        for n, pt in enumerate(pts):
            hit = [int(j) for j in tree.query(pt) if contains(polys[int(j)], pt)]
            owner[n] = hit[0] if hit else int(np.linalg.norm(pc - q[n], axis=1).argmin())
        for j in np.unique(owner):
            elems[j] = np.concatenate([elems[j], free[owner == j]])
    seeds_kept = (np.asarray(pm.seeds)[k_idx] if np.asarray(pm.seeds).shape[0] == len(pm.cell_polys)
                  and getattr(pm, "kind", "voronoi") == "voronoi"
                  else np.asarray([[p.centroid.x, p.centroid.y] for p in polys]))
    out = ParamMesh(pm.p2, pm.t2, seeds_kept, polys, elems, pm.elec_nodes)
    out.kind = getattr(pm, "kind", "voronoi")
    for attr in ("n_rows", "n_cols"):
        if hasattr(pm, attr):
            setattr(out, attr, getattr(pm, attr))
    return out


def build_param_mesh(
    survey: Survey,
    *,
    kind: str = "voronoi",
    wing: float = 5.0,
    depth_factor: float = 4.0,
    surface_div: int = 2,
    padding: str = "free",
    **tri_kw,
) -> ParamMesh:
    """``padding="free"`` (default): the padded zone of the triangular / Voronoi
    schemes consists of free parameters (rect: fixed background frame).
    ``padding="prolong"``: padding is prolonged from the data-region cells
    (:func:`prolong_padding`) for every scheme."""
    if kind not in KINDS:
        raise ValueError(f"discretization kind must be one of {KINDS}, got {kind!r}")
    if padding not in ("free", "prolong"):
        raise ValueError("padding must be 'free' or 'prolong'")

    if kind == "rect":
        pm = rect_parameterization(survey, blocks_per_spacing=surface_div)
        return prolong_padding(pm, survey) if padding == "prolong" else pm

    dom = build_domain(survey, wing=wing, depth_factor=depth_factor,
                       surface_div=surface_div)
    if kind in ("triangle", "voronoi"):
        # Shared underlying FEM mesh: the Voronoi cells are the dual of this very
        # triangulation (cells = its nodes), as in the MATLAB implementation, so
        # both schemes default to the same node set and the Voronoi scheme has
        # ~1/1.75 as many cells as the triangular scheme has triangles.  (The
        # triangular scheme's mesh == parameterization, hence kept moderately coarse.)
        tri_kw.setdefault("surface_h", survey.spacing / 2.0)
        tri_kw.setdefault("h_grad", 1.3)
    centroid_nodes = tri_kw.pop("centroid_nodes", False)
    mesh = triangulate(dom, **tri_kw)
    pm = (voronoi_parameterization(mesh, dom, centroid_nodes=centroid_nodes) if kind == "voronoi"
          else triangle_parameterization(mesh, dom))
    return prolong_padding(pm, survey) if padding == "prolong" else pm


# --------------------------------------------------------------------------- #
def _match_electrodes(p2: np.ndarray, electrodes: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(p2.T[:, None, :] - np.asarray(electrodes)[None, :, :], axis=2)
    return d.argmin(axis=0)


def _assign_owner(cent: np.ndarray, polys: list[Polygon]) -> np.ndarray:
    owner = np.full(len(cent), -1, int)
    tree = STRtree(polys)
    gp = points(cent[:, 0], cent[:, 1])
    for e, pt in enumerate(gp):
        for k in tree.query(pt):
            if contains(polys[int(k)], pt):
                owner[e] = int(k)
                break
    miss = np.nonzero(owner < 0)[0]
    if miss.size:
        pc = np.array([[p.centroid.x, p.centroid.y] for p in polys])
        owner[miss] = np.linalg.norm(cent[miss, None, :] - pc[None, :, :],
                                     axis=2).argmin(axis=1)
    return owner


def _pack(p2, t2, polys, owner, electrodes, kind: str,
          min_elem: int = 1) -> ParamMesh:
    """Fold under-filled blocks (< ``min_elem`` elements, or empty) into the
    edge-adjacent block with the most elements, so the parameter cells tile the
    region with no gaps and no barely-constrained slivers."""
    from shapely.ops import unary_union

    polys = list(polys)
    owner = owner.copy()
    counts = np.bincount(owner[owner >= 0], minlength=len(polys))
    pc = np.array([[p.centroid.x, p.centroid.y] for p in polys])
    for _ in range(8):
        weak = np.nonzero((counts >= 0) & (counts < min_elem))[0]
        if weak.size == 0:
            break
        tree = STRtree(polys)
        progressed = False
        for k in weak:
            nbrs = [int(j) for j in tree.query(polys[k])
                    if j != k and counts[int(j)] >= min_elem
                    and polys[k].intersection(polys[int(j)]).length > 1e-9]
            if not nbrs:
                strong = np.nonzero(counts >= min_elem)[0]
                if strong.size == 0:
                    break
                nbrs = [int(strong[np.linalg.norm(pc[strong] - pc[k], axis=1).argmin()])]
            tgt = max(nbrs, key=lambda j: counts[j])
            polys[tgt] = unary_union([polys[tgt], polys[k]])
            counts[tgt] += max(counts[k], 0)
            owner[owner == k] = tgt
            counts[k] = -1
            progressed = True
        if not progressed:
            break

    keep, cell_elements = [], []
    for k, poly in enumerate(polys):
        if counts[k] < 0:
            continue
        els = np.nonzero(owner == k)[0]
        if els.size:
            keep.append(poly)
            cell_elements.append(els)
    seeds = np.array([[p.centroid.x, p.centroid.y] for p in keep])
    pm = ParamMesh(np.ascontiguousarray(p2), np.ascontiguousarray(t2),
                   seeds, keep, cell_elements, _match_electrodes(p2, electrodes))
    pm.kind = kind
    return pm


# --------------------------------------------------------------------------- #
def triangle_parameterization(mesh: TriMesh, domain) -> ParamMesh:
    p2, t2 = mesh.points, mesh.tris
    polys = [Polygon(p2[:, t2[:, i]].T) for i in range(t2.shape[1])]
    cell_elements = [np.array([i]) for i in range(t2.shape[1])]
    pm = ParamMesh(np.ascontiguousarray(p2), np.ascontiguousarray(t2),
                   mesh.centroids(), polys, cell_elements,
                   _match_electrodes(p2, domain.electrodes))
    pm.kind = "triangle"
    return pm


def rect_parameterization(survey: Survey, *, blocks_per_spacing: int = 2,
                          layer_factor: float = 1.1) -> ParamMesh:
    """Fully-structured rectangular-block model - port of ``meshgena.m``.

    * columns: ``blocks_per_spacing`` blocks between each pair of electrodes
      (constant width, not depth-dependent);
    * rows: geometric depth layers (first ``dz1/2``, ratio ``layer_factor``),
      rescaled so the parametrised region ends exactly at ``z_max``;
    * FEM mesh: every grid cell split into two triangles (no Delaunay);
    * a 3-column border each side and 3 padding rows below stay fixed at the
      background resistivity - only the interior blocks are parameters.
    """
    xe = np.asarray(survey.electrode_x, float)
    ela = float(survey.spacing)
    zmax = float(survey.z_max)
    dz1 = float(getattr(survey, "dz1", 0.0) or ela * 0.5)

    # --- depth levels ------------------------------------------------- #
    dz, tot = [dz1 / 2.0], dz1 / 2.0
    while tot <= zmax * 1.1:
        dz.append(dz[-1] * layer_factor)
        tot += dz[-1]
    zc = np.cumsum(dz)
    zc = zc * (zmax / zc[-1])                          # land exactly on z_max
    n_lay = len(zc)
    z_core = np.concatenate([[0.0], -zc])             # n_lay + 1 levels
    z_pad = z_core[-1] * np.array([1.25, 1.875, 2.8125])
    z_u = np.concatenate([z_core, z_pad])

    # --- column lines ---------------------------------------------- #
    m = max(int(blocks_per_spacing), 1)
    cols = [xe[0]]
    for a, b in zip(xe[:-1], xe[1:]):
        cols.extend(a + (b - a) * np.arange(1, m + 1) / m)
    x_core = np.array(cols)
    x_left = x_core[0] - np.array([6.0, 3.0, 1.0]) * ela
    x_right = x_core[-1] + np.array([1.0, 3.0, 6.0]) * ela
    x_u = np.concatenate([x_left, x_core, x_right])

    n_pad_l = len(x_left)                              # = 3
    n_core_col = len(x_core) - 1                       # parametrised columns

    # --- structured grid -> 2 triangles per cell ------------------- #
    p2, t2, block_tris = _grid_mesh(x_u, z_u)

    polys, cell_elements = [], []
    for i in range(n_pad_l, n_pad_l + n_core_col):
        for j in range(n_lay):
            polys.append(box(x_u[i], z_u[j + 1], x_u[i + 1], z_u[j]))
            cell_elements.append(block_tris[i, j])

    elec_nodes = _match_electrodes(p2, np.column_stack([xe, np.zeros_like(xe)]))
    pm = ParamMesh(p2, t2, np.array([[p.centroid.x, p.centroid.y] for p in polys]),
                   polys, cell_elements, elec_nodes)
    pm.kind = "rect"
    pm.n_cols, pm.n_rows = n_core_col, n_lay
    return pm


def _grid_mesh(xg: np.ndarray, zg: np.ndarray):
    """Structured node grid + two triangles per cell.

    Returns ``p (2,N)``, ``t (3,M)`` and ``block_tris[i, j]`` = the two triangle
    indices of grid cell ``(i, j)``.  Node index = ``i * nz + j``.
    """
    nx, nz = len(xg), len(zg)
    X, Z = np.meshgrid(xg, zg, indexing="ij")
    p = np.vstack([X.ravel(), Z.ravel()])
    tris = np.empty((3, 2 * (nx - 1) * (nz - 1)), int)
    block_tris = np.empty((nx - 1, nz - 1), object)
    c = 0
    for i in range(nx - 1):
        for j in range(nz - 1):
            n00, n01 = i * nz + j, i * nz + j + 1
            n10, n11 = (i + 1) * nz + j, (i + 1) * nz + j + 1
            tris[:, c] = (n00, n10, n11)
            tris[:, c + 1] = (n00, n11, n01)
            block_tris[i, j] = np.array([c, c + 1])
            c += 2
    return np.ascontiguousarray(p), np.ascontiguousarray(tris), block_tris
