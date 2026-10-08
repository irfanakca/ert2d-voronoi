"""Voronoi-polygon parameterization.

Port of the second half of ``meshgen.m`` (the ``VoronoiLimit`` +
re-triangulation block).

Given a FEM mesh, the *unique mesh nodes* are used as Voronoi seeds.  Each
Voronoi cell is clipped to the inversion domain, all clipped-cell vertices plus
the electrode positions are re-triangulated into a second mesh ``(p2, t2)``, and
every ``t2`` element is assigned to the cell that contains its centroid.  The
cells are the inversion parameters.

MATLAB uses the third-party ``VoronoiLimit`` (Mapping Toolbox, ~1000 lines,
fragile).  Here the bounded diagram is built with ``scipy.spatial.Voronoi`` +
``shapely`` clipping.  Result is geometrically equivalent, not vertex-identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay, Voronoi
from shapely.geometry import MultiPoint, Polygon
from shapely.strtree import STRtree

from ert2d.mesh.domain import Domain
from ert2d.mesh.triangulate import TriMesh

_ROUND = 6      # coordinate rounding (decimals) when de-duplicating points


@dataclass
class ParamMesh:
    p2: np.ndarray                 # (2, N2) node coords
    t2: np.ndarray                 # (3, M2) triangle connectivity (0-based)
    seeds: np.ndarray              # (P, 2) Voronoi seed points (= unique FEM nodes)
    cell_polys: list[Polygon]      # clipped Voronoi polygons, one per seed
    cell_elements: list[np.ndarray]  # t2 element indices per cell (the MATLAB ``par(i).ucg``)
    elec_nodes: np.ndarray = None  # (n_elec,) p2 node index for each electrode
    kind: str = "voronoi"          # "voronoi" | "triangle" | "rect"

    @property
    def n_param(self) -> int:
        return len(self.cell_polys)

    @property
    def n_nodes(self) -> int:
        return self.p2.shape[1]

    @property
    def n_tris(self) -> int:
        return self.t2.shape[1]

    def element_to_param(self) -> np.ndarray:
        """(M2,) array mapping each t2 element to its parameter index."""
        m = np.full(self.t2.shape[1], -1, int)
        for k, els in enumerate(self.cell_elements):
            m[els] = k
        return m


def _constrained_retriangulation(cell_polys, electrodes, dom_poly):
    """Constrained Delaunay triangulation of the cell polygons: polygon vertices,
    electrodes and one centroid node per cell; every cell edge is a segment, so no
    triangle crosses a cell boundary (no 'q' refinement: no Steiner nodes)."""
    import triangle as _tri
    from shapely.ops import unary_union

    lines = unary_union([p.exterior for p in cell_polys if not p.is_empty]
                        + [dom_poly.exterior])                 # noded linework
    geoms = getattr(lines, "geoms", [lines])
    verts, index, segs = [], {}, []

    def vid(xy):
        k = (round(xy[0], _ROUND), round(xy[1], _ROUND))
        if k not in index:
            index[k] = len(verts)
            verts.append(k)
        return index[k]

    for g in geoms:
        c = list(g.coords)
        for a, b in zip(c[:-1], c[1:]):
            ia, ib = vid(a), vid(b)
            if ia != ib:
                segs.append((ia, ib))
    for e in electrodes:
        vid(e)
    for p in cell_polys:
        if not p.is_empty:
            c = p.centroid
            vid((c.x, c.y))
    out = _tri.triangulate({"vertices": np.asarray(verts), "segments": np.unique(np.sort(segs, axis=1), axis=0)}, "p")
    return np.ascontiguousarray(out["vertices"].T), np.ascontiguousarray(out["triangles"].T)


def voronoi_parameterization(mesh: TriMesh, domain: Domain, *,
                             centroid_nodes: bool = False) -> ParamMesh:
    """``centroid_nodes=True``: the re-triangulation is constrained to the cell
    boundaries and gets one extra node at every cell centroid."""
    seeds = np.unique(np.round(mesh.points.T, _ROUND), axis=0)
    dom_poly = Polygon(domain.boundary).buffer(0)

    cell_polys = _bounded_voronoi(seeds, dom_poly)

    if centroid_nodes:
        p2, t2 = _constrained_retriangulation(cell_polys, domain.electrodes, dom_poly)
        cent = p2[:, t2].mean(axis=1).T
        inside = _points_in_poly(cent, dom_poly)
        t2, cent = t2[:, inside], cent[inside]
        cell_elements = _assign_elements(cent, cell_polys, seeds)
        d = np.linalg.norm(p2.T[:, None, :] - domain.electrodes[None, :, :], axis=2)
        return ParamMesh(p2, t2, seeds, cell_polys, cell_elements, d.argmin(axis=0))

    # ---- collect points for the re-triangulation --------------------- #
    pieces = [np.asarray(p.exterior.coords)[:-1] for p in cell_polys if not p.is_empty]
    pieces.append(domain.electrodes)
    pts = np.vstack(pieces)
    pts = np.unique(np.round(pts, _ROUND), axis=0)

    tri = Delaunay(pts)
    p2 = np.ascontiguousarray(pts.T)
    t2 = np.ascontiguousarray(tri.simplices.T)

    # keep only triangles whose centroid is inside the domain
    cent = p2[:, t2].mean(axis=1).T
    inside = _points_in_poly(cent, dom_poly)
    t2 = t2[:, inside]
    cent = cent[inside]

    cell_elements = _assign_elements(cent, cell_polys, seeds)

    # electrode -> p2 node index (electrodes were forced into ``pts``)
    d = np.linalg.norm(p2.T[:, None, :] - domain.electrodes[None, :, :], axis=2)
    elec_nodes = d.argmin(axis=0)

    return ParamMesh(p2, t2, seeds, cell_polys, cell_elements, elec_nodes)


# --------------------------------------------------------------------------- #
def _bounded_voronoi(seeds: np.ndarray, dom_poly: Polygon) -> list[Polygon]:
    vor = Voronoi(seeds)
    regions, vertices = _finite_polygons(vor)
    out: list[Polygon] = []
    minx, miny, maxx, maxy = dom_poly.bounds
    big = max(maxx - minx, maxy - miny) * 10.0
    for reg in regions:
        poly = Polygon(vertices[reg])
        if not poly.is_valid:
            poly = poly.buffer(0)
        clipped = poly.intersection(dom_poly)
        if clipped.is_empty:
            out.append(Polygon())
            continue
        if clipped.geom_type == "MultiPolygon":
            clipped = max(clipped.geoms, key=lambda g: g.area)
        out.append(clipped)
    return out


def _finite_polygons(vor: Voronoi):
    """Reconstruct Voronoi regions as finite polygons (infinite ridges extended)."""
    new_regions = []
    new_vertices = list(vor.vertices)
    center = vor.points.mean(axis=0)
    radius = np.ptp(vor.points, axis=0).max() * 10.0

    ridges: dict[int, list] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        ridges.setdefault(p1, []).append((p2, v1, v2))
        ridges.setdefault(p2, []).append((p1, v1, v2))

    for p, region_idx in enumerate(vor.point_region):
        verts = vor.regions[region_idx]
        if all(v >= 0 for v in verts):
            new_regions.append(verts)
            continue
        finite = [v for v in verts if v >= 0]
        for p2, v1, v2 in ridges.get(p, []):
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[p]
            t /= np.linalg.norm(t)
            nrm = np.array([-t[1], t[0]])
            mid = vor.points[[p, p2]].mean(axis=0)
            direction = np.sign(np.dot(mid - center, nrm)) * nrm
            far = vor.vertices[v2] + direction * radius
            finite.append(len(new_vertices))
            new_vertices.append(far)
        pts = np.asarray([new_vertices[v] for v in finite])
        ang = np.arctan2(pts[:, 1] - pts[:, 1].mean(), pts[:, 0] - pts[:, 0].mean())
        new_regions.append([finite[i] for i in np.argsort(ang)])

    return new_regions, np.asarray(new_vertices)


def _points_in_poly(pts: np.ndarray, poly: Polygon) -> np.ndarray:
    from shapely import points as _sp_points
    from shapely import contains as _sp_contains
    gp = _sp_points(pts[:, 0], pts[:, 1])
    return _sp_contains(poly, gp)


def _assign_elements(cent: np.ndarray, cell_polys: list[Polygon],
                     seeds: np.ndarray) -> list[np.ndarray]:
    from shapely import points as _sp_points

    n_cells = len(cell_polys)
    owner = np.full(len(cent), -1, int)

    valid = [i for i, p in enumerate(cell_polys) if not p.is_empty]
    tree = STRtree([cell_polys[i] for i in valid])
    gp = _sp_points(cent[:, 0], cent[:, 1])

    for e, pt in enumerate(gp):
        cand = tree.query(pt)
        for local in cand:
            gi = valid[local]
            if cell_polys[gi].covers(pt):
                owner[e] = gi
                break

    # fallback: nearest seed for any centroid that no polygon covered
    miss = np.nonzero(owner < 0)[0]
    if miss.size:
        d = np.linalg.norm(cent[miss, None, :] - seeds[None, :, :], axis=2)
        owner[miss] = d.argmin(axis=1)

    return [np.nonzero(owner == k)[0] for k in range(n_cells)]
