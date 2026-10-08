"""Model suite for the Voronoi parameterization: models with vertical/lateral discontinuities, stepped and inclined
interfaces and small buried man-made objects, plus mesh-refinement (convergence) runs.

Every model is a background resistivity with shapes painted on top (shapely polygons).  The synthetic data are computed on a
finite-element mesh whose edges coincide with all shape boundaries (no resampling of the model), 3% noise is added and the
data are inverted with the Voronoi, triangular and rectangular parameterizations (``ERT_REG=energy``: one functional for all).

    python scripts/model_suite.py --models step basin arch --mode equal free --seeds 42 1 2 3 4 --out ../../results_v24/suite
    python scripts/model_suite.py --models step basin arch emb --converge --seeds 42 1 --out ../../results_v24/converge
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("ERT_PADDING", "prolong")
os.environ.setdefault("ERT_REG", "energy")
os.environ.setdefault("ERT_CORE_RULE", "intersect")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import shapely
import triangle as tr
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data"


# --------------------------------------------------------------------------------------- model specifications
def _step_basement(x_edges, depths, x_end):
    """Polygon below a staircase interface: depth ``depths[i]`` (positive) for x in [x_edges[i], x_edges[i+1])."""
    pts = [(x_edges[0], -60.0)]
    xs = list(x_edges) + [x_end]
    for i, d in enumerate(depths):
        pts += [(xs[i], -d), (xs[i + 1], -d)]
    pts += [(x_end, -60.0)]
    return Polygon(pts)


def spec_step():
    """Stepped basement (fault scarps) and a vertical lateral contact in the layer."""
    L, wing = 50.0, 5.0
    x0, x1 = -wing, L + wing
    base = _step_basement([x0, 14.0, 26.0, 38.0], [2.4, 3.8, 5.0, 3.2], x1)
    return dict(name="step", n_el=51, spacing=1.0, bg=100.0, noise=0.03, div=1.0,
                shapes=[("conductive block", 30.0, box(30.0, -60.0, x1, 0.0)), ("basement", 400.0, base)],
                objects=[("conductive block", "min", 30.0)], desc="stepped basement, vertical lateral contact")


def spec_basin():
    """Inclined interfaces (basin with 17 degree flanks)."""
    L, wing = 50.0, 5.0
    x0, x1 = -wing, L + wing
    pts = [(x0, -60.0), (x0, -0.8)]
    pts += [(0.0, -0.8), (14.0, -5.0), (36.0, -5.0), (50.0, -0.8), (x1, -0.8)]
    pts += [(x1, -60.0)]
    return dict(name="basin", n_el=51, spacing=1.0, bg=100.0, noise=0.03, div=1.0,
                shapes=[("basement", 500.0, Polygon(pts))], objects=[], desc="inclined interfaces (basin)")


def spec_arch():
    """Small man-made objects: stone wall, air-filled chamber and a conductive pipe in soil above weathered bedrock."""
    L, wing = 20.0, 2.0
    x0, x1 = -wing, L + wing
    return dict(name="arch", n_el=41, spacing=0.5, bg=100.0, noise=0.03, div=1.0,
                shapes=[("bedrock", 400.0, box(x0, -60.0, x1, -3.2)),
                        ("wall", 1000.0, box(5.0, -1.6, 5.6, -0.3)),
                        ("chamber", 5000.0, box(10.0, -2.4, 12.0, -1.4)),
                        ("pipe", 15.0, Point(15.5, -1.0).buffer(0.35, 12))],
                objects=[("wall", "max", 1000.0), ("chamber", "max", 5000.0), ("pipe", "min", 15.0)],
                desc="buried wall, chamber and pipe")


def spec_emb():
    """Embedded-body model of the manuscript (layer with undulating base, conductive rectangle, resistive circle)."""
    L, wing = 50.0, 5.0
    x0, x1 = -wing, L + wing
    xs = np.arange(x0, x1 + 1e-9, 0.25)
    zi = -(3.8 + 0.8 * np.cos(2 * np.pi * (xs - 14.0) / 18.0))
    base = Polygon([(x0, -60.0)] + list(zip(xs, zi)) + [(x1, -60.0)])
    return dict(name="emb", n_el=51, spacing=1.0, bg=100.0, noise=0.03, div=1.0,
                shapes=[("basement", 10000.0, base), ("rectangle", 10.0, box(11.0, -2.5, 17.0, -1.0)),
                        ("circle", 1000.0, Point(33.0, -1.8).buffer(1.2, 12))],
                objects=[("rectangle", "min", 10.0), ("circle", "max", 1000.0)], desc="embedded bodies")


def spec_blob():
    """Arbitrarily shaped bodies: a concave resistive 'boomerang' and an irregular conductive 'amoeba' in a layer over bedrock."""
    L, wing = 50.0, 5.0
    x0, x1 = -wing, L + wing
    th = np.linspace(0, 2 * np.pi, 49)[:-1]
    r = 2.0 * (1 + 0.30 * np.cos(3 * th) + 0.18 * np.sin(5 * th + 0.6))
    amoeba = Polygon(list(zip(34.0 + r * np.cos(th), -2.6 + 0.8 * r * np.sin(th))))
    boomerang = Polygon([(9.0, -0.9), (16.5, -0.9), (16.5, -2.1), (11.2, -2.1), (11.2, -3.9), (9.0, -3.9)])
    return dict(name="blob", n_el=51, spacing=1.0, bg=100.0, noise=0.03, div=1.0,
                shapes=[("bedrock", 400.0, box(x0, -60.0, x1, -5.2)), ("boomerang", 800.0, boomerang), ("amoeba", 15.0, amoeba)],
                objects=[("boomerang", "max", 800.0), ("amoeba", "min", 15.0)], desc="irregularly shaped bodies")


SPECS = {"step": spec_step, "basin": spec_basin, "arch": spec_arch, "emb": spec_emb, "blob": spec_blob}


# --------------------------------------------------------------------------------------- geometry / data
def write_geometry(path: Path, n_el: int, spacing: float) -> int:
    rows = []
    x = lambda i: i * spacing                                                     # noqa: E731
    a_max = max((n_el - 1) // 4, 1)
    for a in range(1, min(12, a_max) + 1):
        for i in range(0, n_el - 3 * a):
            rows.append((x(i), x(i + 3 * a), x(i + a), x(i + 2 * a)))
    for n in range(1, 15):
        for i in range(0, n_el - 2 * n - 1):
            rows.append((x(i), x(i + 2 * n + 1), x(i + n), x(i + n + 1)))
    for n in range(1, 9):
        for i in range(1, n_el - n - 1):
            rows.append((x(i), x(i - 1), x(i + n), x(i + n + 1)))
    with open(path, "w") as f:
        f.write(f"Model-suite test geometry\n{spacing}\n11\n0\nType of measurement (0=apparent resistivity,1=resistance)\n0\n")
        f.write(f"{len(rows)}\n1\n0\n")
        for c1, c2, p1, p2 in rows:
            f.write(f"4 {c1:.3f} 0 {c2:.3f} 0 {p1:.3f} 0 {p2:.3f} 0 100.0\n")
    return len(rows)


def geometry_file(spec) -> Path:
    p = SAMPLE / f"suite_{spec['name']}.dat"
    if not p.exists() or p.stat().st_size < 1000:       # workers only read; main() writes all files up front
        write_geometry(p, spec["n_el"], spec["spacing"])
    return p


def write_all_geometries(names):
    for n in names:
        spec = SPECS[n]()
        write_geometry(SAMPLE / f"suite_{n}.dat", spec["n_el"], spec["spacing"])


def pieces_of(spec, x0, x1, zbot):
    """Disjoint painted pieces (name, rho, polygon) that tile the domain."""
    dom = box(x0, zbot, x1, 0.0)
    pieces, taken = [], None
    for name, rho, poly in reversed(spec["shapes"]):
        p = poly.intersection(dom)
        if taken is not None:
            p = p.difference(taken)
        if not p.is_empty and p.area > 1e-9:
            pieces.append((name, rho, p))
        taken = poly.intersection(dom) if taken is None else unary_union([taken, poly.intersection(dom)])
    bg = dom.difference(taken) if taken is not None else dom
    pieces.append(("bg", spec["bg"], bg))
    return pieces


def build_true_mesh(survey, spec, wing=5.0, depth_factor=4.0, grad=1.12):
    sp = float(survey.spacing)
    xe = np.asarray(survey.electrode_x, float)
    x0, x1 = xe.min() - wing * sp, xe.max() + wing * sp
    zbot = -depth_factor * float(survey.z_max)
    h0 = sp / 4.0
    xs = np.unique(np.concatenate([np.arange(x0, x1 + 1e-9, sp / 2.0), xe]))
    pieces = pieces_of(spec, x0, x1, zbot)
    lines = [LineString([(x, 0.0) for x in xs])]
    for _, _, g in pieces:
        for poly in getattr(g, "geoms", [g]):
            lines.append(LineString(poly.exterior.coords))
            lines += [LineString(r.coords) for r in poly.interiors]
    noded = unary_union(lines)
    verts, key, segs = [], {}, []

    def vid(p):
        k = (round(p[0], 7), round(p[1], 7))
        if k not in key:
            key[k] = len(verts)
            verts.append(k)
        return key[k]

    for ln in getattr(noded, "geoms", [noded]):
        c = list(ln.coords)
        for a, b in zip(c[:-1], c[1:]):
            ia, ib = vid(a), vid(b)
            if ia != ib:
                segs.append((ia, ib))
    h, z = h0, -h0
    pts = []
    while z > zbot + h:
        for x in np.arange(x0 + h / 2.0, x1 - h / 4.0, h):
            if noded.distance(Point(x, z)) > 0.25 * h:
                pts.append((x, z))
        h *= grad
        z -= h
    for p in pts:
        verts.append((float(p[0]), float(p[1])))
    regions, rho_of = [], []
    for k, (_, rho, g) in enumerate(pieces):
        for poly in getattr(g, "geoms", [g]):
            q = poly.representative_point()
            regions.append([q.x, q.y, k, 0])
        rho_of.append(rho)
    out = tr.triangulate({"vertices": np.array(verts), "segments": np.array(segs), "regions": np.array(regions)}, "pq28A")
    P = np.ascontiguousarray(out["vertices"].T, float)
    T = np.ascontiguousarray(out["triangles"].T)
    attr = out["triangle_attributes"].ravel().astype(int)
    rho = np.asarray(rho_of)[attr]
    return P, T, rho, np.column_stack([xe, np.zeros_like(xe)])


def true_log10(spec, X, Z):
    rho = np.full(np.shape(X), spec["bg"], float)
    for _, r, poly in spec["shapes"]:
        rho = np.where(shapely.contains_xy(poly, X, Z), r, rho)
    return np.log10(rho)


def boundary_distance(spec, X, Z, x0, x1, zbot):
    pieces = pieces_of(spec, x0, x1, zbot)
    lines = []
    for _, _, g in pieces:
        for poly in getattr(g, "geoms", [g]):
            lines.append(LineString(poly.exterior.coords))
    b = unary_union(lines)
    return shapely.distance(b, shapely.points(X.ravel(), Z.ravel())).reshape(X.shape)


def synth_data(survey, spec, seeds):
    from ert2d.fem.forward import FemSpace
    from synthetic_truth import add_noise
    P, T, rho, elec = build_true_mesh(survey, spec)
    d = np.linalg.norm(P.T[:, None, :] - elec[None], axis=2)
    en = d.argmin(axis=0)
    assert np.all(d[en, np.arange(len(elec))] < 1e-6), "electrodes must be mesh nodes"
    space = FemSpace.build(P, T, en, 1.0 / survey.z_max)
    rhoa = space.forward(1.0 / rho, survey.quad).rhoa
    out = []
    for s in seeds:
        sv = dataclasses.replace(survey, rhoa=add_noise(rhoa, spec["noise"], np.random.default_rng(s)))
        sv.__post_init__()
        out.append(sv)
    return out, rhoa


# --------------------------------------------------------------------------------------- parameterizations
def core_n(pm, sv):
    from ert2d.mesh import core_region_mask
    return int(core_region_mask(pm, sv).sum())


def build_free(sv, div=1.0, grad=1.15, rect_lf=1.15):
    """Natural settings: Voronoi cells and triangles from the same mesh (surface cells spacing/div, grading ``grad``),
    rectangular grids with one and two blocks per electrode spacing and layer factor ``rect_lf``."""
    from ert2d.mesh import build_param_mesh
    from ert2d.mesh.parameterize import prolong_padding, rect_parameterization
    kw = dict(wing=3.0, h_grad=grad, surface_h=sv.spacing / div, padding="prolong")
    pms = {"voronoi": build_param_mesh(sv, kind="voronoi", **kw), "triangle": build_param_mesh(sv, kind="triangle", **kw)}
    for name, bps in (("rect1", 1), ("rect2", 2)):
        pms[name] = prolong_padding(rect_parameterization(sv, blocks_per_spacing=bps, layer_factor=rect_lf), sv)
    return pms


def build_equal(sv, div=1.0, grad=1.15):
    """Triangular surface size (three grading rates) and rectangular layer factors tuned to the Voronoi count."""
    from ert2d.mesh import build_param_mesh
    from ert2d.mesh.parameterize import prolong_padding, rect_parameterization
    v = build_param_mesh(sv, kind="voronoi", wing=3.0, h_grad=grad, surface_h=sv.spacing / div, padding="prolong")
    target = core_n(v, sv)
    best = None
    for hg in (1.15, 1.3, 1.5):
        for d in np.exp(np.linspace(np.log(0.12), np.log(24.0), 100)):
            pm = build_param_mesh(sv, kind="triangle", surface_h=sv.spacing / d, wing=3.0, h_grad=hg, padding="prolong")
            n = core_n(pm, sv)
            if best is None or abs(n - target) < abs(best[0] - target) - 2:
                best = (n, pm)
    out = {"voronoi": v, "triangle": best[1]}
    for name, bps in (("rect1", 1), ("rect2", 2)):
        bl = None
        for lf in np.round(np.arange(1.0, 2.5, 0.005), 4):
            pm = rect_parameterization(sv, blocks_per_spacing=bps, layer_factor=float(lf))
            n = core_n(pm, sv)
            if bl is None or abs(n - target) < abs(bl[0] - target):
                bl = (n, float(lf))
        out[name] = prolong_padding(rect_parameterization(sv, blocks_per_spacing=bps, layer_factor=bl[1]), sv)
    return out


LEVELS = [  # joint refinement of surface spacing and depth grading
    dict(div=1.0, grad=1.30, bps=1, lf=1.30),
    dict(div=1.5, grad=1.20, bps=2, lf=1.20),
    dict(div=2.5, grad=1.12, bps=3, lf=1.12),
    dict(div=4.0, grad=1.08, bps=4, lf=1.07),
]


def build_level(sv, lev):
    from ert2d.mesh import build_param_mesh
    from ert2d.mesh.parameterize import prolong_padding, rect_parameterization
    kw = dict(wing=3.0, h_grad=lev["grad"], surface_h=sv.spacing / lev["div"], padding="prolong")
    return {"voronoi": build_param_mesh(sv, kind="voronoi", **kw), "triangle": build_param_mesh(sv, kind="triangle", **kw),
            "rect": prolong_padding(rect_parameterization(sv, blocks_per_spacing=lev["bps"], layer_factor=lev["lf"]), sv)}


# --------------------------------------------------------------------------------------- evaluation
def evaluate(spec, sv, pm, rho_cell):
    from fair import grid_field
    sp = float(sv.spacing)
    xs, zs, F, ok = grid_field(pm, rho_cell, sv, dx=0.25 * sp, dz=0.1 * sp)
    X, Z = np.meshgrid(xs, zs)
    T = true_log10(spec, X, Z)
    xe = np.asarray(sv.electrode_x, float)
    dist = boundary_distance(spec, X, Z, xe.min() - 5 * sp, xe.max() + 5 * sp, -4 * float(sv.z_max))
    diff = F - T
    near = dist < 0.75 * sp
    res = dict(err=float(np.sqrt(np.nanmean(diff ** 2))), err_edge=float(np.sqrt(np.nanmean(diff[near] ** 2))),
               err_flat=float(np.sqrt(np.nanmean(diff[~near] ** 2))))
    for name, kind, rho_true in spec["objects"]:
        poly = next(p for n, _, p in spec["shapes"] if n == name)
        inside = shapely.contains_xy(poly, X, Z)
        if inside.any():
            v = 10 ** F[inside]
            res["obj_" + name] = float(np.nanmin(v) if kind == "min" else np.nanmax(v))
    return res


def run_inversion(sv, pm, spec):
    from ert2d.inversion import invert
    t0 = time.perf_counter()
    res = invert(sv, pm, max_iter=40, tol=0.0, target_rms=100.0 * spec["noise"])
    return res, time.perf_counter() - t0


def task(args):
    model, mode, seed, store, level = args
    from ert2d.io.dat_reader import read_dat
    from ert2d.viz.sections import flatten_cell_polys
    spec = SPECS[model]()
    g = read_dat(str(geometry_file(spec)))
    (sv,), _ = synth_data(g, spec, [seed])
    if mode == "equal":
        pms = build_equal(sv, spec["div"])
    elif mode == "free":
        pms = build_free(sv, spec["div"])
    else:
        pms = build_level(sv, LEVELS[level])
    out, models = [], {}
    for kind, pm in pms.items():
        res, t = run_inversion(sv, pm, spec)
        row = dict(model=model, mode=mode, level=level, seed=seed, kind=kind, n=core_n(pm, sv), n_elem=int(pm.n_tris), rms=float(res.misfit),
                   iters=len(res.history), t=t)
        row.update(evaluate(spec, sv, pm, res.rho_cell))
        out.append(row)
        if store:
            pv, ci = flatten_cell_polys(pm.cell_polys)
            models[kind] = dict(poly_verts=pv, cell_idx=ci, rho_cell=np.asarray(res.rho_cell, float), n=row["n"], err=row["err"])
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return out, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["step", "basin", "arch"])
    ap.add_argument("--mode", nargs="+", default=["equal", "free"])
    ap.add_argument("--converge", action="store_true")
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    write_all_geometries(a.models)
    jobs = []
    for m in a.models:
        if a.converge:
            for lev in a.levels:
                for s in a.seeds:
                    jobs.append((m, "level", s, s == a.seeds[0], lev))
        else:
            for mode in a.mode:
                for s in a.seeds:
                    jobs.append((m, mode, s, s == a.seeds[0], 0))
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(task, jobs))
    rows = [r for part, _ in res for r in part]
    models = {(j[0], j[1], j[4]): m for j, (_, m) in zip(jobs, res) if m}
    pickle.dump(dict(rows=rows, models=models, specs={k: SPECS[k]() for k in a.models}), open(a.out / "suite.pkl", "wb"))
    keys = sorted({(r["model"], r["mode"], r["level"]) for r in rows})
    for key in keys:
        print(key)
        for kind in dict.fromkeys(r["kind"] for r in rows):
            s = [r for r in rows if (r["model"], r["mode"], r["level"]) == key and r["kind"] == kind]
            if s:
                print(f"  {kind:8s} n={s[0]['n']:5d} elem {s[0]['n_elem']:5d} err {np.mean([r['err'] for r in s]):.4f} edge {np.mean([r['err_edge'] for r in s]):.4f}"
                      f" flat {np.mean([r['err_flat'] for r in s]):.4f} ± {np.std([r['err'] for r in s], ddof=1) if len(s) > 1 else 0:.4f} t {np.mean([r['t'] for r in s]):.1f}s")


if __name__ == "__main__":
    main()
