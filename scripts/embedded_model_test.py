"""Trial: a layered model with embedded bodies (cf. manuscript Fig. 4c) and a comprehensive data set.

Model (50 m profile, 51 electrodes at 1 m): 100 ohm.m layer with an undulating base at 3.0-4.6 m depth above
a 10000 ohm.m basement; inside the layer a conductive rectangle (10 ohm.m) and a resistive circular body (500 ohm.m)
whose lower boundaries lie well inside the investigation depth.  Data: Wenner (a = 1..12), Wenner-Schlumberger
(a = 1, n = 1..14) and dipole-dipole (a = 1, n = 1..8) readings, 3% noise.

    $env:ERT_REG="energy"; python scripts/embedded_model_test.py --out ../../results_v19/emb
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
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import triangle as tr
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

RHO_LAYER, RHO_COND, RHO_RES, RHO_BASE = 100.0, 10.0, 1000.0, 10000.0
RECT = (11.0, 17.0, -2.5, -1.0)                 # xmin, xmax, zmin, zmax (conductive)
CIRC = (33.0, -1.8, 1.2)                        # centre x, centre z, radius (resistive)
GEOM = Path(__file__).resolve().parents[1] / "sample_data" / "synthetic_emb.dat"


def interface_z(x):
    x = np.asarray(x, float)
    return -(3.8 + 0.8 * np.cos(2 * np.pi * (x - 14.0) / 18.0))


def true_rho(x, z):
    x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
    rho = np.where(z < interface_z(x), RHO_BASE, RHO_LAYER)
    rho = np.where((x >= RECT[0]) & (x <= RECT[1]) & (z >= RECT[2]) & (z <= RECT[3]), RHO_COND, rho)
    rho = np.where((x - CIRC[0]) ** 2 + (z - CIRC[1]) ** 2 <= CIRC[2] ** 2, RHO_RES, rho)
    return rho


def write_geometry(path=GEOM, n_el=51, spacing=1.0):
    rows = []
    x = lambda i: i * spacing                                   # noqa: E731
    for a in range(1, 13):                                       # Wenner  C1 P1 P2 C2
        for i in range(0, n_el - 3 * a):
            rows.append((x(i), x(i + 3 * a), x(i + a), x(i + 2 * a)))
    for n in range(1, 15):                                       # Wenner-Schlumberger, a = 1
        for i in range(0, n_el - 2 * n - 1):
            rows.append((x(i), x(i + 2 * n + 1), x(i + n), x(i + n + 1)))
    for n in range(1, 9):                                        # dipole-dipole, a = 1
        for i in range(1, n_el - n - 1):
            rows.append((x(i), x(i - 1), x(i + n), x(i + n + 1)))
    with open(path, "w") as f:
        f.write("Embedded-body test geometry\n1.0\n11\n0\nType of measurement (0=apparent resistivity,1=resistance)\n0\n")
        f.write(f"{len(rows)}\n1\n0\n")
        for c1, c2, p1, p2 in rows:
            f.write(f"4 {c1:.3f} 0 {c2:.3f} 0 {p1:.3f} 0 {p2:.3f} 0 100.0\n")
    return len(rows)


def build_true_mesh(survey, wing=5.0, depth_factor=4.0, grad=1.12):
    sp = float(survey.spacing)
    xe = np.asarray(survey.electrode_x, float)
    x0, x1 = xe.min() - wing * sp, xe.max() + wing * sp
    zbot = -depth_factor * float(survey.z_max)
    h0 = sp / 4.0
    xs = np.unique(np.concatenate([np.arange(x0, x1 + 1e-9, sp / 2.0), xe]))
    xi = np.arange(x0, x1 + 1e-9, 0.25)
    verts, segs = [], []

    def add(p):
        verts.append(p); return len(verts) - 1

    top = [add((x, 0.0)) for x in xs]
    iface = [add((x, float(interface_z(x)))) for x in xi]
    br = add((x1, zbot)); bl = add((x0, zbot))
    segs += [(a, b) for a, b in zip(top[:-1], top[1:])]
    segs += [(top[-1], iface[-1]), (iface[-1], br), (br, bl), (bl, iface[0]), (iface[0], top[0])]
    segs += [(a, b) for a, b in zip(iface[:-1], iface[1:])]
    bx0, bx1, bz0, bz1 = RECT
    c = [add((bx0, bz0)), add((bx1, bz0)), add((bx1, bz1)), add((bx0, bz1))]
    segs += [(c[i], c[(i + 1) % 4]) for i in range(4)]
    th = np.linspace(0, 2 * np.pi, 49)[:-1]
    ring = [add((CIRC[0] + CIRC[2] * np.cos(t), CIRC[1] + CIRC[2] * np.sin(t))) for t in th]
    segs += [(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]
    constrained = unary_union([LineString([verts[a], verts[b]]) for a, b in segs])
    h, z = h0, -h0
    pts = []
    while z > zbot + h:
        for x in np.arange(x0 + h / 2.0, x1 - h / 4.0, h):
            if constrained.distance(Point(x, z)) > 0.25 * h:
                pts.append((x, z))
        h *= grad
        z -= h
    for p in pts:
        add(p)
    regions = [[x0 + 1.0, -0.15, 1, 0], [0.5 * (x0 + x1), zbot + 2.0, 2, 0],
               [0.5 * (bx0 + bx1), 0.5 * (bz0 + bz1), 3, 0], [CIRC[0], CIRC[1], 4, 0]]
    out = tr.triangulate({"vertices": np.array(verts), "segments": np.array(segs), "regions": regions}, "pq28A")
    P = np.ascontiguousarray(out["vertices"].T, float)
    T = np.ascontiguousarray(out["triangles"].T)
    attr = out["triangle_attributes"].ravel().astype(int)
    rho = np.select([attr == 1, attr == 2, attr == 3, attr == 4], [RHO_LAYER, RHO_BASE, RHO_COND, RHO_RES])
    return P, T, rho, np.column_stack([xe, np.zeros_like(xe)])


def synth_data(survey, seeds, noise):
    from ert2d.fem.forward import FemSpace
    from synthetic_truth import add_noise
    P, T, rho, elec = build_true_mesh(survey)
    d = np.linalg.norm(P.T[:, None, :] - elec[None], axis=2)
    en = d.argmin(axis=0)
    assert np.all(d[en, np.arange(len(elec))] < 1e-6), "electrodes must be mesh nodes"
    space = FemSpace.build(P, T, en, 1.0 / survey.z_max)
    rhoa = space.forward(1.0 / rho, survey.quad).rhoa
    out = []
    for s in seeds:
        sv = dataclasses.replace(survey, rhoa=add_noise(rhoa, noise, np.random.default_rng(s)))
        sv.__post_init__()
        out.append(sv)
    return out, rhoa


def build_free(sv):
    """Each scheme at its natural settings, number of unknowns NOT matched: Voronoi and triangular cells from the
    same mesh (surface cells spacing/4, grading 1.15, margin 3 spacings), rectangular grids with the default
    layer factor 1.1 and two or four columns per electrode spacing."""
    from ert2d.mesh import build_param_mesh
    from ert2d.mesh.parameterize import prolong_padding, rect_parameterization
    kw = dict(wing=3.0, h_grad=1.15, surface_h=sv.spacing / float(os.environ.get("ERT_DIV", "4")), padding="prolong")
    pms = {"voronoi": build_param_mesh(sv, kind="voronoi", **kw), "triangle": build_param_mesh(sv, kind="triangle", **kw)}
    bps = int(os.environ.get("ERT_FREE_BPS", "2"))
    lf = float(os.environ.get("ERT_FREE_LF", "1.1"))
    pms["rect2"] = prolong_padding(rect_parameterization(sv, blocks_per_spacing=bps, layer_factor=lf), sv)
    return pms


def build_equal(sv):
    """Voronoi mesh as in build_free; triangular surface size and rectangular layer factor tuned to the same count."""
    from dense_compare import core_n
    from ert2d.mesh import build_param_mesh
    from ert2d.mesh.parameterize import prolong_padding, rect_parameterization
    kw = dict(wing=3.0, h_grad=1.15, padding="prolong")
    v = build_param_mesh(sv, kind="voronoi", surface_h=sv.spacing / float(os.environ.get("ERT_DIV", "4")), **kw)
    target = core_n(v, sv)
    best = None
    for hg in (1.15, 1.3, 1.5):
        for d in np.exp(np.linspace(np.log(0.12), np.log(24.0), 100)):
            pm = build_param_mesh(sv, kind="triangle", surface_h=sv.spacing / d, wing=3.0, h_grad=hg, padding="prolong")
            n = core_n(pm, sv)
            if best is None or abs(n - target) < abs(best[0] - target) - 2:
                best = (n, pm)
    bps = int(os.environ.get("ERT_FREE_BPS", "2"))
    bl = None
    for lf in np.round(np.arange(1.0, 2.5, 0.005), 4):
        pm = rect_parameterization(sv, blocks_per_spacing=bps, layer_factor=float(lf))
        n = core_n(pm, sv)
        if bl is None or abs(n - target) < abs(bl[0] - target):
            bl = (n, float(lf))
    r = prolong_padding(rect_parameterization(sv, blocks_per_spacing=bps, layer_factor=bl[1]), sv)
    return {"voronoi": v, "triangle": best[1], "rect2": r}


def task(args):
    seed, store, mode = args
    from dense_compare import build_all, core_n
    from fair import grid_field, roughness_metrics
    from ert2d.inversion import invert
    from ert2d.io.dat_reader import read_dat
    from ert2d.viz.sections import flatten_cell_polys
    g = read_dat(str(GEOM))
    (sv,), _ = synth_data(g, [seed], 0.03)
    pms = build_free(g) if mode == "free" else build_equal(g)
    out, models = [], {}
    for kind in ("voronoi", "triangle", "rect2"):
        pm = pms[kind]
        t0 = time.perf_counter()
        res = invert(sv, pm, max_iter=40, tol=0.0, target_rms=3.0)
        t = time.perf_counter() - t0
        xs, zs, F, ok = grid_field(pm, res.rho_cell, sv, dx=0.25, dz=0.1)
        X, Z = np.meshgrid(xs, zs)
        err = float(np.sqrt(np.nanmean((F - np.log10(true_rho(X, Z))) ** 2)))
        inside_r = (X >= RECT[0]) & (X <= RECT[1]) & (Z >= RECT[2]) & (Z <= RECT[3])
        inside_c = (X - CIRC[0]) ** 2 + (Z - CIRC[1]) ** 2 <= CIRC[2] ** 2
        lay = Z > interface_z(X)
        e_lay = float(np.sqrt(np.nanmean(((F - np.log10(true_rho(X, Z)))[lay]) ** 2)))
        e_bas = float(np.sqrt(np.nanmean(((F - np.log10(true_rho(X, Z)))[~lay]) ** 2)))
        out.append(dict(seed=seed, kind=kind, n=core_n(pm, sv), n_elem=int(pm.n_tris), rms=float(res.misfit), iters=len(res.history), t=t,
                        err=err, err_layer=e_lay, err_base=e_bas, rect_min=float(np.nanmin(10 ** F[inside_r])), circ_max=float(np.nanmax(10 ** F[inside_c]))))
        if store:
            pv, ci = flatten_cell_polys(pm.cell_polys)
            models[kind] = dict(poly_verts=pv, cell_idx=ci, rho_cell=np.asarray(res.rho_cell, float), rms=float(res.misfit), n=core_n(pm, sv),
                                err=err)
        print(out[-1], flush=True)
    return out, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--mode", choices=["equal", "free"], default="equal")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    n = write_geometry()
    print("data:", n, "mode:", a.mode)
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(s, s == a.seeds[0], a.mode) for s in a.seeds]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(task, jobs))
    rows = [r for part, _ in res for r in part]
    models = next(m for _, m in res if m)
    pickle.dump(dict(rows=rows, models=models), open(a.out / "emb.pkl", "wb"))
    for k in ("voronoi", "triangle", "rect2"):
        s = [r for r in rows if r["kind"] == k]
        print(f"{k:9s} n={s[0]['n']:5d} elem {s[0]['n_elem']:5d} RMS {np.mean([r['rms'] for r in s]):.2f} err {np.mean([r['err'] for r in s]):.4f} layer {np.mean([r['err_layer'] for r in s]):.4f} base {np.mean([r['err_base'] for r in s]):.4f}"
              f" ± {np.std([r['err'] for r in s], ddof=1):.4f}  rect_min {np.mean([r['rect_min'] for r in s]):.0f}  circ_max {np.mean([r['circ_max'] for r in s]):.0f}"
              f"  t {np.mean([r['t'] for r in s]):.1f}s")


if __name__ == "__main__":
    main()
