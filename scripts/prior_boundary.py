"""Use of a known boundary (e.g. from GPR, a borehole or an excavation): generators in mirrored pairs along the boundary, so that
cell edges follow it, and a reduced smoothing weight on the links that cross it.

Variants per model (all with the same data):
  base      : uniform base mesh, ordinary smoothness (reference)
  weights   : base mesh, links crossing the boundary down-weighted (rectangular grid: staircase)
  aligned   : mirrored generator pairs along the boundary + down-weighted crossing links (Voronoi, triangular)
  shifted   : as aligned, but the boundary used is displaced by (0.5, -0.4) electrode spacings (wrong prior)

    python scripts/prior_boundary.py --models step basin blob arch emb --seeds 42 1 2 3 4 --out ../../results_v24/prior
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("ERT_PADDING", "prolong")
os.environ.setdefault("ERT_REG", "energy")
os.environ.setdefault("ERT_CORE_RULE", "intersect")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from shapely import affinity
from shapely.geometry import MultiLineString, box
from shapely.ops import unary_union

import model_suite as ms

BARRIER_W = 0.1


def known_boundary(spec, sv):
    xe = np.asarray(sv.electrode_x, float)
    sp = float(sv.spacing)
    core = box(xe.min(), -float(sv.z_max), xe.max(), -0.05 * sp)
    lines = unary_union([poly.exterior for _, _, poly in spec["shapes"]]).intersection(core)
    return lines


def mirrored_points(lines, sp, ds=1.0, off=0.35):
    pts = []
    for ln in getattr(lines, "geoms", [lines]):
        if ln.length < 1e-6:
            continue
        n = max(int(ln.length / (ds * sp)), 1)
        for t in (np.arange(n) + 0.5) / n:
            a, b = ln.interpolate(t * ln.length), ln.interpolate(min(t * ln.length + 0.05 * sp, ln.length))
            tx, tz = b.x - a.x, b.y - a.y
            nrm = np.hypot(tx, tz)
            if nrm < 1e-9:
                continue
            nx, nz = -tz / nrm, tx / nrm
            pts += [(a.x + off * sp * nx, a.y + off * sp * nz), (a.x - off * sp * nx, a.y - off * sp * nz)]
    return np.array(pts)


def invert_with(sv, pm, spec, kind, barrier):
    from ert2d.inversion import invert
    from ert2d.inversion.regularization import fv_energy_laplacian
    import time
    pts = pm.seeds if kind == "voronoi" else None
    L = fv_energy_laplacian(pm.cell_polys, pts, barrier=barrier, barrier_weight=BARRIER_W if barrier is not None else 1.0)
    t0 = time.perf_counter()
    res = invert(sv, pm, C=L, roughness_form="C", max_iter=40, tol=0.0, target_rms=100.0 * spec["noise"])
    return res, time.perf_counter() - t0


def task(args):
    model, seed, store = args
    from ert2d.io.dat_reader import read_dat
    from ert2d.mesh import build_param_mesh
    from ert2d.mesh.parameterize import prolong_padding, rect_parameterization
    from ert2d.viz.sections import flatten_cell_polys
    spec = ms.SPECS[model]()
    g = read_dat(str(ms.geometry_file(spec)))
    (sv,), _ = ms.synth_data(g, spec, [seed])
    sp = float(sv.spacing)
    lines = known_boundary(spec, sv)
    wrong = affinity.translate(lines, 0.5 * sp, -0.4 * sp)
    pairs, pairs_wrong = mirrored_points(lines, sp), mirrored_points(wrong, sp)
    kw = dict(wing=3.0, h_grad=1.15, surface_h=sp / spec["div"], padding="prolong")
    out, models = [], {}

    def record(variant, kind, pm, res, t):
        row = dict(model=model, seed=seed, variant=variant, kind=kind, n=ms.core_n(pm, sv), n_elem=int(pm.n_tris), rms=float(res.misfit), t=t)
        row.update(ms.evaluate(spec, sv, pm, res.rho_cell))
        out.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        if store:
            pv, ci = flatten_cell_polys(pm.cell_polys)
            models[(variant, kind)] = dict(poly_verts=pv, cell_idx=ci, rho_cell=np.asarray(res.rho_cell, float), n=row["n"], err=row["err"])

    for kind in ("voronoi", "triangle"):
        pm0 = build_param_mesh(sv, kind=kind, **kw)
        record("base", kind, pm0, *invert_with(sv, pm0, spec, kind, None))
        record("weights", kind, pm0, *invert_with(sv, pm0, spec, kind, lines))
        pm1 = build_param_mesh(sv, kind=kind, extra_points=pairs, smooth_iters=0, **kw)
        record("aligned", kind, pm1, *invert_with(sv, pm1, spec, kind, lines))
        pm2 = build_param_mesh(sv, kind=kind, extra_points=pairs_wrong, smooth_iters=0, **kw)
        record("shifted", kind, pm2, *invert_with(sv, pm2, spec, kind, wrong))
    for name, bps in (("rect1", 1), ("rect2", 2)):
        pm = prolong_padding(rect_parameterization(sv, blocks_per_spacing=bps, layer_factor=1.15), sv)
        record("base", name, pm, *invert_with(sv, pm, spec, name, None))
        record("weights", name, pm, *invert_with(sv, pm, spec, name, lines))
        record("shifted", name, pm, *invert_with(sv, pm, spec, name, wrong))
    return out, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["step", "basin", "blob", "arch", "emb"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ms.write_all_geometries(a.models)
    jobs = [(m, s, s == a.seeds[0]) for m in a.models for s in a.seeds]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(task, jobs))
    rows = [r for part, _ in res for r in part]
    models = {(j[0],) + k: v for j, (_, m) in zip(jobs, res) if m for k, v in m.items()}
    pickle.dump(dict(rows=rows, models=models, specs={k: ms.SPECS[k]() for k in a.models}), open(a.out / "prior.pkl", "wb"))
    for m in a.models:
        print(m)
        for variant in ("base", "weights", "aligned", "shifted"):
            for kind in ("voronoi", "triangle", "rect1", "rect2"):
                s = [x for x in rows if x["model"] == m and x["variant"] == variant and x["kind"] == kind]
                if s:
                    print(f"  {variant:8s} {kind:8s} n={int(np.mean([x['n'] for x in s])):5d} err {np.mean([x['err'] for x in s]):.4f} "
                          f"edge {np.mean([x['err_edge'] for x in s]):.4f} flat {np.mean([x['err_flat'] for x in s]):.4f}")


if __name__ == "__main__":
    main()
