"""Adaptive refinement of the Voronoi (and triangular) parameterization.

An initial coarse mesh is inverted; the links of the cell graph (Delaunay edges between generators) with a large gradient of
ln(rho) are bisected (a node is added at the midpoint of the generators), the mesh and the Voronoi cells are rebuilt and the data
are inverted again.  The cells of the Voronoi scheme are the dual of the rebuilt mesh, so no further construction is needed;
a rectangular grid cannot be refined locally.

    python scripts/adaptive_refine.py --models emb blob arch step --seeds 42 1 2 --out ../../results_v24/adaptive
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
from scipy.spatial import Delaunay

import model_suite as ms

ROUNDS = 3
QUANT = 0.88        # bisect the edges whose |d ln rho| / distance is above this quantile
GRAD = 1.3          # depth-grading rate of the base meshes
BASE_DIV = 0.5      # surface cells of two electrode spacings in the initial mesh
UNIFORM_DIVS = (0.5, 0.7, 1.0, 1.4, 2.0)


def refine_points(pm, rho_cell, extra, quant=QUANT, max_new=400):
    seeds = np.asarray(pm.seeds, float)
    lr = np.log(np.asarray(rho_cell, float))
    tri = Delaunay(seeds)
    e = set()
    for s in tri.simplices:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            i, j = sorted((int(s[a]), int(s[b])))
            e.add((i, j))
    e = np.array(sorted(e))
    d = np.linalg.norm(seeds[e[:, 0]] - seeds[e[:, 1]], axis=1)
    keep = d < 3.0 * np.median(d)                       # drop long hull edges
    e, d = e[keep], d[keep]
    g = np.abs(lr[e[:, 0]] - lr[e[:, 1]]) / d
    sel = np.nonzero(g >= np.quantile(g, quant))[0]
    sel = sel[np.argsort(-g[sel])][:max_new]
    mid = 0.5 * (seeds[e[sel, 0]] + seeds[e[sel, 1]])
    return mid if extra is None else np.vstack([extra, mid])


def task(args):
    model, seed, store = args
    from ert2d.io.dat_reader import read_dat
    from ert2d.mesh import build_param_mesh
    from ert2d.viz.sections import flatten_cell_polys
    spec = ms.SPECS[model]()
    g = read_dat(str(ms.geometry_file(spec)))
    (sv,), _ = ms.synth_data(g, spec, [seed])
    out, models = [], {}
    for kind in ("voronoi", "triangle"):                       # uniform refinement sweep (same grading), for reference
        for div in UNIFORM_DIVS:
            pm = build_param_mesh(sv, kind=kind, wing=3.0, h_grad=GRAD, surface_h=sv.spacing / div, padding="prolong", smooth_iters=10)
            res, t = ms.run_inversion(sv, pm, spec)
            row = dict(model=model, seed=seed, kind=kind, round=-1, div=div, n=ms.core_n(pm, sv), n_elem=int(pm.n_tris), rms=float(res.misfit), t=t)
            row.update(ms.evaluate(spec, sv, pm, res.rho_cell))
            out.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    for kind in ("voronoi", "triangle"):
        extra = None
        for r in range(ROUNDS + 1):
            kw = dict(wing=3.0, h_grad=GRAD, surface_h=sv.spacing / BASE_DIV, padding="prolong", smooth_iters=10)
            if extra is not None:
                kw["extra_points"] = extra
            pm = build_param_mesh(sv, kind=kind, **kw)
            res, t = ms.run_inversion(sv, pm, spec)
            row = dict(model=model, seed=seed, kind=kind, round=r, n=ms.core_n(pm, sv), n_elem=int(pm.n_tris), rms=float(res.misfit), t=t)
            row.update(ms.evaluate(spec, sv, pm, res.rho_cell))
            out.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            if store and r in (0, ROUNDS):
                pv, ci = flatten_cell_polys(pm.cell_polys)
                models[(kind, r)] = dict(poly_verts=pv, cell_idx=ci, rho_cell=np.asarray(res.rho_cell, float), n=row["n"], err=row["err"])
            if r < ROUNDS:
                extra = refine_points(pm, res.rho_cell, extra)
    return out, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["emb", "blob", "arch", "step"])
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
    pickle.dump(dict(rows=rows, models=models, specs={k: ms.SPECS[k]() for k in a.models}), open(a.out / "adaptive.pkl", "wb"))
    for m in a.models:
        for kind in ("voronoi", "triangle"):
            for r in range(ROUNDS + 1):
                s = [x for x in rows if x["model"] == m and x["kind"] == kind and x["round"] == r]
                if s:
                    print(f"{m:6s} {kind:8s} round {r} n={int(np.mean([x['n'] for x in s])):5d} err {np.mean([x['err'] for x in s]):.4f} "
                          f"edge {np.mean([x['err_edge'] for x in s]):.4f} t {np.mean([x['t'] for x in s]):.1f}s")


if __name__ == "__main__":
    main()
