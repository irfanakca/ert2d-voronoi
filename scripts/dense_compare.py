"""Dense regime: Voronoi with the production mesh settings (surface cells = spacing/8,
h_grad 1.15, wing 3 spacings; ``meshgen.m`` ops=1) against triangular and rectangular
parameterizations tuned to the SAME number of core unknowns.  One regularization
functional for every scheme (ERT_REG=energy).  Rect is run with 2 and 4 blocks per
electrode spacing (layer factor tuned to the unknown count).

    $env:ERT_REG="energy"; python scripts/dense_compare.py --out ../../results_v16/dense
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("ERT_REG", "energy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

SEEDS = (42, 1, 2, 3, 4)
MODELS = {"shallow_blocks": 0.03, "checkerboard": 0.02}
H_GRAD, WING = 1.15, 3.0
DIV = float(os.environ.get("ERT_DIV", 4.0))      # surface cells = spacing / DIV (fine regime: 4; MATLAB meshgen.m ops=1 uses 8)
KINDS = ("voronoi", "triangle", "rect2", "rect4")


def core_n(pm, sv):
    from ert2d.mesh import core_region_mask
    return int(core_region_mask(pm, sv).sum())


def build_all(sv, kinds=KINDS):
    from ert2d.mesh import build_param_mesh
    from ert2d.mesh.parameterize import prolong_padding, rect_parameterization
    kw = dict(wing=WING, h_grad=H_GRAD, padding="prolong")
    pms = {"voronoi": build_param_mesh(sv, kind="voronoi", surface_h=sv.spacing / DIV, **kw)}
    target = core_n(pms["voronoi"], sv)
    best = None
    for d in np.exp(np.linspace(np.log(0.8), np.log(24.0), 60)):
        pm = build_param_mesh(sv, kind="triangle", surface_h=sv.spacing / d, **kw)
        n = core_n(pm, sv)
        if best is None or abs(n - target) < abs(best[0] - target):
            best = (n, pm)
    pms["triangle"] = best[1]
    for name, bps in (("rect2", 2), ("rect4", 4)):
        b = None
        for lf in np.round(np.arange(1.0, 2.0, 0.005), 4):
            pm = rect_parameterization(sv, blocks_per_spacing=bps, layer_factor=float(lf))
            n = core_n(pm, sv)
            if b is None or abs(n - target) < abs(b[0] - target):
                b = (n, float(lf))
        pms[name] = prolong_padding(rect_parameterization(sv, blocks_per_spacing=bps, layer_factor=b[1]), sv)
    return pms, target


def task(args):
    model, seed = args
    from rough_analysis import _setup
    from fair import grid_field, roughness_metrics
    from ert2d.inversion import invert
    from ert2d.viz.sections import flatten_cell_polys
    g, sv, rho_fn = _setup(model, seed)
    pms, target = build_all(g)
    out, store = [], {}
    for kind in KINDS:
        pm = pms[kind]
        t0 = time.perf_counter()
        res = invert(sv, pm, max_iter=40, tol=0.0, target_rms=100 * MODELS[model])
        t = time.perf_counter() - t0
        xs, zs, F, ok = grid_field(pm, res.rho_cell, sv)
        X, Z = np.meshgrid(xs, zs)
        err = float(np.sqrt(np.nanmean((F - np.log10(rho_fn(X, Z))) ** 2)))
        rm = roughness_metrics(xs, zs, F, rho_fn)
        out.append(dict(model=model, seed=seed, kind=kind, n=int(pm.n_param), n_core=core_n(pm, sv), n_elem=int(pm.n_tris),
                        rms=float(res.misfit), iters=len(res.history), err=err, t_inv=t, **rm))
        if seed == 42:
            pv, ci = flatten_cell_polys(pm.cell_polys)
            store[kind] = dict(poly_verts=pv, cell_idx=ci, rho_cell=np.asarray(res.rho_cell, float), rms=float(res.misfit),
                               n=int(pm.n_param), err=err)
    return out, (model, seed, store)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(m, s) for m in MODELS for s in SEEDS]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(task, jobs))
    rows = [r for part, _ in res for r in part]
    json.dump(rows, open(a.out / "dense.json", "w"), indent=1)
    pickle.dump({(m, s): st for _, (m, s, st) in res if s == 42}, open(a.out / "dense_models.pkl", "wb"))
    for m in MODELS:
        print("==", m)
        for k in KINDS:
            s = [r for r in rows if r["model"] == m and r["kind"] == k]
            f = lambda key: (np.mean([r[key] for r in s]), np.std([r[key] for r in s], ddof=1))  # noqa: E731
            print(f"  {k:9s} n={s[0]['n_core']:5d} elem {s[0]['n_elem']:5d} RMS {f('rms')[0]:.2f} it {f('iters')[0]:4.1f} "
                  f"err {f('err')[0]:.4f}±{f('err')[1]:.4f} rough {f('rough_all')[0]:.3f} dev {f('dev_homog')[0]:.4f} t {f('t_inv')[0]:.0f}s")


if __name__ == "__main__":
    main()
