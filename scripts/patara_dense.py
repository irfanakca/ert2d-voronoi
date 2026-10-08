"""Patara A6 single-line example in the dense regime (production Voronoi mesh: surface
cells = spacing/8, h_grad 1.15, wing 3 spacings), against triangular and rectangular
parameterizations tuned to the same number of core unknowns.  Full-data inversion to the
target RMS plus 5-fold hold-out cross-validation.

    $env:ERT_REG="energy"; python scripts/patara_dense.py --line 20 --out ../../results_v16/patara_dense
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

ROOT = Path(__file__).resolve().parents[3]
KINDS = ("voronoi", "triangle", "rect2", "rect4")


def job(args):
    dat, kind, fold, nfolds, target = args
    from dense_compare import build_all, core_n
    from real_cv import subset
    from fair import grid_field
    from ert2d.inversion import invert
    from ert2d.io.dat_reader import read_dat
    from ert2d.viz.sections import flatten_cell_polys
    sv = read_dat(dat)
    if fold < 0:                                   # full-data inversion
        pm = build_all(sv)[0][kind]
        t0 = time.perf_counter()
        res = invert(sv, pm, max_iter=40, tol=0.0, target_rms=target)
        t = time.perf_counter() - t0
        pv, ci = flatten_cell_polys(pm.cell_polys)
        xs, zs, F, ok = grid_field(pm, res.rho_cell, sv, dx=0.25, dz=0.1)
        return dict(kind=kind, fold=fold, n=core_n(pm, sv), n_elem=int(pm.n_tris), rms=float(res.misfit),
                    iters=len(res.history), t=t, poly_verts=pv, cell_idx=ci, rho_cell=np.asarray(res.rho_cell, float),
                    grid=(xs, zs, F))
    rng = np.random.default_rng(0)
    perm = rng.permutation(sv.n_data)
    test = np.sort(perm[fold::nfolds])
    train = np.setdiff1d(np.arange(sv.n_data), test)
    s_tr, s_te = subset(sv, train), subset(sv, test)
    pm = build_all(s_tr)[0][kind]
    t0 = time.perf_counter()
    res = invert(s_tr, pm, max_iter=40, tol=0.0, target_rms=target)
    t = time.perf_counter() - t0
    pred = res.space.forward(res.sigma_elem, s_te.quad).rhoa
    held = float(np.sqrt(np.mean((np.log(s_te.rhoa) - np.log(pred)) ** 2)) * 100)
    return dict(kind=kind, fold=fold, n=core_n(pm, s_tr), train_rms=float(res.misfit), held_rms=held, t=t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", type=int, required=True)
    ap.add_argument("--target", type=float, default=2.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dat = str(ROOT / "voronoi_parameterization" / "patara" / f"A6_{a.line:02d}.dat")
    jobs = [(dat, k, f, 5, a.target) for k in KINDS for f in range(-1, 5)]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, jobs))
    full = [r for r in rows if r["fold"] < 0]
    cv = [r for r in rows if r["fold"] >= 0]
    pickle.dump(full, open(a.out / f"full_{a.line}.pkl", "wb"))
    json.dump(cv, open(a.out / f"cv_{a.line}.json", "w"), indent=1)
    print(f"line {a.line}")
    for k in KINDS:
        f = [r for r in full if r["kind"] == k][0]
        h = np.array([r["held_rms"] for r in cv if r["kind"] == k])
        tr = np.mean([r["train_rms"] for r in cv if r["kind"] == k])
        print(f"  {k:9s} n={f['n']:5d} elems {f['n_elem']:5d} RMS {f['rms']:.2f} it {f['iters']:2d} t {f['t']:.1f}s | "
              f"CV train {tr:.2f} held-out {h.mean():.2f} ± {h.std(ddof=1):.2f}")


if __name__ == "__main__":
    main()
