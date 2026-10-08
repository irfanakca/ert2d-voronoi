"""Real-data cross-validation (stability / predictive power) on an ERT profile.

k-fold hold-out: for each fold the model is inverted from 80% of the data (to a fixed
target RMS) and used to predict the 20% that were left out.  The held-out RMS measures
how well a scheme generalizes (a stable, well-regularized model predicts unseen data
better than one that has fitted noise).  Schemes at an equal number of unknowns, plus
a Voronoi model with ~1/1.8 of the unknowns (the cell count it has on the same mesh as
the triangular scheme).

    python scripts/real_cv.py --dat <file.dat> --target 6 --folds 5 --out <dir>
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

H_GRAD = 1.25


def subset(sv, idx):
    kw = dict(quad=sv.quad[idx], rhoa=sv.rhoa[idx])
    for name in ("n_level", "mn", "x_mid", "pseudo_depth", "k_geom"):
        v = getattr(sv, name, None)
        if v is not None:
            kw[name] = np.asarray(v)[idx]
    out = dataclasses.replace(sv, **kw)
    out.z_max = sv.z_max
    return out


def make_pm(sv, spec):
    from ert2d.mesh import build_param_mesh
    from ert2d.mesh.parameterize import prolong_padding, rect_parameterization
    if spec["kind"] in ("rect", "rect_metric"):
        return prolong_padding(rect_parameterization(sv, blocks_per_spacing=2, layer_factor=spec["lf"]), sv)
    return build_param_mesh(sv, kind=spec["kind"], surface_div=2, h_grad=H_GRAD,
                            surface_h=sv.spacing / spec["d"], padding="prolong")


def tune_specs(sv, n_target):
    from ert2d.mesh import core_region_mask
    from ert2d.mesh.parameterize import rect_parameterization
    from geo_scan import tune_tri
    n_rect = int(core_region_mask(rect_parameterization(sv, blocks_per_spacing=2), sv).sum())
    n_target = n_target or n_rect
    return n_rect, [
        dict(label="rect (default)", kind="rect", lf=1.1),
        dict(label="rect metric (default)", kind="rect_metric", lf=1.1),
        dict(label=f"triangle ({n_target})", kind="triangle", d=tune_tri(sv, "triangle", H_GRAD, n_target)),
        dict(label=f"voronoi ({n_target})", kind="voronoi", d=tune_tri(sv, "voronoi", H_GRAD, n_target)),
        dict(label=f"voronoi ({int(n_target / 1.8)})", kind="voronoi", d=tune_tri(sv, "voronoi", H_GRAD, int(n_target / 1.8))),
    ]


def fold_task(args):
    dat, spec, fold, nfolds, target, seed = args
    from ert2d.inversion import invert
    from ert2d.io.dat_reader import read_dat
    sv = read_dat(dat)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(sv.n_data)
    test = np.sort(perm[fold::nfolds])
    train = np.setdiff1d(np.arange(sv.n_data), test)
    s_tr, s_te = subset(sv, train), subset(sv, test)
    pm = make_pm(s_tr, spec)
    t0 = time.perf_counter()
    from fair import operator_for
    res = invert(s_tr, pm, max_iter=40, tol=0.0, target_rms=target, C=operator_for(pm, spec['kind']))
    t = time.perf_counter() - t0
    pred = res.space.forward(res.sigma_elem, s_te.quad).rhoa
    held = float(np.sqrt(np.mean((np.log(s_te.rhoa) - np.log(pred)) ** 2)) * 100)
    return dict(label=spec["label"], fold=fold, n=int(pm.n_param), train_rms=float(res.misfit),
                held_rms=held, iters=len(res.history), t=t, n_elem=int(pm.t2.shape[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dat", type=Path, required=True)
    ap.add_argument("--target", type=float, default=6.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n", type=int, default=0, help="common number of unknowns (0 = rect default)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from ert2d.io.dat_reader import read_dat
    sv = read_dat(a.dat)
    n_rect, specs = tune_specs(sv, a.n)
    print("rect default unknowns:", n_rect, "| specs:", [s["label"] for s in specs], flush=True)
    jobs = [(str(a.dat), sp, f, a.folds, a.target, 0) for sp in specs for f in range(a.folds)]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(fold_task, jobs))
    json.dump(rows, open(a.out / "cv.json", "w"), indent=1)
    for sp in specs:
        r = [x for x in rows if x["label"] == sp["label"]]
        h = np.array([x["held_rms"] for x in r])
        print(f"{sp['label']:20s} n={r[0]['n']:5d} train RMS {np.mean([x['train_rms'] for x in r]):.2f}  "
              f"held-out RMS {h.mean():.2f} ± {h.std(ddof=1):.2f}  iters {np.mean([x['iters'] for x in r]):.1f}  "
              f"t {np.mean([x['t'] for x in r]):.1f}s  elems {np.mean([x['n_elem'] for x in r]):.0f}")


if __name__ == "__main__":
    main()
