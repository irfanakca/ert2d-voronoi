"""Directional bias of the gradient-energy operators actually used in the inversions.

Response of the roughness energy to plane waves cos(k.x + phase) of fixed wavelength and
varying propagation angle: R(phi) = sqrt(mean_i u_i (L u)_i) over interior cells (phase
averaged); u^T L u = sum_links w_ij (u_i - u_j)^2 is the discrete  integral |grad u|^2.
Anisotropy index = (max R - min R) / mean R.

    python scripts/operator_isotropy2.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("ERT_PADDING", "prolong")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from fair import tuned_builder
from ert2d.inversion.regularization import fv_energy_laplacian, grid_laplacian
from ert2d.io.dat_reader import read_dat
from synthetic_truth import DEFAULT_GEOMETRY

OUT = Path(__file__).resolve().parents[1] / "results" / "operator_isotropy.json"
OUT.parent.mkdir(parents=True, exist_ok=True)


def operators(g, n=432):
    build, _ = tuned_builder(g, n)
    out = {}
    for kind in ("voronoi", "triangle", "rect"):
        pm = build(g, kind)
        cent = np.array([[p.centroid.x, p.centroid.y] for p in pm.cell_polys])
        if kind == "rect":
            out["rect (unit weights)"] = (grid_laplacian(pm.n_rows, pm.n_cols), cent, 4)
            out["rect (metric weights)"] = (fv_energy_laplacian(pm.cell_polys), cent, 4)
        elif kind == "triangle":
            out["triangle"] = (fv_energy_laplacian(pm.cell_polys), cent, 3)
        else:
            out["voronoi (finite volume, seeds)"] = (fv_energy_laplacian(pm.cell_polys, pm.seeds), cent, 5)
            out["voronoi (finite volume, centroids)"] = (fv_energy_laplacian(pm.cell_polys), cent, 5)
    return out


def main():
    g = read_dat(DEFAULT_GEOMETRY)
    ops = operators(g)
    phis = np.deg2rad(np.arange(0.0, 180.0, 7.5))
    lams = (6.0, 10.0, 20.0)
    result = {}
    for name, (L, cent, nmin) in ops.items():
        nb = np.asarray((abs(L) > 1e-12).sum(1)).ravel() - 1
        interior = (cent[:, 0] > 10) & (cent[:, 0] < 62) & (cent[:, 1] > -5.5) & (cent[:, 1] < -0.6) & (nb >= nmin)
        row = [f"{name:36s} cells {interior.sum():3d}"]
        for lam in lams:
            k = 2 * np.pi / lam
            R = []
            for phi in phis:
                pw = 0.0
                for delta in (0.0, np.pi / 2):
                    u = np.cos(k * np.cos(phi) * cent[:, 0] + k * np.sin(phi) * cent[:, 1] + delta)
                    pw += float(np.mean((u * (L @ u))[interior]))
                R.append(np.sqrt(max(pw / 2, 0.0)))
            R = np.array(R)
            idx = float((R.max() - R.min()) / R.mean())
            result.setdefault(name, {})[str(int(lam))] = idx
            row.append(f"{lam:4.0f} m: {idx:.2f}")
        print("  ".join(row))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
