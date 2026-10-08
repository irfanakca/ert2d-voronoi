"""Equal-unknowns configuration helpers.

All main comparisons use the same number of *core* unknowns (cells whose centroid
lies in the data region, ``core_region_mask``) for the three parameterizations.
``tuned_builder`` finds, for one survey geometry, the mesh setting of each scheme
that gives the requested core count:

* voronoi / triangle - ``surface_h = spacing / d`` (``h_grad = 1.25``, ``surface_div = 2``),
  both built from the SAME node set for a given ``d`` (Voronoi seeds = mesh nodes);
  ``d`` is tuned separately per scheme because the triangular scheme has ~1.7x more
  cells than the Voronoi scheme on the same mesh;
* rect - 2 blocks per electrode spacing, depth-layer growth factor tuned
  (the number of columns is fixed by the electrode line).

Also provides ``fit_colorbar`` (colour bar exactly as tall as the axes it serves).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ert2d.mesh import build_param_mesh, core_region_mask          # noqa: E402
from ert2d.mesh.parameterize import prolong_padding, rect_parameterization   # noqa: E402
import os

PROLONG = os.environ.get('ERT_PADDING', 'prolong') == 'prolong'

H_GRAD = 1.25


def _core(pm, survey) -> int:
    return int(core_region_mask(pm, survey).sum())


def _tune(survey, kind, target):
    best = None
    if kind == "rect":
        for lf in np.round(np.arange(1.02, 3.0, 0.01), 3):
            pm = rect_parameterization(survey, blocks_per_spacing=2, layer_factor=float(lf))
            n = _core(pm, survey)
            if best is None or abs(n - target) < abs(best[0] - target):
                best = (n, dict(layer_factor=float(lf)))
        return best
    ds = np.exp(np.linspace(np.log(0.4), np.log(10.0), 36))
    cand = []
    for d in ds:
        pm = build_param_mesh(survey, kind=kind, surface_div=2, h_grad=H_GRAD,
                              surface_h=survey.spacing / d)
        cand.append((abs(_core(pm, survey) - target), float(d), _core(pm, survey)))
    cand.sort()
    d0 = cand[0][1]
    for d in np.linspace(d0 * 0.9, d0 * 1.1, 21):
        pm = build_param_mesh(survey, kind=kind, surface_div=2, h_grad=H_GRAD,
                              surface_h=survey.spacing / d)
        n = _core(pm, survey)
        if best is None or abs(n - target) < abs(best[0] - target):
            best = (n, dict(d=float(d)))
    return best


def tuned_builder(survey, target, kinds=("voronoi", "triangle", "rect"), fixed_rect=False):
    """Return (build(survey_like, kind) -> ParamMesh, settings dict).

    ``fixed_rect=True`` keeps the rect scheme at its conventional default
    (layer_factor 1.1) and tunes only voronoi/triangle to ``target``."""
    cfg = {}
    for kind in kinds:
        if kind == "rect" and fixed_rect:
            cfg[kind] = dict(layer_factor=1.1)
            continue
        n, kw = _tune(survey, kind, target)
        kw["core"] = n
        cfg[kind] = kw

    def build(sv, kind):
        k = cfg[kind]
        if kind == "rect":
            pm = rect_parameterization(sv, blocks_per_spacing=2, layer_factor=k["layer_factor"])
            return prolong_padding(pm, sv) if PROLONG else pm
        return build_param_mesh(sv, kind=kind, surface_div=2, h_grad=H_GRAD,
                                surface_h=sv.spacing / k["d"],
                                padding="prolong" if PROLONG else "free")
    return build, cfg


def operator_for(pm, kind):
    """Smoothness operator used for ``kind`` in the manuscript experiments.

    voronoi -> two-point-flux finite-volume Laplacian between generators
    (set ERT_VOR_OP=paper for the earlier area/inverse-distance operator);
    everything else -> the default of ``ert2d.inversion.invert`` (None)."""
    if kind == "rect_metric":                 # rect blocks, finite-volume (edge length / distance) weights
        from ert2d.inversion.regularization import voronoi_smoothness
        pm.kind = "rect_metric"
        return voronoi_smoothness(pm.cell_polys, edge_only=True)
    if kind == "voronoi" and os.environ.get("ERT_VOR_OP", "fv") == "fv":
        from ert2d.inversion.regularization import voronoi_fv_smoothness
        return voronoi_fv_smoothness(pm.cell_polys, pm.seeds)
    return None


def footprint_polygon(survey, widen=0.25):
    """Polygon of the part of the model section that is constrained by the data
    (lower corners cut along the data flank, see ert2d.viz.model_mask)."""
    from ert2d.viz.model_mask import model_footprint
    return model_footprint(survey.x_mid, survey.pseudo_depth, survey.electrode_x, survey.z_max, widen=widen)


def grid_field(pm, rho_cell, survey, dx=0.25, dz=0.1):
    """Recovered log10 resistivity on the common regular grid of the data region.
    Returns (xs, zs, F[z, x], valid mask)."""
    import matplotlib.tri as mtri
    ex = np.asarray(survey.electrode_x, float)
    xs = np.arange(ex.min() + dx / 2, ex.max(), dx)
    zs = -np.arange(dz / 2, float(survey.z_max), dz)
    X, Z = np.meshgrid(xs, zs)
    tri = mtri.Triangulation(pm.p2[0], pm.p2[1], pm.t2.T)
    elem = tri.get_trifinder()(X.ravel(), Z.ravel())
    owner = np.full(pm.t2.shape[1], -1, int)
    for k, els in enumerate(pm.cell_elements):
        owner[els] = k
    safe = np.clip(elem, 0, None)
    ok = (elem >= 0) & (owner[safe] >= 0)
    F = np.log10(np.asarray(rho_cell, float)[owner[safe]])
    F = np.where(ok, F, np.nan).reshape(X.shape)
    return xs, zs, F, ok.reshape(X.shape)


def roughness_metrics(xs, zs, F, rho_fn, margin=1.0):
    """Common-grid roughness of the recovered model: RMS gradient of log10(rho)
    (all grid points) and RMS gradient / RMS deviation from the truth inside the
    true-homogeneous zones, i.e. more than ``margin`` m from any true discontinuity."""
    from scipy import ndimage
    dx, dz = abs(xs[1] - xs[0]), abs(zs[1] - zs[0])
    X, Z = np.meshgrid(xs, zs)
    T = np.log10(rho_fn(X, Z))
    gz, gx = np.gradient(np.nan_to_num(F, nan=float(np.nanmean(F))), dz, dx)
    grad2 = gx ** 2 + gz ** 2
    tz, tx = np.gradient(T, dz, dx)
    jump = (tx ** 2 + tz ** 2) > 1e-12
    dist = ndimage.distance_transform_edt(~jump, sampling=(dz, dx))
    far = dist > margin
    return dict(rough_all=float(np.sqrt(np.mean(grad2))),
                rough_true=float(np.sqrt(np.mean(tx ** 2 + tz ** 2))),
                rough_homog=float(np.sqrt(np.mean(grad2[far]))) if far.any() else float("nan"),
                dev_homog=float(np.sqrt(np.mean((F - T)[far] ** 2))) if far.any() else float("nan"),
                frac_homog=float(far.mean()))


def grid_error(pm, rho_cell, rho_fn, survey, dx=0.25, dz=0.1) -> float:
    """Recovery error on a common regular grid covering the data region
    (electrode span x [0, z_max]): RMS of log10(recovered / true) with equal weight
    per unit area, so it does not depend on how many cells a scheme places where.
    The recovered value at a grid point is that of the cell owning the FEM element
    that contains the point."""
    import matplotlib.tri as mtri
    ex = np.asarray(survey.electrode_x, float)
    xs = np.arange(ex.min() + dx / 2, ex.max(), dx)
    zs = -np.arange(dz / 2, float(survey.z_max), dz)
    X, Z = np.meshgrid(xs, zs)
    tri = mtri.Triangulation(pm.p2[0], pm.p2[1], pm.t2.T)
    elem = tri.get_trifinder()(X.ravel(), Z.ravel())
    owner = np.full(pm.t2.shape[1], -1, int)
    for k, els in enumerate(pm.cell_elements):
        owner[els] = k
    ok = (elem >= 0) & (owner[np.clip(elem, 0, None)] >= 0)
    rec = np.log10(np.asarray(rho_cell, float)[owner[np.clip(elem, 0, None)]])
    true = np.log10(rho_fn(X.ravel(), Z.ravel()))
    return float(np.sqrt(np.mean((rec[ok] - true[ok]) ** 2)))


def fit_colorbar(fig, cbar, axes, gap=0.012, width=0.018):
    """Place the colour bar axes right of ``axes`` with exactly their combined
    vertical extent (after aspect adjustment), then freeze the layout."""
    fig.canvas.draw()
    axes = list(np.ravel(axes))
    for a in axes:
        a.apply_aspect()
    pos = [a.get_position() for a in axes]
    x1 = max(p.x1 for p in pos)
    y0 = min(p.y0 for p in pos)
    y1 = max(p.y1 for p in pos)
    cbar.ax.set_position([x1 + gap, y0, width, y1 - y0])
    try:
        fig.set_layout_engine("none")
    except Exception:                                  # noqa: BLE001
        pass
