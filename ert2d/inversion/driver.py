"""Iterative inversion driver.

Per iteration:

1. forward solve + Jacobian at the current model
2. data residual ``dd = log(rho_a,obs) - log(rho_a,calc)`` and RMS misfit
3. model update - one of:

   * ``lambda_mode="search"`` (default): build a :class:`LambdaScan`
     (one generalised eigensolve) and line-search ``lambda`` over a geometric
     grid, picking the value that minimises the post-step RMS.  The grid is
     wide on the first iteration and then tracks the previous best ``lambda``,
     so damping is chosen adaptively and the curve flattens onto the data
     noise floor instead of sliding indefinitely.
   * ``lambda_mode="matlab"``: the ``pupd.m`` update with a fixed ``lam0`` and
     blind ``lam *= cool`` each iteration.

4. stop when the best achievable RMS reduction drops below ``tol`` (relative),
   or the step cannot reduce the misfit at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ert2d.fem.forward import FemSpace
from ert2d.fem.sensitivity import jacobian
from ert2d.inversion.occam import (build_lambda_scan, misfit_rms,
                                   parameter_update, search_lambda)
import os

from ert2d.inversion.regularization import fv_energy_laplacian, grid_laplacian, voronoi_smoothness
from ert2d.inversion.weights import median_deviation_weights, robust_factor
from ert2d.io.dat_reader import Survey
from ert2d.mesh.voronoi_param import ParamMesh


@dataclass
class IterInfo:
    iteration: int
    lam: float
    misfit_before: float
    misfit_after: float
    rho: np.ndarray = field(repr=False)
    rhoa_calc: np.ndarray = field(repr=False)
    lam_curve: list[tuple[float, float]] = field(default_factory=list, repr=False)


@dataclass
class InversionResult:
    rho_cell: np.ndarray
    sigma_elem: np.ndarray
    history: list[IterInfo]
    rhoa_obs: np.ndarray = field(repr=False)
    weights: np.ndarray = field(repr=False)
    space: FemSpace = field(default=None, repr=False)
    param_mesh: ParamMesh = field(default=None, repr=False)
    jacobian: np.ndarray = field(default=None, repr=False)
    # final IRLS factor per datum in (0, 1] (None = no robust reweighting);
    # ``weights`` stays the static base weights, so ``datum_misfit_pct`` shows
    # the plain misfit of every datum, outliers included
    robust_weights: np.ndarray = field(default=None, repr=False)

    @property
    def misfit(self) -> float:
        return self.history[-1].misfit_after

    @property
    def rhoa_calc(self) -> np.ndarray:
        return self.history[-1].rhoa_calc

    @property
    def residual_log(self) -> np.ndarray:
        return np.log(self.rhoa_obs) - np.log(self.rhoa_calc)

    def datum_misfit_pct(self) -> np.ndarray:
        return 100.0 * self.weights * np.abs(self.residual_log)

    @property
    def rms_curve(self) -> np.ndarray:
        return np.array([i.misfit_after for i in self.history])

    @property
    def lambda_curve(self) -> np.ndarray:
        return np.array([i.lam for i in self.history])

    def rho_elem(self) -> np.ndarray:
        bg = float(np.exp(np.mean(np.log(self.rhoa_obs))))
        out = np.full(self.param_mesh.t2.shape[1], bg)
        for k, els in enumerate(self.param_mesh.cell_elements):
            out[els] = self.rho_cell[k]
        return out


def _elem_sigma(rho_cell: np.ndarray, cell_elements, n_elem: int,
                bg_sigma: float) -> np.ndarray:
    """Element conductivities; elements in no cell keep ``bg_sigma`` (fixed
    background, e.g. the rect-model padding frame)."""
    sig = np.full(n_elem, bg_sigma)
    for k, els in enumerate(cell_elements):
        sig[els] = 1.0 / rho_cell[k]
    return sig


def invert(
    survey: Survey,
    pm: ParamMesh,
    *,
    max_iter: int = 12,
    tol: float = 0.03,
    lambda_mode: str = "search",
    lam0: float | None = None,
    cool: float = 0.8,                     # matlab mode only
    target_rms: float | None = None,       # search mode: stop / Occam target (%)
    slack: float = 0.15,                   # search mode: pick smoothest model
                                           #   within (1+slack) of the best RMS
    lambda_floor: float = 1e-3,            # search mode: absolute lower bound
    n_lambda: int = 11,                    # search mode: candidates per iteration
    lambda_decades: float = 1.5,           # search mode: half-width of the 1st grid
    lambda_step: float = 3.0,              # search mode: max lambda change / iter
    C=None,
    roughness_form: str | None = None,     # "C" (mᵀCm) | "CtC" (‖Cm‖²); None = by kind
    regularization: str | None = None,     # "legacy" | "energy"; None -> $ERT_REG or "legacy"
    keep_jacobian: bool = True,
    callback: Callable[[IterInfo], None] | None = None,
    robust: str = "none",                  # IRLS: "none"|"huber"|"bisquare"|"cauchy"
    robust_c: float | None = None,         # tuning constant (None = method default)
    robust_start: int = 2,                 # first iteration that reweights
    robust_min_weight: float = 0.5,        # lower bound of the IRLS factor
) -> InversionResult:
    """``robust != "none"`` turns on iteratively reweighted least squares: at
    the start of every iteration from ``robust_start`` on, each datum's weight
    is the static base weight times :func:`~ert2d.inversion.weights.robust_factor`
    of the *current* log-residual, so data that keep misfitting badly count
    less. The weights are frozen within an iteration (line search and RMS use
    the same ones), so the RMS values reported with it are robust-weighted.
    Iteration 1 is skipped by default: from the homogeneous start every
    residual is large and structured, which says nothing about noise."""
    if regularization is None:
        regularization = os.environ.get("ERT_REG", "legacy")
    ky = 1.0 / survey.z_max
    space = FemSpace.build(pm.p2, pm.t2, pm.elec_nodes, ky)

    roa =np.asarray(survey.rhoa, float)
    homro = float(np.exp(np.mean(np.log(roa))))
    w0 = median_deviation_weights(roa)
    w = w0                                  # current weights (rebound per iteration)
    rfac = None
    n_elem = pm.t2.shape[1]
    bg_sigma = 1.0 / homro
    kind = getattr(pm, "kind", "voronoi")
    if C is None:
        if kind == "rect" and hasattr(pm, "n_rows"):
            C = grid_laplacian(pm.n_rows, pm.n_cols)          # delsq, like elris.m
        else:
            C = voronoi_smoothness(pm.cell_polys, edge_only=kind != "voronoi")
    # SPD guarantee differs by kind: rect's grid_laplacian is a symmetric
    # diagonally-dominant (hence SPD) graph Laplacian -> "C" penalises mᵀCm
    # directly; voronoi/triangle's area/distance-weighted neighbour Laplacian
    # is NOT guaranteed SPD, so it goes through "CtC" (‖Cm‖²) instead - a real
    # asymmetry between the parameterizations, not a bug.
    roughness = "C" if (kind == "rect" and hasattr(pm, "n_rows")) else "CtC"
    if regularization == "energy" and roughness_form is None:
        # one functional for every scheme: first-order finite-volume gradient energy
        # mᵀLm = Σ w_ij (m_i - m_j)², w_ij = L_ij / d_ij (unit weights for the plain grid)
        if kind == "rect" and hasattr(pm, "n_rows"):
            C = grid_laplacian(pm.n_rows, pm.n_cols)
        else:
            C = fv_energy_laplacian(pm.cell_polys, pm.seeds if kind == "voronoi" else None)
        roughness = "C"
    if roughness_form is not None:
        roughness = roughness_form

    def elem_sigma(rho_cell):
        return _elem_sigma(rho_cell, pm.cell_elements, n_elem, bg_sigma)

    def rhoa_of(rho_cell):
        return space.forward(elem_sigma(rho_cell), survey.quad).rhoa

    def misfit_of(rho_cell) -> float:
        rc = rhoa_of(rho_cell)
        if not np.all(np.isfinite(rc)) or np.any(rc <= 0):
            return np.inf
        return misfit_rms(np.log(roa) - np.log(rc), w)

    rho = np.full(pm.n_param, homro)
    lam = lam0 if lam0 is not None else (
        float(np.std(np.log10(roa), ddof=1) / np.mean(np.log10(roa)))
        if lambda_mode == "matlab" else None)
    history: list[IterInfo] = []
    last_J = None

    for it in range(1, max_iter + 1):
        sigma = elem_sigma(rho)
        fwd = space.forward(sigma, survey.quad)
        dd = np.log(roa) - np.log(fwd.rhoa)
        if robust != "none" and it >= robust_start:
            rfac = robust_factor(dd, robust, robust_c, min_weight=robust_min_weight)
            w = w0 * rfac
        mis_before = misfit_rms(dd, w)

        J = jacobian(space, sigma, survey.quad, fwd.rhoa, rho, pm.cell_elements)
        if keep_jacobian:
            last_J = J

        if lambda_mode == "search":
            scan = build_lambda_scan(J, C, dd, rho, w, roughness=roughness)
            ref = lam if lam is not None else scan.auto_lambda_ref()
            if it == 1 and lam0 is None:
                grid = np.logspace(np.log10(ref) - lambda_decades,
                                   np.log10(ref) + lambda_decades, n_lambda)
            else:                                   # track prev best, bounded step
                d = np.log10(lambda_step)
                grid = ref * np.logspace(-d, d, n_lambda)
            grid = np.clip(grid, max(lambda_floor, 0.02 * scan.auto_lambda_ref()), None)
            sr = search_lambda(scan, misfit_of, grid, target_rms=target_rms,
                               slack=slack, ceiling=mis_before)
            rho_new, mis_after, lam = sr.prho, sr.misfit, sr.lam
            lam_curve = sr.tried
        else:
            upd = parameter_update(J, C, dd, rho, lam, w)
            rho_new = upd.prho
            mis_after = misfit_of(rho_new)
            lam_curve = []

        improved = mis_after < mis_before - 1e-9
        if not improved:
            if callback is not None:                    # let the UI see the try
                rc = history[-1].rhoa_calc if history else fwd.rhoa
                callback(IterInfo(it, float(lam), mis_before, mis_after,
                                  rho.copy(), rc, lam_curve))
            break

        rho = rho_new
        info = IterInfo(it, float(lam), mis_before, mis_after, rho.copy(),
                        rhoa_of(rho), lam_curve)
        history.append(info)
        if callback is not None:
            callback(info)

        if target_rms is not None and mis_after <= target_rms:
            break
        # (mis_before == the previous iteration's misfit_after for static
        # weights; with IRLS it keeps both terms under the same weights)
        if it > 1 and (mis_before - mis_after) / mis_after < tol:
            break
        if lambda_mode == "matlab" and it > 1:
            lam *= cool

    if robust != "none" and rfac is not None:
        # factor from the FINAL residuals (rfac above lags one update behind)
        rfac = robust_factor(np.log(roa) - np.log(rhoa_of(rho)), robust, robust_c,
                             min_weight=robust_min_weight)

    return InversionResult(
        rho_cell=rho,
        sigma_elem=elem_sigma(rho),
        history=history,
        rhoa_obs=roa,
        weights=w0,
        space=space,
        param_mesh=pm,
        jacobian=last_J,
        robust_weights=rfac,
    )
