"""Damped Gauss-Newton parameter update.

Two flavours:

* :func:`parameter_update` - faithful port of ``pupd.m``
  (``A = JᵀRᵀRJ + 2λC``, ``b = JᵀRᵀR dd - 2λ C(1/ρ)``, ``ρ ← ρ·exp(-dp)``),
  used when the driver runs in ``lambda_mode="matlab"``.

* :class:`LambdaScan` / :func:`search_lambda` - the adaptive scheme.  The
  regularisation is the proper quadratic roughness ``λ ‖C m‖²`` with
  ``m = log(1/ρ)``, giving a *symmetric positive* operator ``R = CᵀC``.  One
  generalised eigendecomposition ``M V = (R+εI) V diag(w)`` per iteration lets
  the trial step ``dp(λ)`` be evaluated for **any** ``λ`` in O(n²) - so a whole
  line search over ``λ`` costs one extra eigensolve, not one solve per ``λ``.

``dd = log(ρ_a,obs) - log(ρ_a,calc)``; ``Rd`` is diagonal so ``RᵀR`` is just the
squared data weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp


@dataclass
class Update:
    dp: np.ndarray            # (n_par,) log-conductivity step
    prho: np.ndarray          # (n_par,) updated cell resistivity
    A: np.ndarray             # normal-equation matrix (for reuse / diagnostics)
    b: np.ndarray


def clamp_lambda(lam: float) -> float:
    return 0.01 if lam < 0.001 else lam


def parameter_update(
    J: np.ndarray,
    C: sp.spmatrix | np.ndarray,
    dd: np.ndarray,
    prho: np.ndarray,
    lam: float,
    weights: np.ndarray | None = None,
) -> Update:
    lam = clamp_lambda(float(lam))
    prho = np.asarray(prho, float).ravel()
    w2 = np.ones(J.shape[0]) if weights is None else np.asarray(weights, float).ravel() ** 2

    Cdense = C.toarray() if sp.issparse(C) else np.asarray(C, float)
    JtW = J.T * w2                                   # (n_par, n_data)
    b = JtW @ dd - 2.0 * lam * (Cdense @ (1.0 / prho))
    A = JtW @ J + 2.0 * lam * Cdense
    dp = np.linalg.solve(A, b)

    prho_new = 1.0 / ((1.0 / prho) * np.exp(dp))
    return Update(dp=dp, prho=prho_new, A=A, b=b)


def misfit_rms(dd: np.ndarray, weights: np.ndarray | None = None) -> float:
    """RMS misfit in percent: ``sqrt((Rd dd)ᵀ(Rd dd) / nd) * 100``."""
    w = np.ones_like(dd) if weights is None else np.asarray(weights, float).ravel()
    r = w * dd
    return float(np.sqrt(r @ r / dd.size) * 100.0)


# --------------------------------------------------------------------------- #
# adaptive lambda: one eigensolve, then any lambda in O(n^2)
# --------------------------------------------------------------------------- #
@dataclass
class LambdaScan:
    w: np.ndarray                 # generalised eigenvalues  (M V = Reps V diag(w))
    V: np.ndarray                 # eigenvectors,  Vᵀ Reps V = I,  Vᵀ M V = diag(w)
    p: np.ndarray                 # Vᵀ b0            (data-gradient term)
    q: np.ndarray                 # Vᵀ (Reps @ m)    (roughness-gradient term)
    prho: np.ndarray              # current cell resistivity
    w_pos: np.ndarray = None      # positive eigenvalues (for auto lambda range)

    def dp(self, lam: float) -> np.ndarray:
        """Model step for regularisation weight ``lam``."""
        return self.V @ ((self.p - 2.0 * lam * self.q) / (self.w + 2.0 * lam))

    def model(self, lam: float) -> np.ndarray:
        """Trial cell resistivity for ``lam``:  ρ_new = ρ · exp(-dp)."""
        return self.prho * np.exp(-self.dp(lam))

    def auto_lambda_ref(self) -> float:
        pos = self.w_pos
        return float(np.exp(np.mean(np.log(pos)))) if pos is not None and pos.size else 1.0


def build_lambda_scan(J, C, dd, prho, weights=None, *, eps_rel: float = 1e-6,
                      roughness: str = "CtC") -> LambdaScan:
    """``roughness="CtC"`` penalises ``‖C m‖²`` (use for a non-SPD ``C`` such as
    the Voronoi / triangle Laplacian); ``roughness="C"`` penalises ``mᵀ C m``
    directly (use when ``C`` is already SPD, e.g. the ``delsq`` grid Laplacian -
    matches ``elris.m``)."""
    prho = np.asarray(prho, float).ravel()
    w2 = np.ones(J.shape[0]) if weights is None else np.asarray(weights, float).ravel() ** 2
    JtW = J.T * w2
    M = JtW @ J
    b0 = JtW @ dd

    R = (C if roughness == "C" else C.T @ C)
    R = R.toarray() if sp.issparse(R) else np.asarray(R, float)
    R = 0.5 * (R + R.T)
    eps = eps_rel * (np.trace(R) / R.shape[0] + 1e-30)
    Reps = R + eps * np.eye(R.shape[0])

    m = np.log(1.0 / prho)
    w, V = sla.eigh(M, Reps)                        # M V = Reps V diag(w)
    return LambdaScan(w=w, V=V, p=V.T @ b0, q=V.T @ (Reps @ m), prho=prho,
                      w_pos=w[w > 1e-12 * (w.max() + 1e-30)])


def search_lambda(scan: LambdaScan, misfit_of, lam_grid,
                  target_rms: float | None = None,
                  slack: float = 0.05,
                  ceiling: float | None = None) -> "LambdaSearch":
    """Evaluate ``misfit_of(prho_trial)`` for every ``lam`` in ``lam_grid``.

    ``misfit_of`` is a cheap forward-only RMS-% callback; it must return a large
    finite number (or inf) for an invalid trial.

    Selection (soft Occam - always prefers the *smoothest* model that still
    fits):

    * with ``target_rms`` and it is reachable: the **largest** ``lam`` whose
      RMS is within the target;
    * otherwise: the largest ``lam`` whose RMS is within ``(1 + slack)`` of the
      best RMS on the grid.
    """
    tried: list[tuple[float, float]] = []
    grid = np.atleast_1d(lam_grid).astype(float)
    for lam in grid:
        pr = scan.model(lam)
        if not np.all(np.isfinite(pr)) or np.any(pr <= 0):
            tried.append((float(lam), np.inf))
            continue
        m = misfit_of(pr)
        tried.append((float(lam), float(m) if np.isfinite(m) else np.inf))

    lams = np.array([a for a, _ in tried])
    safe = np.array([b if np.isfinite(b) else np.inf for _, b in tried])
    if not np.isfinite(safe).any():
        mid = len(grid) // 2
        return LambdaSearch(float(grid[mid]), scan.prho.copy(), np.inf, tried)

    if target_rms is not None and (safe <= target_rms).any():
        cand = np.flatnonzero(safe <= target_rms)
    else:
        cand = np.flatnonzero(safe <= safe.min() * (1.0 + slack))
        if ceiling is not None:                         # never accept a worse fit
            below = cand[safe[cand] <= ceiling + 1e-9]
            if below.size:
                cand = below
    idx = int(cand[np.argmax(lams[cand])])              # largest acceptable lambda

    return LambdaSearch(float(lams[idx]), scan.model(lams[idx]),
                        float(safe[idx]), tried)


@dataclass
class LambdaSearch:
    lam: float
    prho: np.ndarray
    misfit: float
    tried: list[tuple[float, float]]
