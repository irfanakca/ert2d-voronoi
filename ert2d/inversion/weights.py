"""Data weighting — port of the weight block in ``main_vor_v00.m``.

    da  = log10(rho_a)
    w   = 1 / (1 + |da - median(da)| + eps)

(down-weights outliers; not normalized).  MATLAB then uses ``Rd = diag(w)``.
"""

from __future__ import annotations

import numpy as np


ROBUST_METHODS = ("none", "huber", "bisquare", "cauchy")
_DEFAULT_C = {"huber": 1.345, "bisquare": 4.685, "cauchy": 2.385}   # 95 % efficiency


def robust_factor(dd: np.ndarray, method: str = "huber", c: float | None = None,
                  *, min_scale: float = 0.005, min_weight: float = 0.5) -> np.ndarray:
    """Residual-based data weight factor in (0, 1] for IRLS.

    ``dd = ln(rho_a,obs) - ln(rho_a,calc)``. The residual scale is the robust
    ``s = 1.4826 * MAD(dd)`` (at least ``min_scale``, i.e. 0.5 %), and with
    ``u = |dd - median(dd)| / (c * s)`` the classical IRLS weight
    ``omega = psi(r)/r`` is

    * ``huber``    ``1`` for ``u <= 1`` else ``1/u``          (mild; never zero)
    * ``bisquare`` ``(1 - u^2)^2`` for ``u < 1`` else ``0``   (rejects gross outliers)
    * ``cauchy``   ``1 / (1 + u^2)``                          (smooth, in between)

    The returned factor is ``sqrt(omega)`` - the data weights enter the
    inversion as ``Rd = diag(w)`` (squared in ``JtW J``), so multiplying ``w``
    by it applies ``omega`` to the squared residual.  ``method="none"`` -> ones.

    ``min_weight`` is a lower bound on that factor (default 0.5: a datum counts
    at least a quarter in the squared misfit).  Without it a coherent real
    anomaly the smooth model cannot yet fit (e.g. a sharp shallow high) is
    down-weighted, the model then fits it even worse, its weight drops further,
    and the anomaly is erased - a feedback loop.  ``min_weight=0`` allows full
    rejection (pure bisquare).
    """
    dd = np.asarray(dd, float).ravel()
    if method in (None, "none"):
        return np.ones_like(dd)
    if method not in _DEFAULT_C:
        raise ValueError(f"unknown robust method {method!r}; use one of {ROBUST_METHODS}")
    c = _DEFAULT_C[method] if c is None else float(c)
    r = dd - np.median(dd)
    s = max(1.4826 * float(np.median(np.abs(r))), min_scale)
    u = np.abs(r) / (c * s)
    if method == "huber":
        omega = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12))
    elif method == "bisquare":
        omega = np.where(u < 1.0, (1.0 - u ** 2) ** 2, 0.0)
    else:
        omega = 1.0 / (1.0 + u ** 2)
    return np.clip(np.sqrt(omega), float(min_weight), 1.0)


def median_deviation_weights(rhoa: np.ndarray) -> np.ndarray:
    da = np.log10(np.asarray(rhoa, float).ravel())
    dev = np.abs(da - np.median(da))
    return 1.0 / (1.0 + dev + np.finfo(float).eps)
