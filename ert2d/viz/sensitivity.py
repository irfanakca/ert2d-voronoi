"""Parameter sensitivity -> per-element transparency.

Port of the ``senscalc`` sub-function in ``main_vor_v00.m``:

    s      = sqrt(sum(J**2, axis=0))          # per-parameter L2 sensitivity
    slog   = log10(s + eps)
    clip   = clip(slog, p5, p95)              # robust 5-95 percentile
    sn     = (clip - p5) / (p95 - p5)         # 0..1
    alpha  = amin + (amax - amin) * sn**gamma
"""

from __future__ import annotations

import numpy as np


def parameter_alpha(J: np.ndarray, *, amin: float = 0.12, amax: float = 1.0,
                    gamma: float = 1.2) -> np.ndarray:
    s = np.sqrt(np.sum(np.asarray(J, float) ** 2, axis=0))
    slog = np.log10(s + np.finfo(float).eps)
    p5, p95 = np.percentile(slog, [5, 95])
    clip = np.clip(slog, p5, p95)
    sn = (clip - p5) / (p95 - p5 + np.finfo(float).eps)
    return amin + (amax - amin) * sn ** gamma


def element_alpha(J: np.ndarray, cell_elements, n_elem: int, **kw) -> np.ndarray:
    """Broadcast :func:`parameter_alpha` onto the FEM elements."""
    a_par = parameter_alpha(J, **kw)
    out = np.ones(n_elem)
    for k, els in enumerate(cell_elements):
        out[els] = a_par[k]
    return out
