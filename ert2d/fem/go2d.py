"""Apparent resistivity from nodal potentials — port of ``go2d.m``.

``V`` is the (n_elec, n_elec) matrix of surface potentials, ``V[k, s]`` = potential
at electrode ``k`` for a unit current source at electrode ``s``.

For a four-electrode measurement with current pair ``(C1, C2)`` and potential
pair ``(P1, P2)`` the potential difference is

    dV = V[C1,P1] - V[C1,P2] - V[C2,P1] + V[C2,P2]

(using reciprocity, this is the reading at P1-P2 for injection at C1-C2).
Missing electrodes (poles) are encoded as ``-1`` in ``quad`` and drop out.
Apparent resistivity is the ratio to the homogeneous reference ``V1``.
"""

from __future__ import annotations

import numpy as np


def _pair(V: np.ndarray, r: np.ndarray, c: np.ndarray) -> np.ndarray:
    r = np.asarray(r)
    c = np.asarray(c)
    out = np.zeros(r.shape[0])
    ok = (r >= 0) & (c >= 0)
    out[ok] = V[r[ok], c[ok]]
    return out


def voltage_diff(V: np.ndarray, quad: np.ndarray) -> np.ndarray:
    c1, c2, p1, p2 = quad.T
    return (_pair(V, c1, p1) - _pair(V, c1, p2)
            - _pair(V, c2, p1) + _pair(V, c2, p2))


def apparent_resistivity(V0: np.ndarray, V1: np.ndarray,
                         quad: np.ndarray) -> np.ndarray:
    return voltage_diff(V0, quad) / voltage_diff(V1, quad)
