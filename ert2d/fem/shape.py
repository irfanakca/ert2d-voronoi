"""Linear-triangle geometry — Python port of ``pdetrgm.m`` (MathWorks ``pdetrg``).

For each element returns the area and the (2*area)-scaled shape-function
gradient coefficients ``b_i = dN_i/dx * 2A``, ``c_i = dN_i/dy * 2A`` used by the
FEM stiffness assembly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TriGeom:
    area: np.ndarray        # (M,)   triangle area (MATLAB ``delta``)
    b: np.ndarray           # (3, M) b1,b2,b3  (== MATLAB b1,b2,b3)
    c: np.ndarray           # (3, M) c1,c2,c3


def tri_geom(points: np.ndarray, tris: np.ndarray) -> TriGeom:
    """points (2, N), tris (3, M) 0-based -> :class:`TriGeom`.

    Matches ``[delta,b1,c1,b2,c2,b3,c3] = pdetrgm(p, t)``:
        b1 = y2 - y3   c1 = x3 - x2
        b2 = y3 - y1   c2 = x1 - x3
        b3 = y1 - y2   c3 = x2 - x1
        delta = |(x2-x1)(y3-y1) - (x3-x1)(y2-y1)| / 2
    """
    x, y = points[0], points[1]
    a1, a2, a3 = tris[0], tris[1], tris[2]
    x1, x2, x3 = x[a1], x[a2], x[a3]
    y1, y2, y3 = y[a1], y[a2], y[a3]

    area = 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    return TriGeom(area=area, b=b, c=c)
