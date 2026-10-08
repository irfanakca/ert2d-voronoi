"""2.5-D DC-resistivity FEM stiffness assembly.

Port of the element-matrix loop shared by ``k1f.m`` and ``forward.m``.

Element matrix for wavenumber ``ky`` and element conductivity ``sigma_e``::

    K_ij = sigma_e/(4A) (b_i b_j + c_i c_j)  +  sigma_e (ky^2/6) A * m_ij

with the consistent mass matrix ``m_ii = 1``, ``m_ij = 1/2`` (MATLAB uses
``a`` on the diagonal and ``a/2`` off-diagonal, ``a = sigma (1/6) ky^2 A``).

``accumarray(x, k)`` in MATLAB becomes a ``scipy.sparse`` COO build (duplicate
(row, col) pairs are summed).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ert2d.fem.shape import TriGeom, tri_geom

# element-matrix entry order, matching MATLAB
#   x = [x1,x1; x1,x2; x1,x3; x2,x1; x2,x2; x2,x3; x3,x1; x3,x2; x3,x3]
_RI = (0, 0, 0, 1, 1, 1, 2, 2, 2)
_CI = (0, 1, 2, 0, 1, 2, 0, 1, 2)


def element_entries(geom: TriGeom, sigma: np.ndarray, ky: float) -> np.ndarray:
    """Return the 9 element-matrix entries per element, shape (9, M)."""
    b, c, A = geom.b, geom.c, geom.area
    p1 = sigma / (4.0 * A)
    a = sigma * (ky * ky / 6.0) * A
    half = a / 2.0

    ent = np.empty((9, A.size))
    for k, (i, j) in enumerate(zip(_RI, _CI)):
        stiff = p1 * (b[i] * b[j] + c[i] * c[j])
        mass = a if i == j else half
        ent[k] = stiff + mass
    return ent


def assemble(points: np.ndarray, tris: np.ndarray, sigma: np.ndarray,
             ky: float, *, geom: TriGeom | None = None) -> sp.csc_matrix:
    """Global stiffness matrix (n_nodes x n_nodes), CSC, symmetric."""
    if geom is None:
        geom = tri_geom(points, tris)
    sigma = np.broadcast_to(np.asarray(sigma, float), (tris.shape[1],))
    ent = element_entries(geom, sigma, ky)

    rows = np.concatenate([tris[i] for i in _RI])
    cols = np.concatenate([tris[j] for j in _CI])
    data = ent.reshape(-1)
    n = points.shape[1]
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc()


def assemble_from_unit(k_unit: np.ndarray, x_idx: np.ndarray, n: int) -> sp.csc_matrix:
    """Rebuild K from the flat ``k1`` entry vector + ``x`` index pairs
    (the arrays ``k1f.m`` returns).  Used to check the assembler against the
    MATLAB reference."""
    k_unit = np.asarray(k_unit, float).reshape(-1)
    r = np.asarray(x_idx)[:, 0]
    c = np.asarray(x_idx)[:, 1]
    return sp.coo_matrix((k_unit, (r, c)), shape=(n, n)).tocsc()
