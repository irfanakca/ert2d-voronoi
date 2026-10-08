"""Jacobian of log apparent-resistivity w.r.t. log cell-resistivity.

Port of the ``opt == 1`` branch of ``forward.m`` + ``jacob.m``.

MATLAB computes, for every Voronoi cell ``k``, an extra linear solve
``fitur_k = K \\ (-ccd_k phi)`` (inside a ``parfor``).  That is
``O(n_param)`` solves and does not scale.

Here the identical quantity is obtained with the reciprocity / Geselowitz
identity: for datum ``i`` with current field ``phi_A`` (columns ``C1-C2`` of the
forward solution) and potential field ``phi_M`` (columns ``P1-P2``),

    d V_MN,i / d sigma_k  =  -(1/2pi) * phi_A,i^T (dK/dsigma_k) phi_M,i
                          =  -(1/2pi) * sum_{e in k} phi_A,i^e . Ke_unit . phi_M,i^e

so **no per-parameter solve is needed** - only the forward solution ``phi`` that
we already have.  ``Ke_unit`` is the unit-conductivity element matrix (mass term
included, exactly the entries ``k1f.m`` stores in ``k1``), so the result matches
the MATLAB adjoint value, not just the classic gradient-only sensitivity.

The slow textbook version is kept as :func:`jacobian_adjoint` for cross-checks.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla

from ert2d.fem.assembly import _CI, _RI, assemble, element_entries
from ert2d.fem.forward import FemSpace
from ert2d.fem.go2d import voltage_diff

_THRESH = 1e-2


def _forward_fields(space: FemSpace, sigma: np.ndarray):
    K = assemble(space.points, space.tris, sigma, space.ky, geom=space.geom)
    lu = spla.splu(K.tocsc())
    phi = lu.solve(space.source.toarray())          # (n, n_elec), source = 2*pi
    return lu, phi


def _pair_field(phi: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """phi[:, a] - phi[:, b] with pole (-1) columns treated as zero."""
    n = phi.shape[0]
    out = np.zeros((n, a.shape[0]))
    oka = a >= 0
    out[:, oka] += phi[:, a[oka]]
    okb = b >= 0
    out[:, okb] -= phi[:, b[okb]]
    return out


def jacobian(
    space: FemSpace,
    sigma: np.ndarray,
    quad: np.ndarray,
    rhoa_calc: np.ndarray,
    prho: np.ndarray,
    cell_elements: list[np.ndarray],
) -> np.ndarray:
    sigma = np.asarray(sigma, float)
    prho = np.asarray(prho, float)
    quad = np.asarray(quad)
    n_data, n_par = quad.shape[0], len(cell_elements)

    _, phi = _forward_fields(space, sigma)

    PA = _pair_field(phi, quad[:, 0], quad[:, 1])    # current field  (n, n_data)
    PM = _pair_field(phi, quad[:, 2], quad[:, 3])    # potential field (n, n_data)

    ent = element_entries(space.geom, np.ones(space.tris.shape[1]), space.ky)  # (9, M)
    PA_e = PA[space.tris]                            # (3, M, n_data)
    PM_e = PM[space.tris]

    # g[e, i] = PA_e^T Ke_unit PM_e   (symmetric bilinear form, per element/datum)
    g = np.zeros((space.tris.shape[1], n_data))
    for k, (i, j) in enumerate(zip(_RI, _CI)):
        g += ent[k][:, None] * PA_e[i] * PM_e[j]

    # sum g over the elements of each cell  ->  d V_MN / d sigma_k  (x -2pi)
    e2p = np.full(space.tris.shape[1], -1)
    for k, els in enumerate(cell_elements):
        e2p[els] = k
    dV = np.zeros((n_par, n_data))
    np.add.at(dV, e2p[e2p >= 0], g[e2p >= 0])
    dV *= -1.0 / (2.0 * np.pi)                       # (n_par, n_data)

    # a[k, i] = go2d(dV0_k, V1)[i] = dV[k,i] / voltage_diff(phi_unit)[i]
    scale = space.ky / np.pi
    dref_unit = voltage_diff(space.V_ref, quad) / scale
    a = dV / dref_unit[None, :]
    a[np.abs(a) <= _THRESH] = 0.0

    J = (a / prho[:, None] / rhoa_calc[None, :]).T   # (n_data, n_par)
    return J


# --------------------------------------------------------------------------- #
def jacobian_adjoint(
    space: FemSpace,
    sigma: np.ndarray,
    quad: np.ndarray,
    rhoa_calc: np.ndarray,
    prho: np.ndarray,
    cell_elements: list[np.ndarray],
    *,
    chunk: int = 128,
) -> np.ndarray:
    """Reference implementation: one batched adjoint solve per cell (slow)."""
    import scipy.sparse as sp

    sigma = np.asarray(sigma, float)
    prho = np.asarray(prho, float)
    n = space.points.shape[1]
    ne = len(space.elec_nodes)
    n_data, n_par = quad.shape[0], len(cell_elements)

    lu, phi = _forward_fields(space, sigma)
    ent = element_entries(space.geom, np.ones(space.tris.shape[1]), space.ky)
    grow = np.stack([space.tris[i] for i in _RI])
    gcol = np.stack([space.tris[j] for j in _CI])
    dref = voltage_diff(space.V_ref, quad)
    scale = space.ky / np.pi

    J = np.zeros((n_data, n_par))
    for k, els in enumerate(cell_elements):
        if els.size == 0:
            continue
        r = grow[:, els].reshape(-1)
        c = gcol[:, els].reshape(-1)
        v = ent[:, els].reshape(-1)
        ccd = sp.coo_matrix((v, (r, c)), shape=(n, n)).tocsr()
        dphi = lu.solve(-(ccd @ phi))
        dV0 = dphi[space.elec_nodes] * scale
        a = voltage_diff(dV0, quad) / dref
        a[np.abs(a) <= _THRESH] = 0.0
        J[:, k] = a / prho[k] / rhoa_calc
    return J
