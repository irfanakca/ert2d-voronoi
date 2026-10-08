"""2.5-D DC forward operator — port of ``forward.m`` (+ ``initial.m`` source setup).

The MATLAB code uses a single wavenumber ``ky = 1/z_max`` and a crude inverse
transform ``V = phi * ky / pi``; reproduced verbatim (apparent resistivity is a
ratio so the scaling cancels).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ert2d.fem.assembly import assemble
from ert2d.fem.go2d import apparent_resistivity
from ert2d.fem.shape import TriGeom, tri_geom

I_SOURCE = 2.0 * np.pi          # MATLAB ``I = 2*pi``


@dataclass
class FemSpace:
    """Fixed geometry: mesh, electrode nodes, wavenumber, source RHS, reference."""

    points: np.ndarray                 # (2, N)
    tris: np.ndarray                   # (3, M) 0-based
    elec_nodes: np.ndarray             # (n_elec,) node index per electrode (MATLAB ``akel``)
    ky: float
    geom: TriGeom = field(default=None, repr=False)
    source: sp.csc_matrix = field(default=None, repr=False)   # (N, n_elec)
    V_ref: np.ndarray = field(default=None, repr=False)       # (n_elec, n_elec)

    @classmethod
    def build(cls, points, tris, elec_nodes, ky):
        n = points.shape[1]
        ne = len(elec_nodes)
        geom = tri_geom(points, tris)
        src = sp.csc_matrix(
            (np.full(ne, I_SOURCE), (elec_nodes, np.arange(ne))), shape=(n, ne))
        self = cls(points, tris, np.asarray(elec_nodes), float(ky), geom, src)
        # reference potentials at UNIT conductivity, exactly like ``k1f.m``
        self.V_ref = self._surface_potentials(np.ones(tris.shape[1]))
        return self

    # ---------------------------------------------------------------- #
    def _surface_potentials(self, sigma: np.ndarray) -> np.ndarray:
        K = assemble(self.points, self.tris, sigma, self.ky, geom=self.geom)
        lu = spla.splu(K.tocsc())
        phi = lu.solve(self.source.toarray())
        return phi[self.elec_nodes] * self.ky / np.pi

    def forward(self, sigma: np.ndarray, quad: np.ndarray) -> "ForwardResult":
        V0 = self._surface_potentials(np.asarray(sigma, float))
        rhoa = apparent_resistivity(V0, self.V_ref, quad)
        return ForwardResult(rhoa=rhoa, potentials=V0)


@dataclass
class ForwardResult:
    rhoa: np.ndarray
    potentials: np.ndarray
