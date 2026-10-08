"""Load MATLAB reference ``.mat`` dumps with 1-based -> 0-based conversion.

Used only for validating the port against ``export_reference.m`` output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


class MatRef:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._d = sio.loadmat(self.path)

    # -- raw ------------------------------------------------------------ #
    def __contains__(self, key: str) -> bool:
        return key in self._d

    def raw(self, key: str):
        return self._d[key]

    def scalar(self, key: str) -> float:
        return float(np.asarray(self._d[key]).ravel()[0])

    # -- mesh (0-based, MATLAB (dim, N) layout kept) ------------------- #
    @property
    def p2(self) -> np.ndarray:
        return np.asarray(self._d["p2"], float)

    @property
    def t2(self) -> np.ndarray:
        """(3, M) 0-based connectivity (drops the subdomain row)."""
        return np.asarray(self._d["t2"][:3], np.int64) - 1

    @property
    def akel(self) -> np.ndarray:
        return np.asarray(self._d["akel"], np.int64).ravel() - 1

    @property
    def x_idx(self) -> np.ndarray:
        """(9M, 2) assembly index pairs, 0-based."""
        return np.asarray(self._d["x"], np.int64) - 1

    @property
    def k1(self) -> np.ndarray:
        v = self._d["k1"]
        return (v.toarray().ravel() if sp.issparse(v) else np.asarray(v).ravel())

    @property
    def so(self) -> sp.csc_matrix:
        v = self._d["so"]
        return v.tocsc() if sp.issparse(v) else sp.csc_matrix(v)

    @property
    def V1(self) -> np.ndarray:
        return np.asarray(self._d["V1"], float)

    @property
    def C(self) -> sp.csc_matrix:
        v = self._d["C"]
        return v.tocsc() if sp.issparse(v) else sp.csc_matrix(v)

    @property
    def sig(self) -> np.ndarray:
        return np.asarray(self._d["sig"], float).ravel()

    @property
    def prho(self) -> np.ndarray:
        return np.asarray(self._d["prho"], float).ravel()

    @property
    def par_ucg(self) -> list[np.ndarray]:
        """List of element-index arrays per Voronoi cell, 0-based."""
        par = self._d["par"]
        if par.dtype.names and "ucg" in par.dtype.names:      # struct array
            return [np.asarray(par[0, i]["ucg"], np.int64).ravel() - 1
                    for i in range(par.shape[1])]
        # cell array fallback (par_ucg)
        pc = self._d.get("par_ucg", par)
        return [np.asarray(pc[i, 0], np.int64).ravel() - 1 for i in range(pc.shape[0])]

    @property
    def dp(self) -> np.ndarray:
        return np.asarray(self._d["dp"], float).ravel()
