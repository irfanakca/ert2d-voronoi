"""Numerical validation of the FEM + inversion core against MATLAB dumps.

Reference files (produced by ``scripts/export_reference.m`` or a manual save):

* ``k3_ref_initial_sonra.mat``  - state right after ``initial()``
* ``k3_ref_pupd_sonra.mat``     - state after the first ``pupd()``

Each test is skipped when its reference file is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from ert2d.fem.assembly import assemble, assemble_from_unit
from ert2d.fem.forward import FemSpace
from ert2d.fem.sensitivity import jacobian, jacobian_adjoint
from ert2d.inversion.occam import misfit_rms, parameter_update
from ert2d.inversion.regularization import voronoi_smoothness
from ert2d.inversion.weights import median_deviation_weights
from ert2d.io.dat_reader import read_dat
from ert2d.io.matref import MatRef
from ert2d.mesh.voronoi_param import _bounded_voronoi

ROOT = Path(__file__).resolve().parents[2]
REF = Path(__file__).resolve().parent / "reference"
K3 = ROOT / "kelenderis" / "k3.dat"
pytestmark = pytest.mark.skipif(not K3.is_file(),
                                reason="kelenderis/k3.dat not present (private sample data)")


def _ref(name: str) -> MatRef:
    for cand in (REF / name, REF / f"{name}.mat"):
        if cand.exists():
            return MatRef(cand)
    pytest.skip(f"{name} not present")


@pytest.fixture(scope="module")
def init_ref():
    return _ref("k3_ref_initial_sonra.mat")


@pytest.fixture(scope="module")
def pupd_ref():
    return _ref("k3_ref_pupd_sonra.mat")


@pytest.fixture(scope="module")
def survey():
    return read_dat(K3)


# --------------------------------------------------------------------------- #
def test_assembly_matches_accumarray(init_ref):
    m = init_ref
    n = m.p2.shape[1]
    ky = 1.0 / (-m.p2[1].min() / 4.0)
    K_ml = assemble_from_unit(m.k1, m.x_idx, n)
    K_py = assemble(m.p2, m.t2, sigma=1.0, ky=ky)
    assert abs(K_py - K_ml).max() / abs(K_ml).max() < 1e-10


def test_reference_potentials(init_ref):
    m = init_ref
    n = m.p2.shape[1]
    ky = 1.0 / (-m.p2[1].min() / 4.0)
    K = assemble(m.p2, m.t2, sigma=1.0, ky=ky)
    lu = spla.splu(K.tocsc())
    V1 = lu.solve(m.so.toarray())[m.akel] * ky / np.pi
    assert np.abs(V1 - m.V1).max() / np.abs(m.V1).max() < 1e-10


def test_forward_homogeneous_returns_homro(init_ref, survey):
    m = init_ref
    ky = 1.0 / (-m.p2[1].min() / 4.0)
    homro = float(np.exp(np.mean(np.log(survey.rhoa))))
    space = FemSpace.build(m.p2, m.t2, m.akel, ky)
    fwd = space.forward(np.full(m.t2.shape[1], 1.0 / homro), survey.quad)
    assert np.allclose(fwd.rhoa, homro, rtol=1e-9)


def test_first_update_matches_matlab(pupd_ref, init_ref, survey):
    m = pupd_ref
    ky = 1.0 / (-m.p2[1].min() / 4.0)
    homro = float(np.exp(np.mean(np.log(survey.rhoa))))
    prho0 = init_ref.prho
    space = FemSpace.build(m.p2, m.t2, m.akel, ky)

    sigma0 = np.full(m.t2.shape[1], 1.0 / homro)
    fwd = space.forward(sigma0, survey.quad)
    dd = np.log(survey.rhoa) - np.log(fwd.rhoa)
    w = median_deviation_weights(survey.rhoa)
    lam = np.std(np.log10(survey.rhoa), ddof=1) / np.mean(np.log10(survey.rhoa))

    J = jacobian(space, sigma0, survey.quad, fwd.rhoa, prho0, m.par_ucg)
    upd = parameter_update(J, m.C, dd, prho0, lam, w)

    assert np.abs(upd.dp - m.dp).max() / np.abs(m.dp).max() < 1e-8
    assert np.abs(upd.prho - m.prho).max() / np.abs(m.prho).max() < 1e-8

    sig1 = np.empty(m.t2.shape[1])
    for k, els in enumerate(m.par_ucg):
        sig1[els] = 1.0 / upd.prho[k]
    dd1 = np.log(survey.rhoa) - np.log(space.forward(sig1, survey.quad).rhoa)
    assert misfit_rms(dd1, w) == pytest.approx(m.scalar("misfit"), rel=1e-5)


def test_fast_jacobian_equals_adjoint(pupd_ref, init_ref, survey):
    """The Geselowitz reformulation must reproduce the per-cell adjoint solve."""
    m = pupd_ref
    ky = 1.0 / (-m.p2[1].min() / 4.0)
    homro = float(np.exp(np.mean(np.log(survey.rhoa))))
    space = FemSpace.build(m.p2, m.t2, m.akel, ky)
    sig0 = np.full(m.t2.shape[1], 1.0 / homro)
    fwd = space.forward(sig0, survey.quad)

    Jf = jacobian(space, sig0, survey.quad, fwd.rhoa, init_ref.prho, m.par_ucg)
    Ja = jacobian_adjoint(space, sig0, survey.quad, fwd.rhoa, init_ref.prho, m.par_ucg)
    assert np.abs(Jf - Ja).max() / np.abs(Ja).max() < 1e-10


def test_regularization_matches_matlab(init_ref):
    m = init_ref
    p = np.asarray(m.raw("p"), float)
    seeds = np.unique(np.round(p.T, 6), axis=0)
    from shapely.geometry import Polygon
    box = Polygon([
        (seeds[:, 0].min(), seeds[:, 1].min()), (seeds[:, 0].max(), seeds[:, 1].min()),
        (seeds[:, 0].max(), seeds[:, 1].max()), (seeds[:, 0].min(), seeds[:, 1].max())]).buffer(0)
    C_py = voronoi_smoothness(_bounded_voronoi(seeds, box)).tocsr()
    C_ml = m.C.tocsr()

    assert C_py.shape == C_ml.shape
    assert np.allclose(C_py.diagonal(), 1.0)
    assert np.abs(np.asarray(C_py.sum(1)).ravel()).max() < 1e-12
    both = (np.abs(C_py) > 1e-12).multiply(np.abs(C_ml) > 1e-12)
    overlap = both.nnz / (np.abs(C_ml) > 1e-12).nnz
    assert overlap > 0.98
    pv = np.asarray(C_py[both.nonzero()]).ravel()
    mv = np.asarray(C_ml[both.nonzero()]).ravel()
    assert np.corrcoef(pv, mv)[0, 1] > 0.999
