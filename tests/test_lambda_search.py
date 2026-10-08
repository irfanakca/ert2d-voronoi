"""Adaptive-lambda line search."""

from __future__ import annotations

import numpy as np
import pytest

from ert2d.inversion import (build_lambda_scan, invert, misfit_rms,
                             parameter_update, search_lambda)
from ert2d.io.dat_reader import read_dat
from ert2d.mesh import build_domain, triangulate, voronoi_parameterization
from tests._sample_survey import default_survey_path


@pytest.fixture(scope="module")
def setup():
    s = read_dat(default_survey_path())
    dom = build_domain(s)
    pm = voronoi_parameterization(triangulate(dom), dom)
    return s, pm


def test_scan_matches_direct_solve(setup):
    """LambdaScan.dp(lam) must equal the dense (M + 2 lam R) solve."""
    import scipy.linalg as sla

    from ert2d.fem.forward import FemSpace
    from ert2d.fem.sensitivity import jacobian
    from ert2d.inversion.driver import _elem_sigma
    from ert2d.inversion.weights import median_deviation_weights
    from ert2d.inversion.regularization import voronoi_smoothness

    s, pm = setup
    space = FemSpace.build(pm.p2, pm.t2, pm.elec_nodes, 1.0 / s.z_max)
    homro = float(np.exp(np.mean(np.log(s.rhoa))))
    rho = np.full(pm.n_param, homro)
    sig = _elem_sigma(rho, pm.cell_elements, pm.t2.shape[1], 1.0 / homro)
    fwd = space.forward(sig, s.quad)
    dd = np.log(s.rhoa) - np.log(fwd.rhoa)
    w = median_deviation_weights(s.rhoa)
    J = jacobian(space, sig, s.quad, fwd.rhoa, rho, pm.cell_elements)
    C = voronoi_smoothness(pm.cell_polys)

    scan = build_lambda_scan(J, C, dd, rho, w)

    w2 = w ** 2
    M = (J.T * w2) @ J
    b0 = (J.T * w2) @ dd
    R = (C.T @ C).toarray()
    R = 0.5 * (R + R.T)
    R = R + 1e-6 * (np.trace(R) / R.shape[0]) * np.eye(R.shape[0])   # == build_lambda_scan
    m = np.log(1.0 / rho)
    for lam in (0.001, 0.02, 0.5):
        A = M + 2 * lam * R
        b = b0 - 2 * lam * (R @ m)
        dp_direct = sla.solve(A, b, assume_a="sym")
        dp_scan = scan.dp(lam)
        assert np.abs(dp_scan - dp_direct).max() / (np.abs(dp_direct).max() + 1e-12) < 1e-6


def test_search_is_damped_and_stops(setup):
    s, pm = setup
    r = invert(s, pm, max_iter=20, lambda_mode="search")
    rms = r.rms_curve
    assert rms[0] > rms[-1]                       # improved overall
    assert np.all(np.diff(rms) <= 1e-6)           # monotone (no blow-ups)
    assert len(r.history) < 20                    # converged / stopped early
    # damped shape: last decrement much smaller than the first
    assert (rms[-2] - rms[-1]) < 0.25 * (rms[0] - rms[1])


def test_target_rms_stops_at_target(setup):
    s, pm = setup
    r = invert(s, pm, max_iter=20, lambda_mode="search", target_rms=2.5)
    assert r.misfit <= 2.5 + 1e-6


def test_matlab_mode_still_available(setup):
    s, pm = setup
    r = invert(s, pm, max_iter=4, tol=0.0, lambda_mode="matlab", lam0=0.05)
    assert len(r.history) == 4
    assert np.all(np.diff(r.rms_curve) < 0)
