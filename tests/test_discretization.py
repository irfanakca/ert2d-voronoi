"""The three model-discretization kinds: triangle / rect / voronoi."""

from __future__ import annotations

import numpy as np
import pytest

from ert2d.inversion import invert
from ert2d.io.dat_reader import read_dat
from ert2d.mesh import KINDS, build_param_mesh, core_region_mask
from tests._sample_survey import default_survey_path


@pytest.fixture(scope="module")
def survey():
    return read_dat(default_survey_path())


@pytest.mark.parametrize("kind", KINDS)
def test_build_param_mesh(survey, kind):
    pm = build_param_mesh(survey, kind=kind, surface_div=2)
    assert pm.kind == kind
    assert pm.n_param > 20
    assert pm.p2.shape[0] == 2 and pm.t2.shape[0] == 3
    # every parameter owns at least one element
    assert all(len(e) >= 1 for e in pm.cell_elements)
    # electrode nodes sit on the surface
    z = pm.p2[1, pm.elec_nodes]
    assert np.allclose(z, 0.0, atol=1e-6)
    # cell polygons cover a sensible area (>= half the electrode-line box)
    span = float(np.ptp(survey.electrode_x))
    area = sum(p.area for p in pm.cell_polys if not p.is_empty)
    assert area > 0.4 * span * survey.z_max


def test_triangle_is_identity_partition(survey):
    pm = build_param_mesh(survey, kind="triangle")
    assert pm.n_param == pm.t2.shape[1]
    assert all(len(e) == 1 for e in pm.cell_elements)


def test_rect_is_structured(survey):
    pm = build_param_mesh(survey, kind="rect", surface_div=2)
    assert hasattr(pm, "n_cols") and hasattr(pm, "n_rows")
    assert pm.n_param == pm.n_cols * pm.n_rows
    assert all(len(e) == 2 for e in pm.cell_elements)      # 2 triangles / block
    # padding frame: some elements belong to no parameter
    covered = np.concatenate([e for e in pm.cell_elements])
    assert len(covered) < pm.t2.shape[1]


@pytest.mark.parametrize("kind", KINDS)
def test_core_region_mask(survey, kind):
    pm = build_param_mesh(survey, kind=kind, surface_div=2)
    mask = core_region_mask(pm, survey)
    assert mask.dtype == bool
    assert mask.shape == (pm.n_param,)
    # every full-domain build parameterizes some padding too (wing/depth
    # factor for triangle/voronoi; rect has none), so the core count must
    # never exceed the total, and must be strictly positive.
    assert 0 < mask.sum() <= pm.n_param
    # every "core" cell's polygon centroid is really inside the box.
    x_lo, x_hi = survey.electrode_x.min(), survey.electrode_x.max()
    for k in np.nonzero(mask)[0]:
        c = pm.cell_polys[k].centroid
        assert x_lo - 1e-9 <= c.x <= x_hi + 1e-9
        assert -survey.z_max - 1e-9 <= c.y <= 1e-9
    if kind == "rect":
        # rect's own geometry already spans exactly the core box (no wing/
        # depth_factor padding is ever parameterized for rect - see
        # rect_parameterization docstring), so every cell should be "core".
        assert mask.all()
    else:
        # triangle/voronoi parameterize the full wing/depth_factor-padded
        # domain, so some cells should fall outside the core box.
        assert not mask.all()


@pytest.mark.parametrize("kind", KINDS)
def test_inversion_runs_each_kind(survey, kind):
    pm = build_param_mesh(survey, kind=kind, surface_div=2)
    res = invert(survey, pm, max_iter=6)
    assert res.misfit < survey.rhoa.std()               # improved a lot
    assert np.all(np.diff(res.rms_curve) <= 1e-6)        # monotone
    rho_e = res.rho_elem()
    assert np.isfinite(rho_e).all() and (rho_e > 0).all()
