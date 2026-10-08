"""Smoke tests for the plotting layer (headless)."""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from ert2d.inversion import invert
from ert2d.io.dat_reader import read_dat
from ert2d.mesh import build_domain, triangulate, voronoi_parameterization
from ert2d.viz import data_fit_figure, diagnostics_figure
from tests._sample_survey import default_survey_path


@pytest.fixture(scope="module")
def result():
    survey = read_dat(default_survey_path())
    dom = build_domain(survey)
    pm = voronoi_parameterization(triangulate(dom), dom)
    return survey, invert(survey, pm, max_iter=3)


def test_data_fit_figure(result):
    survey, res = result
    fig = data_fit_figure(survey, res)
    assert len(fig.axes) >= 3
    fig.canvas.draw()


def test_diagnostics_figure(result):
    survey, res = result
    fig = diagnostics_figure(survey, res)
    assert len(fig.axes) >= 3
    fig.canvas.draw()


def test_result_diagnostics_arrays(result):
    _, res = result
    assert res.datum_misfit_pct().shape == res.rhoa_obs.shape
    assert res.rms_curve.size == len(res.history)
    assert res.jacobian is not None
