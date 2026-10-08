"""Structural tests for the mesh + Voronoi parameterization pipeline.

The mesh is only required to be *equivalent* to the MATLAB one (Triangle vs
initmesh), so these check invariants, not exact node counts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

from ert2d.io.dat_reader import read_dat
from ert2d.mesh import build_domain, triangulate, voronoi_parameterization

ROOT = Path(__file__).resolve().parents[2]
REF = Path(__file__).resolve().parent / "reference"
K3 = ROOT / "kelenderis" / "k3.dat"
pytestmark = pytest.mark.skipif(not K3.is_file(),
                                reason="kelenderis/k3.dat not present (private sample data)")


@pytest.fixture(scope="module")
def k3():
    survey = read_dat(ROOT / "kelenderis" / "k3.dat")
    dom = build_domain(survey)
    mesh = triangulate(dom)
    pm = voronoi_parameterization(mesh, dom)
    return survey, dom, mesh, pm


@pytest.fixture(scope="module")
def k3_matlab_settings():
    """Mesh built with the MATLAB meshgen constants for a fair comparison."""
    survey = read_dat(ROOT / "kelenderis" / "k3.dat")
    dom = build_domain(survey, wing=3.0, surface_div=4, depth_factor=4.0)
    mesh = triangulate(dom, min_angle=30, h_grad=1.12)
    pm = voronoi_parameterization(mesh, dom)
    return survey, dom, mesh, pm


def test_domain_matches_matlab_bounds(k3):
    """With the MATLAB constants (wing=3, depth=4) the domain must match
    meshgen.m exactly, regardless of the current package defaults."""
    survey, *_ = k3
    dom = build_domain(survey, wing=3.0, surface_div=4, depth_factor=4.0)
    assert dom.x_bounds == pytest.approx((-3.75, 33.75))
    assert dom.z_bounds[0] == pytest.approx(-4 * 3.091625, rel=1e-6)


def test_electrodes_are_mesh_nodes(k3):
    survey, _, mesh, _ = k3
    P = mesh.points.T
    for x in survey.electrode_x:
        d = np.hypot(P[:, 0] - x, P[:, 1] - 0.0)
        assert d.min() < 1e-6, f"electrode x={x} not a node"


def test_mesh_covers_domain(k3):
    _, dom, mesh, _ = k3
    area_tris = mesh.areas().sum()
    area_dom = Polygon(dom.boundary).area
    assert area_tris == pytest.approx(area_dom, rel=1e-3)


def test_partition_is_exact(k3):
    _, _, mesh, pm = k3
    lens = np.array([len(e) for e in pm.cell_elements])
    assert lens.sum() == pm.n_tris                 # every element assigned once
    e2p = pm.element_to_param()
    assert (e2p >= 0).all()                        # none unassigned
    all_els = np.concatenate([e for e in pm.cell_elements if len(e)])
    assert len(all_els) == len(np.unique(all_els))  # none double-assigned


def test_cells_tile_the_domain(k3):
    _, dom, _, pm = k3
    total = sum(p.area for p in pm.cell_polys if not p.is_empty)
    assert total == pytest.approx(Polygon(dom.boundary).area, rel=1e-3)


@pytest.mark.skipif(not (REF / "k3_ref.mat.mat").exists()
                    and not (REF / "k3_ref.mat").exists(),
                    reason="MATLAB reference not present")
def test_structurally_close_to_matlab(k3_matlab_settings):
    import scipy.io as sio
    _, _, mesh, pm = k3_matlab_settings
    f = REF / "k3_ref.mat.mat"
    if not f.exists():
        f = REF / "k3_ref.mat"
    ref = sio.loadmat(f)
    n_nodes_ref = ref["p"].shape[1]
    n_par_ref = ref["par"].shape[1]
    assert 0.8 < mesh.n_nodes / n_nodes_ref < 1.25
    assert 0.8 < pm.n_param / n_par_ref < 1.25
