"""Tests for ert2d.io.dat_reader against the sample .dat files."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from ert2d.io.dat_reader import read_dat

ROOT = Path(__file__).resolve().parents[2]          # MATLAB project dir

SAMPLES = sorted(ROOT.glob("*.dat"))

# device-calibration tables that are not surveys
NOT_SURVEYS = {"calib_info.dat", "main_calib.dat", "main_calib150-2000.dat"}

_RE_3D_HEADER = re.compile(r"^\s*\d+\s*,\s*\d+\s*$")


def _is_3d_collated(path: Path) -> bool:
    """A 3-D collated file (ELRIS3D format) has 'nelx,nely' on its 2nd line -
    it belongs to ert3d, not this 2-D reader."""
    lines = [ln for ln in path.read_text(errors="replace").splitlines() if ln.strip()]
    return len(lines) > 1 and bool(_RE_3D_HEADER.match(lines[1]))


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.name)
def test_parses_without_error(path: Path):
    if path.name in NOT_SURVEYS or _is_3d_collated(path):
        pytest.skip("not a 2-D survey file")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = read_dat(path)
    assert s.n_data > 0
    assert s.n_electrodes >= 2
    assert s.quad.shape == (s.n_data, 4)
    # every datum must reference at least one current and one potential electrode
    has_c = (s.quad[:, 0] >= 0) | (s.quad[:, 1] >= 0)
    has_p = (s.quad[:, 2] >= 0) | (s.quad[:, 3] >= 0)
    assert has_c.all() and has_p.all()
    # electrode indices in range
    assert s.quad.max() < s.n_electrodes
    assert np.isfinite(s.rhoa).all()


def _sample(name: str) -> Path:
    p = ROOT / name
    if not p.is_file():
        pytest.skip(f"{name} not present (not bundled outside the full dev repo)")
    return p


def test_wenner_mp0_geometry():
    """w2_trimmed.dat: Wenner, midpoint=0, spacing 10, first datum a=10."""
    s = read_dat(_sample("w2_trimmed.dat"))
    assert s.array_type == 1
    c1, c2, p1, p2 = s.quad[0]
    x = s.electrode_x
    # C1 P1 P2 C2 spaced by a = 10
    assert x[p1] - x[c1] == pytest.approx(10.0)
    assert x[p2] - x[p1] == pytest.approx(10.0)
    assert x[c2] - x[p2] == pytest.approx(10.0)
    assert s.rhoa[0] == pytest.approx(43.61, abs=1e-2)


def test_dipole_dipole_levels():
    s = read_dat(_sample("various_dd2.dat"))
    assert s.array_type == 3
    assert s.n_level.min() >= 1
    # geometric factor should be finite & positive for dd
    assert np.isfinite(s.k_geom).all()


def test_general_array_roundtrip():
    s = read_dat(_sample("a5.dat"))
    assert s.array_type == 11
    assert s.n_data == 206
    # first row: C1=1, C2=0, P1=2, P2=3  -> a dipole-dipole style quad
    x = s.electrode_x
    c1, c2, p1, p2 = s.quad[0]
    assert {round(x[c1]), round(x[c2]), round(x[p1]), round(x[p2])} == {0, 1, 2, 3}
