"""RES2DINV ``.dat`` reader.

Python port of the MATLAB ``read_data.m`` / ``oku`` pair. Reads the classic
RES2DINV single-line-per-datum formats (Wenner, pole-pole, dipole-dipole,
pole-dipole, Wenner-Schlumberger) and the general array format (type 11), and
returns one normalized :class:`Survey`.

Normalization
-------------
Whatever the input array type, a survey is reduced to:

* ``electrode_x`` / ``electrode_z`` - sorted unique electrode positions
* ``quad`` - ``(n_data, 4)`` integer array of electrode indices
  ``[C1, C2, P1, P2]``; ``-1`` marks an absent (pole) electrode
* ``rhoa`` - apparent resistivity per datum

Electrode-position reconstruction for the classic types reproduces the formulas
in ``read_data.m`` so downstream meshing / pseudo-section code matches the MATLAB
reference. The general format keeps positions as given in the file.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np

__all__ = ["Survey", "ArrayType", "read_dat"]

_DELIM = re.compile(r"[\s,]+")
_MERGE_TOL = 1e-3


class ArrayType(IntEnum):
    CENTRAL = 0
    WENNER = 1
    POLE_POLE = 2
    DIPOLE_DIPOLE = 3
    POLE_DIPOLE = 6
    SCHLUMBERGER = 7
    GENERAL = 11


_ARRAY_NAMES = {
    0: "Central-potential", 1: "Wenner", 2: "Pole-pole", 3: "Dipole-dipole",
    6: "Pole-dipole", 7: "Wenner-Schlumberger", 11: "General array",
}

# median depth-of-investigation coefficients (from read_data.m): depth = -s*(a*n + b)
_DOI = {1: (0.519, 0.0), 2: (0.867, 0.0), 3: (0.2587, 0.1759),
        6: (0.3874, 0.1489), 7: (0.3874, 0.1489), 0: (0.3874, 0.1489)}
_DZ1 = {1: 0.519, 2: 0.867, 3: 0.42, 6: 0.41, 7: 0.5, 0: 0.41, 11: 0.4}


# --------------------------------------------------------------------------- #
class _Tokens:
    """Whitespace/comma separated token cursor with whole-line skipping."""

    def __init__(self, text: str):
        self._lines = text.splitlines()
        self._li = 0
        self._buf: list[str] = []

    def _fill(self) -> bool:
        while not self._buf and self._li < len(self._lines):
            self._buf = [t for t in _DELIM.split(self._lines[self._li].strip()) if t]
            self._li += 1
        return bool(self._buf)

    def next(self) -> str:
        if not self._fill():
            raise EOFError("unexpected end of file")
        return self._buf.pop(0)

    def next_int(self) -> int:
        return int(round(float(self.next())))

    def next_float(self) -> float:
        return float(self.next())

    def floats(self, n: int) -> list[float]:
        return [self.next_float() for _ in range(n)]

    def skip_line(self) -> None:
        """Discard the remainder of the current line (like MATLAB ``fgetl``)."""
        if not self._buf:
            self._fill()          # ensure a line is loaded, then drop it
        self._buf = []


# --------------------------------------------------------------------------- #
@dataclass
class Survey:
    title: str
    spacing: float
    array_type: int
    electrode_x: np.ndarray
    electrode_z: np.ndarray
    quad: np.ndarray                       # (n_data, 4) -> [C1, C2, P1, P2]
    rhoa: np.ndarray                       # (n_data,)
    ip: bool = False
    chargeability: np.ndarray | None = None
    topography: np.ndarray | None = None
    resistance_input: bool = False
    source_path: str = ""

    n_level: np.ndarray | None = field(default=None, repr=False)
    mn: np.ndarray | None = field(default=None, repr=False)   # MN / dipole spacing per datum
    x_mid: np.ndarray | None = field(default=None, repr=False)
    pseudo_depth: np.ndarray | None = field(default=None, repr=False)
    z_max: float = 0.0
    dz1: float = 0.0
    k_geom: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        self.electrode_x = np.asarray(self.electrode_x, float)
        self.electrode_z = np.asarray(self.electrode_z, float)
        self.quad = np.asarray(self.quad, int)
        self.rhoa = np.asarray(self.rhoa, float).ravel()
        if self.n_level is None:
            self.n_level = self._infer_levels()
        else:
            self.n_level = np.asarray(self.n_level, float).ravel()
        if self.mn is not None:
            self.mn = np.asarray(self.mn, float).ravel()
        self.k_geom = self._geometric_factor()
        self._pseudo_geometry()

    # ---- sizes -------------------------------------------------------- #
    @property
    def n_data(self) -> int:
        return self.quad.shape[0]

    @property
    def n_electrodes(self) -> int:
        return self.electrode_x.size

    @property
    def array_name(self) -> str:
        return _ARRAY_NAMES.get(self.array_type, f"type {self.array_type}")

    # ---- geometry --------------------------------------------------- #
    def electrode_pos(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx)
        out = np.full(idx.shape + (2,), np.nan)
        ok = idx >= 0
        out[ok, 0] = self.electrode_x[idx[ok]]
        out[ok, 1] = self.electrode_z[idx[ok]]
        return out

    def _geometric_factor(self) -> np.ndarray:
        c1, c2, p1, p2 = (self.electrode_pos(self.quad[:, k]) for k in range(4))

        def inv_r(a, b):
            d = np.linalg.norm(a - b, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = 1.0 / d
            r[~np.isfinite(r)] = 0.0
            return r

        g = inv_r(c1, p1) - inv_r(c1, p2) - inv_r(c2, p1) + inv_r(c2, p2)
        with np.errstate(divide="ignore", invalid="ignore"):
            k = 2.0 * np.pi / g
        k[~np.isfinite(k)] = np.nan
        return k

    def _infer_levels(self) -> np.ndarray:
        c1 = self.electrode_pos(self.quad[:, 0])[:, 0]
        p1 = self.electrode_pos(self.quad[:, 2])[:, 0]
        return np.abs(p1 - c1) / self.spacing

    def _general_array_depth(self, allpos: np.ndarray) -> np.ndarray:
        """Per-datum depth of investigation for the general/merged format
        (type 11), detected from each reading's real electrode coordinates -
        NOT the file's declared sub-array code (already distrusted elsewhere
        in this reader: a merged file can freely mix array shapes under one
        code). Recognises the same classic patterns ``_DOI`` already has
        coefficients for; a row that matches none of them falls back to the
        coarse quarter-of-span estimate used before this existed.

        Coefficients (median DOI, ``read_data.m``): Wenner ``0.519 a n``,
        dipole-dipole ``a(0.2587 n + 0.1759)``, pole-dipole /
        Wenner-Schlumberger ``a(0.3874 n + 0.1489)`` (``read_data.m``'s own
        general-array reader uses one identical formula for both), pole-pole
        ``0.867 a n``.
        """
        ela = float(self.spacing)
        c1, c2, p1, p2 = (allpos[:, k, 0] for k in range(4))
        tol = max(1e-3 * ela, 1e-9)

        def close(a, b):
            return np.abs(a - b) <= tol

        has_c2 = np.isfinite(c2)
        has_p2 = np.isfinite(p2)
        d_c1p1 = p1 - c1
        d_p1p2 = p2 - p1
        d_p2c2 = c2 - p2

        depth = np.empty(c1.shape)
        left = np.ones(c1.shape, dtype=bool)

        # Wenner: C1..P1..P2..C2, three equal (signed) segments
        m = left & has_c2 & has_p2 & close(d_c1p1, d_p1p2) & close(d_p1p2, d_p2c2) & ~close(d_c1p1, 0.0)
        a, b = _DOI[1]
        depth[m] = -ela * (a * np.abs(d_c1p1[m]) / ela + b)
        left &= ~m

        # dipole-dipole: adjacent current pair AND adjacent potential pair (each ~= a)
        m = left & has_c2 & has_p2 & close(np.abs(c2 - c1), ela) & close(np.abs(d_p1p2), ela)
        a, b = _DOI[3]
        depth[m] = -ela * (a * np.abs(p1[m] - c2[m]) / ela + b)
        left &= ~m

        # pole-dipole / Wenner-Schlumberger: potential pair fixed at ~= a
        # (C2 may or may not be present - both use the same DOI formula)
        m = left & has_p2 & close(np.abs(d_p1p2), ela) & ~close(d_c1p1, 0.0)
        a, b = _DOI[6]
        depth[m] = -ela * (a * np.abs(d_c1p1[m]) / ela + b)
        left &= ~m

        # pole-pole: only C1, P1 present
        m = left & ~has_c2 & ~has_p2
        a, b = _DOI[2]
        depth[m] = -ela * (a * np.abs(d_c1p1[m]) / ela + b)
        left &= ~m

        # unrecognised: coarse fallback (quarter of the largest separation)
        if left.any():
            pos = np.stack([c1, c2, p1, p2], axis=1)
            with np.errstate(invalid="ignore"):
                span = np.nanmax(pos[left], axis=1) - np.nanmin(pos[left], axis=1)
            depth[left] = -0.25 * span
            frac = left.sum() / left.size
            if frac > 0.2:
                warnings.warn(
                    f"{left.sum()}/{left.size} general-array ({frac:.0%}) readings did not "
                    "match a recognised electrode pattern (Wenner / dipole-dipole / "
                    "pole-dipole / pole-pole); their depth of investigation used the "
                    "coarse quarter-of-span fallback and may be inaccurate.")
        return depth

    def _pseudo_geometry(self) -> None:
        allpos = self.electrode_pos(self.quad)
        with np.errstate(invalid="ignore"):
            if self.array_type == 6:
                # read_data.m (pole-dipole, classic format): xd = (A + N) / 2 -
                # the current electrode and the FAR potential electrode, not the
                # mean of all three real electrodes (A, M and N would skew the
                # plotted position toward M).  quad = [C1(=A), C2(absent), P1(=M),
                # P2(=N)].
                self.x_mid = (allpos[:, 0, 0] + allpos[:, 3, 0]) / 2.0
            else:
                self.x_mid = np.nanmean(allpos[:, :, 0], axis=1)

        if self.array_type == 11:
            self.pseudo_depth = self._general_array_depth(allpos)
            self.z_max = float(np.nanmax(np.abs(self.pseudo_depth)))
        else:
            a, b = _DOI.get(self.array_type, (0.3, 0.0))
            base = self.mn if (self.array_type == 7 and self.mn is not None) else self.spacing
            self.pseudo_depth = -base * (a * self.n_level + b)
            self.z_max = float(np.nanmax(np.abs(self.pseudo_depth)))
        self.dz1 = _DZ1.get(self.array_type, 0.4) * self.spacing

    @classmethod
    def from_file(cls, path: str | Path) -> "Survey":
        return read_dat(path)


# --------------------------------------------------------------------------- #
def read_dat(path: str | Path) -> Survey:
    if isinstance(path, str):
        path = path.strip().strip('"').strip("'")     # tolerate copy-pasted paths
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"data file not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    tok = _Tokens(text)

    title = _first_line(text)
    tok.skip_line()
    spacing = tok.next_float()
    array_type = tok.next_int()

    resistance_input = False
    if array_type == ArrayType.GENERAL:
        tok.next_int()                       # sub-array type (unreliable)
        tok.skip_line()                      # "Type of measurement ( ... )"
        resistance_input = bool(tok.next_int())

    n_data = tok.next_int()
    midpoint = tok.next_int()
    ip_flag = bool(tok.next_int())

    if array_type == ArrayType.GENERAL:
        survey = _parse_general(tok, n_data, ip_flag, resistance_input)
        _peek_topo(tok, survey)
    else:
        raw = _parse_classic(tok, int(array_type), n_data, midpoint, ip_flag)
        topo = _read_topo(tok)
        survey = _build_classic(raw, spacing, topo)

    survey.title = title
    survey.spacing = spacing
    survey.array_type = int(array_type)
    survey.ip = ip_flag
    survey.resistance_input = resistance_input
    survey.source_path = str(path)
    survey.__post_init__()
    return survey


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


# --------------------------------------------------------------------------- #
# classic per-datum formats
# --------------------------------------------------------------------------- #
@dataclass
class _ClassicRaw:
    at: int
    midpoint: int
    xd: np.ndarray
    a_spac: np.ndarray
    n_lev: np.ndarray
    rhoa: np.ndarray
    chg: np.ndarray | None


def _parse_classic(tok, at, n_data, midpoint, ip_flag) -> _ClassicRaw:
    xd = np.empty(n_data)
    a_spac = np.empty(n_data)
    n_lev = np.full(n_data, np.nan)
    rhoa = np.empty(n_data)
    chg = np.empty(n_data) if ip_flag else None

    two_col = at in (1, 2)
    ncol = (4 if ip_flag else 3) if two_col else (5 if ip_flag else 4)
    got = n_data
    for i in range(n_data):
        try:
            row = tok.floats(ncol)
        except EOFError:
            got = i
            warnings.warn(
                f"{at}-type file declares {n_data} data but only {got} present; "
                "truncating", stacklevel=2)
            break
        xd[i] = row[0]
        a_spac[i] = row[1]
        if two_col:
            rhoa[i] = row[2]
            if ip_flag:
                chg[i] = row[3]
        else:
            n_lev[i] = row[2]
            rhoa[i] = row[3]
            if ip_flag:
                chg[i] = row[4]
    if got < n_data:
        xd, a_spac, n_lev, rhoa = xd[:got], a_spac[:got], n_lev[:got], rhoa[:got]
        chg = chg[:got] if chg is not None else None
    return _ClassicRaw(at, midpoint, xd, a_spac, n_lev, rhoa, chg)


def _build_classic(raw: _ClassicRaw, spacing: float, topo: np.ndarray | None) -> Survey:
    at, mp = raw.at, raw.midpoint
    xd, a, n = raw.xd, raw.a_spac, raw.n_lev
    if at in (1, 2):
        n = a / spacing

    c1 = c2 = p1 = p2 = None
    if at == 1:
        if mp == 1:
            t = n * spacing
            c1, c2 = xd - 1.5 * t, xd + 1.5 * t
            p1, p2 = c1 + t, c1 + 2 * t
        else:
            t = n * spacing
            c1 = xd.copy(); p1 = c1 + t; p2 = p1 + t; c2 = p2 + t
    elif at == 2:
        t = n * spacing
        c1 = xd - 0.5 * t if mp == 1 else xd.copy()
        p1 = c1 + t
    elif at == 3:
        if mp == 1:
            c1 = xd - (n * spacing) / 2.0
            c2 = c1 - spacing
            p1 = c1 + n * spacing
            p2 = p1 + a
        else:
            c2 = xd.copy(); c1 = c2 + spacing
            p1 = c2 + n * spacing; p2 = p1 + a
    elif at == 6:
        # faithful port of read_data.m case 6: mp==1 stores xd already offset to
        # the A electrode by a fixed (nlev+1)/2 * spacing convention; mp==0 stores
        # xd == A directly (see the xd reassignment below, and _pseudo_geometry).
        c1 = xd - ((n + 1) * spacing) / 2.0 if mp == 1 else xd.copy()
        p1 = c1 + n * spacing
        p2 = p1 + a
    elif at in (7, 0):
        if mp == 1:
            c1 = xd - a / 2.0 - a * n
            c2 = xd + a / 2.0 + a * n
            p1 = c1 + n * a
            p2 = p1 + a
        else:
            c1 = xd.copy(); p1 = c1 + n * spacing
            p2 = p1 + a; c2 = p2 + n * spacing
    else:
        raise ValueError(f"unsupported classic array type {at}")

    cols = [c1, c2, p1, p2]
    ex = _unique_positions(np.concatenate([c for c in cols if c is not None]))
    quad = np.full((raw.xd.size, 4), -1, int)
    for j, col in enumerate(cols):
        if col is not None:
            quad[:, j] = np.abs(col[:, None] - ex[None, :]).argmin(axis=1)

    surv = Survey(
        title="", spacing=spacing, array_type=at,
        electrode_x=ex, electrode_z=_topo_z(ex, topo),
        quad=quad, rhoa=raw.rhoa, chargeability=raw.chg, topography=topo,
    )
    surv.n_level = np.asarray(n, float)
    surv.mn = np.asarray(raw.a_spac, float)
    return surv


# --------------------------------------------------------------------------- #
# general array format (type 11)
# --------------------------------------------------------------------------- #
def _parse_general(tok, n_data, ip_flag, resistance_input) -> Survey:
    coords = np.full((n_data, 4, 2), np.nan)
    vals = np.empty(n_data)
    chg = np.empty(n_data) if ip_flag else None

    for i in range(n_data):
        ne = tok.next_int()
        pts = [(tok.next_float(), tok.next_float()) for _ in range(ne)]
        vals[i] = tok.next_float()
        if ip_flag:
            chg[i] = tok.next_float()
        if ne == 4:
            coords[i] = pts
        elif ne == 3:                        # C1 P1 P2
            coords[i, 0], coords[i, 2], coords[i, 3] = pts
        elif ne == 2:                        # C1 P1
            coords[i, 0], coords[i, 2] = pts

    flat = coords.reshape(-1, 2)
    flat = flat[np.isfinite(flat[:, 0])]
    ex, ez = _unique_positions_xy(flat)

    quad = np.full((n_data, 4), -1, int)
    ep = np.column_stack([ex, ez])
    for j in range(4):
        ok = np.isfinite(coords[:, j, 0])
        if ok.any():
            d = np.linalg.norm(coords[ok, j][:, None, :] - ep[None, :, :], axis=2)
            quad[ok, j] = d.argmin(axis=1)

    surv = Survey(
        title="", spacing=1.0, array_type=11,
        electrode_x=ex, electrode_z=ez, quad=quad, rhoa=vals,
        chargeability=chg, resistance_input=resistance_input,
    )
    if resistance_input:
        surv.k_geom = surv._geometric_factor()
        surv.rhoa = vals * surv.k_geom
    return surv


# --------------------------------------------------------------------------- #
# trailer
# --------------------------------------------------------------------------- #
def _read_topo(tok) -> np.ndarray | None:
    try:
        topog = tok.next_int()
    except (EOFError, ValueError):
        return None
    if not topog:
        return None
    try:
        n_topo = tok.next_int()
        return np.array(tok.floats(2 * n_topo)).reshape(n_topo, 2)
    except (EOFError, ValueError):
        return None


def _peek_topo(tok, survey: Survey) -> None:
    topo = _read_topo(tok)
    if topo is not None:
        survey.topography = topo
        survey.electrode_z = _topo_z(survey.electrode_x, topo)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _unique_positions(x: np.ndarray) -> np.ndarray:
    xs = np.sort(x)
    keep = [xs[0]]
    for v in xs[1:]:
        if v - keep[-1] > _MERGE_TOL:
            keep.append(v)
    return np.asarray(keep)


def _unique_positions_xy(pts: np.ndarray):
    key = np.round(pts / _MERGE_TOL).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    uniq = pts[np.sort(idx)]
    uniq = uniq[np.lexsort((uniq[:, 1], uniq[:, 0]))]
    return uniq[:, 0].copy(), uniq[:, 1].copy()


def _topo_z(ex: np.ndarray, topo: np.ndarray | None) -> np.ndarray:
    if topo is None or len(topo) == 0:
        return np.zeros_like(ex)
    return np.interp(ex, topo[:, 0], topo[:, 1])
