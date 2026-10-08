"""``.loc`` line-location files.

One ``.loc`` sits next to each ``.dat`` with the same stem.  Minimal format:
two coordinate lines - the start and end of the profile in a projected CRS
(easting, northing, metres) - blank lines and ``#`` comments ignored::

    # Kelenderis line 3
    412340.5   4041880.2
    412370.5   4041862.7

A per-electrode listing is also accepted, with a leading index column
ignored (only the start and end rows are used)::

    1   0.0000    0.0000
    2   1.2500   -0.0654
    ...
    48  58.7500  -3.0747

For each line the *last* two numbers found are taken as (x, y) - so an
optional leading index never gets mistaken for a coordinate.

(``northing easting`` order is detected and swapped, see
:func:`_as_easting_northing`.)

Electrodes are assumed equally spaced along the straight segment between the
two endpoints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass
class LineLocation:
    start: np.ndarray          # (2,) easting, northing
    end: np.ndarray            # (2,)
    source_path: str = ""

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def azimuth_deg(self) -> float:
        d = self.end - self.start
        return float((np.degrees(np.arctan2(d[0], d[1]))) % 360.0)

    def positions_along(self, x_local: np.ndarray) -> np.ndarray:
        """Map along-line distances (metres from the first electrode) to
        projected coordinates.  Returns ``(n, 2)``."""
        x = np.asarray(x_local, float)
        span = x.max() - x.min()
        t = np.zeros_like(x) if span == 0 else (x - x.min()) / span
        return self.start[None, :] + t[:, None] * (self.end - self.start)[None, :]


def read_loc(path: str | Path) -> LineLocation:
    path = Path(path)
    rows = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        nums = [float(m) for m in _NUM.findall(line)]
        if len(nums) >= 2:
            rows.append(nums[-2:])
    if len(rows) < 2:
        raise ValueError(f"{path.name}: need two coordinate lines, got {len(rows)}")
    return LineLocation(_as_easting_northing(np.array(rows[0])),
                        _as_easting_northing(np.array(rows[-1])), str(path))


def _as_easting_northing(xy: np.ndarray) -> np.ndarray:
    """Accept ``northing easting`` as well as ``easting northing``.

    A projected easting (UTM/TM zones) never reaches 1 000 000 m, while a
    northing above 1 000 000 m is the norm outside the tropics - so a pair
    whose first value is >= 1e6 and second < 1e6 is northing-first and is
    swapped.  Local (small) coordinates are left as they are."""
    if xy[0] >= 1.0e6 and abs(xy[1]) < 1.0e6:
        return xy[::-1].copy()
    return xy


def loc_path_for(dat_path: str | Path) -> Path:
    return Path(dat_path).with_suffix(".loc")
