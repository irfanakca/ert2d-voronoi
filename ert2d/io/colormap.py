"""Load Geosoft-style ``.clr`` colour tables as matplotlib colormaps.

A ``.clr`` file has one header line (``ColorMap <r> <g>`` - ignored) followed
by rows of ``position  R  G  B``, ``position`` running 0-100 and R/G/B 0-255::

    ColorMap 2 1
        0          0     0   255
        2.6316     0    85   255
        ...
      100.0000   255   159   255
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_clr(path: str | Path, name: str | None = None):
    """Parse a ``.clr`` file into a :class:`matplotlib.colors.LinearSegmentedColormap`."""
    from matplotlib.colors import LinearSegmentedColormap

    path = Path(path)
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue                            # header line / blank / malformed
        try:
            pos, r, g, b = (float(p) for p in parts)
        except ValueError:
            continue
        rows.append((pos, r, g, b))
    if len(rows) < 2:
        raise ValueError(f"{path.name}: no colour stops found")

    rows.sort(key=lambda row: row[0])
    lo, hi = rows[0][0], rows[-1][0]
    span = hi - lo or 1.0
    stops = [((pos - lo) / span, (r / 255.0, g / 255.0, b / 255.0))
             for pos, r, g, b in rows]
    return LinearSegmentedColormap.from_list(name or path.stem, stops)


def register_clr(path: str | Path, name: str | None = None, *, force: bool = True) -> str:
    """Load a ``.clr`` file and register it (plus its ``_r`` reverse) with
    matplotlib's global colormap registry, so it can be used anywhere a
    colormap *name* string is expected. Returns the registered name."""
    import warnings

    import matplotlib as mpl

    cmap = load_clr(path, name)
    name = cmap.name
    with warnings.catch_warnings():
        # re-registering on GUI restart / repeated "Load…" is expected, not a bug
        warnings.filterwarnings("ignore", message="Overwriting the cmap")
        mpl.colormaps.register(cmap, name=name, force=force)
        mpl.colormaps.register(cmap.reversed(), name=name + "_r", force=force)
    return name
