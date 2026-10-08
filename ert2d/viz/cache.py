"""Cache computed mesh/inversion results to disk, separately from plotting.

Every figure-producing script under ``scripts/`` follows the same pattern
now:

    data = compute(...)                 # expensive: mesh generation, inversion
    save_cache(npz_path, data, meta)    # cheap: plain arrays/scalars only
    plot(data)                          # cheap, pure function of `data`

and a figure can be regenerated later purely from disk, without recomputing
any mesh or inversion, via ``--replot-from <npz_path>``:

    data, meta = load_cache(npz_path)
    plot(data)

``arrays`` values may be numpy arrays, plain Python scalars/strings, or
ragged sequences of per-cell arrays (e.g. polygon vertex rings from
``ert2d.viz.sections.flatten_cell_polys``, one variable-length ``(n_i, 2)``
array per polygon) - the latter are stored as ``dtype=object`` arrays via
``np.savez_compressed(..., allow_pickle)`` and come back exactly as a plain
Python ``list`` of arrays after :func:`load_cache`.

``meta`` is for small JSON-safe bookkeeping (CLI args used, dataset name,
free-form notes) that's convenient to read without touching ``numpy`` at all;
it is optional and stored alongside the ``.npz`` as ``<stem>.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _is_ragged(v: Any) -> bool:
    """True for a list/tuple of numpy arrays that don't all share a shape
    (e.g. one polygon's vertex ring per parameter cell) - needs an
    ``object``-dtype array to round-trip through ``.npz``."""
    if not isinstance(v, (list, tuple)) or not v:
        return False
    if not all(isinstance(e, np.ndarray) for e in v):
        return False
    shapes = {e.shape for e in v}
    return len(shapes) != 1


def save_cache(npz_path: str | Path, arrays: dict[str, Any],
               meta: dict[str, Any] | None = None) -> Path:
    """Write ``arrays`` to ``npz_path`` (``.npz``) and, if given, ``meta`` to
    the sibling ``.json`` (same stem). Returns the ``.npz`` path."""
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    to_save: dict[str, np.ndarray] = {}
    for k, v in arrays.items():
        if isinstance(v, np.ndarray):
            to_save[k] = v
        elif _is_ragged(v):
            arr = np.empty(len(v), dtype=object)
            for i, e in enumerate(v):
                arr[i] = e
            to_save[k] = arr
        else:
            to_save[k] = np.asarray(v)
    np.savez_compressed(npz_path, **to_save)

    if meta is not None:
        npz_path.with_suffix(".json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return npz_path


def load_cache(npz_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Inverse of :func:`save_cache`. Returns ``(arrays, meta)``, where
    object-dtype (ragged) entries come back as plain ``list``s of arrays and
    everything else stays a numpy array (0-d arrays for scalars - use
    ``.item()`` or rely on numpy's usual scalar broadcasting). ``meta`` is
    ``{}`` if no companion ``.json`` file exists."""
    npz_path = Path(npz_path)
    arrays: dict[str, Any] = {}
    with np.load(npz_path, allow_pickle=True) as d:
        for k in d.files:
            v = d[k]
            arrays[k] = list(v) if v.dtype == object else v

    json_path = npz_path.with_suffix(".json")
    meta = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
    return arrays, meta
