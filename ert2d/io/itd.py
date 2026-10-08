"""``.itd`` — a saved inversion result, next to its ``.dat`` file.

Same stem, ``.itd`` extension. A pickle of :class:`~ert2d.inversion.driver.
InversionResult` (with the FEM solve state stripped out - ``FemSpace.solver``
wraps a SuperLU factorisation, which isn't picklable and isn't needed to
redraw or resample the model) so a previous result can be shown, or fed into
a depth map, without re-inverting.
"""

from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

from ert2d.inversion.driver import InversionResult


def itd_path_for(dat_path: str | Path) -> Path:
    return Path(dat_path).with_suffix(".itd")


def save_itd(path: str | Path, result: InversionResult) -> None:
    light = replace(result, space=None)
    with open(path, "wb") as f:
        pickle.dump(light, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_itd(path: str | Path) -> InversionResult:
    with open(path, "rb") as f:
        result = pickle.load(f)
    if not isinstance(result, InversionResult):
        raise ValueError(f"{path}: not an ert2d .itd result")
    return result
