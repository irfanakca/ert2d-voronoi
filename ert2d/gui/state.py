"""Application state: the set of loaded data files and their results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ert2d.io.dat_reader import Survey, read_dat
from ert2d.io.loc import LineLocation, loc_path_for, read_loc


@dataclass
class MeshSettings:
    kind: str = "voronoi"               # "voronoi" | "triangle" | "rect"
    wing: float = 5.0
    depth_factor: float = 4.0
    surface_div: int = 2


@dataclass
class InversionSettings:
    max_iter: int = 12
    lambda_mode: str = "search"          # "search" | "matlab"
    lam0: float | None = None
    cooling: float = 0.8                 # matlab mode
    target_rms: float | None = None      # search mode (%)
    tol: float = 0.03
    robust: str = "none"                 # IRLS data weighting: none|huber|cauchy|bisquare
    robust_min_weight: float = 0.5       # lower bound of the residual-based weight factor


@dataclass
class VisualSettings:
    colormap: str = "turbo"             # shared by model AND data panels
    reverse_colormap: bool = False
    log_scale: bool = True
    model_style: str = "cells"          # "cells" | "contour"
    show_points: bool = True
    show_electrodes: bool = False
    show_contour_lines: bool = False
    show_labels: bool = False
    sensitivity_alpha: bool = False
    # blank the model's lower-left / lower-right corners the data never reach
    mask_corners: bool = True
    mask_widen_pct: float = 25.0        # 0 = data flank, 100 = no cut
    mask_shape: str = "triangle"        # "triangle" | "ellipse"
    # colour-scale limits in Ω·m (linear); None = auto from the data/model range
    data_clim_min: float | None = None
    data_clim_max: float | None = None
    model_clim_min: float | None = None
    model_clim_max: float | None = None

    @property
    def cmap(self) -> str:
        return self.colormap + "_r" if self.reverse_colormap else self.colormap

    @property
    def data_clim(self) -> tuple[float, float] | None:
        return _valid_clim(self.data_clim_min, self.data_clim_max)

    @property
    def model_clim(self) -> tuple[float, float] | None:
        return _valid_clim(self.model_clim_min, self.model_clim_max)


@dataclass
class ContourSettings:
    """Depth Map contouring + raster filters (see ``ert2d.viz.depth_map``).

    ``interp_method="triangulation"`` reproduces the original Delaunay
    contouring (no extrapolation past the data hull); any other method, or
    turning on a filter (all of which need a regular grid to operate on),
    grids the depth slice first."""
    n_levels: int = 48
    fill: bool = True                    # filled bands vs. contour lines only
    interp_method: str = "triangulation"  # "triangulation"|"linear"|"cubic"|"nearest"
    grid_step: float = 1.0               # metres
    smooth_on: bool = False
    smooth_sigma: float = 1.0            # grid cells
    sharpen_on: bool = False
    sharpen_amount: float = 1.0
    emboss_on: bool = False
    emboss_strength: float = 1.0
    hillshade_on: bool = False
    hillshade_azimuth: float = 315.0
    hillshade_altitude: float = 45.0
    hillshade_strength: float = 1.0


def _valid_clim(lo, hi) -> tuple[float, float] | None:
    """Both bounds set, positive (Ω·m plots are always log-capable) and
    ordered - otherwise fall back to the auto range rather than plot garbage."""
    if lo is None or hi is None or lo <= 0 or hi <= 0 or lo >= hi:
        return None
    return (lo, hi)


@dataclass
class FileItem:
    path: Path
    project: str = ""
    survey: Survey | None = None
    location: LineLocation | None = None
    result: "object | None" = None            # InversionResult (final)
    partial: "object | None" = None           # live result during iteration
    load_error: str | None = None
    mesh: MeshSettings = field(default_factory=MeshSettings)
    inversion: InversionSettings = field(default_factory=InversionSettings)

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def is_loaded(self) -> bool:
        return self.survey is not None

    @property
    def is_inverted(self) -> bool:
        return self.result is not None

    @property
    def display_result(self):
        """Final result if present, else the live partial one (or None)."""
        return self.result or self.partial

    def load(self) -> None:
        try:
            self.survey = read_dat(self.path)
            self.load_error = None
        except Exception as exc:                # noqa: BLE001
            self.load_error = str(exc)
            self.survey = None
        loc = loc_path_for(self.path)
        if loc.is_file():
            try:
                self.location = read_loc(loc)
            except Exception:                   # noqa: BLE001
                self.location = None
        itd = self.path.with_suffix(".itd")
        if itd.is_file():
            try:
                from ert2d.io.itd import load_itd
                self.result = load_itd(itd)
            except Exception:                   # noqa: BLE001
                self.result = None


class AppState:
    """All loaded files, grouped by project (their parent folder name)."""

    def __init__(self):
        self.files: dict[Path, FileItem] = {}

    # ------------------------------------------------------------------ #
    def add_paths(self, paths, project: str | None = None,
                  *, skip_invalid: bool = False) -> list[FileItem]:
        added = []
        for p in paths:
            p = Path(p)
            if p.suffix.lower() != ".dat" or p in self.files:
                continue
            item = FileItem(p, project or p.parent.name)
            item.load()
            if skip_invalid and not item.is_loaded:
                continue                             # not a readable data file
            self.files[p] = item
            added.append(item)
        return added

    def add_folder(self, folder) -> list[FileItem]:
        folder = Path(folder)
        return self.add_paths(sorted(folder.glob("*.dat")), project=folder.name,
                              skip_invalid=True)

    def remove(self, path) -> None:
        self.files.pop(Path(path), None)

    def by_project(self) -> dict[str, list[FileItem]]:
        out: dict[str, list[FileItem]] = {}
        for item in self.files.values():
            out.setdefault(item.project, []).append(item)
        for v in out.values():
            v.sort(key=lambda it: _natural_key(it.name))
        return out


def _natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]
