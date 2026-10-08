"""End-to-end inversion of one .dat file + the two GUI-style figures.

    python scripts/invert_demo.py ../kelenderis/k3.dat
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ert2d.io.dat_reader import read_dat                       # noqa: E402
from ert2d.mesh import build_domain, triangulate, voronoi_parameterization  # noqa: E402
from ert2d.inversion import invert                             # noqa: E402


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "../A12_23.dat"
    dat = Path(arg.strip().strip('"').strip("'")).expanduser()
    if not dat.is_file():
        sys.exit(f"file not found: {dat}")
    survey = read_dat(dat)
    print(f"{dat.name}: {survey.array_name}, {survey.n_data} data, "
          f"{survey.n_electrodes} electrodes")

    t0 = time.perf_counter()
    dom = build_domain(survey)
    mesh = triangulate(dom)
    pm = voronoi_parameterization(mesh, dom)
    print(f"mesh: {mesh.n_nodes} nodes / {mesh.n_tris} tris, "
          f"{pm.n_param} Voronoi cells  ({time.perf_counter() - t0:.1f}s)")

    def cb(info):
        print(f"  iter {info.iteration:2d}  lambda {info.lam:7.4f}  "
              f"RMS {info.misfit_before:7.3f}% -> {info.misfit_after:7.3f}%")

    t1 = time.perf_counter()
    res = invert(survey, pm, max_iter=8, callback=cb)
    print(f"done in {time.perf_counter() - t1:.1f}s   final RMS {res.misfit:.3f}%")
    print(f"model rho: min {res.rho_cell.min():.1f}  median "
          f"{np.median(res.rho_cell):.1f}  max {res.rho_cell.max():.1f}  ohm-m")

    try:
        import matplotlib
        matplotlib.use("Agg")
        from ert2d.viz import data_fit_figure, diagnostics_figure
    except ImportError:
        return
    out = Path(__file__).resolve().parents[1]
    f1 = data_fit_figure(survey, res)
    f1.savefig(out / f"tab1_{dat.stem}.png", dpi=130)
    f2 = diagnostics_figure(survey, res)
    f2.savefig(out / f"tab2_{dat.stem}.png", dpi=130)
    print(f"wrote tab1_{dat.stem}.png, tab2_{dat.stem}.png")


if __name__ == "__main__":
    main()
