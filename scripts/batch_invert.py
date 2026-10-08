"""Invert every .dat file in a folder — the Python equivalent of the top-level
loop in ``main_vor_v00.m``.

    python scripts/batch_invert.py  <folder>  [--glob "P*.dat"] [--iters 10]
                                    [--figs] [--start N]

For each file: build mesh, invert, save ``<name>_inv.npz`` next to it, and
(with ``--figs``) write ``<name>_tab1.png`` / ``<name>_tab2.png``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ert2d.io.dat_reader import read_dat                        # noqa: E402
from ert2d.io.results import save_result                        # noqa: E402
from ert2d.mesh import build_domain, triangulate, voronoi_parameterization  # noqa: E402
from ert2d.inversion import invert                              # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--glob", default="*.dat")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--start", type=int, default=0, help="skip the first N files")
    ap.add_argument("--figs", action="store_true")
    args = ap.parse_args()

    files = sorted(args.folder.glob(args.glob))[args.start:]
    if not files:
        sys.exit(f"no files match {args.folder}/{args.glob}")
    print(f"{len(files)} files\n")

    if args.figs:
        import matplotlib
        matplotlib.use("Agg")
        from ert2d.viz import data_fit_figure, diagnostics_figure

    rows = []
    for f in files:
        t0 = time.perf_counter()
        try:
            survey = read_dat(f)
            dom = build_domain(survey)
            pm = voronoi_parameterization(triangulate(dom), dom)
            res = invert(survey, pm, max_iter=args.iters)
            save_result(f.with_name(f.stem + "_inv"), res, survey)
            if args.figs:
                data_fit_figure(survey, res).savefig(
                    f.with_name(f.stem + "_tab1.png"), dpi=120)
                diagnostics_figure(survey, res).savefig(
                    f.with_name(f.stem + "_tab2.png"), dpi=120)
            dt = time.perf_counter() - t0
            print(f"{f.name:28} {survey.array_name:20} nd={survey.n_data:4d} "
                  f"cells={pm.n_param:4d}  RMS {res.rms_curve[0]:6.2f} -> "
                  f"{res.misfit:6.2f}%  ({dt:4.1f}s)")
            rows.append((f.name, res.misfit))
        except Exception as exc:  # noqa: BLE001
            print(f"{f.name:28} FAILED  {exc!r}")

    if rows:
        m = np.array([r[1] for r in rows])
        print(f"\n{len(rows)} inverted   final RMS  mean {m.mean():.2f}%  "
              f"median {np.median(m):.2f}%  worst {m.max():.2f}%")


if __name__ == "__main__":
    main()
