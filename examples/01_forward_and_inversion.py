"""Forward modelling and inversion with the three parameterizations.

A layered model with two embedded bodies (the "arch" model of scripts/model_suite.py) is computed with the 2.5-D finite-element
forward solver on a mesh that follows the body boundaries, 3% noise is added, and the data are inverted with Voronoi, triangular
and rectangular cells (same Occam-type Gauss-Newton scheme and one gradient-energy smoothness functional for all).

    python examples/01_forward_and_inversion.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("ERT_REG", "energy")                  # one functional for every scheme
os.environ.setdefault("ERT_CORE_RULE", "intersect")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import model_suite as ms                                    # noqa: E402
from ert2d.io.dat_reader import read_dat                    # noqa: E402

spec = ms.SPECS["arch"]()
ms.write_all_geometries(["arch"])
geometry = read_dat(str(ms.geometry_file(spec)))
(survey,), noise_free = ms.synth_data(geometry, spec, [42])          # forward problem on the true mesh + noise
print(f"{len(survey.rhoa)} apparent resistivities, electrode spacing {survey.spacing} m")

for kind, pm in ms.build_free(survey).items():                        # inverse problem
    res, seconds = ms.run_inversion(survey, pm, spec)
    err = ms.evaluate(spec, survey, pm, res.rho_cell)
    print(f"{kind:9s} unknowns {ms.core_n(pm, survey):5d}  RMS {res.misfit:5.2f}%  recovery error {err['err']:.3f}  {seconds:.1f} s")
