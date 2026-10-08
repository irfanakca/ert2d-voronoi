"""Synthetic true-resistivity models + a forward-modeled synthetic ``.dat``
generator, for testing recovery (not just convergence / roughness / RMS) of
the three model discretizations against a KNOWN answer.

Pipeline: named true model ``rho(x, z)`` -> forward-modeled on a mesh finer
than any inversion mesh would use (avoids an "inverse crime") -> optional
calibrated noise -> written out as a RES2DINV-compatible ``.dat`` (general
array / type 11, so exact electrode coordinates are stored per reading and no
array-type-specific reconstruction formula is needed) + a companion
``.truth.npz`` raster of the true model for later recovery-error checks.

Two named models ship here:

* ``"two_blocks"`` - a background + one conductive and one resistive
  rectangular block, 12 m apart edge-to-edge (widened from an original 2 m
  gap so the two anomalies read as clearly separate features in plotted
  sections; see ``_two_blocks`` for exact positions).
  ``sample_data/synthetic_ws.dat`` is titled "Two Blocks 2 m. apart" but
  ships DATA ONLY, no recorded ground truth; this is OUR OWN plausible
  reconstruction of a model that would produce a qualitatively
  similar-looking pseudosection, not a verified recovery of whatever the
  original file's author actually used (and the block separation has since
  been deliberately changed from the title's "2 m" for legibility).
* ``"four_blocks"`` - an independently-defined 2x2 checkerboard-like layout
  (shallow/deep x left/right), unrelated to any existing sample file.

CLI:
    python scripts/synthetic_truth.py --model two_blocks --noise 0.03 --out synth_two_blocks.dat
    python scripts/synthetic_truth.py --model four_blocks --noise 0.0

Programmatic use (reused by checkerboard_test.py / compare_synthetic.py):
    from synthetic_truth import synthesize, get_true_model
    survey, rho_fn, rhoa_true = synthesize(geometry_survey, "two_blocks", noise=0.03, seed=1)
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ert2d.fem.forward import FemSpace            # noqa: E402
from ert2d.io.dat_reader import Survey, read_dat  # noqa: E402
from ert2d.mesh.domain import build_domain        # noqa: E402
from ert2d.mesh.triangulate import triangulate    # noqa: E402

RhoFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]

import os as _os

DEFAULT_GEOMETRY = (Path(_os.environ["ERT_SYN_GEOM"]) if "ERT_SYN_GEOM" in _os.environ
                    else Path(__file__).resolve().parents[1] / "sample_data" / "synthetic_ws.dat")


# --------------------------------------------------------------------------- #
# named true-resistivity models  --  rho(x, z), z <= 0 (depth negative down)
# --------------------------------------------------------------------------- #
def _two_blocks(x: np.ndarray, z: np.ndarray, bg: float = 100.0) -> np.ndarray:
    # Block separation widened from the original 2 m edge-to-edge gap
    # (conductive 30-34 m, resistive 36-40 m) to 12 m (6x the synthetic_ws.dat
    # electrode spacing of 2 m), per author feedback that the two anomalies
    # read as one blurred feature rather than two distinct ones in plotted
    # sections. New positions: conductive 26-30 m, resistive 42-46 m (block
    # centres 28 m and 44 m, 16 m apart) - both still well inside the
    # synthetic_ws.dat electrode line (0-72 m) and roughly centred on it.
    rho = np.full(np.broadcast(x, z).shape, bg, dtype=float)
    conductive = (x >= 26.0) & (x <= 30.0) & (z <= -2.0) & (z >= -6.0)
    resistive = (x >= 42.0) & (x <= 46.0) & (z <= -2.0) & (z >= -6.0)  # 12 m gap: 30-42
    rho = np.where(conductive, 10.0, rho)
    rho = np.where(resistive, 1000.0, rho)
    return rho


def _four_blocks(x: np.ndarray, z: np.ndarray, bg: float = 100.0) -> np.ndarray:
    rho = np.full(np.broadcast(x, z).shape, bg, dtype=float)
    shallow = (z <= -1.5) & (z >= -4.5)
    deep = (z <= -5.0) & (z >= -8.0)
    left = (x >= 24.0) & (x <= 32.0)
    right = (x >= 40.0) & (x <= 48.0)
    rho = np.where(shallow & left, 20.0, rho)
    rho = np.where(shallow & right, 500.0, rho)
    rho = np.where(deep & left, 500.0, rho)
    rho = np.where(deep & right, 20.0, rho)
    return rho


def _shallow_blocks(x: np.ndarray, z: np.ndarray, bg: float = 100.0) -> np.ndarray:
    """Two shallow, wide-and-thin anomalies (archaeological scale): a conductive
    (pit fill) and a resistive (wall) block, 4 m wide and 2 m high."""
    x = np.asarray(x, float) - float(_os.environ.get("ERT_MODEL_SHIFT", 0.0))   # lateral shift (grid-alignment test)
    rho = np.full(np.broadcast(x, z).shape, bg, dtype=float)
    zone = (z <= -1.2) & (z >= -3.2)
    rho = np.where(zone & (x >= 26.0) & (x <= 30.0), 10.0, rho)
    rho = np.where(zone & (x >= 42.0) & (x <= 46.0), 1000.0, rho)
    return rho


TRUE_MODELS: dict[str, RhoFunc] = {
    "shallow_blocks": _shallow_blocks,
    "two_blocks": _two_blocks,
    "four_blocks": _four_blocks,
}


def get_true_model(name: str) -> RhoFunc:
    if name not in TRUE_MODELS:
        raise ValueError(f"unknown model {name!r}, choose from {sorted(TRUE_MODELS)}")
    return TRUE_MODELS[name]


# --------------------------------------------------------------------------- #
# forward machinery (fine mesh, independent of any inversion parameterization)
# --------------------------------------------------------------------------- #
def build_fine_mesh(survey: Survey, *, refine: float = 2.5, **domain_kw):
    """A FEM mesh noticeably finer at the surface than ``build_param_mesh``'s
    default (``surface_h = spacing/2`` for ``kind="triangle"``), so
    forward-modeling the truth here and later inverting on a coarser
    parameter mesh doesn't commit an "inverse crime" (same mesh generating
    and recovering the data)."""
    dom = build_domain(survey, **domain_kw)
    mesh = triangulate(dom, surface_h=survey.spacing / refine, h_grad=1.12)
    return mesh, dom


def _match_electrodes(p2: np.ndarray, electrodes: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(p2.T[:, None, :] - electrodes[None, :, :], axis=2)
    return d.argmin(axis=0)


def forward_true_rhoa(survey: Survey, rho_of_xz: RhoFunc, *, refine: float = 2.5
                      ) -> tuple[np.ndarray, FemSpace]:
    """Forward-model apparent resistivity for ``rho_of_xz`` sampled at fine-mesh
    element centroids, through the same :class:`FemSpace` the inversion uses."""
    mesh, dom = build_fine_mesh(survey, refine=refine)
    elec_nodes = _match_electrodes(mesh.points, dom.electrodes)
    ky = 1.0 / survey.z_max
    space = FemSpace.build(mesh.points, mesh.tris, elec_nodes, ky)

    cent = mesh.points[:, mesh.tris].mean(axis=1)      # (2, M)
    rho = rho_of_xz(cent[0], cent[1])
    sigma = 1.0 / np.asarray(rho, float)
    rhoa = space.forward(sigma, survey.quad).rhoa
    return rhoa, space


def add_noise(rhoa: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative log-normal noise: ``rhoa_noisy = rhoa * exp(noise * randn())``.

    For small ``noise`` this is ~``noise`` fractional (percent) Gaussian noise
    on ``rhoa`` and EXACTLY additive Gaussian noise of std ``noise`` on
    ``log(rhoa)`` - the space ``ert2d.inversion.weights`` reasons in (median-
    deviation weights on ``log10(rhoa)``) and that ``ert2d.inversion.occam.
    misfit_rms`` reports as RMS percent.  ``noise=0.03`` ~ "3% noise"."""
    if noise <= 0:
        return rhoa.copy()
    return rhoa * np.exp(noise * rng.standard_normal(rhoa.shape))


def synthesize(
    geometry: Survey,
    model: str | RhoFunc,
    *,
    noise: float = 0.0,
    seed: int | None = None,
    refine: float = 2.5,
) -> tuple[Survey, RhoFunc, np.ndarray]:
    """Build a synthetic :class:`Survey` with the SAME geometry/quad as
    ``geometry`` but ``rhoa`` replaced by the forward-modeled (+ noisy) true
    model.  Returns ``(survey, rho_fn, rhoa_true_noise_free)``."""
    rho_fn = get_true_model(model) if isinstance(model, str) else model
    rhoa_true, _ = forward_true_rhoa(geometry, rho_fn, refine=refine)
    rng = np.random.default_rng(seed)
    rhoa_noisy = add_noise(rhoa_true, noise, rng)

    survey = dataclasses.replace(geometry, rhoa=rhoa_noisy)
    survey.__post_init__()
    return survey, rho_fn, rhoa_true


# --------------------------------------------------------------------------- #
# RES2DINV writer - general-array / type 11 format: explicit (x, z) per
# electrode per reading, so it's exact regardless of any array-type-specific
# position-reconstruction formula (see ert2d.io.dat_reader._build_classic).
# --------------------------------------------------------------------------- #
def write_dat(path: str | Path, survey: Survey, *, title: str = "synthetic") -> Path:
    path = Path(path)
    lines = [
        title,
        f"{survey.spacing:.6f}",
        "11",                                            # array type = GENERAL
        "11",                                             # sub-array type (unused by the reader)
        "Type of measurement (0 = normal resistance, 1 = resistance)",
        "0",                                              # resistance_input = False (values are rhoa)
        f"{survey.n_data}",
        "0",                                              # midpoint flag (unused for GENERAL)
        "0",                                              # ip_flag = False
    ]
    for row, rhoa in zip(survey.quad, survey.rhoa):
        used = row[row >= 0]
        n_used = int(used.size)
        pts = " ".join(f"{survey.electrode_x[i]:.4f} {survey.electrode_z[i]:.4f}"
                       for i in row if i >= 0)
        lines.append(f"{n_used} {pts} {rhoa:.6f}")
    lines.append("0")                                     # no topography trailer
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_truth_npz(path: str | Path, model_name: str, rho_fn: RhoFunc, survey: Survey,
                   *, noise: float, seed: int | None, nx: int = 300, nz: int = 120
                   ) -> Path:
    """Raster of the true model over the survey extent, for later
    recovery-error comparisons (a recovered model can be resampled onto the
    same grid, or ``rho_fn`` evaluated directly at any point of interest)."""
    path = Path(path)
    x = np.linspace(survey.electrode_x.min(), survey.electrode_x.max(), nx)
    z = np.linspace(-1.2 * survey.z_max, 0.0, nz)
    X, Z = np.meshgrid(x, z)
    rho = rho_fn(X, Z)
    np.savez_compressed(
        path, model=model_name, x=x, z=z, rho=rho,
        noise=float(noise), seed=(-1 if seed is None else int(seed)),
        electrode_x=survey.electrode_x, electrode_z=survey.electrode_z,
    )
    return path


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(TRUE_MODELS), default="two_blocks")
    ap.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY,
                    help="template .dat providing electrode positions + quad "
                         "(array geometry only - its rhoa is discarded)")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="log-normal noise std, fractional (0.03 ~ 3%% noise)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--refine", type=float, default=2.5,
                    help="fine forward mesh surface_h = spacing/refine")
    ap.add_argument("--out", type=Path, default=None,
                    help="output .dat path (default synth_<model>.dat)")
    args = ap.parse_args()

    geometry = read_dat(args.geometry)
    survey, rho_fn, rhoa_true = synthesize(
        geometry, args.model, noise=args.noise, seed=args.seed, refine=args.refine)

    out = args.out or Path(f"synth_{args.model}.dat")
    write_dat(out, survey,
              title=f"Synthetic {args.model} (ert2d synthetic_truth.py, noise={args.noise})")
    truth_path = out.with_suffix(".truth.npz")
    save_truth_npz(truth_path, args.model, rho_fn, survey, noise=args.noise, seed=args.seed)

    print(f"{args.model}: {survey.n_data} data, {survey.n_electrodes} electrodes "
          f"(geometry from {args.geometry.name})")
    print(f"  noise-free rhoa range {rhoa_true.min():.2f} - {rhoa_true.max():.2f} ohm-m"
          f"  (noise={args.noise})")
    print(f"  wrote {out}")
    print(f"  wrote {truth_path}  (true-model raster, for recovery-error checks)")

    check = read_dat(out)
    print(f"  round-trip check: read_dat parsed {check.n_data} data, "
          f"{check.n_electrodes} electrodes, array={check.array_name}")


if __name__ == "__main__":
    main()
