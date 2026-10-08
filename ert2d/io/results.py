"""Save / load inversion results and resample the model to a regular grid.

Replaces the ``save([folder,'\\',name,'.mat'], ...)`` call at the end of
``invert()`` in ``main_vor_v00.m``.  Uses ``.npz`` (portable, no MATLAB needed);
``scipy.io.savemat`` is offered too for interoperability.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def to_regular_grid(result, *, dx: float | None = None, dz: float | None = None,
                    survey=None):
    """Interpolate the element-wise model resistivity onto a regular grid,
    like the final ``griddata`` block in ``main_vor_v00`` (node values ->
    ``meshgrid`` -> ``RO``)."""
    from scipy.interpolate import griddata

    pm = result.param_mesh
    cent = pm.p2[:, pm.t2].mean(axis=1)          # (2, M) element centroids
    rho_e = result.rho_elem()

    x0, x1 = pm.p2[0].min(), pm.p2[0].max()
    z0, z1 = pm.p2[1].min(), 0.0
    if survey is not None:
        x0, x1 = survey.electrode_x.min(), survey.electrode_x.max()
        z0 = -survey.z_max
    dx = dx or (x1 - x0) / 200.0
    dz = dz or (z1 - z0) / 60.0

    xi = np.arange(x0, x1 + dx / 2, dx)
    zi = np.arange(z0, z1 + dz / 2, dz)
    XI, ZI = np.meshgrid(xi, zi)
    RO = griddata((cent[0], cent[1]), rho_e, (XI, ZI), method="linear")
    RO_n = griddata((cent[0], cent[1]), rho_e, (XI, ZI), method="nearest")
    RO = np.where(np.isfinite(RO), RO, RO_n)
    return xi, zi, RO


def save_result(path: str | Path, result, survey=None, *, matlab: bool = False):
    path = Path(path)
    pm = result.param_mesh
    data = {
        "rho_cell": result.rho_cell,
        "rho_elem": result.rho_elem(),
        "sigma_elem": result.sigma_elem,
        "p2": pm.p2,
        "t2": pm.t2,
        "rhoa_obs": result.rhoa_obs,
        "rhoa_calc": result.rhoa_calc,
        "residual_log": result.residual_log,
        "rms_curve": result.rms_curve,
        "weights": result.weights,
    }
    if result.jacobian is not None:
        data["jacobian"] = result.jacobian
    if survey is not None:
        xi, zi, RO = to_regular_grid(result, survey=survey)
        data.update(xi=xi, zi=zi, RO=RO,
                    electrode_x=survey.electrode_x, electrode_z=survey.electrode_z)

    if matlab:
        from scipy.io import savemat
        savemat(path.with_suffix(".mat"), data)
    else:
        np.savez_compressed(path.with_suffix(".npz"), **data)
    return path


def load_result(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix == ".mat":
        from scipy.io import loadmat
        return {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
