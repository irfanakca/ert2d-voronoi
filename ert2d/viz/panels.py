"""Composite figures matching the planned two-tab GUI layout.

Tab 1 (:func:`data_fit_figure`)   - measured / calculated pseudo-section + model
Tab 2 (:func:`diagnostics_figure`) - per-datum misfit pseudo-section, RMS curve,
                                     measured-vs-calculated crossplot
"""

from __future__ import annotations

import numpy as np

from ert2d.viz.sections import (crossplot, model_section, pseudosection,
                                rms_curve)
from ert2d.viz.sensitivity import parameter_alpha


def _lazy_mpl():
    import matplotlib.pyplot as plt
    return plt


def data_fit_figure(survey, result, *, figsize=(9, 10), sensitivity_alpha=True):
    """Three stacked panels: measured pseudo-section, calculated pseudo-section,
    inverted model section (MATLAB ``subplot(311/312/313)``)."""
    plt = _lazy_mpl()
    fig, ax = plt.subplots(3, 1, figsize=figsize)

    x, pd = survey.x_mid, survey.pseudo_depth
    elec = survey.electrode_x
    obs, cal = result.rhoa_obs, result.rhoa_calc

    lo = float(np.log10(np.nanmin([obs.min(), cal.min()])))
    hi = float(np.log10(np.nanmax([obs.max(), cal.max()])))

    cf0 = pseudosection(x, pd, obs, ax=ax[0], clim=(lo, hi), electrodes=elec,
                        title="Measured apparent resistivity")
    fig.colorbar(cf0, ax=ax[0], label=r"$\log_{10}\rho_a$")

    cf1 = pseudosection(x, pd, cal, ax=ax[1], clim=(lo, hi), electrodes=elec,
                        title="Calculated apparent resistivity")
    fig.colorbar(cf1, ax=ax[1], label=r"$\log_{10}\rho_a$")

    alpha = None
    if sensitivity_alpha and result.jacobian is not None:
        alpha = parameter_alpha(result.jacobian)
    model_section(
        result.param_mesh, result.rho_cell, ax=ax[2], alpha_cell=alpha,
        electrodes=elec, x_lim=(elec.min(), elec.max()), z_lim=survey.z_max,
        title=f"Inverted model   (iter {len(result.history)}, "
              f"RMS {result.misfit:.2f}%)")

    fig.tight_layout()
    return fig


def diagnostics_figure(survey, result, *, figsize=(9, 9)):
    """Per-datum misfit pseudo-section (top), RMS-vs-iteration and crossplot
    (bottom row)."""
    plt = _lazy_mpl()
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])
    ax_ps = fig.add_subplot(gs[0, :])
    ax_rms = fig.add_subplot(gs[1, 0])
    ax_xp = fig.add_subplot(gs[1, 1])

    cf = pseudosection(
        survey.x_mid, survey.pseudo_depth, result.datum_misfit_pct(),
        ax=ax_ps, log10=False, cmap="magma_r", electrodes=survey.electrode_x,
        title="Per-datum weighted misfit (%)")
    fig.colorbar(cf, ax=ax_ps, label="misfit (%)")

    rms_curve(result.rms_curve, ax=ax_rms)
    crossplot(result.rhoa_obs, result.rhoa_calc, ax=ax_xp)

    fig.tight_layout()
    return fig
