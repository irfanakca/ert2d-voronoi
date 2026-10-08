"""Pseudo-section and model-section plots.

``pseudosection`` fills only the actual data footprint (a Delaunay triangulation
of the data points, with over-stretched boundary triangles masked) - no
extrapolation into empty regions, matching ``pseudo.m``'s ``griddata`` +
``contourf`` behaviour where natural-neighbour interpolation returns NaN outside
the data hull.

``model_section`` draws the Voronoi parameter cells directly (piecewise-constant
resistivity), so there are no internal FEM-triangle edges.
"""

from __future__ import annotations

import numpy as np


def _lazy_mpl():
    import matplotlib.pyplot as plt
    return plt


def format_log_colorbar(cbar, n_ticks: int = 6) -> None:
    """Relabel a colorbar whose data is ``log10(x)`` to show linear ``x``
    (whole numbers, no decimals)."""
    lo, hi = cbar.mappable.get_clim()
    ticks = np.linspace(lo, hi, n_ticks)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(
        [f"{10 ** v:.0f}" if 10 ** v >= 1 else f"{10 ** v:.2g}" for v in ticks])


# --------------------------------------------------------------------------- #
# cache / decouple-compute-from-plot helpers
# --------------------------------------------------------------------------- #
def flatten_cell_polys(cell_polys):
    """Flatten a ``ParamMesh.cell_polys`` list (one ``Polygon``/``MultiPolygon``
    per parameter cell, possibly empty) into a flat, ``.npz``-cacheable form:

        poly_verts : list[np.ndarray (n_i, 2)]   -- one array per polygon RING
                                                     (a MultiPolygon cell
                                                     contributes >1 entry)
        cell_idx   : np.ndarray[int] (n_polys,)  -- the originating parameter
                                                     cell index ``k`` for each
                                                     entry of ``poly_verts``,
                                                     so ``rho_cell[cell_idx]``
                                                     (etc.) lines up 1:1 with
                                                     ``poly_verts``.

    This is exactly the geometry a ``PolyCollection`` needs, with no shapely
    objects left in it - safe to round-trip through ``np.savez`` (as a ragged
    ``dtype=object`` array) and reload without shapely even being imported.
    """
    poly_verts, cell_idx = [], []
    for k, poly in enumerate(cell_polys):
        if poly is None or poly.is_empty:
            continue
        geoms = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
        for g in geoms:
            poly_verts.append(np.asarray(g.exterior.coords))
            cell_idx.append(k)
    return poly_verts, np.asarray(cell_idx, dtype=int)


def panel_figsize(x_extent, z_extent, depth_exaggeration: float = 3.0, *,
                  width: float = 5.0, margin: float = 1.5,
                  min_height: float = 2.4, max_height: float = 11.0):
    """Figure size for a single (x, z) section panel drawn at
    ``ax.set_aspect(depth_exaggeration)`` (see ``model_section``), sized from
    the panel's ACTUAL data extent rather than a fixed default - so panels
    stay legible regardless of how thin/tall the true survey geometry is.

    ``width`` is the axes data-width budget (inches); the data-height is then
    ``width * (dz / dx) * depth_exaggeration`` (matplotlib's own
    ``set_aspect(a)`` convention: a circle of equal x/z data extent is drawn
    ``a`` times taller than wide), plus ``margin`` inches for title/labels/
    colorbar, clipped to ``[min_height, max_height]`` so pathological extents
    (e.g. a single electrode line) don't produce a degenerate figure.
    """
    dx = max(float(x_extent[1] - x_extent[0]), 1e-6)
    dz = max(float(abs(z_extent[1] - z_extent[0])), 1e-6)
    data_h = width * (dz / dx) * float(depth_exaggeration)
    height = min(max(data_h + margin, min_height), max_height)
    return (width, height)


# --------------------------------------------------------------------------- #
def _masked_triangulation(x, z, mask_ratio: float):
    """Delaunay of the data points with long/thin boundary triangles removed."""
    import matplotlib.tri as mtri

    tri = mtri.Triangulation(x, z)
    if mask_ratio and tri.triangles.size:
        t = tri.triangles
        # edge lengths per triangle
        d = np.stack([
            np.hypot(x[t[:, 0]] - x[t[:, 1]], z[t[:, 0]] - z[t[:, 1]]),
            np.hypot(x[t[:, 1]] - x[t[:, 2]], z[t[:, 1]] - z[t[:, 2]]),
            np.hypot(x[t[:, 2]] - x[t[:, 0]], z[t[:, 2]] - z[t[:, 0]]),
        ], axis=1)
        med = np.median(d)
        tri.set_mask(d.max(axis=1) > mask_ratio * med)
    return tri


def pseudosection(
    x_mid: np.ndarray,
    pseudo_depth: np.ndarray,
    values: np.ndarray,
    *,
    ax=None,
    log10: bool = True,
    levels: int = 48,
    cmap: str = "jet",
    clim: tuple[float, float] | None = None,
    title: str | None = None,
    electrodes: np.ndarray | None = None,
    show_electrodes: bool = True,
    xlim: tuple[float, float] | None = None,
    xlabel: str = "Distance (m)",
    ylabel: str = "Pseudo-depth (m)",
    mask_ratio: float = 4.0,
    show_points: bool = True,
    contour_lines: bool = False,
    contour_labels: bool = False,
    fill: bool = True,
):
    """Filled-contour pseudo-section, drawn only where data exist.

    ``electrodes`` is used both for the marker triangles (when
    ``show_electrodes``) and, unless ``xlim`` is given, to fix the x-limits -
    so toggling the markers never changes the axis range.

    ``fill=False`` draws coloured contour *lines* only (no filled bands) -
    ``contour_lines``/``contour_labels`` then apply to those lines directly
    instead of a separate black overlay.
    """
    plt = _lazy_mpl()
    if ax is None:
        _, ax = plt.subplots()

    x = np.asarray(x_mid, float)
    z = np.asarray(pseudo_depth, float)
    v = np.asarray(values, float)
    if log10:
        v = np.log10(v)

    if clim is None:
        lev = np.linspace(np.nanmin(v), np.nanmax(v), levels)
    else:
        lev = np.linspace(clim[0], clim[1], levels)
        v = np.clip(v, clim[0], clim[1])

    tri = _masked_triangulation(x, z, mask_ratio)
    if fill:
        cf = ax.tricontourf(tri, v, levels=lev, cmap=cmap, extend="neither")
        if contour_lines:
            cs = ax.tricontour(tri, v, levels=np.linspace(lev[0], lev[-1], 9),
                               colors="k", linewidths=0.4)
            if contour_labels:
                ax.clabel(cs, fmt=(lambda x: f"{10 ** x:.0f}") if log10 else "%.0f",
                          fontsize=7)
        try:                                # kill faint lines between levels
            cf.set_edgecolor("face")
        except Exception:                   # noqa: BLE001  (old matplotlib)
            for coll in getattr(cf, "collections", []):
                coll.set_edgecolor("face")
    else:
        cf = ax.tricontour(tri, v, levels=lev, cmap=cmap)
        if contour_labels:
            ax.clabel(cf, fmt=(lambda x: f"{10 ** x:.0f}") if log10 else "%.0f",
                      fontsize=7)

    if show_points:
        ax.plot(x, z, "+", color="0.25", ms=2, alpha=0.5)
    if electrodes is not None and show_electrodes:
        ax.plot(electrodes, np.zeros_like(electrodes), "v", color="0.1", ms=4,
                clip_on=False)

    if xlim is not None:
        ax.set_xlim(*xlim)
    elif electrodes is not None:
        ax.set_xlim(float(np.min(electrodes)), float(np.max(electrodes)))
    else:
        ax.set_xlim(x.min(), x.max())

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return cf


# --------------------------------------------------------------------------- #
def _render_model_polys(
    ax, poly_verts, vals, *, alphas=None, cmap="turbo", vmin=None, vmax=None,
    x_shown=None, z_shown=None, electrodes=None, show_electrodes=True,
    xlabel="Distance (m)", ylabel="Depth (m)", title=None,
    depth_exaggeration: float | None = 3.0, equal_aspect: bool | None = None,
    clip_verts=None,
):
    """Shared PolyCollection renderer behind :func:`model_section` (which
    starts from a live ``ParamMesh``) and :func:`model_section_from_arrays`
    (which starts from cached, plain-array polygon geometry) - so a figure
    built from disk-cached data goes through the exact same drawing code as
    one built right after inversion.
    """
    from matplotlib.collections import PolyCollection

    vals = np.asarray(vals, float)
    if vmin is None or vmax is None:
        vmin = float(vals.min()) if vmin is None else vmin
        vmax = float(vals.max()) if vmax is None else vmax

    if alphas is None:
        pc = PolyCollection(poly_verts, array=vals, cmap=cmap,
                            edgecolors="none", linewidths=0.0)
        pc.set_clim(vmin, vmax)
        mappable = pc
    else:
        # fade low-sensitivity cells toward white in RGB so fills stay opaque
        # (true alpha leaves hairline seams between polygons)
        import matplotlib as mpl
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        norm = Normalize(vmin, vmax)
        rgba = mpl.colormaps[cmap](norm(vals))
        a = np.clip(np.asarray(alphas, float), 0.0, 1.0)[:, None]
        rgb = a * rgba[:, :3] + (1.0 - a) * 1.0
        pc = PolyCollection(poly_verts, facecolors=np.c_[rgb, np.ones(len(rgb))],
                            edgecolors="none", linewidths=0.0)
        mappable = ScalarMappable(norm, cmap)
        mappable.set_array(vals)
    ax.add_collection(pc)
    if clip_verts is not None:
        # leave the region outside the polygon empty (e.g. the corners no data
        # or sensitivity reaches) - see ert2d.viz.model_mask
        from matplotlib.patches import Polygon
        clip = Polygon(np.asarray(clip_verts, float), closed=True, fc="none",
                       ec="none", transform=ax.transData)
        pc.set_clip_path(clip)

    if electrodes is not None and show_electrodes:
        ax.plot(electrodes, np.zeros_like(electrodes), "v", color="0.1", ms=4,
                clip_on=False)

    # `equal_aspect` (deprecated) wins if explicitly set, for back-compat
    # with older callers (e.g. the GUI); otherwise use `depth_exaggeration` -
    # a normal cartographic vertical exaggeration instead of a strict 1:1
    # ("equal") aspect, which the manuscript author found produced overly
    # thin/long panels. See ``model_section`` docstring.
    if equal_aspect is True:
        ax.set_aspect("equal")
    elif equal_aspect is False:
        ax.set_aspect("auto")
    elif depth_exaggeration:
        ax.set_aspect(float(depth_exaggeration))

    ax.set_xlim(*x_shown)
    ax.set_ylim(*z_shown)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return mappable


def model_section(
    param_mesh,
    rho_cell: np.ndarray,
    *,
    ax=None,
    alpha_cell: np.ndarray | None = None,
    cmap: str = "turbo",
    clim: tuple[float, float] | None = None,
    z_lim: float | None = None,
    x_lim: tuple[float, float] | None = None,
    title: str | None = None,
    electrodes: np.ndarray | None = None,
    show_electrodes: bool = True,
    xlabel: str = "Distance (m)",
    ylabel: str = "Depth (m)",
    colorbar: bool = True,
    depth_exaggeration: float = 3.0,
    equal_aspect: bool | None = None,
    linear_colorbar: bool = True,
    clip_verts=None,
):
    """Filled Voronoi cells coloured by ``log10(rho)`` - no triangle edges.

    ``clip_verts`` (an ``(m, 2)`` polygon in x/z data coordinates, e.g. from
    :func:`ert2d.viz.model_mask.model_footprint`) leaves everything outside it
    empty; it works the same for Voronoi, triangle and rectangular cells.

    When ``clim`` is not given, the auto colour range is taken only from
    cells inside the displayed ``x_lim`` / ``z_lim`` window (same window the
    axes end up cropped to). ``cell_polys`` also carries the mesh's outer
    padding - the fixed-background border columns and the rows below the
    depth of investigation (``rect_parameterization``'s "3-column border...
    stay fixed at the background resistivity") - which data has no
    sensitivity to; including it would let that background value skew the
    scale for the part of the model actually shown.

    ``depth_exaggeration`` (default 3.0) draws 1 m of depth ``exaggeration``x
    taller than 1 m of horizontal distance (``ax.set_aspect(exaggeration)``,
    matplotlib's convention: a circle of equal x/z data extent is drawn
    ``exaggeration`` times taller than wide) - a normal, moderate ERT-section
    convention, deliberately less extreme than a strict ``aspect="equal"``
    (exaggeration=1), which produced very thin/long panels the manuscript
    author found hard to lay out. Pass ``equal_aspect=True``/``False``
    (deprecated) for the old literal-1:1 / matplotlib-"auto" behaviour.
    """
    plt = _lazy_mpl()
    if ax is None:
        _, ax = plt.subplots()

    if x_lim is not None:
        x_shown = (float(x_lim[0]), float(x_lim[1]))
    elif electrodes is not None:
        x_shown = (float(electrodes.min()), float(electrodes.max()))
    else:
        x_shown = (float(param_mesh.p2[0].min()), float(param_mesh.p2[0].max()))
    z_shown = (-abs(float(z_lim)), 0.0) if z_lim else (float(param_mesh.p2[1].min()), 0.0)

    rho_cell = np.asarray(rho_cell, float)
    poly_verts, cell_idx = flatten_cell_polys(param_mesh.cell_polys)
    vals = np.log10(rho_cell[cell_idx]) if len(cell_idx) else np.asarray([])
    alphas = None if alpha_cell is None else np.asarray(alpha_cell, float)[cell_idx]

    shown = np.array([
        x_shown[0] <= v[:, 0].mean() <= x_shown[1] and z_shown[0] <= v[:, 1].mean() <= z_shown[1]
        for v in poly_verts
    ], dtype=bool)
    if clim is None:
        auto_vals = vals[shown] if shown.any() else vals
        vmin, vmax = float(auto_vals.min()), float(auto_vals.max())
    else:
        vmin, vmax = np.log10(clim[0]), np.log10(clim[1])

    mappable = _render_model_polys(
        ax, poly_verts, vals, alphas=alphas, cmap=cmap, vmin=vmin, vmax=vmax,
        x_shown=x_shown, z_shown=z_shown, electrodes=electrodes,
        show_electrodes=show_electrodes, xlabel=xlabel, ylabel=ylabel,
        title=title, depth_exaggeration=depth_exaggeration,
        equal_aspect=equal_aspect, clip_verts=clip_verts)

    if colorbar:
        cb = ax.figure.colorbar(mappable, ax=ax)
        if linear_colorbar:
            format_log_colorbar(cb)
            cb.set_label(r"$\rho$  ($\Omega\!\cdot$m)")
        else:
            cb.set_label(r"$\log_{10}\rho$  ($\Omega\!\cdot$m)")
    return mappable


def model_section_from_arrays(
    poly_verts, cell_idx, rho_cell: np.ndarray, *, ax=None,
    alpha_cell: np.ndarray | None = None, cmap: str = "turbo",
    clim: tuple[float, float] | None = None, x_lim, z_lim, title: str | None = None,
    electrodes: np.ndarray | None = None, show_electrodes: bool = True,
    xlabel: str = "Distance (m)", ylabel: str = "Depth (m)", colorbar: bool = True,
    depth_exaggeration: float = 3.0, linear_colorbar: bool = True,
    clip_verts=None,
):
    """Cache-replay counterpart of :func:`model_section`: draws the identical
    figure from already-flattened polygon geometry (:func:`flatten_cell_polys`
    output, as saved by ``ert2d.viz.cache.save_cache``) instead of a live
    ``ParamMesh`` - no mesh/inversion objects needed, so a figure can be
    regenerated purely from a ``.npz`` cache on disk. ``x_lim``/``z_lim`` are
    required (the live version can fall back to the mesh's own extent; the
    cached version has no mesh to fall back to).
    """
    plt = _lazy_mpl()
    if ax is None:
        _, ax = plt.subplots()

    x_shown = (float(x_lim[0]), float(x_lim[1]))
    z_shown = (-abs(float(z_lim)), 0.0) if np.isscalar(z_lim) else \
        (float(z_lim[0]), float(z_lim[1]))

    rho_cell = np.asarray(rho_cell, float)
    cell_idx = np.asarray(cell_idx, int)
    vals = np.log10(rho_cell[cell_idx]) if len(cell_idx) else np.asarray([])
    alphas = None if alpha_cell is None else np.asarray(alpha_cell, float)[cell_idx]

    if clim is None:
        vmin, vmax = (float(vals.min()), float(vals.max())) if len(vals) else (0.0, 1.0)
    else:
        vmin, vmax = np.log10(clim[0]), np.log10(clim[1])

    mappable = _render_model_polys(
        ax, list(poly_verts), vals, alphas=alphas, cmap=cmap, vmin=vmin, vmax=vmax,
        x_shown=x_shown, z_shown=z_shown, electrodes=electrodes,
        show_electrodes=show_electrodes, xlabel=xlabel, ylabel=ylabel,
        title=title, depth_exaggeration=depth_exaggeration,
        clip_verts=clip_verts)

    if colorbar:
        cb = ax.figure.colorbar(mappable, ax=ax)
        if linear_colorbar:
            format_log_colorbar(cb)
            cb.set_label(r"$\rho$  ($\Omega\!\cdot$m)")
        else:
            cb.set_label(r"$\log_{10}\rho$  ($\Omega\!\cdot$m)")
    return mappable


# --------------------------------------------------------------------------- #
def crossplot(rhoa_obs, rhoa_calc, *, ax=None, equal_aspect: bool = True,
              xlabel: str = r"measured $\rho_a$  ($\Omega\!\cdot$m)",
              ylabel: str = r"calculated $\rho_a$  ($\Omega\!\cdot$m)",
              title: str | None = "Data crossplot"):
    plt = _lazy_mpl()
    if ax is None:
        _, ax = plt.subplots()
    o = np.asarray(rhoa_obs, float)
    c = np.asarray(rhoa_calc, float)
    lo, hi = min(o.min(), c.min()), max(o.max(), c.max())
    ax.plot([lo, hi], [lo, hi], "-", color="0.5", lw=1)
    ax.scatter(o, c, s=12, alpha=0.6, edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if equal_aspect:
        ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return ax


def rms_curve(rms, *, ax=None, xlabel: str = "iteration",
              ylabel: str = "RMS misfit (%)", title: str | None = "Convergence"):
    plt = _lazy_mpl()
    if ax is None:
        _, ax = plt.subplots()
    rms = np.asarray(rms, float)
    ax.plot(np.arange(1, rms.size + 1), rms, "o-")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    return ax
