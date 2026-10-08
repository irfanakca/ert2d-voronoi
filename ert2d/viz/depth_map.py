"""Draw one :class:`~ert2d.io.depth_map.DepthMap` slice - shared by the GUI's
Depth Map tab and the animation export, so both stay pixel-for-pixel
consistent.

Figures that use :func:`draw_depth_slice` should use matplotlib's
``layout="compressed"`` (not ``"constrained"``): the map is drawn with
``ax.set_aspect("equal")``, which on an elongated survey (a long line, a
narrow grid) can leave the axes much smaller than its nominal box.
``"constrained"`` sizes the colour bar to that nominal box (too tall, and -
combined with a manually added ``axes_grid1`` divider - it can even clip the
tick labels off the figure); ``"compressed"`` is matplotlib's layout mode
built for exactly this fixed-aspect-axes-plus-colourbar case and keeps a
plain ``fig.colorbar(cf, ax=ax)`` matched to the true axes height.
"""

from __future__ import annotations

import numpy as np

from matplotlib.ticker import FuncFormatter, MaxNLocator

from ert2d.viz.raster import grid_points, shade_rgba, sharpen, smooth
from ert2d.viz.sections import format_log_colorbar, pseudosection


def integer_coordinate_ticks(ax):
    """Whole-metre tick labels without matplotlib's offset / scientific
    notation (``+4.2316e6``), so map coordinates stay readable."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    fmt = FuncFormatter(lambda v, _: f"{v:.0f}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(axis="x", labelrotation=30)


def line_footprint_path(line_traces, default_halfwidth: float = 0.5):
    """Union of one band per profile, as a single matplotlib ``Path``.

    Each band runs exactly from the line's start to its end (so nothing is
    drawn past where a line stops) and reaches half-way to the nearest
    roughly-parallel neighbouring line sideways (so adjacent bands abut and
    the space between lines is filled). Lines with no parallel neighbour use
    the median half-width of the others (``default_halfwidth`` if none)."""
    from matplotlib.path import Path

    segs = [np.asarray(xy, float) for _, xy in line_traces]
    if not segs:
        return None
    mid = [s.mean(axis=0) for s in segs]
    vec = [s[1] - s[0] for s in segs]
    length = [float(np.hypot(*v)) for v in vec]
    unit = [v / L if L > 0 else np.array([1.0, 0.0]) for v, L in zip(vec, length)]

    half = []
    for i in range(len(segs)):
        best = None
        for j in range(len(segs)):
            if j == i or abs(float(unit[i] @ unit[j])) < 0.866:   # > 30 deg apart
                continue
            n_j = np.array([-unit[j][1], unit[j][0]])
            d = abs(float((mid[i] - mid[j]) @ n_j))              # perpendicular gap
            if d > 1e-6 and (best is None or d < best):
                best = d
        half.append(None if best is None else 0.5 * best)
    known = [h for h in half if h is not None]
    fallback = float(np.median(known)) if known else default_halfwidth
    half = [fallback if h is None else h for h in half]

    paths = []
    for s, u, h in zip(segs, unit, half):
        n = np.array([-u[1], u[0]]) * h        # left normal -> same winding for all
        verts = [s[0] - n, s[1] - n, s[1] + n, s[0] + n, s[0] - n]
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        paths.append(Path(verts, codes))
    return Path.make_compound_path(*paths)


def _draw_gridded_slice(
    ax, E, N, rho, *, bounds, cmap, log, lclim, n_levels, fill,
    contour_lines, contour_labels, method, grid_step,
    smooth_on, smooth_sigma, sharpen_on, sharpen_amount,
    emboss_on, emboss_strength,
    hillshade_on, hillshade_azimuth, hillshade_altitude, hillshade_strength,
):
    """Grid + optionally filter/shade one slice, then contour it. Shared body
    of :func:`draw_depth_slice`'s non-triangulation branch."""
    import matplotlib as mpl

    v = np.log10(np.asarray(rho, float)) if log else np.asarray(rho, float)
    Xg, Yg, Zg = grid_points(E, N, v, bounds, grid_step, method=method)

    if smooth_on:
        Zg = smooth(Zg, smooth_sigma)
    if sharpen_on:
        Zg = sharpen(Zg, sharpen_amount, sigma=max(smooth_sigma, 1.0))

    if lclim is not None:
        lev = np.linspace(lclim[0], lclim[1], n_levels)
        Zg = np.clip(Zg, lclim[0], lclim[1])
    else:
        finite = Zg[np.isfinite(Zg)]
        if finite.size == 0:
            return None
        lev = np.linspace(float(finite.min()), float(finite.max()), n_levels)

    cmap_obj = mpl.colormaps[cmap]
    if hillshade_on or emboss_on:
        blend_mode = "overlay" if emboss_on else "soft"
        strength = emboss_strength if emboss_on else hillshade_strength
        rgba = shade_rgba(Zg, cmap=cmap_obj, vmin=lev[0], vmax=lev[-1],
                          azimuth=hillshade_azimuth, altitude=hillshade_altitude,
                          blend_mode=blend_mode, vert_exag=strength)
        e0, e1, n0, n1 = bounds
        ax.imshow(rgba, extent=(e0, e1, n0, n1), origin="lower",
                 interpolation="bilinear")
        cf = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(lev[0], lev[-1]), cmap=cmap_obj)
        cf.set_array(Zg)
        if contour_lines:
            ax.contour(Xg, Yg, Zg, levels=np.linspace(lev[0], lev[-1], 9),
                       colors="k", linewidths=0.4)
    elif fill:
        cf = ax.contourf(Xg, Yg, Zg, levels=lev, cmap=cmap, extend="neither")
        if contour_lines:
            cs = ax.contour(Xg, Yg, Zg, levels=np.linspace(lev[0], lev[-1], 9),
                            colors="k", linewidths=0.4)
            if contour_labels:
                ax.clabel(cs, fmt=(lambda x: f"{10 ** x:.0f}") if log else "%.0f",
                          fontsize=7)
    else:
        cf = ax.contour(Xg, Yg, Zg, levels=lev, cmap=cmap)
        if contour_labels:
            ax.clabel(cf, fmt=(lambda x: f"{10 ** x:.0f}") if log else "%.0f",
                      fontsize=7)
    return cf


def draw_depth_slice(
    ax, dm, k: int, *,
    cmap: str = "turbo",
    log: bool = True,
    clim: tuple[float, float] | None = None,
    n_levels: int = 48,
    fill: bool = True,
    contour_lines: bool = False,
    contour_labels: bool = False,
    interp_method: str = "triangulation",       # "triangulation"|"linear"|"cubic"|"nearest"
    grid_step: float = 1.0,
    smooth_on: bool = False,
    smooth_sigma: float = 1.0,
    sharpen_on: bool = False,
    sharpen_amount: float = 1.0,
    emboss_on: bool = False,
    emboss_strength: float = 1.0,
    hillshade_on: bool = False,
    hillshade_azimuth: float = 315.0,
    hillshade_altitude: float = 45.0,
    hillshade_strength: float = 1.0,
    xlabel: str = "",
    ylabel: str = "",
    no_data_text: str = "No data at this depth",
    show_lines: bool = True,
    show_line_names: bool = True,
    colorbar: bool = True,
    cbar_label: str = r"$\rho$  ($\Omega\!\cdot$m)",
    clip_to_lines: bool = True,
):
    """Contour one depth level onto ``ax`` (masked to the data's convex hull,
    same as ``pseudosection``). ``clim`` is the raw (Ω·m) model colour range -
    log10-transformed here if ``log``, matching ``model_section``. Returns the
    contour mappable (for a colour bar), or ``None`` if there was no data.

    ``clip_to_lines`` additionally clips the contours to the profiles'
    footprint (:func:`line_footprint_path`), so where a shorter line ends the
    map stays empty instead of being interpolated across to longer neighbours.

    ``interp_method="triangulation"`` (default) reproduces the original
    Delaunay-triangulation contouring (no extrapolation past the data hull).
    Any other method (``"linear"``, ``"cubic"``, ``"nearest"``), or enabling
    a filter (smoothing/sharpening/emboss/hillshade - which all need a
    regular array to operate on), grids the scattered points first via
    :func:`ert2d.viz.raster.grid_points` and contours that instead."""
    e0, e1, n0, n1 = dm.bounds
    E, N, rho = dm.points[k]

    lclim = None
    if clim is not None:
        lo, hi = clim
        lclim = (float(np.log10(lo)), float(np.log10(hi))) if log else (float(lo), float(hi))

    use_grid = interp_method != "triangulation" or smooth_on or sharpen_on \
        or emboss_on or hillshade_on

    cf = None
    if len(rho) >= 3:
        try:
            if use_grid:
                cf = _draw_gridded_slice(
                    ax, E, N, rho, bounds=(e0, e1, n0, n1), cmap=cmap, log=log,
                    lclim=lclim, n_levels=n_levels, fill=fill,
                    contour_lines=contour_lines, contour_labels=contour_labels,
                    method=interp_method if interp_method != "triangulation" else "linear",
                    grid_step=grid_step, smooth_on=smooth_on, smooth_sigma=smooth_sigma,
                    sharpen_on=sharpen_on, sharpen_amount=sharpen_amount,
                    emboss_on=emboss_on, emboss_strength=emboss_strength,
                    hillshade_on=hillshade_on, hillshade_azimuth=hillshade_azimuth,
                    hillshade_altitude=hillshade_altitude,
                    hillshade_strength=hillshade_strength)
            else:
                cf = pseudosection(
                    E, N, rho, ax=ax, log10=log, cmap=cmap, clim=lclim,
                    levels=n_levels, fill=fill,
                    xlim=(e0, e1), xlabel=xlabel, ylabel=ylabel, show_points=False,
                    contour_lines=contour_lines, contour_labels=contour_labels)
        except Exception:                      # noqa: BLE001  (degenerate points)
            cf = None
    if cf is None:
        ax.text(0.5, 0.5, no_data_text, ha="center", va="center",
                transform=ax.transAxes)
    else:
        if use_grid:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        if clip_to_lines:
            from matplotlib.patches import PathPatch

            fp = line_footprint_path(dm.line_traces)
            if fp is not None:
                clip = PathPatch(fp, transform=ax.transData, fc="none", ec="none")
                # filled/line contours live in ax.collections; a shaded
                # (hillshade/emboss) slice is drawn as an ax.images entry
                for artist in list(ax.collections) + list(ax.images):
                    artist.set_clip_path(clip)

    if show_lines:
        for name, xy in dm.line_traces:
            ax.plot(xy[:, 0], xy[:, 1], "-", lw=1.0, color="0.15")
            if show_line_names:
                ax.annotate(name, xy[0], fontsize=7, color="0.15")

    ax.set_xlim(e0, e1)
    ax.set_ylim(n0, n1)
    ax.set_aspect("equal")                     # x = easting, y = northing: north up
    integer_coordinate_ticks(ax)

    if cf is not None and colorbar:
        cb = ax.figure.colorbar(cf, ax=ax)
        if log:
            format_log_colorbar(cb)
        cb.set_label(cbar_label)

    return cf


def export_depth_slice_pngs(
    folder, dm, *,
    prefix: str = "ertslice",
    size_inches: tuple[float, float] = (6.0, 6.0),
    dpi: int = 150,
    cmap: str = "turbo",
    log: bool = True,
    clim: tuple[float, float] | None = None,
    n_levels: int = 48,
    fill: bool = True,
    contour_lines: bool = False,
    contour_labels: bool = False,
    interp_method: str = "triangulation",
    grid_step: float = 1.0,
    smooth_on: bool = False,
    smooth_sigma: float = 1.0,
    sharpen_on: bool = False,
    sharpen_amount: float = 1.0,
    emboss_on: bool = False,
    emboss_strength: float = 1.0,
    hillshade_on: bool = False,
    hillshade_azimuth: float = 315.0,
    hillshade_altitude: float = 45.0,
    hillshade_strength: float = 1.0,
    show_lines: bool = True,
    show_line_names: bool = True,
    xlabel: str = "X (m)",
    ylabel: str = "Y (m)",
    cbar_label: str = r"$\rho$  ($\Omega\!\cdot$m)",
    title_fmt=None,                            # callable(depth: float) -> str
) -> list:
    """Save every level of ``dm`` as its own PNG, ``<prefix>_<depth in cm>.png``
    (0.20 m -> ``ertslice_020.png``, 1.2 m -> ``ertslice_120.png``).

    Same figure as the GUI's *Export image* (the on-screen figure at
    ``dpi``): ``size_inches`` should be the on-screen canvas size and the
    layout is ``"compressed"`` like the Depth Map tab, so each PNG looks like
    a screenshot of that depth. Existing files of the same name are
    overwritten. Returns the written paths."""
    from pathlib import Path

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=size_inches, dpi=dpi, layout="compressed")
    FigureCanvasAgg(fig)

    paths = []
    for k, depth in enumerate(dm.depths):
        fig.clear()
        ax = fig.add_subplot(111)
        draw_depth_slice(ax, dm, k, cmap=cmap, log=log, clim=clim,
                         n_levels=n_levels, fill=fill,
                         contour_lines=contour_lines, contour_labels=contour_labels,
                         interp_method=interp_method, grid_step=grid_step,
                         smooth_on=smooth_on, smooth_sigma=smooth_sigma,
                         sharpen_on=sharpen_on, sharpen_amount=sharpen_amount,
                         emboss_on=emboss_on, emboss_strength=emboss_strength,
                         hillshade_on=hillshade_on, hillshade_azimuth=hillshade_azimuth,
                         hillshade_altitude=hillshade_altitude,
                         hillshade_strength=hillshade_strength,
                         show_lines=show_lines, show_line_names=show_line_names,
                         xlabel=xlabel, ylabel=ylabel, cbar_label=cbar_label)
        if title_fmt is not None:
            ax.set_title(title_fmt(float(depth)))
        out = folder / f"{prefix}_{int(round(float(depth) * 100)):03d}.png"
        fig.savefig(out, dpi=dpi)
        paths.append(out)
    return paths


def export_depth_map_gif(
    path, dm, *,
    cmap: str = "turbo",
    log: bool = True,
    clim: tuple[float, float] | None = None,
    n_levels: int = 48,
    fill: bool = True,
    contour_lines: bool = False,
    contour_labels: bool = False,
    interp_method: str = "triangulation",
    grid_step: float = 1.0,
    smooth_on: bool = False,
    smooth_sigma: float = 1.0,
    sharpen_on: bool = False,
    sharpen_amount: float = 1.0,
    emboss_on: bool = False,
    emboss_strength: float = 1.0,
    hillshade_on: bool = False,
    hillshade_azimuth: float = 315.0,
    hillshade_altitude: float = 45.0,
    hillshade_strength: float = 1.0,
    show_lines: bool = True,
    show_line_names: bool = True,
    xlabel: str = "X (m)",
    ylabel: str = "Y (m)",
    cbar_label: str = r"$\rho$  ($\Omega\!\cdot$m)",
    title_fmt=None,                            # callable(depth: float) -> str
    fps: float = 2.0,
    dpi: int = 140,
    base_size: float = 5.5,
) -> None:
    """Render every level of ``dm`` with :func:`draw_depth_slice` and save the
    sequence as an animated GIF - a "flythrough" of the depth-slice map.

    The figure's aspect ratio is derived from the map's own (easting,
    northing) extent, so an elongated survey doesn't leave wide blank margins;
    every frame is then cropped to one shared tight bounding box (computed
    once, from the first frame, and reused for the rest) so the frames stay
    exactly the same size - required for a valid GIF - while still trimming
    everything outside the map, its labels and its title. All frames share one
    colour palette (no per-frame requantizing, no dithering) so the animation
    doesn't flicker between frames.
    """
    import io

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from PIL import Image

    e0, e1, n0, n1 = dm.bounds
    data_ar = max(e1 - e0, 1e-9) / max(n1 - n0, 1e-9)
    fig_h = base_size
    fig_w = float(np.clip(fig_h * data_ar, base_size * 0.6, base_size * 2.4))

    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi, layout="compressed")
    FigureCanvasAgg(fig)

    frames = []
    tight_bbox = None
    for k in range(len(dm.depths)):
        fig.clear()
        ax = fig.add_subplot(111)
        draw_depth_slice(ax, dm, k, cmap=cmap, log=log, clim=clim,
                         n_levels=n_levels, fill=fill,
                         contour_lines=contour_lines,
                         contour_labels=contour_labels,
                         interp_method=interp_method, grid_step=grid_step,
                         smooth_on=smooth_on, smooth_sigma=smooth_sigma,
                         sharpen_on=sharpen_on, sharpen_amount=sharpen_amount,
                         emboss_on=emboss_on, emboss_strength=emboss_strength,
                         hillshade_on=hillshade_on, hillshade_azimuth=hillshade_azimuth,
                         hillshade_altitude=hillshade_altitude,
                         hillshade_strength=hillshade_strength,
                         show_lines=show_lines, show_line_names=show_line_names,
                         xlabel=xlabel, ylabel=ylabel, cbar_label=cbar_label)
        if title_fmt is not None:
            ax.set_title(title_fmt(float(dm.depths[k])))
        fig.canvas.draw()
        if tight_bbox is None:
            tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.08)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches=tight_bbox)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
        buf.close()

    if not frames:
        raise ValueError("no frames to export")

    # One shared palette, quantized from a composite of every frame - not just
    # the first. A single frame (e.g. a sparsely-covered shallowest depth) can
    # be almost entirely white/background; building the palette from that one
    # frame alone starves it of real colour entries and every other frame
    # then quantizes down to near-grayscale. Pooling all frames avoids that.
    tile = 64
    composite = Image.new("RGB", (tile * len(frames), tile))
    for i, im in enumerate(frames):
        composite.paste(im.resize((tile, tile)), (tile * i, 0))
    base_pal = composite.quantize(colors=256, method=Image.MEDIANCUT,
                                  dither=Image.Dither.NONE)
    pal_frames = [im.quantize(palette=base_pal, dither=Image.Dither.NONE)
                 for im in frames]
    duration_ms = max(int(round(1000.0 / fps)), 20)
    pal_frames[0].save(path, save_all=True, append_images=pal_frames[1:],
                       duration=duration_ms, loop=0)
