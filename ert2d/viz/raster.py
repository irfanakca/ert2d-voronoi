"""Grid a scattered depth-slice (E, N, value) onto a regular raster and apply
optional NaN-aware image filters - the building block behind the Depth Map
tab's "grid" contouring mode (as opposed to its default raw Delaunay
triangulation, see :func:`ert2d.viz.sections.pseudosection`), which needs a
regular array to smooth/sharpen/shade in the first place.
"""

from __future__ import annotations

import numpy as np


def grid_points(E, N, v, bounds, step: float, method: str = "linear"):
    """Interpolate scattered ``(E, N, v)`` onto a regular grid over ``bounds``
    (``e0, e1, n0, n1``) at ``step`` metres. ``method`` is one of scipy's
    ``griddata`` methods (``"linear"``, ``"cubic"``, ``"nearest"``).

    ``linear``/``cubic`` already return NaN outside the data's convex hull;
    ``nearest`` extrapolates everywhere, so it is masked to the hull here too,
    keeping the same "no extrapolation" footprint as the triangulated mode.
    """
    from scipy.interpolate import griddata

    e0, e1, n0, n1 = bounds
    nx = max(int(round((e1 - e0) / step)), 2)
    ny = max(int(round((n1 - n0) / step)), 2)
    xg = np.linspace(e0, e1, nx)
    yg = np.linspace(n0, n1, ny)
    Xg, Yg = np.meshgrid(xg, yg)

    pts = np.column_stack([np.asarray(E, float), np.asarray(N, float)])
    Zg = griddata(pts, np.asarray(v, float), (Xg, Yg), method=method)

    if method == "nearest":
        from scipy.spatial import Delaunay

        try:
            hull = Delaunay(pts)
            inside = hull.find_simplex(np.column_stack([Xg.ravel(), Yg.ravel()])) >= 0
            Zg = np.where(inside.reshape(Xg.shape), Zg, np.nan)
        except Exception:                        # noqa: BLE001  (too few/degenerate points)
            pass                                 # leave unmasked
    return Xg, Yg, Zg


def nan_gaussian_filter(Z: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-blur ``Z`` while ignoring NaNs (normalised convolution), so the
    no-data footprint (outside the hull) doesn't bleed grey/zero into the
    smoothed edges."""
    from scipy.ndimage import gaussian_filter

    if sigma <= 0:
        return Z
    mask = np.isfinite(Z)
    if not mask.any():
        return Z
    filled = np.where(mask, Z, 0.0)
    weight = gaussian_filter(mask.astype(float), sigma)
    blurred = gaussian_filter(filled, sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = blurred / weight
    out[~mask] = np.nan
    return out


def smooth(Z: np.ndarray, sigma: float) -> np.ndarray:
    """Denoise the interpolated grid (Gaussian blur, "yumuşatma")."""
    return nan_gaussian_filter(Z, sigma)


def sharpen(Z: np.ndarray, amount: float, sigma: float = 1.0) -> np.ndarray:
    """Unsharp-mask sharpening ("keskinleştirme"): push each cell away from its
    local blurred average, amplifying gradients/edges in the resistivity
    surface. ``amount`` 0 = no change; ~1 = a typical unsharp mask."""
    if amount <= 0:
        return Z
    blurred = nan_gaussian_filter(Z, sigma)
    return Z + amount * (Z - blurred)


def shade_rgba(Z: np.ndarray, *, cmap, vmin: float, vmax: float,
               azimuth: float = 315.0, altitude: float = 45.0,
               blend_mode: str = "soft", vert_exag: float = 1.0):
    """Colour-map ``Z`` and relief-shade it in one pass using matplotlib's
    :class:`~matplotlib.colors.LightSource` - used for both the "gölgelendirme"
    (hillshade, ``blend_mode="soft"``) and "kabartma" (emboss,
    ``blend_mode="overlay"``) filters: same illumination model, a starker
    blend for the latter. The colour mapping (and hence the colour bar) is
    unaffected by the shading - only its apparent brightness/contrast changes.
    """
    from matplotlib.colors import LightSource

    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    masked = np.ma.masked_invalid(Z)
    return ls.shade(masked, cmap=cmap, vmin=vmin, vmax=vmax,
                    blend_mode=blend_mode, vert_exag=vert_exag)
