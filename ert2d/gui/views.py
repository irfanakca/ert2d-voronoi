"""Centre-panel views: Line View (3 stacked panels), Data, Map View."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QHBoxLayout, QHeaderView,
                               QLabel, QPushButton, QSlider, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ert2d.gui.i18n import t, translator
from ert2d.gui.state import FileItem
from ert2d.inversion.weights import median_deviation_weights
from ert2d.viz.depth_map import draw_depth_slice, integer_coordinate_ticks
from ert2d.viz.model_mask import model_footprint
from ert2d.viz.sections import (crossplot, format_log_colorbar, model_section,
                                pseudosection, rms_curve)
from ert2d.viz.sensitivity import parameter_alpha


class _Canvas(QWidget):
    """A bare matplotlib canvas - no navigation toolbar."""

    def __init__(self, layout: str = "constrained"):
        super().__init__()
        self.figure = Figure(figsize=(6, 6), layout=layout)
        self.canvas = FigureCanvasQTAgg(self.figure)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

    def draw(self):
        self.canvas.draw_idle()


def _visual(v, name, default):
    return getattr(v, name, default) if v is not None else default


# --------------------------------------------------------------------------- #
class LineView(QWidget):
    """Measured / calculated pseudo-section (shared colour bar) + model,
    all three panels the same size and x-range, colour bars aligned."""

    def __init__(self):
        super().__init__()
        self._c = _Canvas()
        self._msg = QLabel("")
        self._msg.setAlignment(Qt.AlignCenter)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._msg)
        lay.addWidget(self._c)
        self._item = self._visuals = None
        translator.language_changed.connect(
            lambda _: self.show_item(self._item, self._visuals))

    def show_item(self, item: FileItem | None, visuals=None):
        self._item, self._visuals = item, visuals
        fig = self._c.figure
        fig.clear()

        if item is None or not item.is_loaded:
            self._msg.setText(t("panel.no_selection"))
            self._msg.show(); self._c.hide()
            return
        self._msg.hide(); self._c.show()

        s = item.survey
        v = visuals
        res = item.display_result
        xlim = (float(s.electrode_x.min()), float(s.electrode_x.max()))
        cmap = _visual(v, "cmap", "turbo")
        log = _visual(v, "log_scale", True)
        show_e = _visual(v, "show_electrodes", False)
        pts = _visual(v, "show_points", True)
        clines = _visual(v, "show_contour_lines", False)
        clabels = _visual(v, "show_labels", False)

        # 3 stacked panels, one shared colormap; each colour bar is sized to a
        # single panel's height and centred on the group it belongs to
        ax_m, ax_c, ax_mod = fig.subplots(3, 1, sharex=True)
        ax_c.sharey(ax_m)

        obs = s.rhoa
        cal = res.rhoa_calc if res is not None else None
        data_clim = _visual(v, "data_clim", None)
        if data_clim is not None:
            lo, hi = data_clim
            clim = (float(np.log10(lo)), float(np.log10(hi))) if log else (float(lo), float(hi))
        elif cal is not None:
            both = np.r_[obs, cal]
            clim = ((float(np.log10(both.min())), float(np.log10(both.max())))
                    if log else (float(both.min()), float(both.max())))
        else:
            clim = None

        def _ps(ax, values, title):
            return pseudosection(
                s.x_mid, s.pseudo_depth, values, ax=ax, log10=log, cmap=cmap,
                clim=clim, electrodes=s.electrode_x, show_electrodes=show_e,
                xlim=xlim, xlabel=t("axis.distance"),
                ylabel=t("axis.pseudo_depth"), show_points=pts,
                contour_lines=clines, contour_labels=clabels, title=title)

        cf = _ps(ax_m, obs, t("panel.measured"))
        if cal is not None:
            cf = _ps(ax_c, cal, t("panel.calculated"))
        else:
            ax_c.set_title(t("panel.calculated"))

        # shrink=0.5 of the 2-panel span == 1 panel's height, vertically
        # centred on the pair by default
        cb = fig.colorbar(cf, ax=[ax_m, ax_c], shrink=0.5, aspect=15, pad=0.02)
        if log:
            format_log_colorbar(cb)
        cb.set_label(t("cbar.rhoa"))

        if res is not None:
            alpha = None
            if _visual(v, "sensitivity_alpha", False) and res.jacobian is not None:
                alpha = parameter_alpha(res.jacobian)
            clip = None
            if _visual(v, "mask_corners", False):
                clip = model_footprint(
                    s.x_mid, s.pseudo_depth, s.electrode_x, s.z_max,
                    widen=_visual(v, "mask_widen_pct", 25.0) / 100.0,
                    shape=_visual(v, "mask_shape", "triangle"))
            m = model_section(
                res.param_mesh, res.rho_cell, ax=ax_mod, alpha_cell=alpha,
                cmap=cmap, clim=_visual(v, "model_clim", None),
                electrodes=s.electrode_x, show_electrodes=show_e,
                x_lim=xlim, z_lim=s.z_max, equal_aspect=False, colorbar=False,
                clip_verts=clip,
                xlabel=t("axis.distance"), ylabel=t("axis.depth"),
                title=t("panel.model") + f"   (RMS {res.misfit:.2f}%)")
            cbm = fig.colorbar(m, ax=ax_mod, aspect=15, pad=0.02)
            format_log_colorbar(cbm)
            cbm.set_label(t("cbar.rho"))
        else:
            ax_mod.set_title(t("panel.model"))
            ax_mod.set_xlim(*xlim)
            ax_mod.text(0.5, 0.5, t("panel.not_inverted"), ha="center",
                        va="center", transform=ax_mod.transAxes)
        self._c.draw()


# --------------------------------------------------------------------------- #
class DataView(QWidget):
    """Full-width stretched table over: misfit pseudo-section (+ colour bar)
    and a row of convergence / residual histogram / crossplot."""

    def __init__(self):
        super().__init__()
        self.table = QTableWidget()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self._c = _Canvas()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self.table, 1)
        lay.addWidget(self._c, 2)
        self._item = self._visuals = None
        translator.language_changed.connect(
            lambda _: self.show_item(self._item, self._visuals))

    def show_item(self, item: FileItem | None, visuals=None):
        self._item, self._visuals = item, visuals
        fig = self._c.figure
        fig.clear()
        self.table.clear()

        if item is None or not item.is_loaded:
            self.table.setRowCount(0)
            self._c.draw()
            return

        s = item.survey
        res = item.display_result
        self._fill_table(s, res)
        cmap = _visual(visuals, "cmap", "turbo")

        if res is not None:
            gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0])
            ax_ps = fig.add_subplot(gs[0, :])
            ax_a = fig.add_subplot(gs[1, 0])
            ax_b = fig.add_subplot(gs[1, 1])
            ax_c = fig.add_subplot(gs[1, 2])

            cf = pseudosection(
                s.x_mid, s.pseudo_depth, res.datum_misfit_pct(), ax=ax_ps,
                log10=False, cmap=cmap, electrodes=s.electrode_x,
                xlim=(float(s.electrode_x.min()), float(s.electrode_x.max())),
                xlabel=t("axis.distance"), ylabel=t("axis.pseudo_depth"),
                title=t("data.misfit_section"))
            fig.colorbar(cf, ax=ax_ps, location="right", fraction=0.05, pad=0.02,
                         label="%")

            rms_curve(res.rms_curve, ax=ax_a, xlabel=t("axis.iteration"),
                      ylabel=t("axis.rms"), title=t("data.convergence"))
            lam = getattr(res, "lambda_curve", None)
            if lam is not None and len(lam) == len(res.rms_curve) and np.any(lam > 0):
                axl = ax_a.twinx()
                axl.plot(range(1, len(lam) + 1), lam, "s--", color="C1", ms=3)
                axl.set_yscale("log")
                axl.set_ylabel("λ", color="C1")
                axl.tick_params(axis="y", colors="C1")
            ax_b.hist(res.residual_log, bins=25, color="C0", edgecolor="none")
            ax_b.set_xlabel(r"$\ln\rho_{a,obs} - \ln\rho_{a,calc}$")
            ax_b.set_ylabel(t("axis.count"))
            ax_b.set_title(t("data.residuals"))
            crossplot(res.rhoa_obs, res.rhoa_calc, ax=ax_c, equal_aspect=False,
                      xlabel=t("axis.rhoa_meas"), ylabel=t("axis.rhoa_calc"),
                      title=t("data.crossplot"))
        else:
            ax = fig.add_subplot(111)
            ax.hist(np.log10(s.rhoa), bins=30, color="C0", edgecolor="none")
            ax.set_xlabel(r"$\log_{10}\rho_a$")
            ax.set_ylabel(t("axis.count"))
            ax.set_title(t("data.rhoa_hist"))
        self._c.draw()

    def _fill_table(self, s, res):
        cols = ["C1", "C2", "P1", "P2", "ρₐ", "z*"]
        if res is not None:
            cols += ["ρₐ calc", "misfit %"]
        cols.append("w")
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(s.n_data)
        mis = res.datum_misfit_pct() if res is not None else None
        cal = res.rhoa_calc if res is not None else None
        # weight the inversion uses: static base weight x the residual-based
        # (robust / IRLS) factor when that is on; base weight before any result
        if res is not None:
            wt = np.asarray(res.weights, float)
            rw = getattr(res, "robust_weights", None)
            if rw is not None:
                wt = wt * np.asarray(rw, float)
        else:
            wt = median_deviation_weights(s.rhoa)
        for i in range(s.n_data):
            row = [*(int(x) for x in s.quad[i]),
                   f"{s.rhoa[i]:.2f}", f"{s.pseudo_depth[i]:.2f}"]
            if res is not None:
                row += [f"{cal[i]:.2f}", f"{mis[i]:.2f}"]
            row.append(f"{wt[i]:.3f}")
            for j, val in enumerate(row):
                it = QTableWidgetItem(str(val))
                it.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, j, it)


# --------------------------------------------------------------------------- #
class MapView(QWidget):
    """Profile lines from the ``.loc`` endpoint files."""

    def __init__(self):
        super().__init__()
        self._c = _Canvas()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._c)
        self._cache = ([], None)
        translator.language_changed.connect(
            lambda _: self.show_items(*self._cache))

    def show_items(self, items: list[FileItem], highlight: FileItem | None = None):
        self._cache = (items, highlight)
        fig = self._c.figure
        fig.clear()
        ax = fig.add_subplot(111)
        any_loc = False
        for it in items:
            if it.location is None:
                continue
            any_loc = True
            L = it.location
            xy = np.vstack([L.start, L.end])
            hot = it is highlight
            ax.plot(xy[:, 0], xy[:, 1], "-", lw=2.5 if hot else 1.2,
                    color="C3" if hot else "C0")
            # start = blue square, end = red square
            ms = 8 if hot else 6
            ax.plot(*L.start, "s", ms=ms, mfc="blue", mec="k", mew=0.6, zorder=3)
            ax.plot(*L.end, "s", ms=ms, mfc="red", mec="k", mew=0.6, zorder=3)
            ax.annotate(it.name, L.start, fontsize=8, color="C3" if hot else "0.3",
                        xytext=(6, 4), textcoords="offset points")
        if any_loc:
            # north up (x = easting, y = northing, 1:1), plain integer ticks
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel(t("axis.easting"))
            ax.set_ylabel(t("axis.northing"))
            ax.grid(True, alpha=0.3)
            integer_coordinate_ticks(ax)
            ax.legend(
                handles=[Line2D([], [], ls="", marker="s", ms=6, mfc="blue",
                                mec="k", mew=0.6, label=t("map.start")),
                         Line2D([], [], ls="", marker="s", ms=6, mfc="red",
                                mec="k", mew=0.6, label=t("map.end"))],
                loc="best", fontsize=8, framealpha=0.85)
        else:
            ax.text(0.5, 0.5, t("map.no_loc"), ha="center", va="center",
                    transform=ax.transAxes)
        self._c.draw()


# --------------------------------------------------------------------------- #
class DepthMapView(QWidget):
    """Depth-slice resistivity map combining several georeferenced lines
    (:class:`ert2d.io.depth_map.DepthMap`), with a slider to pick the depth
    and an animation export (a fixed depth step, saved as a GIF)."""

    animate_requested = Signal(float)          # step (m)

    def __init__(self):
        super().__init__()
        self._c = _Canvas(layout="compressed")
        self._msg = QLabel("")
        self._msg.setAlignment(Qt.AlignCenter)
        self._l_depth = QLabel()
        self._depth_val = QLabel()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        # Redrawing (grid interpolation + filters + contouring) is too heavy
        # to run on every intermediate value while dragging - only the depth
        # label updates immediately; the actual figure redraw is debounced to
        # once the slider has been still for a moment (see _on_slider_changed).
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(120)
        self._redraw_timer.timeout.connect(self._redraw)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.btn_prev = QPushButton("◀")
        self.btn_next = QPushButton("▶")
        for b in (self.btn_prev, self.btn_next):
            b.setEnabled(False)
            b.setMaximumWidth(28)
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next.clicked.connect(lambda: self._step(1))
        self.cb_lines = QCheckBox()
        self.cb_lines.setChecked(True)
        self.cb_lines.toggled.connect(self._on_lines_toggled)
        self.cb_names = QCheckBox()
        self.cb_names.setChecked(True)
        self.cb_names.toggled.connect(self._redraw)
        row = QHBoxLayout()
        row.addWidget(self._l_depth)
        row.addWidget(self.btn_prev)
        row.addWidget(self.slider, 1)
        row.addWidget(self.btn_next)
        row.addWidget(self._depth_val)
        row.addWidget(self.cb_lines)
        row.addWidget(self.cb_names)

        self._l_step = QLabel()
        self.step = QDoubleSpinBox()
        self.step.setRange(0.01, 1000.0)
        self.step.setDecimals(2)
        self.step.setSingleStep(0.1)
        self.step.setValue(0.2)
        self.step.setSuffix(" m")
        self.btn_animate = QPushButton()
        self.btn_animate.setEnabled(False)
        self.btn_animate.clicked.connect(
            lambda: self.animate_requested.emit(self.step.value()))
        anim_row = QHBoxLayout()
        anim_row.addWidget(self._l_step)
        anim_row.addWidget(self.step)
        anim_row.addWidget(self.btn_animate)
        anim_row.addStretch(1)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._msg)
        lay.addWidget(self._c, 1)
        lay.addLayout(row)
        lay.addLayout(anim_row)
        self._dm = self._visuals = self._contour = None
        translator.language_changed.connect(lambda _: self._retranslate())
        self._retranslate()
        self._c.hide()

    def _on_lines_toggled(self, on: bool):
        self.cb_names.setEnabled(on)            # names belong to the lines
        self._redraw()

    def _step(self, delta: int):
        self.slider.setValue(self.slider.value() + delta)

    def _on_slider_changed(self, *_):
        dm = self._dm
        if dm is not None:
            k = self.slider.value()
            if 0 <= k < len(dm.depths):
                self._depth_val.setText(f"{dm.depths[k]:.2f} m")
        self._redraw_timer.start()             # coalesce rapid drag events

    def _retranslate(self):
        self._l_depth.setText(t("depthmap.depth"))
        self.btn_prev.setToolTip(t("depthmap.prev"))
        self.btn_next.setToolTip(t("depthmap.next"))
        self.cb_lines.setText(t("depthmap.show_lines"))
        self.cb_names.setText(t("depthmap.show_names"))
        self._l_step.setText(t("depthmap.anim_step"))
        self.btn_animate.setText(t("depthmap.build_animation"))
        if self._dm is None:
            self._msg.setText(t("panel.no_depth_map"))
        self._redraw()

    def show_depth_map(self, dm, visuals=None, contour=None):
        self._dm, self._visuals, self._contour = dm, visuals, contour
        if dm is None:
            self.slider.setEnabled(False)
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            self.btn_animate.setEnabled(False)
            self._msg.setText(t("panel.no_depth_map"))
            self._msg.show(); self._c.hide()
            return
        self._msg.hide(); self._c.show()
        self.slider.setEnabled(True)
        self.btn_prev.setEnabled(True)
        self.btn_next.setEnabled(True)
        self.btn_animate.setEnabled(True)
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(dm.depths) - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._redraw()

    def refresh(self, visuals=None, contour=None):
        self._visuals = visuals
        if contour is not None:
            self._contour = contour
        self._redraw()

    def _redraw(self, *_):
        dm = self._dm
        fig = self._c.figure
        fig.clear()
        if dm is None:
            return

        k = self.slider.value()
        ax = fig.add_subplot(111)
        cmap = _visual(self._visuals, "cmap", "turbo")
        log = _visual(self._visuals, "log_scale", True)
        clines = _visual(self._visuals, "show_contour_lines", False)
        clabels = _visual(self._visuals, "show_labels", False)
        model_clim = _visual(self._visuals, "model_clim", None)
        c = self._contour

        draw_depth_slice(
            ax, dm, k, cmap=cmap, log=log, clim=model_clim,
            n_levels=_visual(c, "n_levels", 48), fill=_visual(c, "fill", True),
            contour_lines=clines, contour_labels=clabels,
            interp_method=_visual(c, "interp_method", "triangulation"),
            grid_step=_visual(c, "grid_step", 1.0),
            smooth_on=_visual(c, "smooth_on", False),
            smooth_sigma=_visual(c, "smooth_sigma", 1.0),
            sharpen_on=_visual(c, "sharpen_on", False),
            sharpen_amount=_visual(c, "sharpen_amount", 1.0),
            emboss_on=_visual(c, "emboss_on", False),
            emboss_strength=_visual(c, "emboss_strength", 1.0),
            hillshade_on=_visual(c, "hillshade_on", False),
            hillshade_azimuth=_visual(c, "hillshade_azimuth", 315.0),
            hillshade_altitude=_visual(c, "hillshade_altitude", 45.0),
            hillshade_strength=_visual(c, "hillshade_strength", 1.0),
            show_lines=self.cb_lines.isChecked(),
            show_line_names=self.cb_names.isChecked(),
            xlabel=t("axis.easting" if dm.is_global else "axis.x_dist"),
            ylabel=t("axis.northing" if dm.is_global else "axis.y_dist"),
            no_data_text=t("depthmap.no_data"), cbar_label=t("cbar.rho"))

        ax.set_title(t("depthmap.title", depth=float(dm.depths[k])))
        self._depth_val.setText(f"{dm.depths[k]:.2f} m")
        self._c.draw()
