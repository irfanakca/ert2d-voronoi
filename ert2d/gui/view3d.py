"""3D View tab: isosurfaces + one-or-more X/Y/Z orthogonal slice planes
through the Depth Map's stacked resistivity volume, rendered with PyVista
(embedded via ``pyvistaqt.QtInteractor``).

Each axis' slice position is a plain :class:`~PySide6.QtWidgets.QSlider`
(arrow-key steppable, exactly like the Depth Map tab's depth slider) indexing
into that axis' real grid coordinates - not a mouse-dragged VTK widget, to
match that same keyboard-driven interaction. Redraws are debounced (see
``DepthMapView`` for the same lag fix) since rebuilding slice/isosurface
meshes on every intermediate slider tick would make dragging feel sluggish.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QSlider, QSpinBox,
                               QVBoxLayout, QWidget)

from ert2d.gui.i18n import t, translator


def _visual(v, name, default):
    return getattr(v, name, default) if v is not None else default


class _AxisSliceRow(QWidget):
    """One axis' slice controls: on/off, prev/next + position slider (index
    into that axis' coordinate array), and how many evenly-spaced parallel
    slices to draw along it."""

    changed = Signal()

    def __init__(self):
        super().__init__()
        self.cb = QCheckBox()
        self.cb.setChecked(True)
        self.btn_prev = QPushButton("◀"); self.btn_prev.setMaximumWidth(24)
        self.btn_next = QPushButton("▶"); self.btn_next.setMaximumWidth(24)
        self.slider = QSlider(Qt.Horizontal)
        self.val_label = QLabel()
        self.val_label.setMinimumWidth(90)
        self._l_count = QLabel()
        self.count = QSpinBox(); self.count.setRange(1, 10); self.count.setValue(1)
        self._coords = np.asarray([0.0])
        self._fmt = "{:.2f}"

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.cb)
        lay.addWidget(self.btn_prev)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.btn_next)
        lay.addWidget(self.val_label)
        lay.addWidget(self._l_count)
        lay.addWidget(self.count)

        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next.clicked.connect(lambda: self._step(1))
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.count.valueChanged.connect(lambda _: self.changed.emit())
        self.cb.toggled.connect(self._on_toggled)

    def set_count_label(self, text: str) -> None:
        self._l_count.setText(text)

    def _on_toggled(self, on: bool) -> None:
        for w in (self.btn_prev, self.btn_next, self.slider, self.count):
            w.setEnabled(on)
        self.changed.emit()

    def _step(self, delta: int) -> None:
        self.slider.setValue(self.slider.value() + delta)

    def _on_slider_changed(self, *_):
        self._update_label()
        self.changed.emit()

    def set_coords(self, coords: np.ndarray, fmt: str, default: str = "max") -> None:
        """``default`` picks the initial slider position: "min"/"max" (an
        outer boundary slice - so the three axes together start as a closed
        box) or "mid" (a through-the-middle cross-section)."""
        self._coords = np.asarray(coords, float)
        self._fmt = fmt
        last = max(self._coords.size - 1, 0)
        start = {"min": 0, "max": last, "mid": self._coords.size // 2}.get(default, last)
        self.slider.blockSignals(True)
        self.slider.setRange(0, last)
        self.slider.setValue(start)
        self.slider.blockSignals(False)
        self._update_label()

    def _update_label(self) -> None:
        if self._coords.size:
            k = min(self.slider.value(), self._coords.size - 1)
            self.val_label.setText(self._fmt.format(self._coords[k]))

    def position(self) -> float:
        k = min(self.slider.value(), self._coords.size - 1)
        return float(self._coords[k])

    def index(self) -> int:
        return min(self.slider.value(), max(self._coords.size - 1, 0))

    def enabled(self) -> bool:
        return self.cb.isChecked()

    def slice_count(self) -> int:
        return self.count.value()


class View3D(QWidget):
    """Isosurfaces + X/Y/Z slice planes through the Depth Map's volume."""

    def __init__(self):
        super().__init__()
        from pyvistaqt import QtInteractor

        self.plotter = QtInteractor(self)
        self._msg = QLabel("")
        self._msg.setAlignment(Qt.AlignCenter)

        self.cb_iso = QCheckBox()
        self._l_iso = QLabel()
        self.iso_values = QLineEdit()
        self.iso_values.editingFinished.connect(self._request_redraw)
        self.cb_outline = QCheckBox()
        self.cb_outline.setChecked(True)

        iso_row = QHBoxLayout()
        iso_row.addWidget(self.cb_iso)
        iso_row.addWidget(self._l_iso)
        iso_row.addWidget(self.iso_values, 1)
        iso_row.addWidget(self.cb_outline)

        self.cb_grid = QCheckBox()
        self.cb_grid.setChecked(True)
        self._l_grid_step = QLabel()
        self.grid_step = QDoubleSpinBox()
        self.grid_step.setRange(0.1, 1000.0)
        self.grid_step.setDecimals(1)
        self.grid_step.setSingleStep(1.0)
        self.grid_step.setValue(5.0)
        self.grid_step.setSuffix(" m")

        grid_row = QHBoxLayout()
        grid_row.addWidget(self.cb_grid)
        grid_row.addWidget(self._l_grid_step)
        grid_row.addWidget(self.grid_step)
        grid_row.addStretch(1)

        self.row_x = _AxisSliceRow()
        self.row_y = _AxisSliceRow()
        self.row_z = _AxisSliceRow()
        self._l_x = QLabel("X"); self._l_y = QLabel("Y"); self._l_z = QLabel()

        # axis label sits to the left of each row's own on/off checkbox
        rows_lay = QVBoxLayout()
        for lab, row in ((self._l_x, self.row_x), (self._l_y, self.row_y),
                         (self._l_z, self.row_z)):
            r = QHBoxLayout()
            r.addWidget(lab)
            r.addWidget(row)
            rows_lay.addLayout(r)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._msg)
        lay.addWidget(self.plotter, 1)
        lay.addLayout(iso_row)
        lay.addLayout(grid_row)
        lay.addLayout(rows_lay)

        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(150)
        self._redraw_timer.timeout.connect(self._redraw)

        self.cb_iso.toggled.connect(self._request_redraw)
        self.cb_outline.toggled.connect(self._request_redraw)
        self.cb_grid.toggled.connect(self._request_redraw)
        self.grid_step.valueChanged.connect(self._request_redraw)
        for row in (self.row_x, self.row_y, self.row_z):
            row.changed.connect(self._request_redraw)

        self._grid = self._vol = self._visuals = self._dm = None
        translator.language_changed.connect(lambda _: self._retranslate())
        self._retranslate()
        self.plotter.hide()

    # ------------------------------------------------------------------ #
    def _retranslate(self):
        self._l_iso.setText(t("view3d.iso_values"))
        self.iso_values.setPlaceholderText(t("view3d.iso_placeholder"))
        self.cb_iso.setText(t("view3d.show_iso"))
        self.cb_outline.setText(t("view3d.show_outline"))
        self.cb_grid.setText(t("view3d.show_grid"))
        self._l_grid_step.setText(t("view3d.grid_step"))
        self._l_z.setText(t("view3d.axis_z"))
        for row in (self.row_x, self.row_y, self.row_z):
            row.set_count_label(t("view3d.slice_count"))
        if self._grid is None:
            self._msg.setText(t("panel.no_depth_map"))

    def _request_redraw(self, *_):
        self._redraw_timer.start()

    # ------------------------------------------------------------------ #
    def show_volume(self, grid, vol, visuals=None, dm=None):
        self._grid, self._vol, self._visuals, self._dm = grid, vol, visuals, dm
        if grid is None:
            self._msg.setText(t("panel.no_depth_map"))
            self._msg.show(); self.plotter.hide()
            return
        self._msg.hide(); self.plotter.show()
        # outer-boundary slices on every axis, so the view opens as a closed
        # box (Z defaults to the shallowest depth, like the Depth Map tab's
        # own initial level)
        self.row_x.set_coords(vol.x, "X = {:.2f} m", default="max")
        self.row_y.set_coords(vol.y, "Y = {:.2f} m", default="max")
        self.row_z.set_coords(vol.z, t("view3d.depth_fmt"), default="min")
        self._redraw()
        self.plotter.reset_camera()

    def refresh(self, visuals=None):
        self._visuals = visuals
        self._request_redraw()

    # ------------------------------------------------------------------ #
    def _parse_iso_values(self):
        txt = self.iso_values.text().strip()
        if not txt:
            return None
        vals = []
        for part in txt.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                vals.append(float(part))
            except ValueError:
                pass
        return vals or None

    def _remove(self, name: str):
        try:
            self.plotter.remove_actor(name)
        except Exception:                       # noqa: BLE001
            pass

    def _redraw(self):
        grid = self._grid
        if grid is None:
            return
        from ert2d.viz.volume3d import axis_slice_mesh, z_slice_from_scatter

        v = self._visuals
        cmap = _visual(v, "cmap", "turbo")
        log_scale = bool(_visual(v, "log_scale", True))
        clim = _visual(v, "model_clim", None)
        finite = grid["rho"][np.isfinite(grid["rho"])]
        if clim is None:
            clim = (float(finite.min()), float(finite.max())) if finite.size else (1.0, 100.0)
        elif log_scale and clim[0] <= 0:
            clim = (max(clim[0], 1e-3), clim[1])  # log-scaled colour mapping needs > 0
        # Discrete colour bands instead of a smooth per-vertex gradient - a
        # continuously-interpolated raster otherwise looks far more
        # "over-interpolated"/smoothed-out than the 2D tabs' filled contours,
        # which paint one flat colour per level band.
        n_colors = 16

        if self.cb_outline.isChecked():
            self.plotter.add_mesh(grid.outline(), color="black", name="outline")
        else:
            self._remove("outline")

        if self.cb_grid.isChecked():
            b = grid.bounds
            step = max(self.grid_step.value(), 1e-6)

            def _n_labels(lo, hi):
                return max(int(round((hi - lo) / step)) + 1, 2)

            self.plotter.show_bounds(
                bounds=b, grid="back", location="outer", color="black",
                font_size=10, fmt="%.1f",
                xtitle=t("view3d.axis_x_title"), ytitle=t("view3d.axis_y_title"),
                ztitle=t("view3d.axis_z_title"),
                n_xlabels=_n_labels(b[0], b[1]), n_ylabels=_n_labels(b[2], b[3]),
                n_zlabels=_n_labels(b[4], b[5]))
        else:
            try:
                self.plotter.remove_bounds_axes()
            except Exception:                   # noqa: BLE001
                pass

        for row, axis in ((self.row_x, "x"), (self.row_y, "y"), (self.row_z, "z")):
            name = f"slice_{axis}"
            mesh = None
            if row.enabled():
                if axis == "z" and row.slice_count() == 1 and self._dm is not None:
                    # exact match to the 2D Depth Map tab: the same masked
                    # Delaunay triangulation of that level's raw scatter,
                    # instead of a regridded (smoother-looking) raster
                    mesh = z_slice_from_scatter(self._dm, row.index())
                if mesh is None:
                    pos = -row.position() if axis == "z" else row.position()
                    mesh = axis_slice_mesh(grid, axis, pos, count=row.slice_count())
            if mesh is not None:
                self.plotter.add_mesh(mesh, scalars="rho", cmap=cmap, clim=clim,
                                      log_scale=log_scale, name=name, n_colors=n_colors,
                                      show_scalar_bar=True,
                                      scalar_bar_args={"title": t("cbar.rho")})
            else:
                self._remove(name)

        if self.cb_iso.isChecked():
            vals = self._parse_iso_values()
            if vals is None and finite.size:
                vals = list(np.percentile(finite, [25, 50, 75]))
            iso = grid.contour(isosurfaces=sorted(vals), scalars="rho") if vals else None
            if iso is not None and iso.n_points:
                self.plotter.add_mesh(iso, scalars="rho", cmap=cmap, clim=clim,
                                      log_scale=log_scale, opacity=0.55, name="iso",
                                      n_colors=n_colors, show_scalar_bar=True,
                                      scalar_bar_args={"title": t("cbar.rho")})
            else:
                self._remove("iso")
        else:
            self._remove("iso")

        self.plotter.render()
