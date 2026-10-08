"""Right-hand settings panels: Visuals, Inversion, Processing, Acquisition,
Contour & Filters."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from ert2d.gui.i18n import t, translator
from ert2d.gui.state import (ContourSettings, FileItem, InversionSettings,
                             MeshSettings, VisualSettings)

_CMAPS = ["turbo", "jet", "viridis", "magma", "RdYlBu_r", "Spectral_r", "gray"]

_BUNDLED_CLR = Path(__file__).parent / "resources" / "geosoft.clr"


def _register_bundled_colormaps() -> list[str]:
    """Register the ``.clr`` colour tables shipped with the GUI (currently
    just ``geosoft.clr``) and return their names, to seed ``_CMAPS``."""
    from ert2d.io.colormap import register_clr

    names = []
    if _BUNDLED_CLR.is_file():
        try:
            names.append(register_clr(_BUNDLED_CLR))
        except Exception:                       # noqa: BLE001
            pass
    return names


_CMAPS += _register_bundled_colormaps()


def _opt_float(txt: str):
    txt = txt.strip()
    if not txt:
        return None
    try:
        return float(txt)
    except ValueError:
        return None


class _Panel(QWidget):
    def __init__(self):
        super().__init__()
        translator.language_changed.connect(lambda _: self.retranslate())

    def retranslate(self):  # overridden
        pass


class _RangeEdit(QWidget):
    """Two small "auto"-placeholder fields side by side: lower – upper."""

    changed = Signal()

    def __init__(self):
        super().__init__()
        self.lo = QLineEdit(); self.lo.setPlaceholderText("auto")
        self.hi = QLineEdit(); self.hi.setPlaceholderText("auto")
        for w in (self.lo, self.hi):
            w.setMaximumWidth(64)
            w.editingFinished.connect(self.changed)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.lo)
        lay.addWidget(QLabel("–"))
        lay.addWidget(self.hi)
        lay.addStretch(1)

    def values(self) -> tuple[float | None, float | None]:
        return _opt_float(self.lo.text()), _opt_float(self.hi.text())


# --------------------------------------------------------------------------- #
class VisualsPanel(_Panel):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.v = VisualSettings()
        form = QFormLayout(self)

        self.cmap = QComboBox(); self.cmap.addItems(_CMAPS)
        self.cmap.setCurrentText(self.v.colormap)
        self.btn_load_cmap = QPushButton("…")
        self.btn_load_cmap.setMaximumWidth(28)
        self.btn_load_cmap.clicked.connect(self._load_colormap)
        cmap_row = QHBoxLayout()
        cmap_row.setContentsMargins(0, 0, 0, 0)
        cmap_row.addWidget(self.cmap, 1)
        cmap_row.addWidget(self.btn_load_cmap)
        self.cb_rev = QCheckBox(); self.cb_rev.setChecked(self.v.reverse_colormap)
        self.scale = QComboBox()
        self.style = QComboBox()
        self.cb_points = QCheckBox(); self.cb_points.setChecked(self.v.show_points)
        self.cb_elec = QCheckBox(); self.cb_elec.setChecked(self.v.show_electrodes)
        self.cb_clines = QCheckBox()
        self.cb_labels = QCheckBox()
        self.cb_sens = QCheckBox(); self.cb_sens.setChecked(self.v.sensitivity_alpha)
        self.cb_mask = QCheckBox(); self.cb_mask.setChecked(self.v.mask_corners)
        self.mask_pct = QDoubleSpinBox()
        self.mask_pct.setRange(0.0, 100.0)
        self.mask_pct.setDecimals(0)
        self.mask_pct.setSingleStep(5.0)
        self.mask_pct.setValue(self.v.mask_widen_pct)
        self.mask_shape = QComboBox()
        self.data_range = _RangeEdit()
        self.model_range = _RangeEdit()

        self._l_mask_pct = QLabel(); self._l_mask_shape = QLabel()
        self._l_cmap = QLabel(); self._l_scale = QLabel()
        self._l_style = QLabel()
        self._l_data_range = QLabel(); self._l_model_range = QLabel()
        form.addRow(self._l_cmap, cmap_row)
        form.addRow(self.cb_rev)
        form.addRow(self._l_scale, self.scale)
        form.addRow(self._l_style, self.style)
        form.addRow(self._l_data_range, self.data_range)
        form.addRow(self._l_model_range, self.model_range)
        form.addRow(self.cb_points)
        form.addRow(self.cb_elec)
        form.addRow(self.cb_clines)
        form.addRow(self.cb_labels)
        form.addRow(self.cb_sens)
        form.addRow(self.cb_mask)
        form.addRow(self._l_mask_pct, self.mask_pct)
        form.addRow(self._l_mask_shape, self.mask_shape)
        self._form = form
        self.retranslate()

        for w in (self.cmap, self.scale, self.style, self.mask_shape):
            w.currentIndexChanged.connect(self._sync)
        for w in (self.cb_rev, self.cb_points, self.cb_elec, self.cb_clines,
                  self.cb_labels, self.cb_sens, self.cb_mask):
            w.toggled.connect(self._sync)
        self.mask_pct.valueChanged.connect(self._sync)
        for w in (self.data_range, self.model_range):
            w.changed.connect(self._sync)

    def _load_colormap(self):
        f, _ = QFileDialog.getOpenFileName(
            self, t("vis.load_colormap"), "", "Colour table (*.clr)")
        if not f:
            return
        from ert2d.io.colormap import register_clr

        try:
            name = register_clr(f)
        except Exception as exc:                # noqa: BLE001
            QMessageBox.warning(self, t("vis.load_colormap"), str(exc))
            return
        if self.cmap.findText(name) < 0:
            self.cmap.addItem(name)
        self.cmap.setCurrentText(name)

    def retranslate(self):
        self._l_cmap.setText(t("vis.colormap"))
        self.btn_load_cmap.setToolTip(t("vis.load_colormap"))
        self.cb_rev.setText(t("vis.reverse_colormap"))
        self._l_scale.setText(t("vis.scale"))
        self._l_style.setText(t("vis.style"))
        self._l_data_range.setText(t("vis.data_range"))
        self._l_model_range.setText(t("vis.model_range"))
        self.cb_points.setText(t("vis.show_points"))
        self.cb_elec.setText(t("vis.show_electrodes"))
        self.cb_clines.setText(t("vis.show_contour_lines"))
        self.cb_labels.setText(t("vis.show_labels"))
        self.cb_sens.setText(t("vis.sensitivity"))
        self.cb_mask.setText(t("vis.mask"))
        self._l_mask_pct.setText(t("vis.mask_pct"))
        self._l_mask_shape.setText(t("vis.mask_shape"))
        for combo, keys in ((self.scale, ("vis.scale.log", "vis.scale.linear")),
                            (self.style, ("vis.style.cells", "vis.style.contour")),
                            (self.mask_shape, ("vis.mask_shape.triangle",
                                               "vis.mask_shape.ellipse"))):
            cur = combo.currentIndex()
            combo.blockSignals(True); combo.clear()
            combo.addItems([t(k) for k in keys])
            combo.setCurrentIndex(max(cur, 0))
            combo.blockSignals(False)

    def _sync(self):
        self.v.colormap = self.cmap.currentText()
        self.v.reverse_colormap = self.cb_rev.isChecked()
        self.v.log_scale = self.scale.currentIndex() == 0
        self.v.model_style = "cells" if self.style.currentIndex() == 0 else "contour"
        self.v.show_points = self.cb_points.isChecked()
        self.v.show_electrodes = self.cb_elec.isChecked()
        self.v.show_contour_lines = self.cb_clines.isChecked()
        self.v.show_labels = self.cb_labels.isChecked()
        self.v.sensitivity_alpha = self.cb_sens.isChecked()
        self.v.mask_corners = self.cb_mask.isChecked()
        self.v.mask_widen_pct = self.mask_pct.value()
        self.v.mask_shape = ("triangle", "ellipse")[max(self.mask_shape.currentIndex(), 0)]
        self.mask_pct.setEnabled(self.v.mask_corners)
        self.mask_shape.setEnabled(self.v.mask_corners)
        self.v.data_clim_min, self.v.data_clim_max = self.data_range.values()
        self.v.model_clim_min, self.v.model_clim_max = self.model_range.values()
        self.changed.emit()


# --------------------------------------------------------------------------- #
class InversionPanel(_Panel):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.mesh = MeshSettings()
        self.inv = InversionSettings()
        root = QVBoxLayout(self)

        self.gb_disc = QGroupBox()
        f1 = QFormLayout(self.gb_disc)
        self.cell_type = QComboBox()
        self._l_cell = QLabel()
        f1.addRow(self._l_cell, self.cell_type)
        self.wing = QDoubleSpinBox(); self.wing.setRange(1, 20); self.wing.setValue(self.mesh.wing)
        self.depth = QDoubleSpinBox(); self.depth.setRange(1, 12); self.depth.setValue(self.mesh.depth_factor)
        self.sdiv = QSpinBox(); self.sdiv.setRange(1, 8); self.sdiv.setValue(self.mesh.surface_div)
        self._l_wing = QLabel(); self._l_depth = QLabel(); self._l_sdiv = QLabel()
        f1.addRow(self._l_wing, self.wing)
        f1.addRow(self._l_depth, self.depth)
        f1.addRow(self._l_sdiv, self.sdiv)

        self.gb_method = QGroupBox()
        f2 = QFormLayout(self.gb_method)
        self.mode = QComboBox()
        self.lam0 = QLineEdit(); self.lam0.setPlaceholderText("auto")
        self.target = QLineEdit(); self.target.setPlaceholderText("auto")
        self.cool = QDoubleSpinBox(); self.cool.setRange(0.1, 1.0); self.cool.setSingleStep(0.05)
        self.cool.setValue(self.inv.cooling)
        self.robust = QComboBox()
        self.rmin = QDoubleSpinBox(); self.rmin.setRange(0.0, 1.0)
        self.rmin.setSingleStep(0.1); self.rmin.setDecimals(2)
        self.rmin.setValue(self.inv.robust_min_weight)
        self.rmin.setEnabled(False)          # robust weighting is off by default
        self._l_mode = QLabel(); self._l_lam = QLabel()
        self._l_target = QLabel(); self._l_cool = QLabel()
        self._l_robust = QLabel(); self._l_rmin = QLabel()
        f2.addRow(self._l_mode, self.mode)
        f2.addRow(self._l_lam, self.lam0)
        f2.addRow(self._l_target, self.target)
        f2.addRow(self._l_cool, self.cool)
        f2.addRow(self._l_robust, self.robust)
        f2.addRow(self._l_rmin, self.rmin)

        self.gb_prog = QGroupBox()
        f3 = QFormLayout(self.gb_prog)
        self.max_iter = QSpinBox(); self.max_iter.setRange(1, 50); self.max_iter.setValue(self.inv.max_iter)
        self.tol = QDoubleSpinBox(); self.tol.setRange(0.0, 0.5); self.tol.setSingleStep(0.01)
        self.tol.setDecimals(3); self.tol.setValue(self.inv.tol)
        self._l_iter = QLabel(); self._l_tol = QLabel()
        f3.addRow(self._l_iter, self.max_iter)
        f3.addRow(self._l_tol, self.tol)

        root.addWidget(self.gb_disc)
        root.addWidget(self.gb_method)
        root.addWidget(self.gb_prog)
        root.addStretch(1)
        self.retranslate()

        for w in (self.wing, self.depth, self.cool, self.tol, self.rmin):
            w.valueChanged.connect(self._sync)
        self.robust.currentIndexChanged.connect(self._sync)
        self.sdiv.valueChanged.connect(self._sync)
        self.max_iter.valueChanged.connect(self._sync)
        self.cell_type.currentIndexChanged.connect(self._sync)
        self.mode.currentIndexChanged.connect(self._sync)
        self.lam0.editingFinished.connect(self._sync)
        self.target.editingFinished.connect(self._sync)
        self.cool.setEnabled(False)          # search mode is the default

    def retranslate(self):
        self.gb_disc.setTitle(t("inv.discretization"))
        self.gb_method.setTitle(t("inv.method"))
        self.gb_prog.setTitle(t("inv.progress"))
        self._l_cell.setText(t("inv.cell_type"))
        cur = max(self.cell_type.currentIndex(), 0)
        self.cell_type.blockSignals(True)
        self.cell_type.clear()
        self.cell_type.addItems([t("inv.cell.voronoi"), t("inv.cell.triangle"),
                                 t("inv.cell.rect")])
        self.cell_type.setCurrentIndex(cur)
        self.cell_type.blockSignals(False)
        self._l_wing.setText(t("inv.mesh_wing"))
        self._l_depth.setText(t("inv.mesh_depth"))
        self._l_sdiv.setText(t("inv.surface_div"))
        self._l_mode.setText(t("inv.lambda_mode"))
        self._l_lam.setText(t("inv.lambda0"))
        self._l_target.setText(t("inv.target_rms"))
        self._l_cool.setText(t("inv.cooling"))
        self._l_robust.setText(t("inv.robust"))
        self._l_rmin.setText(t("inv.robust_min"))
        cur = max(self.robust.currentIndex(), 0)
        self.robust.blockSignals(True)
        self.robust.clear()
        self.robust.addItems([t("inv.robust.none"), t("inv.robust.huber"),
                              t("inv.robust.cauchy"), t("inv.robust.bisquare")])
        self.robust.setCurrentIndex(cur)
        self.robust.blockSignals(False)
        self._l_iter.setText(t("inv.max_iter"))
        self._l_tol.setText(t("inv.tol"))
        cur = self.mode.currentIndex()
        self.mode.blockSignals(True)
        self.mode.clear()
        self.mode.addItems([t("inv.mode.search"), t("inv.mode.matlab")])
        self.mode.setCurrentIndex(max(cur, 0))
        self.mode.blockSignals(False)

    def _sync(self):
        self.mesh.kind = ("voronoi", "triangle", "rect")[self.cell_type.currentIndex()]
        self.mesh.wing = self.wing.value()
        self.mesh.depth_factor = self.depth.value()
        self.mesh.surface_div = self.sdiv.value()
        self.inv.lambda_mode = "search" if self.mode.currentIndex() == 0 else "matlab"
        self.inv.cooling = self.cool.value()
        self.inv.max_iter = self.max_iter.value()
        self.inv.tol = self.tol.value()
        self.inv.robust = ("none", "huber", "cauchy", "bisquare")[max(self.robust.currentIndex(), 0)]
        self.inv.robust_min_weight = self.rmin.value()
        self.rmin.setEnabled(self.inv.robust != "none")
        self.inv.lam0 = _opt_float(self.lam0.text())
        self.inv.target_rms = _opt_float(self.target.text())
        search = self.inv.lambda_mode == "search"
        self.target.setEnabled(search)
        self.cool.setEnabled(not search)
        self.changed.emit()

    def apply_to(self, item: FileItem):
        item.mesh = MeshSettings(**vars(self.mesh))
        item.inversion = InversionSettings(**vars(self.inv))


# --------------------------------------------------------------------------- #
class AcquisitionPanel(_Panel):
    def __init__(self):
        super().__init__()
        self._form = QFormLayout(self)
        self._rows: list[tuple[QLabel, QLabel]] = []
        for _ in range(6):
            lab, val = QLabel(), QLabel()
            self._form.addRow(lab, val)
            self._rows.append((lab, val))
        self._item = None
        self.retranslate()

    def show_item(self, item: FileItem | None):
        self._item = item
        keys = ["acq.array", "acq.spacing", "acq.n_electrodes", "acq.n_data",
                "acq.rhoa_range", "acq.pseudo_depth"]
        if item is None or not item.is_loaded:
            vals = [""] * 6
        else:
            s = item.survey
            vals = [s.array_name, f"{s.spacing:g} m", str(s.n_electrodes),
                    str(s.n_data),
                    f"{s.rhoa.min():.1f} – {s.rhoa.max():.1f} Ω·m",
                    f"{abs(s.pseudo_depth).max():.2f} m"]
        for (lab, val), k, v in zip(self._rows, keys, vals):
            lab.setText(t(k)); val.setText(v)

    def retranslate(self):
        self.show_item(self._item)


# --------------------------------------------------------------------------- #
class ContourFilterPanel(_Panel):
    """Depth Map contouring options + raster filters (smoothing, sharpening,
    emboss, hillshade) - see ``ert2d.viz.depth_map.draw_depth_slice``."""

    changed = Signal()

    def __init__(self):
        super().__init__()
        self.v = ContourSettings()
        root = QVBoxLayout(self)

        self.gb_contour = QGroupBox()
        f1 = QFormLayout(self.gb_contour)
        self.interp = QComboBox()
        self.n_levels = QSpinBox(); self.n_levels.setRange(3, 200)
        self.n_levels.setValue(self.v.n_levels)
        self.cb_fill = QCheckBox(); self.cb_fill.setChecked(self.v.fill)
        self.grid_step = QDoubleSpinBox(); self.grid_step.setRange(0.05, 100.0)
        self.grid_step.setDecimals(2); self.grid_step.setSingleStep(0.25)
        self.grid_step.setSuffix(" m"); self.grid_step.setValue(self.v.grid_step)
        self._l_interp = QLabel(); self._l_levels = QLabel(); self._l_step = QLabel()
        f1.addRow(self._l_interp, self.interp)
        f1.addRow(self._l_levels, self.n_levels)
        f1.addRow(self.cb_fill)
        f1.addRow(self._l_step, self.grid_step)

        self.gb_filters = QGroupBox()
        f2 = QFormLayout(self.gb_filters)
        self.cb_smooth = QCheckBox()
        self.smooth_sigma = QDoubleSpinBox(); self.smooth_sigma.setRange(0.1, 20.0)
        self.smooth_sigma.setDecimals(1); self.smooth_sigma.setSingleStep(0.5)
        self.smooth_sigma.setValue(self.v.smooth_sigma)
        self.cb_sharpen = QCheckBox()
        self.sharpen_amount = QDoubleSpinBox(); self.sharpen_amount.setRange(0.1, 5.0)
        self.sharpen_amount.setDecimals(1); self.sharpen_amount.setSingleStep(0.1)
        self.sharpen_amount.setValue(self.v.sharpen_amount)
        self.cb_emboss = QCheckBox()
        self.emboss_strength = QDoubleSpinBox(); self.emboss_strength.setRange(0.1, 10.0)
        self.emboss_strength.setDecimals(1); self.emboss_strength.setSingleStep(0.5)
        self.emboss_strength.setValue(self.v.emboss_strength)
        self.cb_hillshade = QCheckBox()
        self.hs_azimuth = QDoubleSpinBox(); self.hs_azimuth.setRange(0.0, 360.0)
        self.hs_azimuth.setDecimals(0); self.hs_azimuth.setSuffix("°")
        self.hs_azimuth.setValue(self.v.hillshade_azimuth)
        self.hs_altitude = QDoubleSpinBox(); self.hs_altitude.setRange(1.0, 90.0)
        self.hs_altitude.setDecimals(0); self.hs_altitude.setSuffix("°")
        self.hs_altitude.setValue(self.v.hillshade_altitude)
        self.hs_strength = QDoubleSpinBox(); self.hs_strength.setRange(0.1, 10.0)
        self.hs_strength.setDecimals(1); self.hs_strength.setSingleStep(0.5)
        self.hs_strength.setValue(self.v.hillshade_strength)
        self._l_smooth_sigma = QLabel(); self._l_sharpen_amount = QLabel()
        self._l_emboss_strength = QLabel(); self._l_hs_azimuth = QLabel()
        self._l_hs_altitude = QLabel(); self._l_hs_strength = QLabel()
        f2.addRow(self.cb_smooth)
        f2.addRow(self._l_smooth_sigma, self.smooth_sigma)
        f2.addRow(self.cb_sharpen)
        f2.addRow(self._l_sharpen_amount, self.sharpen_amount)
        f2.addRow(self.cb_emboss)
        f2.addRow(self._l_emboss_strength, self.emboss_strength)
        f2.addRow(self.cb_hillshade)
        f2.addRow(self._l_hs_azimuth, self.hs_azimuth)
        f2.addRow(self._l_hs_altitude, self.hs_altitude)
        f2.addRow(self._l_hs_strength, self.hs_strength)

        root.addWidget(self.gb_contour)
        root.addWidget(self.gb_filters)
        root.addStretch(1)
        self.retranslate()

        self.interp.currentIndexChanged.connect(self._sync)
        self.n_levels.valueChanged.connect(self._sync)
        self.grid_step.valueChanged.connect(self._sync)
        for w in (self.cb_fill, self.cb_smooth, self.cb_sharpen, self.cb_emboss,
                  self.cb_hillshade):
            w.toggled.connect(self._sync)
        for w in (self.smooth_sigma, self.sharpen_amount, self.emboss_strength,
                  self.hs_azimuth, self.hs_altitude, self.hs_strength):
            w.valueChanged.connect(self._sync)

    def retranslate(self):
        self.gb_contour.setTitle(t("contour.group"))
        self.gb_filters.setTitle(t("contour.filters_group"))
        self._l_interp.setText(t("contour.interp"))
        self._l_levels.setText(t("contour.n_levels"))
        self.cb_fill.setText(t("contour.fill"))
        self._l_step.setText(t("contour.grid_step"))
        cur = max(self.interp.currentIndex(), 0)
        self.interp.blockSignals(True)
        self.interp.clear()
        self.interp.addItems([t("contour.interp.triangulation"),
                              t("contour.interp.linear"),
                              t("contour.interp.cubic"),
                              t("contour.interp.nearest")])
        self.interp.setCurrentIndex(cur)
        self.interp.blockSignals(False)
        self.cb_smooth.setText(t("contour.smooth"))
        self._l_smooth_sigma.setText(t("contour.smooth_sigma"))
        self.cb_sharpen.setText(t("contour.sharpen"))
        self._l_sharpen_amount.setText(t("contour.sharpen_amount"))
        self.cb_emboss.setText(t("contour.emboss"))
        self._l_emboss_strength.setText(t("contour.emboss_strength"))
        self.cb_hillshade.setText(t("contour.hillshade"))
        self._l_hs_azimuth.setText(t("contour.hs_azimuth"))
        self._l_hs_altitude.setText(t("contour.hs_altitude"))
        self._l_hs_strength.setText(t("contour.hs_strength"))

    def _sync(self):
        methods = ("triangulation", "linear", "cubic", "nearest")
        self.v.interp_method = methods[max(self.interp.currentIndex(), 0)]
        self.v.n_levels = self.n_levels.value()
        self.v.fill = self.cb_fill.isChecked()
        self.v.grid_step = self.grid_step.value()
        self.v.smooth_on = self.cb_smooth.isChecked()
        self.v.smooth_sigma = self.smooth_sigma.value()
        self.v.sharpen_on = self.cb_sharpen.isChecked()
        self.v.sharpen_amount = self.sharpen_amount.value()
        self.v.emboss_on = self.cb_emboss.isChecked()
        self.v.emboss_strength = self.emboss_strength.value()
        self.v.hillshade_on = self.cb_hillshade.isChecked()
        self.v.hillshade_azimuth = self.hs_azimuth.value()
        self.v.hillshade_altitude = self.hs_altitude.value()
        self.v.hillshade_strength = self.hs_strength.value()
        self.smooth_sigma.setEnabled(self.v.smooth_on)
        self.sharpen_amount.setEnabled(self.v.sharpen_on)
        self.emboss_strength.setEnabled(self.v.emboss_on)
        for w in (self.hs_azimuth, self.hs_altitude, self.hs_strength):
            w.setEnabled(self.v.hillshade_on)
        self.changed.emit()


# --------------------------------------------------------------------------- #
class ProcessingPanel(_Panel):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        self._label = QLabel()
        self._label.setWordWrap(True)
        lay.addWidget(self._label)
        lay.addStretch(1)
        self._item = None
        self.retranslate()

    def show_item(self, item: FileItem | None):
        self._item = item
        if item and item.is_inverted:
            r = item.result
            self._label.setText(
                f"iterations: {len(r.history)}\n"
                f"final RMS: {r.misfit:.3f}%\n"
                f"cells: {r.param_mesh.n_param}\n"
                f"mesh nodes: {r.param_mesh.n_nodes}")
        else:
            self._label.setText(t("proc.none"))

    def retranslate(self):
        self.show_item(self._item)
