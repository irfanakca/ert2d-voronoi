"""ELRIS 2D main window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (QApplication, QFileDialog, QMainWindow, QMessageBox,
                               QProgressBar, QSplitter, QTabWidget, QLabel)

from ert2d.gui.i18n import t, translator
from ert2d.gui.icons import icon, layers
from ert2d.gui.panels import (AcquisitionPanel, ContourFilterPanel, InversionPanel,
                              ProcessingPanel, VisualsPanel)
from ert2d.gui.state import AppState
from ert2d.gui.tree import ProjectTree
from ert2d.gui.view3d import View3D
from ert2d.gui.views import DataView, DepthMapView, LineView, MapView
from ert2d.gui.worker import InversionWorker, start_worker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.state = AppState()
        self._thread = self._worker = None
        self._depth_map_items = None
        self._last_depth_map_items = None

        self.tree = ProjectTree(self.state)
        self.line_view = LineView()
        self.data_view = DataView()
        self.map_view = MapView()
        self.depth_map_view = DepthMapView()
        self.view3d = View3D()
        self.centre = QTabWidget()
        self.centre.setTabPosition(QTabWidget.South)
        self.centre.addTab(self.line_view, "")
        self.centre.addTab(self.data_view, "")
        self.centre.addTab(self.map_view, "")
        self.centre.addTab(self.depth_map_view, "")
        self.centre.addTab(self.view3d, "")

        self.visuals = VisualsPanel()
        self.inversion = InversionPanel()
        self.processing = ProcessingPanel()
        self.acquisition = AcquisitionPanel()
        self.contour = ContourFilterPanel()
        self.right = QTabWidget()
        self.right.setTabPosition(QTabWidget.South)
        for w in (self.visuals, self.inversion, self.processing, self.acquisition,
                  self.contour):
            self.right.addTab(w, "")
        self.right.setMaximumWidth(430)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.tree)
        split.addWidget(self.centre)
        split.addWidget(self.right)
        split.setStretchFactor(1, 1)
        split.setSizes([220, 700, 380])
        self.setCentralWidget(split)

        self._build_menus()
        self._build_toolbar()
        self._build_statusbar()

        self.tree.selection_changed.connect(self._on_select)
        self.tree.check_changed.connect(self._refresh_map)
        self.tree.items_removed.connect(self._on_items_removed)
        self.visuals.changed.connect(self._refresh_line_view)
        self.contour.changed.connect(self._refresh_depth_map)
        self.depth_map_view.animate_requested.connect(self.export_depth_animation)
        translator.language_changed.connect(self.retranslate)

        self.resize(1200, 850)
        self.retranslate()

    # ------------------------------------------------------------------ #
    def _build_menus(self):
        mb = self.menuBar()
        self.m_file = mb.addMenu("")
        self.a_open_folder = self.m_file.addAction("", self.open_folder)
        self.a_open_files = self.m_file.addAction("", self.open_files)
        self.m_file.addSeparator()
        self.a_save = self.m_file.addAction("", self.save_result)
        self.a_export = self.m_file.addAction("", self.export_image)
        self.m_file.addSeparator()
        self.a_quit = self.m_file.addAction("", self.close)
        self.a_quit.setShortcut(QKeySequence.Quit)

        self.m_edit = mb.addMenu("")
        self.a_remove = self.m_edit.addAction("", self.remove_selected)
        self.m_edit.addSeparator()
        self.a_gen_loc = self.m_edit.addAction("", self.generate_loc_files)
        self.m_view = mb.addMenu("")
        self.m_export = mb.addMenu("")
        self.m_settings = mb.addMenu("")
        self.m_lang = self.m_settings.addMenu("")
        grp = QActionGroup(self)
        self.a_lang_tr = self._lang_action("tr", grp)
        self.a_lang_en = self._lang_action("en", grp)
        self.m_lang.addActions([self.a_lang_tr, self.a_lang_en])

        self.m_help = mb.addMenu("")
        self.a_about = self.m_help.addAction("", self._about)

    def _lang_action(self, code, group):
        a = QAction("", self, checkable=True)
        a.setChecked(translator.language == code)
        a.triggered.connect(lambda: translator.set_language(code))
        group.addAction(a)
        return a

    def _build_toolbar(self):
        tb = self.addToolBar("main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonIconOnly)
        tb.setIconSize(QSize(16, 16))
        self.a_tb_open = tb.addAction(icon("open_folder"), "", self.open_folder)
        self.a_tb_files = tb.addAction(icon("open_files"), "", self.open_files)
        tb.addSeparator()
        self.a_run = tb.addAction(icon("run"), "", self.run_inversion)
        self.a_stop = tb.addAction(icon("stop"), "", self.stop_inversion)
        self.a_stop.setEnabled(False)
        tb.addSeparator()
        self.a_depth_map = tb.addAction(layers(), "", self.build_depth_map)
        tb.addSeparator()
        self.a_tb_save = tb.addAction(icon("save"), "", self.save_result)
        self.a_tb_export = tb.addAction(icon("export"), "", self.export_image)

    def _build_statusbar(self):
        self.status_label = QLabel()
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setVisible(False)
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.progress)

    # ------------------------------------------------------------------ #
    def retranslate(self):
        self.setWindowTitle(t("app.title"))
        self.m_file.setTitle(t("menu.file"))
        self.m_edit.setTitle(t("menu.edit"))
        self.m_view.setTitle(t("menu.view"))
        self.m_export.setTitle(t("menu.export"))
        self.m_settings.setTitle(t("menu.settings"))
        self.m_lang.setTitle(t("menu.language"))
        self.m_help.setTitle(t("menu.help"))
        self.a_open_folder.setText(t("action.open_folder"))
        self.a_open_files.setText(t("action.open_files"))
        self.a_save.setText(t("action.save_result"))
        self.a_export.setText(t("action.export_image"))
        self.a_quit.setText(t("action.quit"))
        self.a_remove.setText(t("action.remove_from_list"))
        self.a_gen_loc.setText(t("action.gen_loc"))
        self.a_about.setText(t("action.about"))
        self.a_lang_tr.setText(t("lang.tr"))
        self.a_lang_en.setText(t("lang.en"))
        for act, key in ((self.a_run, "action.run"), (self.a_stop, "action.stop"),
                         (self.a_depth_map, "action.depth_map"),
                         (self.a_tb_open, "action.open_folder"),
                         (self.a_tb_files, "action.open_files"),
                         (self.a_tb_save, "action.save_result"),
                         (self.a_tb_export, "action.export_image")):
            act.setText(t(key))
            act.setToolTip(t(key))
        for i, k in enumerate(("tab.line_view", "tab.data", "tab.map_view",
                               "tab.depth_map", "tab.view3d")):
            self.centre.setTabText(i, t(k))
        for i, k in enumerate(("tab.visuals", "tab.inversion", "tab.processing",
                               "tab.acquisition", "tab.contour_filters")):
            self.right.setTabText(i, t(k))
        if not self.status_label.text():
            self.status_label.setText(t("status.ready"))

    # ------------------------------------------------------------------ #
    def open_folder(self):
        d = QFileDialog.getExistingDirectory(self, t("action.open_folder"))
        if d:
            self.state.add_folder(d)
            self.tree.rebuild()
            self._refresh_map()

    def open_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, t("action.open_files"), "", "RES2DINV (*.dat);;All files (*)")
        if files:
            self.state.add_paths(files)
            self.tree.rebuild()
            self._refresh_map()

    def remove_selected(self):
        items = self.tree.current_or_checked()
        if not items:
            return
        for it in items:
            self.state.remove(it.path)
        self.tree.rebuild()
        self._on_items_removed()

    def generate_loc_files(self):
        items = self.tree.current_or_checked()
        if not items:
            QMessageBox.information(self, t("action.gen_loc"), t("locgen.none_selected"))
            return
        from ert2d.gui.dialogs import LocGeneratorDialog
        from ert2d.io.loc_generator import generate_loc_files

        dlg = LocGeneratorDialog(self, items)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        written = generate_loc_files(items, dlg.spec())
        from ert2d.io.loc import loc_path_for, read_loc

        for it in items:                       # re-read the .loc just written
            loc = loc_path_for(it.path)
            try:
                it.location = read_loc(loc) if loc.is_file() else None
            except Exception:                  # noqa: BLE001
                it.location = None
        self.tree.refresh_decorations()
        self._refresh_map()
        if self._current_item() in items:
            self._on_select(self._current_item())
        self.status_label.setText(t("locgen.done", n=len(written)))

    def save_result(self):
        it = self._current_item()
        if not it or not it.is_inverted:
            return
        from ert2d.io.results import save_result
        out = it.path.with_name(it.name + "_inv")
        save_result(out, it.result, it.survey)
        self.status_label.setText(f"saved {out.name}.npz")

    def export_image(self):
        f, _ = QFileDialog.getSaveFileName(self, t("action.export_image"),
                                           "", "PNG (*.png);;PDF (*.pdf)")
        if f:
            self.centre.currentWidget()._c.figure.savefig(f, dpi=150)

    # ------------------------------------------------------------------ #
    def _current_item(self):
        cur = self.tree.currentItem()
        from ert2d.gui.tree import _ROLE_ITEM
        return cur.data(0, _ROLE_ITEM) if cur else None

    def _on_select(self, item):
        self.line_view.show_item(item, self.visuals.v)
        self.data_view.show_item(item, self.visuals.v)
        self.acquisition.show_item(item)
        self.processing.show_item(item)
        self._refresh_map()
        if item:
            self.status_label.setText(t("status.loaded", name=item.name))

    def _refresh_line_view(self):
        self.line_view.show_item(self._current_item(), self.visuals.v)
        self.data_view.show_item(self._current_item(), self.visuals.v)
        self.depth_map_view.refresh(self.visuals.v, self.contour.v)
        self.view3d.refresh(self.visuals.v)

    def _refresh_depth_map(self):
        self.depth_map_view.refresh(self.visuals.v, self.contour.v)

    def _refresh_map(self):
        items = list(self.state.files.values())
        self.map_view.show_items(items, self._current_item())

    def _on_items_removed(self):
        self._on_select(None)
        self._refresh_map()

    # ------------------------------------------------------------------ #
    def _start_batch(self, items):
        self._thread, self._worker = start_worker(items)
        w: InversionWorker = self._worker
        w.file_started.connect(self._on_file_started)
        w.iteration.connect(self._on_iteration)
        w.file_finished.connect(self._on_file_finished)
        w.file_failed.connect(self._on_file_failed)
        w.batch_finished.connect(self._on_batch_finished)
        w.stopped.connect(self._on_batch_finished)
        self._thread.finished.connect(self._cleanup_thread)

        self.a_run.setEnabled(False)
        self.a_depth_map.setEnabled(False)
        self.a_stop.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self._thread.start()

    def run_inversion(self):
        if self._thread is not None:
            return
        items = self.tree.current_or_checked()
        if not items:
            return
        for it in items:
            self.inversion.apply_to(it)
        self._start_batch(items)

    def build_depth_map(self):
        if self._thread is not None:
            return
        items = self.tree.current_or_checked()
        if not items:
            return
        missing = [it for it in items if it.location is None]
        if missing:
            QMessageBox.warning(
                self, t("action.depth_map"),
                t("depthmap.missing_loc", names=", ".join(it.name for it in missing)))
            return
        self._depth_map_items = items
        to_invert = [it for it in items if it.result is None]
        if not to_invert:
            self._finish_depth_map(items)
            return
        for it in to_invert:
            self.inversion.apply_to(it)
        self._start_batch(to_invert)

    def stop_inversion(self):
        if self._worker:
            self._worker.request_stop()
            self.a_stop.setEnabled(False)

    def _on_file_started(self, item):
        self.status_label.setText(t("status.running", name=item.name))

    def _on_iteration(self, item, info):
        self.status_label.setText(
            t("status.iter", name=item.name, it=info.iteration,
              rms=info.misfit_after))
        if item is self._current_item():
            self.line_view.show_item(item, self.visuals.v)
            self.data_view.show_item(item, self.visuals.v)

    def _on_file_finished(self, item):
        self.status_label.setText(
            t("status.done", name=item.name, rms=item.result.misfit))
        self.tree.refresh_decorations()
        if item is self._current_item():
            self._on_select(item)

    def _on_file_failed(self, item, msg):
        self.status_label.setText(t("status.error", msg=f"{item.name}: {msg}"))

    def _on_batch_finished(self, n=0):
        self.status_label.setText(t("status.batch_done", n=n))
        items, self._depth_map_items = self._depth_map_items, None
        if items:
            self._finish_depth_map(items)

    def _finish_depth_map(self, items):
        ok_items = [it for it in items if it.result is not None]
        if len(ok_items) < 2:
            QMessageBox.warning(self, t("action.depth_map"), t("depthmap.need_two"))
            return
        from ert2d.io.depth_map import build_depth_map

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            dm = build_depth_map(ok_items)
        except Exception as exc:                # noqa: BLE001
            QMessageBox.warning(self, t("action.depth_map"), str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._last_depth_map_items = ok_items
        self.depth_map_view.show_depth_map(dm, self.visuals.v, self.contour.v)
        self.centre.setCurrentWidget(self.depth_map_view)
        self.status_label.setText(t("depthmap.done", n=len(ok_items)))
        self._build_volume3d(dm)

    def _build_volume3d(self, dm):
        from ert2d.io.volume import build_volume
        from ert2d.viz.volume3d import to_image_data

        try:
            vol = build_volume(dm, grid_step=self.contour.v.grid_step)
            grid = to_image_data(vol)
        except Exception:                        # noqa: BLE001  (e.g. too few points)
            self.view3d.show_volume(None, None, self.visuals.v, dm=None)
            return
        self.view3d.show_volume(grid, vol, self.visuals.v, dm=dm)

    def export_depth_animation(self, step_m: float):
        items = self._last_depth_map_items
        if not items:
            QMessageBox.information(self, t("depthmap.build_animation"),
                                    t("depthmap.anim_none"))
            return
        z_max_common = max(float(it.survey.z_max) for it in items)
        depths = np.arange(0.0, z_max_common + step_m / 2, step_m)
        if depths.size < 2:
            QMessageBox.warning(self, t("depthmap.build_animation"),
                                t("depthmap.anim_too_few"))
            return

        f, _ = QFileDialog.getSaveFileName(self, t("depthmap.anim_save"),
                                           "", "GIF (*.gif)")
        if not f:
            return

        from ert2d.io.depth_map import build_depth_map
        from ert2d.viz.depth_map import export_depth_map_gif, export_depth_slice_pngs

        v = self.visuals.v
        c = self.contour.v
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            dm = build_depth_map(items, depths=depths)
            common = dict(
                cmap=v.cmap, log=v.log_scale, clim=v.model_clim,
                n_levels=c.n_levels, fill=c.fill,
                contour_lines=v.show_contour_lines, contour_labels=v.show_labels,
                interp_method=c.interp_method, grid_step=c.grid_step,
                smooth_on=c.smooth_on, smooth_sigma=c.smooth_sigma,
                sharpen_on=c.sharpen_on, sharpen_amount=c.sharpen_amount,
                emboss_on=c.emboss_on, emboss_strength=c.emboss_strength,
                hillshade_on=c.hillshade_on, hillshade_azimuth=c.hillshade_azimuth,
                hillshade_altitude=c.hillshade_altitude,
                hillshade_strength=c.hillshade_strength,
                show_lines=self.depth_map_view.cb_lines.isChecked(),
                show_line_names=self.depth_map_view.cb_names.isChecked(),
                xlabel=t("axis.easting" if dm.is_global else "axis.x_dist"),
                ylabel=t("axis.northing" if dm.is_global else "axis.y_dist"),
                cbar_label=t("cbar.rho"),
                title_fmt=lambda d: t("depthmap.title", depth=d))
            export_depth_map_gif(f, dm, **common)
            # one PNG per step next to the GIF, same look as "Export image"
            fig_size = tuple(self.depth_map_view._c.figure.get_size_inches())
            export_depth_slice_pngs(Path(f).parent, dm, size_inches=fig_size, **common)
        except Exception as exc:                # noqa: BLE001
            QMessageBox.warning(self, t("depthmap.build_animation"), str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.status_label.setText(t("depthmap.anim_done", n=depths.size))

    def _cleanup_thread(self):
        self._thread = self._worker = None
        self.a_run.setEnabled(True)
        self.a_depth_map.setEnabled(True)
        self.a_stop.setEnabled(False)
        self.progress.setVisible(False)

    def _about(self):
        QMessageBox.about(self, t("app.title"),
                          "ELRIS 2D — Voronoi-parameterised DC resistivity "
                          "inversion.\nPython port of main_vor_v00.")
