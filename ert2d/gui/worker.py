"""Background inversion worker.

Runs :func:`ert2d.inversion.invert` for one or more :class:`FileItem`s on a
``QThread`` and reports progress via signals so the UI stays responsive.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal

import numpy as np

from ert2d.gui.state import FileItem
from ert2d.inversion import invert
from ert2d.inversion.driver import IterInfo
from ert2d.io.itd import save_itd
from ert2d.mesh import build_param_mesh


class LiveResult:
    """Minimal stand-in for ``InversionResult`` during iteration, so the views
    can redraw before the run finishes."""

    def __init__(self, param_mesh, rhoa_obs, weights, history):
        self.param_mesh = param_mesh
        self.rhoa_obs = np.asarray(rhoa_obs, float)
        self.weights = np.asarray(weights, float)
        self.history = list(history)
        self.jacobian = None

    @property
    def rho_cell(self):
        return self.history[-1].rho

    @property
    def rhoa_calc(self):
        return self.history[-1].rhoa_calc

    @property
    def misfit(self):
        return self.history[-1].misfit_after

    @property
    def residual_log(self):
        return np.log(self.rhoa_obs) - np.log(self.rhoa_calc)

    def datum_misfit_pct(self):
        return 100.0 * self.weights * np.abs(self.residual_log)

    @property
    def rms_curve(self):
        return np.array([i.misfit_after for i in self.history])

    @property
    def lambda_curve(self):
        return np.array([i.lam for i in self.history])

    def rho_elem(self):
        bg = float(np.exp(np.mean(np.log(self.rhoa_obs))))
        out = np.full(self.param_mesh.t2.shape[1], bg)
        for k, els in enumerate(self.param_mesh.cell_elements):
            out[els] = self.rho_cell[k]
        return out


class InversionWorker(QObject):
    file_started = Signal(object)                 # FileItem
    iteration = Signal(object, object)            # FileItem, IterInfo
    file_finished = Signal(object)                # FileItem  (result set on it)
    file_failed = Signal(object, str)             # FileItem, message
    batch_finished = Signal(int)                  # count
    stopped = Signal()

    def __init__(self, items: list[FileItem]):
        super().__init__()
        self._items = items
        self._abort = False

    def request_stop(self) -> None:
        self._abort = True

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        done = 0
        for item in self._items:
            if self._abort:
                break
            if not item.is_loaded:
                self.file_failed.emit(item, item.load_error or "not loaded")
                continue
            self.file_started.emit(item)
            item.partial = None
            try:
                item.result = self._invert_one(item)
                item.partial = None
                self.file_finished.emit(item)
                done += 1
            except Exception as exc:             # noqa: BLE001
                self.file_failed.emit(item, str(exc))
        if self._abort:
            self.stopped.emit()
        else:
            self.batch_finished.emit(done)

    def _invert_one(self, item: FileItem):
        s = item.survey
        pm = build_param_mesh(s, kind=item.mesh.kind, wing=item.mesh.wing,
                              depth_factor=item.mesh.depth_factor,
                              surface_div=item.mesh.surface_div)

        iv = item.inversion
        history: list[IterInfo] = []

        def cb(info: IterInfo):
            if self._abort:
                raise _Aborted()
            history.append(info)
            item.partial = LiveResult(pm, s.rhoa,
                                      _weights(s.rhoa), history)
            self.iteration.emit(item, info)

        result = invert(s, pm, max_iter=iv.max_iter, lambda_mode=iv.lambda_mode,
                        lam0=iv.lam0, cool=iv.cooling, target_rms=iv.target_rms,
                        tol=iv.tol, callback=cb, robust=iv.robust,
                        robust_min_weight=iv.robust_min_weight)
        try:
            save_itd(item.path.with_suffix(".itd"), result)
        except Exception:                        # noqa: BLE001  (non-fatal)
            pass
        return result


class _Aborted(Exception):
    pass


def _weights(rhoa):
    from ert2d.inversion.weights import median_deviation_weights
    return median_deviation_weights(rhoa)


def start_worker(items: list[FileItem]):
    """Create a (thread, worker) pair; caller must keep both alive and
    ``thread.start()``."""
    thread = QThread()
    worker = InversionWorker(items)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    for sig in (worker.batch_finished, worker.stopped):
        sig.connect(thread.quit)
    return thread, worker
