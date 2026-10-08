"""Headless smoke tests for the PySide6 GUI."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

import matplotlib                                  # noqa: E402
matplotlib.use("QtAgg")

from PySide6.QtWidgets import QApplication         # noqa: E402

from ert2d.gui.i18n import translator              # noqa: E402
from ert2d.gui.main_window import MainWindow       # noqa: E402
from ert2d.gui.worker import InversionWorker       # noqa: E402

SAMPLE_DATA = Path(__file__).resolve().parents[1] / "sample_data"


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(app):
    w = MainWindow()
    w.state.add_folder(SAMPLE_DATA)          # bundled synthetic data - always present
    w.tree.rebuild()
    return w


def test_window_builds_and_loads(win):
    assert len(win.state.files) > 0
    top = win.tree.topLevelItem(0)
    assert top.childCount() > 0


def test_language_switch_retranslates(win):
    translator.set_language("en")
    assert win.windowTitle() == "ELRIS 2D"
    translator.set_language("tr")
    assert win.centre.tabText(0) == "Hat Görünümü"


def test_select_shows_views(win):
    top = win.tree.topLevelItem(0)
    child = top.child(0)
    win.tree.setCurrentItem(child)
    it = win._current_item()
    assert it is not None and it.is_loaded
    win.line_view.show_item(it, win.visuals.v)
    win.data_view.show_item(it)
    win.acquisition.show_item(it)


def test_inversion_then_redraw(win):
    top = win.tree.topLevelItem(0)
    child = top.child(0)
    win.tree.setCurrentItem(child)
    it = win._current_item()
    it.result = InversionWorker([it])._invert_one(it)
    assert it.is_inverted and it.result.misfit < 20
    win.line_view.show_item(it, win.visuals.v)      # must not raise
    win.processing.show_item(it)
