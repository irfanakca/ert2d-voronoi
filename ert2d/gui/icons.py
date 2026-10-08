"""Icon helpers - bundled Fugue PNGs plus small painted status dots."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap

_DIR = Path(__file__).parent / "icons"


@lru_cache(maxsize=64)
def icon(name: str) -> QIcon:
    p = _DIR / f"{name}.png"
    return QIcon(str(p)) if p.exists() else QIcon()


@lru_cache(maxsize=8)
def dot(color: str, size: int = 10) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    pr = QPainter(pm)
    pr.setRenderHint(QPainter.Antialiasing)
    pr.setBrush(QColor(color))
    pr.setPen(Qt.NoPen)
    pr.drawEllipse(1, 1, size - 2, size - 2)
    pr.end()
    return QIcon(pm)


@lru_cache(maxsize=4)
def layers(color: str = "#444444", size: int = 16) -> QIcon:
    """Three stacked bars, narrowing with depth - stands in for a
    depth-slice map when no bundled PNG exists."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    pr = QPainter(pm)
    pr.setRenderHint(QPainter.Antialiasing)
    pr.setPen(Qt.NoPen)
    qc = QColor(color)
    bar_h = size / 4.5
    gap = bar_h * 0.6
    widths = (size - 2, size - 6, size - 10)
    for i, w in enumerate(widths):
        pr.setOpacity(1.0 - 0.22 * i)
        pr.setBrush(qc)
        x = (size - w) / 2
        y = 1 + i * (bar_h + gap)
        pr.drawRect(QRectF(x, y, w, bar_h))
    pr.end()
    return QIcon(pm)
