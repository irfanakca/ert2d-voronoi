"""Modal dialogs - currently just the automatic ``.loc`` generator."""

from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QLabel, QVBoxLayout)

from ert2d.gui.i18n import t, translator
from ert2d.io.loc_generator import LocGenSpec


class LocGeneratorDialog(QDialog):
    """Collect origin / azimuth / line spacing, then hand back a
    :class:`~ert2d.io.loc_generator.LocGenSpec` via :meth:`spec`.

    ``items`` (already in the order profiles will be stacked in - the tree's
    natural-sort order) is used only to show the user what will be written
    and how many profiles it covers; ``items[0]`` is the one that lands
    exactly at the chosen origin."""

    def __init__(self, parent, items):
        super().__init__(parent)
        self._items = items
        self.setModal(True)

        self.origin_x = QDoubleSpinBox()
        self.origin_y = QDoubleSpinBox()
        for w in (self.origin_x, self.origin_y):
            w.setRange(-1.0e7, 1.0e7)
            w.setDecimals(3)
            w.setSuffix(" m")
        self.azimuth = QDoubleSpinBox()
        self.azimuth.setRange(0.0, 360.0)
        self.azimuth.setDecimals(2)
        self.azimuth.setSuffix("°")
        self.azimuth.setValue(90.0)
        self.line_spacing = QDoubleSpinBox()
        self.line_spacing.setRange(0.0, 1000.0)
        self.line_spacing.setDecimals(3)
        self.line_spacing.setSuffix(" m")
        self.side = QComboBox()

        self._l_origin_x = QLabel(); self._l_origin_y = QLabel()
        self._l_azimuth = QLabel(); self._l_line_spacing = QLabel()
        self._l_side = QLabel()
        self._info = QLabel()
        self._info.setWordWrap(True)

        form = QFormLayout()
        form.addRow(self._l_origin_x, self.origin_x)
        form.addRow(self._l_origin_y, self.origin_y)
        form.addRow(self._l_azimuth, self.azimuth)
        form.addRow(self._l_line_spacing, self.line_spacing)
        form.addRow(self._l_side, self.side)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addWidget(self._info)
        lay.addLayout(form)
        lay.addWidget(self.buttons)

        translator.language_changed.connect(lambda _: self._retranslate())
        self._retranslate()

    def _retranslate(self):
        self.setWindowTitle(t("locgen.title"))
        self._l_origin_x.setText(t("locgen.origin_x"))
        self._l_origin_y.setText(t("locgen.origin_y"))
        self._l_azimuth.setText(t("locgen.azimuth"))
        self._l_line_spacing.setText(t("locgen.line_spacing"))
        self._l_side.setText(t("locgen.side"))
        cur = max(self.side.currentIndex(), 0)
        self.side.blockSignals(True)
        self.side.clear()
        self.side.addItems([t("locgen.side.right"), t("locgen.side.left")])
        self.side.setCurrentIndex(cur)
        self.side.blockSignals(False)
        names = ", ".join(it.name for it in self._items[:6])
        if len(self._items) > 6:
            names += ", …"
        no_survey = sum(1 for it in self._items if it.survey is None)
        extra = t("locgen.skipped", n=no_survey) if no_survey else ""
        self._info.setText(t("locgen.info", n=len(self._items), names=names) + extra)

    def spec(self) -> LocGenSpec:
        return LocGenSpec(
            origin_x=self.origin_x.value(), origin_y=self.origin_y.value(),
            azimuth_deg=self.azimuth.value(), line_spacing=self.line_spacing.value(),
            side="right" if self.side.currentIndex() == 0 else "left")
