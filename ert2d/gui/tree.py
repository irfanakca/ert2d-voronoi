"""Left panel: checkable project / file tree."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QMenu, QTreeWidget, QTreeWidgetItem

from ert2d.gui.i18n import t, translator
from ert2d.gui.icons import dot
from ert2d.gui.state import AppState, FileItem

_ROLE_ITEM = Qt.UserRole + 1

_STYLE = """
QTreeWidget { outline: 0; }
QTreeWidget::item { padding: 2px 0; }
QTreeWidget::item:selected {
    background: #2f6fb3;
    color: white;
    border: 0;
}
QTreeWidget::indicator {
    width: 13px; height: 13px;
    border: 1px solid #888;
    background: white;
}
QTreeWidget::indicator:checked {
    background: #2f6fb3;
    border: 1px solid #2f6fb3;
    image: none;
}
"""


class ProjectTree(QTreeWidget):
    selection_changed = Signal(object)          # FileItem | None
    check_changed = Signal()
    items_removed = Signal()

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.setColumnCount(1)
        self.setHeaderLabels([t("tree.header")])
        self.setAlternatingRowColors(True)
        self.setStyleSheet(_STYLE)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.itemChanged.connect(self._on_item_changed)
        self.currentItemChanged.connect(self._on_current)
        self.customContextMenuRequested.connect(self._context_menu)
        translator.language_changed.connect(
            lambda _: self.setHeaderLabels([t("tree.header")]))

    # ------------------------------------------------------------------ #
    def rebuild(self):
        self.blockSignals(True)
        self.clear()
        for project, items in sorted(self.state.by_project().items()):
            top = QTreeWidgetItem(self, [project])
            top.setFlags(top.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
            top.setCheckState(0, Qt.Unchecked)
            for it in items:
                child = QTreeWidgetItem(top, [it.name])
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                child.setCheckState(0, Qt.Unchecked)
                child.setData(0, _ROLE_ITEM, it)
                self._decorate(child, it)
            top.setExpanded(True)
        self.blockSignals(False)

    # ------------------------------------------------------------------ #
    def checked_items(self) -> list[FileItem]:
        out = []
        for i in range(self.topLevelItemCount()):
            top = self.topLevelItem(i)
            for j in range(top.childCount()):
                c = top.child(j)
                if c.checkState(0) == Qt.Checked:
                    out.append(c.data(0, _ROLE_ITEM))
        return out

    def current_or_checked(self) -> list[FileItem]:
        chk = self.checked_items()
        if chk:
            return chk
        cur = self.currentItem()
        it = cur.data(0, _ROLE_ITEM) if cur else None
        return [it] if it else []

    def refresh_decorations(self):
        for i in range(self.topLevelItemCount()):
            top = self.topLevelItem(i)
            for j in range(top.childCount()):
                c = top.child(j)
                self._decorate(c, c.data(0, _ROLE_ITEM))

    @staticmethod
    def _decorate(child, it: FileItem | None):
        """Status via an icon dot + font weight - never a text colour, so the
        name stays readable (white) when the row is selected."""
        if it is None:
            return
        f = child.font(0)
        if it.load_error:
            child.setIcon(0, dot("#d64545"))
            child.setToolTip(0, it.load_error)
            f.setBold(False)
        elif it.is_inverted:
            child.setIcon(0, dot("#3a9a4e"))
            child.setToolTip(0, "")
            f.setBold(True)
        else:
            child.setIcon(0, dot("#b0b0b0"))
            child.setToolTip(0, "")
            f.setBold(False)
        child.setFont(0, f)

    # ------------------------------------------------------------------ #
    def _on_current(self, cur, _prev):
        it = cur.data(0, _ROLE_ITEM) if cur else None
        self.selection_changed.emit(it)

    def _on_item_changed(self, item, _col):
        self.check_changed.emit()

    # ------------------------------------------------------------------ #
    def _targets_for(self, item: QTreeWidgetItem) -> list[FileItem]:
        """What a right-click on ``item`` should remove: a whole project group,
        the current checked set (if the clicked file is part of it), or just
        the clicked file."""
        if item.parent() is None:
            return [c.data(0, _ROLE_ITEM) for j in range(item.childCount())
                    for c in [item.child(j)] if c.data(0, _ROLE_ITEM) is not None]
        it = item.data(0, _ROLE_ITEM)
        checked = self.checked_items()
        if it in checked:
            return checked
        return [it] if it is not None else []

    def _context_menu(self, pos):
        item = self.itemAt(pos)
        if item is None:
            return
        targets = self._targets_for(item)
        if not targets:
            return
        menu = QMenu(self)
        menu.addAction(t("action.remove_from_list"), lambda: self._remove(targets))
        menu.exec(self.viewport().mapToGlobal(pos))

    def _remove(self, targets: list[FileItem]) -> None:
        for it in targets:
            self.state.remove(it.path)
        self.rebuild()
        self.items_removed.emit()
