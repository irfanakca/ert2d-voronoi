"""Entry point:  ``python -m ert2d.gui``  or  ``ert2d-gui``."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    import matplotlib
    matplotlib.use("QtAgg")
    from PySide6.QtWidgets import QApplication

    from ert2d.gui.main_window import MainWindow

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("ELRIS 2D")
    app.setStyle("Fusion")            # flat, square controls; consistent look
    win = MainWindow()

    # optional: folder / files on the command line
    args = (argv if argv is not None else sys.argv)[1:]
    from pathlib import Path
    paths = [Path(a) for a in args if Path(a).exists()]
    folders = [p for p in paths if p.is_dir()]
    files = [p for p in paths if p.is_file()]
    for f in folders:
        win.state.add_folder(f)
    if files:
        win.state.add_paths(files)
    if paths:
        win.tree.rebuild()
        win._refresh_map()

    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
