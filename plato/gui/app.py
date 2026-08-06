"""GUI entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ..config import Config
from .main_window import MainWindow
from .session import Session


def run(cfg: Config | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    session = Session()
    if cfg is not None:
        session.add_plate(cfg)
    window = MainWindow(session)
    window.show()
    return app.exec()
