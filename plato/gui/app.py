"""GUI entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ..config import Config
from .branding import app_icon
from .main_window import MainWindow
from ..data.session import Session
from .theme import apply_theme


def run(cfg: Config | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("PLATO")
    app.setWindowIcon(app_icon())
    # Restore the saved theme, defaulting to dark. Goes through themes.apply
    # rather than apply_theme so the light palette is reachable at startup and
    # not only after a toggle.
    from PySide6.QtCore import QSettings

    from . import themes
    from .settings import APPLICATION, ORGANISATION

    saved = str(QSettings(ORGANISATION, APPLICATION).value("theme", themes.DARK))
    themes.apply(saved if saved in themes.THEMES else themes.DARK, app)
    session = Session()
    if cfg is not None:
        session.add_plate(cfg)
    window = MainWindow(session)
    window.show()
    return app.exec()
