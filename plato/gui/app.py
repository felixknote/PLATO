"""GUI entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ..config import Config
from .branding import app_icon
from .main_window import MainWindow
from ..data.session import Session
from .theme import apply_theme


def _detect_os_theme(app: QApplication) -> str:
    """The OS's own light/dark setting, for a first launch with nothing saved.

    Qt's styleHints().colorScheme() reads the platform's actual setting --
    verified against Windows' own AppsUseLightTheme registry value, which
    this agreed with exactly on the machine this was built on. Falls back to
    dark, this app's long-standing default, for the Unknown case: an
    older Qt, or a platform/window manager that does not report a preference
    at all (common on Linux without a desktop portal running).
    """
    from PySide6.QtCore import Qt

    from . import themes

    try:
        scheme = app.styleHints().colorScheme()
    except AttributeError:  # pragma: no cover - Qt < 6.5
        return themes.DARK
    if scheme == Qt.ColorScheme.Light:
        return themes.LIGHT
    if scheme == Qt.ColorScheme.Dark:
        return themes.DARK
    return themes.DARK


def run(cfg: Config | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("PLATO")
    app.setWindowIcon(app_icon())
    # Restore the saved theme; on first run (nothing saved yet), detect the
    # OS's own light/dark setting instead of always defaulting to dark. Goes
    # through themes.apply rather than apply_theme so the light palette is
    # reachable at startup and not only after a toggle. The manual Ctrl+D
    # toggle in the View menu still overrides this for the session and is
    # what gets saved from then on -- this only decides the very first launch.
    from PySide6.QtCore import QSettings

    from . import themes
    from .settings import APPLICATION, ORGANISATION

    settings = QSettings(ORGANISATION, APPLICATION)
    saved = settings.value("theme", None)
    if saved is None:
        saved = _detect_os_theme(app)
    else:
        saved = str(saved)
    themes.apply(saved if saved in themes.THEMES else themes.DARK, app)
    session = Session()
    if cfg is not None:
        session.add_plate(cfg)
    window = MainWindow(session)
    # No show() here: the window shows itself in _restore_geometry, either
    # maximised or at its saved size. Calling show() after showMaximized()
    # is what used to collapse it to the layout minimum.
    return app.exec()
