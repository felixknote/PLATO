"""Where this installation keeps its data.

Nothing here is baked into the code. A path is found, in order, from:

1. an environment variable (``PLATO_EMBEDDING_ROOT``, ``PLATO_DATA_ROOT``,
   ``PLATO_IMAGE_ROOT``) -- how a lab pins a shared location for everyone;
2. what the user last chose in the app, remembered via ``QSettings``;
3. nothing, in which case the UI asks.

There is deliberately no fallback to a drive letter. A default like
``Z:\\Analysis\\DINO`` works on exactly one machine and fails silently and
confusingly everywhere else -- a colleague, a laptop without the share mapped,
or the same machine after the folder is renamed. Asking once and remembering
the answer is correct on every machine instead.

The same reasoning applies to *finding* things inside those roots: identify a
dataset by what it contains (a ``features_metadata.csv``), never by a folder
name, because names change.
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QSettings

ORGANISATION = "plato"
APPLICATION = "plato"

# Environment variable -> QSettings key, for each configurable root.
EMBEDDING_ROOT = ("PLATO_EMBEDDING_ROOT", "paths/embedding_root")
DATA_LIBRARY = ("PLATO_DATA_ROOT", "paths/data_library")
IMAGE_ROOT = ("PLATO_IMAGE_ROOT", "paths/image_root")


def _settings() -> QSettings:
    return QSettings(ORGANISATION, APPLICATION)


def get_root(spec: tuple[str, str]) -> Path | None:
    """The configured directory for ``spec``, or None if there is not one.

    A configured path that no longer exists returns None rather than a dead
    Path: the caller's job is then to ask, which is the same thing it would do
    if nothing had been configured.
    """
    env_name, key = spec
    raw = os.environ.get(env_name) or _settings().value(key, "")
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    return path if path.is_dir() else None


def set_root(spec: tuple[str, str], path: Path | None) -> None:
    """Remember ``path`` for next time. Passing None forgets it."""
    _, key = spec
    settings = _settings()
    if path is None:
        settings.remove(key)
    else:
        settings.setValue(key, str(Path(path)))
    settings.sync()


def guess_data_library(image_root: Path | None) -> Path | None:
    """The folder screens live in, inferred from one screen inside it.

    Saves asking twice: choosing a screen's own folder tells us where its
    siblings are.
    """
    if image_root is None:
        return None
    parent = Path(image_root).parent
    return parent if parent.is_dir() else None
