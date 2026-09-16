"""Remembering which folder holds a dataset's images, across restarts.

Locating images is real, sometimes manual work: an automatic hunt across a
data library, or the user picking a folder by hand in Locate. Before this,
that answer lived only in ``EmbeddingExplorer._resolver_cache``, a plain
in-memory dict -- so every new PLATO session re-hunted or re-asked for every
dataset, including ones already confirmed correct a dozen times before.

The dataset directory is the natural key: it is what the resolver cache in
``explorer.py`` already keys on, and it is stable across restarts for any
real on-disk export. Only the resolved ROOT PATHS are persisted, not the
scanned file index -- a saved index would be a second copy of the filesystem
to keep in sync (renamed, moved or deleted files would silently go stale);
storing paths and re-probing them against the frame on each load always
reflects what is actually on disk right now.

Re-probing is NOT cheap, though it was once believed to be: it re-walks each
remembered root exactly as a fresh scan would (there being no way to check a
file still exists, or a new one appeared, without listing the directory), and
on a real network share that is minutes, not seconds -- measured at ~12
minutes across six real roots. ``rebuild_resolver`` therefore takes an
``on_progress`` callback and its caller (``EmbeddingExplorer._start_resolving_images``)
runs it off the GUI thread, the same way the very first hunt for a dataset
already does -- a remembered answer must never be assumed instant.

This remembers the choice by default. It is not a lock: Locate's "Choose
folder..." always overrides it for the one dataset being relocated, and
overwrites the remembered answer for next time -- "usually the connection
should not change" means auto-apply, not that it cannot be changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .explorer_model import ImageResolver


class ResolverStore:
    """dataset directory (str) -> list of image root paths, on disk as JSON."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._data: dict[str, list[str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(raw, dict):
            self._data = {
                str(key): [str(p) for p in value]
                for key, value in raw.items()
                if isinstance(value, list)
            }

    def roots_for(self, dataset_directory: Path | str) -> list[Path] | None:
        """The remembered roots for this dataset, or None if never recorded."""
        paths = self._data.get(str(dataset_directory))
        if not paths:
            return None
        return [Path(p) for p in paths]

    def remember(self, dataset_directory: Path | str, roots: list[Path]) -> None:
        """Record ``roots`` as the answer for ``dataset_directory``, and save.

        Called every time a resolver is confirmed -- by the automatic hunt or
        by Locate -- so a folder chosen by hand becomes tomorrow's automatic
        answer without anything else changing.
        """
        key = str(dataset_directory)
        paths = [str(r) for r in roots]
        if self._data.get(key) == paths:
            return  # Nothing changed; skip the write.
        self._data[key] = paths
        self._save()

    def forget(self, dataset_directory: Path | str) -> None:
        """Drop the remembered answer for one dataset, if any."""
        key = str(dataset_directory)
        if key in self._data:
            del self._data[key]
            self._save()

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        except OSError:
            # A cache that cannot be written costs a re-hunt next time,
            # nothing more -- never block on it.
            pass


def rebuild_resolver(roots: list[Path], frame, *, on_progress=None) -> ImageResolver | None:
    """Re-scan ``roots`` and probe them against ``frame``, folding them
    together exactly as LocateAllDialog does when a dataset's images are
    split across more than one folder.

    Returns None if none of the roots still resolve anything -- a remembered
    folder can go stale (moved, deleted, a share unmounted), and that must
    read as "not located" rather than crash or silently show nothing.

    ``on_progress``, when given, is passed straight to
    ``ImageResolver.for_root`` for each root in turn -- see
    ``image_lookup._walk_into`` for when it fires. This is the expensive call
    in this module (see the module docstring); a caller on the GUI thread
    must never invoke this without routing the work to a background task.
    """
    from .explorer_model import ImageResolver

    combined: ImageResolver | None = None
    for root in roots:
        if not Path(root).is_dir():
            continue
        resolver = ImageResolver.for_root(root, frame, on_progress=on_progress)
        if resolver.report is None or not resolver.report.ok:
            continue
        combined = resolver if combined is None else combined.combined_with(resolver, frame)
    return combined
