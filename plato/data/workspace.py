"""Many embeddings, open at once: the object the explorer now works from.

The explorer used to hold one ``EmbeddingDataset`` in ``self.dataset`` and one
frame in ``self.frame``. That made "switch dataset" mean "throw everything away
and load again", and made questions that span embeddings -- does the DINO
layout agree with the descriptor layout, do these two screens overlap --
impossible to ask.

The model is:

    Dataset  ->  Embedding(s)  ->  Points  ->  Metadata  ->  Images

A **dataset** is a directory of exported vectors plus its metadata CSV. An
**embedding** is one set of vectors over those points: usually the export's
own, but equally a second model's output, a differently preprocessed variant,
or descriptors computed from the images. Several embeddings can therefore
share one dataset's points and metadata, and each keeps its own provenance.

A **projection** (UMAP/t-SNE coordinates) is a property of an embedding plus
its parameters, and is cached separately -- see ``plato.data.projection``.
Switching between two projections of the same embedding must never recompute
either, which is why they hang off the entry rather than off the explorer.

Nothing here is a widget. The workspace is what a view renders and what a test
can build without a QApplication.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .embeddings import EmbeddingDataset

# How an embedding's vectors were obtained. Recorded per entry so the UI can
# say where a layout came from, and so two entries over the same points remain
# distinguishable when their names collide.
SOURCE_EXPORT = "export"      # the vectors shipped with the dataset
SOURCE_COMPUTED = "computed"  # descriptors computed from the images here
SOURCE_EXTERNAL = "external"  # loaded from elsewhere and attached
# A concatenation of several OTHER entries, projected together as one fit so
# that position is comparable across them -- see plato.data.joint_projection.
# info["source_names"] carries which entries went into it, for the label.
SOURCE_JOINT = "joint"


@dataclass
class EmbeddingEntry:
    """One embedding: vectors, the points they describe, and its provenance.

    ``frame`` is the resolved metadata frame (from ``explorer_model.build_frame``),
    not the raw CSV, so every entry offers the same canonical columns whatever
    its export happened to write.
    """

    name: str
    dataset: EmbeddingDataset
    frame: pd.DataFrame
    source: str = SOURCE_EXPORT
    # Free-form provenance for display: model name, dimensions, preprocessing.
    info: dict = field(default_factory=dict)
    # Stable identity, independent of the name, which the user may duplicate.
    key: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # Cached per-entry state. Projections are keyed by ProjectionParams.key()
    # so switching UMAP <-> t-SNE, or between two parameter sets, is instant
    # after the first run of each.
    projections: dict = field(default_factory=dict)
    # Which entry in `projections` was on screen when this entry was last
    # the active one, so switching back restores exactly that view rather
    # than an arbitrary one when several parameter sets have been tried.
    last_result_key: str | None = None
    # Derived numeric columns (entropy, image statistics), computed once.
    derived: dict = field(default_factory=dict)
    # The image resolver for this entry's dataset, once found. None means
    # "not looked for yet"; a resolved entry caches it so switching back does
    # not re-walk the share.
    resolver: object | None = None
    resolver_searched: bool = False

    @property
    def vectors(self) -> np.ndarray:
        return self.dataset.vectors

    @property
    def n_points(self) -> int:
        return int(self.dataset.n_points)

    @property
    def n_dimensions(self) -> int:
        return int(self.dataset.n_dimensions)

    @property
    def directory(self) -> Path:
        return self.dataset.directory

    def fingerprint(self) -> str:
        """Cache key for this entry's vectors."""
        return self.dataset.fingerprint()

    def label(self) -> str:
        """What to show in a chooser: name plus what makes it distinct."""
        bits = [self.name]
        model = str(self.info.get("model", "") or "")
        if model:
            bits.append(model)
        if self.source == SOURCE_COMPUTED:
            bits.append("computed")
        elif self.source == SOURCE_EXTERNAL:
            bits.append("external")
        elif self.source == SOURCE_JOINT:
            bits.append("joint")
        head, *rest = bits
        return f"{head} ({', '.join(rest)})" if rest else head

    def describe(self) -> str:
        """A sentence of provenance, for the panel under the chooser."""
        parts = [f"{self.n_points:,} points × {self.n_dimensions} dimensions"]
        model = str(self.info.get("model", "") or "")
        if model:
            parts.append(f"model: {model}")
        if self.source == SOURCE_COMPUTED:
            parts.append("descriptors computed from images")
        elif self.source == SOURCE_JOINT:
            names = self.info.get("source_names", [])
            if names:
                parts.append(f"combining {', '.join(names)}")
        return " · ".join(parts)


class Workspace:
    """Every embedding open in this session, and which one is current.

    Deliberately not a QObject: the view watches it by calling, not by signal,
    so the model stays testable headless. The explorer owns exactly one of
    these and rebuilds its controls when it changes.
    """

    def __init__(self) -> None:
        self._entries: list[EmbeddingEntry] = []
        self._current: str | None = None
        # Selection is workspace state, not per-view state, so that a grid,
        # a single plot and the preview column cannot disagree. Keyed by
        # entry so switching embeddings does not silently carry a selection
        # onto points it was not made against.
        self._selection: dict[str, np.ndarray] = {}

    # -- entries -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[EmbeddingEntry]:
        return list(self._entries)

    def add(self, entry: EmbeddingEntry, *, make_current: bool = True) -> EmbeddingEntry:
        """Add an embedding. Loading the same directory twice is allowed.

        Two entries over one directory is a legitimate thing to want -- the
        export's vectors and descriptors computed from the same images, say --
        so this does not deduplicate. Identity is the entry key.
        """
        self._entries.append(entry)
        if make_current or self._current is None:
            self._current = entry.key
        return entry

    def remove(self, key: str) -> None:
        self._entries = [e for e in self._entries if e.key != key]
        self._selection.pop(key, None)
        if self._current == key:
            self._current = self._entries[0].key if self._entries else None

    def clear(self) -> None:
        self._entries.clear()
        self._selection.clear()
        self._current = None

    def get(self, key: str) -> EmbeddingEntry | None:
        for entry in self._entries:
            if entry.key == key:
                return entry
        return None

    def find_by_directory(self, directory: Path) -> list[EmbeddingEntry]:
        """Every entry loaded from ``directory``."""
        target = str(Path(directory))
        return [e for e in self._entries if str(e.directory) == target]

    # -- the current entry -------------------------------------------------

    @property
    def current_key(self) -> str | None:
        return self._current

    @property
    def current(self) -> EmbeddingEntry | None:
        return self.get(self._current) if self._current else None

    def set_current(self, key: str) -> bool:
        """Switch entries. Returns False if the key is not present."""
        if self.get(key) is None:
            return False
        self._current = key
        return True

    # -- selection ---------------------------------------------------------
    #
    # One selection model, per entry. Every view reads and writes it here
    # rather than keeping its own, which is what keeps the scatter, the grid,
    # the preview column and the cluster panel consistent.

    def selection(self, key: str | None = None) -> np.ndarray:
        """Selected row indices for an entry (the current one by default)."""
        key = key or self._current
        if key is None:
            return np.empty(0, dtype=np.int64)
        return self._selection.get(key, np.empty(0, dtype=np.int64))

    def set_selection(self, rows, key: str | None = None) -> np.ndarray:
        """Replace an entry's selection. Returns what was stored."""
        key = key or self._current
        if key is None:
            return np.empty(0, dtype=np.int64)
        rows = np.unique(np.asarray(rows, dtype=np.int64).ravel())
        entry = self.get(key)
        if entry is not None and len(rows):
            # A selection can outlive the frame it was made against, e.g.
            # after an entry is reloaded with fewer rows.
            rows = rows[(rows >= 0) & (rows < entry.n_points)]
        self._selection[key] = rows
        return rows

    def toggle_selection(self, row: int, key: str | None = None) -> np.ndarray:
        """Add a row, or remove it if already selected."""
        key = key or self._current
        current = self.selection(key)
        row = int(row)
        if row in current:
            rows = current[current != row]
        else:
            rows = np.append(current, row)
        return self.set_selection(rows, key)

    def clear_selection(self, key: str | None = None) -> None:
        self.set_selection(np.empty(0, dtype=np.int64), key)

    # -- convenience for views ---------------------------------------------

    @property
    def frame(self) -> pd.DataFrame | None:
        entry = self.current
        return entry.frame if entry is not None else None

    def combined_frame(self, keys: list[str] | None = None) -> pd.DataFrame:
        """One frame over several entries, with an ``embedding`` column.

        Used by grouping modes that compare embeddings against each other.
        Row indices are NOT preserved -- the result is for aggregate
        questions ("how many points per dataset"), not for indexing back into
        a single entry, which is what ``EmbeddingEntry.frame`` is for.
        """
        chosen = [e for e in self._entries if keys is None or e.key in keys]
        if not chosen:
            return pd.DataFrame()
        parts = []
        for entry in chosen:
            part = entry.frame.copy()
            part[EMBEDDING_COLUMN] = entry.label()
            part[DATASET_COLUMN] = entry.dataset.name
            parts.append(part)
        return pd.concat(parts, ignore_index=True)


# Columns the workspace adds when several entries are viewed together. Named
# here rather than in explorer_model because they describe the *session*, not
# any one export's metadata.
EMBEDDING_COLUMN = "embedding"
DATASET_COLUMN = "dataset"
