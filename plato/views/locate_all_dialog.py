"""Locate source images for every open embedding, in one popup.

The old flow located images for whichever embedding happened to be current,
one dialog per dataset -- so with several embeddings open (the workspace now
supports that; see plato.data.workspace) locating them all meant reopening
the same dialog once per entry with no memory of what had already been
resolved.

This mirrors the "Browse embeddings" flow instead: one list, one row per open
embedding, each independently scanned and accepted. An entry already resolved
shows its current folder and match quality; choosing a new folder for one row
never disturbs the others.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtWidgets import QDialog as _QDialog

from ..data.explorer_model import ImageResolver
from ..data.workspace import SOURCE_JOINT, EmbeddingEntry
from ..gui import themes

MAX_SUGGESTIONS = 8


def _inherited_resolver(
    entry: EmbeddingEntry, resolver_for: dict[str, ImageResolver]
) -> tuple[ImageResolver, list[str]] | None:
    """A joint entry's resolver, folded from its sources' resolvers.

    A joint entry's rows are exactly its sources' rows concatenated, so once
    every source dataset's images are found there is nothing left to locate
    by hand -- the answer is "all of the folders already found for the
    entries this was built from," which is precisely what combined_with
    already does for a dataset split across several folders.

    ``resolver_for`` is keyed by entry key and passed in rather than read off
    each source EmbeddingEntry directly, because inside the dialog the
    authoritative answer is a row's pending_resolver -- which may already be
    ahead of entry.resolver if the user just resolved that row in this same
    session, before accepting the dialog. Only sources present in the map
    with a resolver contribute; a source missing or still unresolved just
    makes the fold partial, which the returned resolver's own report
    (probed against the joint frame) will say honestly.

    Returns None for a non-joint entry, or a joint entry with nothing yet to
    inherit -- the caller then falls back to asking the user, same as today.
    Otherwise returns the folded resolver plus the names of the sources it
    came from, for the row's status text.
    """
    if entry.source != SOURCE_JOINT:
        return None
    keys = entry.info.get("source_keys") or []
    names = entry.info.get("source_names") or []
    by_key_name = dict(zip(keys, names))
    available = [k for k in keys if resolver_for.get(k) is not None]
    if not available:
        return None
    combined = resolver_for[available[0]]
    for key in available[1:]:
        combined = combined.combined_with(resolver_for[key], entry.frame)
    # Even a single source's resolver has to be re-probed against the JOINT
    # frame, not left with its report from the source's own (smaller) frame
    # -- otherwise the row would show the source's match fraction, over the
    # source's row count, mislabelled as the joint entry's own result.
    if len(available) == 1:
        combined = combined.reprobed_against(entry.frame)
    source_names = [by_key_name.get(k, k) for k in available]
    return combined, source_names


class _ScanSignals(QObject):
    done = Signal(str, object, bool)  # entry key, ImageResolver, is_addition
    failed = Signal(str, str)  # entry key, message


class _ScanTask(QRunnable):
    """Indexes one candidate folder against one entry's frame, off the GUI thread.

    ``base`` is None for a fresh "Choose folder…" scan, which replaces
    whatever the row already had. When it is set (from "Add another
    folder…"), the newly-scanned root is merged into it instead -- some
    exports split their images across more than one folder, and a second
    folder should add matches rather than throw away the first one's.
    """

    def __init__(
        self,
        key: str,
        root: Path,
        frame,
        signals: _ScanSignals,
        *,
        base: ImageResolver | None = None,
    ) -> None:
        super().__init__()
        self._key = key
        self._root = root
        self._frame = frame
        self._signals = signals
        self._base = base

    def run(self) -> None:  # pragma: no cover - worker thread
        try:
            resolver = ImageResolver.for_root(self._root, self._frame)
            if self._base is not None:
                resolver = self._base.combined_with(resolver, self._frame)
        except Exception as exc:  # noqa: BLE001 - reported in the dialog
            try:
                self._signals.failed.emit(self._key, str(exc))
            except RuntimeError:
                pass
            return
        try:
            self._signals.done.emit(self._key, resolver, self._base is not None)
        except RuntimeError:
            pass


class _EntryRow(QWidget):
    """One embedding: its name, current status, and a Choose folder button."""

    choose_requested = Signal(str)  # entry key
    add_folder_requested = Signal(str)  # entry key
    suggestion_clicked = Signal(str, str)  # entry key, path

    def __init__(self, entry: EmbeddingEntry, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = entry.key
        self.entry = entry
        # Result actually accepted for this row -- distinct from the entry's
        # resolver, which is only updated when the whole dialog is accepted,
        # so cancelling the dialog leaves every entry exactly as it was.
        self.pending_resolver: ImageResolver | None = entry.resolver
        # True only while pending_resolver came from set_inherited and has
        # never been touched by the user. A source resolving further (a
        # second arm found after the first) should freely improve this row's
        # inherit; anything the user chose (a scan, or an inherit they kept
        # by opening the dialog again later) must never be silently replaced.
        self._auto_inherited = False

        self.name_label = QLabel(f"<b>{entry.label()}</b>")
        self.name_label.setWordWrap(True)

        self.status_label = QLabel(self._initial_status())
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("hint")

        self.choose_button = QPushButton("Choose folder…")
        self.choose_button.clicked.connect(lambda: self.choose_requested.emit(self.key))

        # Only useful once a folder has been chosen -- there is nothing to
        # add a folder TO otherwise. Some exports genuinely split their
        # images across more than one folder (an arm per folder, a plate per
        # drive), which is why this merges rather than replacing.
        self.add_folder_button = QPushButton("Add another folder…")
        self.add_folder_button.setToolTip(
            "If this dataset's images are split across more than one folder, "
            "add the others here -- rows resolve as long as their image is "
            "under any of the folders added."
        )
        self.add_folder_button.clicked.connect(
            lambda: self.add_folder_requested.emit(self.key)
        )
        self.add_folder_button.setVisible(self.pending_resolver is not None)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        self.progress.hide()

        self.suggestions = QListWidget()
        self.suggestions.setMaximumHeight(96)
        self.suggestions.itemDoubleClicked.connect(
            lambda item: self.suggestion_clicked.emit(
                self.key, item.data(Qt.ItemDataRole.UserRole)
            )
        )
        self.suggestions.hide()

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.name_label, 1)
        top.addWidget(self.add_folder_button)
        top.addWidget(self.choose_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        layout.addLayout(top)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.suggestions)
        self.setLayout(layout)
        self.restyle()

    def restyle(self) -> None:
        colours = themes.current()
        border = colours.border
        self.setStyleSheet(
            f"_EntryRow {{ background: {colours.surface}; "
            f"border: 1px solid {border}; border-radius: 4px; }}"
        )

    @staticmethod
    def _roots_label(resolver: ImageResolver) -> str:
        roots = resolver.roots
        if len(roots) <= 1:
            return str(resolver.root)
        return " + ".join(str(r) for r in roots)

    def _initial_status(self) -> str:
        if self.entry.resolver is not None and self.entry.resolver.report is not None:
            report = self.entry.resolver.report
            percent = report.fraction * 100
            mark = "✓" if report.ok else "⚠"
            return (
                f"{mark} {self._roots_label(self.entry.resolver)}\n"
                f"{report.n_files:,} images, {percent:.0f}% of sampled rows resolved"
            )
        return "Not located yet."

    # -- driving the row from outside --------------------------------------

    def set_scanning(self, root: Path) -> None:
        self.status_label.setText(f"Looking in {root}…")
        self.progress.show()
        self.suggestions.hide()
        self.choose_button.setEnabled(False)
        self.add_folder_button.setEnabled(False)

    def set_result(self, resolver: ImageResolver, is_addition: bool = False) -> None:
        self.progress.hide()
        self.choose_button.setEnabled(True)
        report = resolver.report
        if report is None:
            self.status_label.setText("Nothing to check against.")
            self.add_folder_button.setEnabled(self.pending_resolver is not None)
            return

        label = self._roots_label(resolver)
        if report.ok:
            self.pending_resolver = resolver
            self._auto_inherited = False
            self.add_folder_button.setVisible(True)
            self.add_folder_button.setEnabled(True)
            percent = report.fraction * 100
            now = "now " if is_addition else ""
            if report.fraction >= 0.95:
                self.status_label.setText(
                    f"✓ {label}\n{report.n_files:,} images found, "
                    f"every sampled row {now}resolved."
                )
            else:
                self.status_label.setText(
                    f"⚠ {label}\n{report.n_files:,} images found, but "
                    f"only {percent:.0f}% of sampled rows {now}resolved. "
                    "Add another folder if the rest live elsewhere."
                )
            self._offer_neighbours(resolver.root)
            return

        self.add_folder_button.setEnabled(self.pending_resolver is not None)
        self.status_label.setText(f"✗ {label}\n{report.describe()}.")
        self._offer_neighbours(resolver.root)

    def set_inherited(self, resolver: ImageResolver, source_names: list[str]) -> None:
        """A joint row's resolver, folded from its already-resolved sources.

        Distinct wording from set_result: nothing was scanned for THIS row --
        the folders came from the entries it was built from, so the status
        says whose work is being reused rather than implying a folder was
        just picked.
        """
        self.pending_resolver = resolver
        self._auto_inherited = True
        self.add_folder_button.setVisible(True)
        self.add_folder_button.setEnabled(True)
        report = resolver.report
        names = " + ".join(source_names)
        if report is None or not report.ok:
            self.status_label.setText(
                f"Inherited from {names}, but that does not cover this "
                "combination's rows. Choose a folder to add what is missing."
            )
            return
        percent = report.fraction * 100
        mark = "✓" if percent >= 95 else "⚠"
        self.status_label.setText(
            f"{mark} Inherited from {names}\n"
            f"{report.n_files:,} images, {percent:.0f}% of sampled rows resolved."
        )

    def set_failed(self, message: str) -> None:
        self.progress.hide()
        self.choose_button.setEnabled(True)
        self.add_folder_button.setEnabled(self.pending_resolver is not None)
        self.status_label.setText(f"Could not read that folder:\n{message}")

    def _offer_neighbours(self, root: Path) -> None:
        candidates: list[Path] = []
        parent = root.parent
        if parent.is_dir() and parent != root:
            candidates.append(parent)
        for source in (root, parent):
            try:
                candidates.extend(
                    d for d in sorted(source.iterdir()) if d.is_dir() and d != root
                )
            except OSError:
                continue
            if len(candidates) >= MAX_SUGGESTIONS:
                break

        self.suggestions.clear()
        for candidate in candidates[:MAX_SUGGESTIONS]:
            label = "⬆ " if candidate == root.parent else "   "
            item = QListWidgetItem(f"{label}{candidate.name or candidate}")
            item.setData(Qt.ItemDataRole.UserRole, str(candidate))
            item.setToolTip(str(candidate))
            self.suggestions.addItem(item)
        self.suggestions.setVisible(self.suggestions.count() > 0)


class LocateAllDialog(_QDialog):
    """Locate images for every embedding open in the workspace, at once."""

    def __init__(self, entries: list[EmbeddingEntry], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Locate source data")
        self.setMinimumSize(620, 520)

        self._pool = QThreadPool.globalInstance()
        self._signals = _ScanSignals()
        self._signals.done.connect(self._on_scanned)
        self._signals.failed.connect(self._on_failed)
        self._rows: dict[str, _EntryRow] = {}
        self._entries_by_key = {entry.key: entry for entry in entries}

        intro = QLabel(
            "Choose the folder holding the raw microscopy images for each "
            "embedding below. Any arrangement works — one folder per plate, "
            "arms in subfolders, or everything together. If a dataset's "
            "images are split across more than one folder, use “Add another "
            "folder…” to add each one in turn."
        )
        intro.setWordWrap(True)
        intro.setObjectName("muted")

        rows_layout = QVBoxLayout()
        rows_layout.setSpacing(8)
        for entry in entries:
            row = _EntryRow(entry)
            row.choose_requested.connect(self._choose_for)
            row.add_folder_requested.connect(self._add_folder_for)
            row.suggestion_clicked.connect(self._scan)
            self._rows[entry.key] = row
            rows_layout.addWidget(row)
        rows_layout.addStretch(1)

        # A joint entry combining already-resolved sources needs no folder
        # dialog at all -- fill those rows in immediately, before the dialog
        # is even shown, the same way _on_scanned does for one resolved
        # mid-session below.
        for entry in entries:
            self._apply_inherited(entry)

        rows_container = QWidget()
        rows_container.setLayout(rows_layout)
        scroll = QScrollArea()
        scroll.setWidget(rows_container)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addWidget(intro)
        layout.addWidget(scroll, 1)
        layout.addWidget(self.buttons)
        self.setLayout(layout)

    def restyle(self) -> None:
        for row in self._rows.values():
            row.restyle()

    # -- scanning ------------------------------------------------------

    def _choose_for(self, key: str) -> None:
        row = self._rows.get(key)
        if row is None:
            return
        start = str(row.pending_resolver.root) if row.pending_resolver else ""
        chosen = QFileDialog.getExistingDirectory(self, "Choose the image folder", start)
        if chosen:
            self._scan(key, chosen)

    def _add_folder_for(self, key: str) -> None:
        """Scan a second (or third...) folder and merge it into what the row has.

        Distinct from "Choose folder…", which replaces -- this is for a
        dataset whose images are genuinely split across more than one
        folder, where the existing matches must be kept, not thrown away.
        """
        row = self._rows.get(key)
        if row is None or row.pending_resolver is None:
            return
        already = {str(r) for r in row.pending_resolver.roots}
        chosen = QFileDialog.getExistingDirectory(self, "Add another image folder", "")
        if not chosen:
            return
        if str(Path(chosen)) in already:
            return
        self._scan(key, chosen, base=row.pending_resolver)

    def _scan(self, key: str, root: str, *, base: ImageResolver | None = None) -> None:
        row = self._rows.get(key)
        if row is None:
            return
        path = Path(root)
        row.set_scanning(path)
        self._pool.start(_ScanTask(key, path, row.entry.frame, self._signals, base=base))

    def _on_scanned(self, key: str, resolver: ImageResolver, is_addition: bool) -> None:
        row = self._rows.get(key)
        if row is not None:
            row.set_result(resolver, is_addition)
        # A source entry just got (or improved) a resolver -- any joint row
        # built from it may now be inheritable, or inheritable with a better
        # match than before. Re-check every row rather than tracking which
        # joint entries depend on this one key: with a handful of open
        # embeddings this is a handful of dict lookups, not worth the extra
        # bookkeeping of a reverse index.
        for other_key, other_entry in self._entries_by_key.items():
            if other_key != key:
                self._apply_inherited(other_entry)

    def _apply_inherited(self, entry: EmbeddingEntry) -> None:
        """Pre-fill or refresh a joint row from its sources' CURRENT resolvers.

        Skips a row the user has actually chosen a folder for (by hand, or
        via "Add another folder...") -- that must never be silently replaced.
        A row that only has a previous auto-inherit, though, is refreshed
        freely: if the dialog is still open when a second source resolves,
        the joint row should pick up the improvement immediately rather than
        being stuck with the first, partial fold until the dialog is
        reopened.
        """
        row = self._rows.get(entry.key)
        if row is None or (row.pending_resolver is not None and not row._auto_inherited):
            return
        resolver_for = {
            other_key: other_row.pending_resolver
            for other_key, other_row in self._rows.items()
        }
        result = _inherited_resolver(entry, resolver_for)
        if result is None:
            return
        resolver, source_names = result
        row.set_inherited(resolver, source_names)

    def _on_failed(self, key: str, message: str) -> None:
        row = self._rows.get(key)
        if row is not None:
            row.set_failed(message)

    # -- result ----------------------------------------------------------

    def results(self) -> dict[str, ImageResolver]:
        """entry key -> resolver, for every row that ended up with one.

        A row whose resolver is unchanged from what the entry already had is
        still included; the caller applying results back onto the workspace
        is a cheap no-op for those and it keeps this simple rather than
        tracking "did this one actually change."
        """
        return {
            key: row.pending_resolver
            for key, row in self._rows.items()
            if row.pending_resolver is not None
        }
