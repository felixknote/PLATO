"""The Embedding Explorer: UMAP / t-SNE of precomputed features, coloured by
metadata, with the original micrograph one hover away.

Layout mirrors the browser panel deliberately -- controls left, plot centre,
detail right -- so moving between the two tabs does not mean relearning where
things are.

The expensive part (the projection) runs on a worker thread and is cached on
disk by ``plato.data.projection``, so switching UMAP <-> t-SNE after the first
run is instant. Colouring and filtering never recompute a projection: they
re-draw the same coordinates, which is what keeps exploration fluid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..data.annotations import (
    UNANNOTATED,
    AnnotationTable,
    find_default_moa,
    find_default_pathway,
    load_or_empty,
)
from ..data.embeddings import (
    EmbeddingDataset,
    EmbeddingError,
    discover_datasets,
    load_dataset,
)
from ..data.explorer_model import (
    CONCENTRATION,
    DRUG,
    GENE,
    IMAGE_NAME,
    MOA,
    PATHWAY,
    PLATE,
    ROLE,
    WELL,
    ImageResolver,
    build_frame,
    colour_fields,
    distinct_values,
    field_label,
    filter_fields,
)
from ..data.index.db import ImageRow
from ..data.projection import METHODS, TSNE, UMAP, ProjectionCache, ProjectionParams, project
from ..gui.theme import BORDER, SURFACE, TEXT, TEXT_FAINT, TEXT_MUTED
from ..gui.viewer import ImageWindow
from .palette import UNKNOWN_COLOUR, colours_for, is_unknown, sort_values
from .preview import PreviewPane
from .scatter import EmbeddingScatter

# Legend gets unreadable long before this; past it, colour still encodes the
# grouping but the legend is replaced by a count.
MAX_LEGEND_ENTRIES = 24

# Filter lists longer than this are searchable rather than fully listed.
MAX_FILTER_VALUES = 120


class _WorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)


class _ProjectionTask(QRunnable):
    """Runs one projection off the GUI thread."""

    def __init__(self, vectors, params, fingerprint, cache, signals) -> None:
        super().__init__()
        self._vectors = vectors
        self._params = params
        self._fingerprint = fingerprint
        self._cache = cache
        self._signals = signals

    def run(self) -> None:  # pragma: no cover - worker thread
        try:
            result = project(
                self._vectors,
                self._params,
                fingerprint=self._fingerprint,
                cache=self._cache,
                progress=self._signals.progress.emit,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            try:
                self._signals.failed.emit(str(exc))
            except RuntimeError:
                pass
            return
        try:
            self._signals.finished.emit(result)
        except RuntimeError:
            pass


class FilterList(QGroupBox):
    """Multi-select list for one column. Nothing selected = no constraint.

    Same contract as the browser's FilterBox, but driven by a pandas frame
    rather than SQL, so the two panels behave identically from the user's side.
    """

    changed = Signal()

    def __init__(self, column: str, label: str, values: list[str]) -> None:
        super().__init__(label)
        self.column = column
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.setMaximumHeight(120)
        for value in values:
            item = QListWidgetItem(value or "(blank)")
            item.setData(Qt.ItemDataRole.UserRole, value)
            self.list.addItem(item)
        self.list.itemSelectionChanged.connect(self.changed.emit)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 4, 6, 4)
        layout.addWidget(self.list)
        self.setLayout(layout)

    def selected(self) -> list[str]:
        return [
            item.data(Qt.ItemDataRole.UserRole) for item in self.list.selectedItems()
        ]

    def clear(self) -> None:
        self.list.clearSelection()


class EmbeddingExplorer(QWidget):
    """Controls + scatter + preview, over one embedding dataset."""

    status = Signal(str)

    def __init__(self, session, work_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self.work_dir = Path(work_dir)
        self.cache = ProjectionCache(self.work_dir / "projections")

        self.dataset: EmbeddingDataset | None = None
        self.frame = None
        self.resolver: ImageResolver | None = None
        self.result = None
        self._visible_rows = np.empty(0, dtype=np.int64)
        self._windows: list[ImageWindow] = []
        self._busy = False

        # Redraws are coalesced. Dragging a slider emits a value per pixel of
        # travel, and a full regroup + repaint of 32k points per tick turns a
        # smooth drag into a slideshow. One redraw after the control settles
        # looks identical and costs a fraction as much.
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(60)
        self._redraw_timer.timeout.connect(self._redraw)
        self._pending_reset = False

        search_roots = [Path(__file__).resolve().parents[2]]
        self.moa_table = load_or_empty(find_default_moa(search_roots))
        self.pathway_table = load_or_empty(find_default_pathway(search_roots))

        self._build_ui()

    # -- construction -----------------------------------------------------

    def _build_ui(self) -> None:
        # --- left: data + projection controls
        self.dataset_box = QComboBox()
        self.dataset_box.setMinimumWidth(180)
        self.dataset_box.currentIndexChanged.connect(self._on_dataset_changed)

        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_for_dataset)

        source_row = QHBoxLayout()
        source_row.setContentsMargins(0, 0, 0, 0)
        source_row.addWidget(self.dataset_box, 1)
        source_row.addWidget(browse)
        source_widget = QWidget()
        source_widget.setLayout(source_row)

        self.method_box = QComboBox()
        self.method_box.addItems(METHODS)

        self.neighbours_box = QComboBox()
        self.neighbours_box.addItems(["15", "50", "100", "200", "500"])
        self.neighbours_box.setCurrentText("500")

        self.perplexity_box = QComboBox()
        self.perplexity_box.addItems(["10", "30", "50", "100"])
        self.perplexity_box.setCurrentText("30")

        self.max_points_box = QComboBox()
        self.max_points_box.addItems(["all", "5000", "10000", "20000"])

        self.run_button = QPushButton("Compute projection")
        self.run_button.setDefault(True)
        self.run_button.clicked.connect(self.compute)

        projection_form = QFormLayout()
        projection_form.setContentsMargins(6, 4, 6, 4)
        projection_form.addRow("Dataset", source_widget)
        projection_form.addRow("Method", self.method_box)
        projection_form.addRow("Neighbours", self.neighbours_box)
        projection_form.addRow("Perplexity", self.perplexity_box)
        projection_form.addRow("Points", self.max_points_box)
        projection_form.addRow(self.run_button)
        projection_group = QGroupBox("Projection")
        projection_group.setLayout(projection_form)
        self.method_box.currentTextChanged.connect(self._sync_method_options)

        # --- display controls
        self.colour_box = QComboBox()
        self.colour_box.currentIndexChanged.connect(self.schedule_redraw)

        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(2, 16)
        self.size_slider.setValue(6)
        self.size_slider.valueChanged.connect(self.schedule_redraw)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(85)
        self.opacity_slider.valueChanged.connect(self.schedule_redraw)

        self.legend_box = QCheckBox("Show legend")
        self.legend_box.setChecked(True)
        self.legend_box.toggled.connect(self._redraw)

        self.dim_others_box = QCheckBox("Grey out unannotated")
        self.dim_others_box.setChecked(True)
        self.dim_others_box.setToolTip(
            "Draw points with no value for the chosen field in grey, so an "
            "unannotated condition never reads as a category of its own."
        )
        self.dim_others_box.toggled.connect(self._redraw)

        display_form = QFormLayout()
        display_form.setContentsMargins(6, 4, 6, 4)
        display_form.addRow("Colour by", self.colour_box)
        display_form.addRow("Point size", self.size_slider)
        display_form.addRow("Opacity", self.opacity_slider)
        display_form.addRow(self.legend_box)
        display_form.addRow(self.dim_others_box)
        display_group = QGroupBox("Display")
        display_group.setLayout(display_form)

        # --- filters
        self.filter_layout = QVBoxLayout()
        self.filter_layout.setContentsMargins(0, 0, 0, 0)
        self.filter_layout.setSpacing(8)
        self.filters: list[FilterList] = []

        clear_filters = QPushButton("Clear filters")
        clear_filters.clicked.connect(self.clear_filters)

        left = QVBoxLayout()
        left.setContentsMargins(10, 10, 10, 10)
        left.setSpacing(10)
        left.addWidget(projection_group)
        left.addWidget(display_group)
        left.addLayout(self.filter_layout)
        left.addStretch(1)
        left.addWidget(clear_filters)

        left_container = QWidget()
        left_container.setLayout(left)
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_container)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(240)
        left_scroll.setMaximumWidth(300)

        # --- centre: the plot
        self.scatter = EmbeddingScatter()
        self.scatter.point_hovered.connect(self._on_hover)
        self.scatter.point_clicked.connect(self._on_click)

        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setStyleSheet(f"color: {TEXT_MUTED}; padding: 18px;")

        self.count_label = QLabel("—")
        self.count_label.setStyleSheet(f"color: {TEXT_MUTED};")

        reset_view = QPushButton("Reset view")
        reset_view.clicked.connect(self.scatter.reset_view)
        export_png = QPushButton("Export PNG…")
        export_png.clicked.connect(lambda: self.export("png"))
        export_svg = QPushButton("Export SVG…")
        export_svg.clicked.connect(lambda: self.export("svg"))

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(10, 6, 10, 6)
        toolbar.addWidget(self.count_label, 1)
        toolbar.addWidget(reset_view)
        toolbar.addWidget(export_png)
        toolbar.addWidget(export_svg)
        toolbar_widget = QWidget()
        toolbar_widget.setLayout(toolbar)

        centre = QVBoxLayout()
        centre.setContentsMargins(0, 0, 0, 0)
        centre.setSpacing(0)
        centre.addWidget(toolbar_widget)
        centre.addWidget(self.message)
        centre.addWidget(self.scatter, 1)
        centre_widget = QWidget()
        centre_widget.setLayout(centre)

        # --- right: hover preview
        heading = QLabel("Point")
        heading.setObjectName("panelHeading")
        self.preview = PreviewPane()
        hint = QLabel("Hover to preview · click to open full resolution")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 11px;")

        right = QVBoxLayout()
        right.setContentsMargins(12, 10, 12, 10)
        right.setSpacing(8)
        right.addWidget(heading)
        right.addWidget(self.preview, 1)
        right.addWidget(hint)
        right_container = QWidget()
        right_container.setLayout(right)
        right_container.setMinimumWidth(280)
        right_container.setMaximumWidth(340)

        splitter = QSplitter()
        splitter.addWidget(left_scroll)
        splitter.addWidget(centre_widget)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([260, 1100, 300])
        splitter.setChildrenCollapsible(False)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self.setLayout(layout)

        self._sync_method_options()
        self._set_message("Choose an embedding dataset to begin.")

    def _sync_method_options(self) -> None:
        is_umap = self.method_box.currentText() == UMAP
        self.neighbours_box.setEnabled(is_umap)
        self.perplexity_box.setEnabled(not is_umap)

    def _set_message(self, text: str) -> None:
        self.message.setText(text)
        self.message.setVisible(bool(text))
        self.scatter.setVisible(not text)

    # -- dataset loading --------------------------------------------------

    def set_dataset_root(self, root: Path) -> None:
        """Populate the dataset dropdown from a directory of exports."""
        root = Path(root)
        directories = discover_datasets(root)
        self.dataset_box.blockSignals(True)
        self.dataset_box.clear()
        for directory in directories:
            self.dataset_box.addItem(directory.name, str(directory))
        self.dataset_box.blockSignals(False)
        if directories:
            self._on_dataset_changed()
        else:
            self._set_message(f"No embedding datasets found under\n{root}")

    def _browse_for_dataset(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose an embeddings folder")
        if not chosen:
            return
        path = Path(chosen)
        # Accept either a dataset directory or a parent holding several.
        directories = discover_datasets(path)
        if not directories:
            QMessageBox.information(
                self,
                "Embeddings",
                f"No embedding export found in\n{path}\n\n"
                "Expected a folder containing features_metadata.csv.",
            )
            return
        self.set_dataset_root(path)

    def _on_dataset_changed(self) -> None:
        directory = self.dataset_box.currentData()
        if not directory:
            return
        try:
            self.dataset = load_dataset(Path(directory))
        except EmbeddingError as exc:
            self.dataset = None
            self.frame = None
            self._clear_filters()
            self._set_message(str(exc))
            self.status.emit("embeddings unavailable")
            return
        except (OSError, ValueError) as exc:
            self.dataset = None
            self._set_message(f"Could not read this dataset:\n{exc}")
            return

        image_root = self._image_root_guess()
        self.frame, self.resolver = build_frame(
            self.dataset,
            moa_table=self.moa_table,
            pathway_table=self.pathway_table,
            image_root=image_root,
        )
        self.result = None
        self._rebuild_colour_options()
        self._rebuild_filters()
        run = self.dataset.run_info.get("model", "")
        self._set_message(
            f"{self.dataset.name}: {self.dataset.n_points:,} points × "
            f"{self.dataset.n_dimensions} dimensions"
            f"{' · ' + str(run) if run else ''}\n\n"
            "Choose a method and press Compute projection."
        )
        self.status.emit(
            f"{self.dataset.name}: {self.dataset.n_points:,} embeddings"
            + ("" if self.resolver else " · original images not found")
        )

    def _image_root_guess(self) -> Path | None:
        """Where the original micrographs live.

        Prefers a loaded plate's image directory (its parent, since the export
        is organised one folder per plate), because that is a path the user has
        already confirmed by loading it. Falls back to nothing rather than
        guessing at the filesystem.
        """
        candidates: list[Path] = []
        for plate in getattr(self.session, "plates", []):
            directory = Path(plate.cfg.images.dir)
            candidates.extend((directory, directory.parent))
        for candidate in candidates:
            if candidate.is_dir():
                probe = ImageResolver.detect(candidate, self.frame if self.frame is not None else None) if self.frame is not None else None
                if probe is not None:
                    return candidate
        return candidates[0] if candidates else None

    # -- controls ---------------------------------------------------------

    def _rebuild_colour_options(self) -> None:
        current = self.colour_box.currentData()
        self.colour_box.blockSignals(True)
        self.colour_box.clear()
        for column in colour_fields(self.frame):
            self.colour_box.addItem(field_label(column), column)
        # Prefer a field that says something biological on first open.
        if current is not None:
            index = self.colour_box.findData(current)
            if index >= 0:
                self.colour_box.setCurrentIndex(index)
        else:
            for preferred in (MOA, GENE, DRUG, ROLE):
                index = self.colour_box.findData(preferred)
                if index >= 0:
                    self.colour_box.setCurrentIndex(index)
                    break
        self.colour_box.blockSignals(False)

    def _clear_filters(self) -> None:
        self.filters.clear()
        while self.filter_layout.count():
            item = self.filter_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _rebuild_filters(self) -> None:
        self._clear_filters()
        if self.frame is None:
            return
        for column in filter_fields(self.frame, max_values=MAX_FILTER_VALUES):
            values = distinct_values(self.frame, column)
            box = FilterList(column, field_label(column), values)
            box.changed.connect(self.schedule_redraw)
            self.filters.append(box)
            self.filter_layout.addWidget(box)

    def clear_filters(self) -> None:
        for box in self.filters:
            box.clear()
        self._redraw()

    def current_filters(self) -> dict[str, list[str]]:
        return {box.column: box.selected() for box in self.filters if box.selected()}

    # -- projection -------------------------------------------------------

    def _params(self) -> ProjectionParams:
        text = self.max_points_box.currentText()
        return ProjectionParams(
            method=self.method_box.currentText(),
            n_neighbors=int(self.neighbours_box.currentText()),
            perplexity=float(self.perplexity_box.currentText()),
            max_points=None if text == "all" else int(text),
        )

    def compute(self) -> None:
        if self.dataset is None:
            QMessageBox.information(
                self, "Embeddings", "Load an embedding dataset first."
            )
            return
        if self._busy:
            return
        params = self._params()

        cached = self.cache.load(self.dataset.fingerprint(), params)
        if cached is not None:
            self._on_projection(cached)
            return

        self._busy = True
        self.run_button.setEnabled(False)
        self._set_message(
            f"Computing {params.method} over {self.dataset.n_points:,} points…\n"
            "This runs once per parameter set and is then cached."
        )
        signals = _WorkerSignals()
        signals.finished.connect(self._on_projection)
        signals.failed.connect(self._on_projection_failed)
        signals.progress.connect(lambda text: self.status.emit(text))
        self._signals = signals  # keep alive for the task's lifetime
        QThreadPool.globalInstance().start(
            _ProjectionTask(
                self.dataset.vectors,
                params,
                self.dataset.fingerprint(),
                self.cache,
                signals,
            )
        )

    def _on_projection(self, result) -> None:
        self._busy = False
        self.run_button.setEnabled(True)
        self.result = result
        self._set_message("")
        origin = "cached" if result.from_cache else f"{result.seconds:.1f}s"
        self.status.emit(
            f"{result.params.method}: {len(result.coords):,} points ({origin})"
        )
        self._redraw(reset_view=True)

    def _on_projection_failed(self, message: str) -> None:
        self._busy = False
        self.run_button.setEnabled(True)
        self._set_message(f"Projection failed:\n{message}")
        self.status.emit("projection failed")

    # -- drawing ----------------------------------------------------------

    def schedule_redraw(self, *_args) -> None:
        """Coalesce rapid control changes into one redraw."""
        self._redraw_timer.start()

    def _redraw(self, *_args, reset_view: bool = False) -> None:
        # A queued redraw may still be pending when an immediate one runs
        # (e.g. a fresh projection); dropping it avoids drawing twice.
        self._redraw_timer.stop()
        if self.result is None or self.frame is None:
            return

        rows = self.result.row_indices
        subset = self.frame.iloc[rows]

        mask = np.ones(len(rows), dtype=bool)
        for column, values in self.current_filters().items():
            if column in subset.columns:
                mask &= subset[column].isin(values).to_numpy()

        coords = self.result.coords[mask]
        visible = rows[mask]
        self._visible_rows = visible
        if len(coords) == 0:
            self.scatter.set_points(
                np.empty((0, 2), dtype=np.float32),
                np.empty(0, dtype=np.int64),
                [],
                legend_entries=None,
            )
            self.count_label.setText("<b>0</b> of %d points" % len(rows))
            return

        column = self.colour_box.currentData()
        values = subset[column].astype(str).to_numpy()[mask] if column else np.array([""] * len(coords))
        ordered = sort_values(list(dict.fromkeys(values.tolist())))
        mapping, continuous = colours_for(ordered)
        if self.dim_others_box.isChecked():
            for value in ordered:
                if is_unknown(value):
                    mapping[value] = UNKNOWN_COLOUR

        # Group positions by colour in one pass. Several values can share a
        # colour (the palette wraps, and every unknown maps to grey), so
        # entries accumulate rather than overwrite.
        #
        # Unknown values are emitted first so that real categories are drawn
        # over the grey bulk rather than buried under it.
        collected: dict[str, list[np.ndarray]] = {}
        for value in sorted(ordered, key=is_unknown, reverse=True):
            positions = np.flatnonzero(values == value)
            if len(positions):
                collected.setdefault(mapping[value], []).append(positions)
        groups = {
            colour: (parts[0] if len(parts) == 1 else np.concatenate(parts))
            for colour, parts in collected.items()
        }

        legend_entries = None
        if self.legend_box.isChecked() and len(ordered) <= MAX_LEGEND_ENTRIES:
            legend_entries = [(v or "(blank)", mapping[v]) for v in ordered]

        self.scatter.set_points(
            coords,
            visible,
            [],
            groups=groups,
            point_size=self.size_slider.value(),
            opacity=self.opacity_slider.value() / 100.0,
            legend_entries=legend_entries,
            reset_view=reset_view,
        )

        label = field_label(column) if column else ""
        extra = "" if len(ordered) <= MAX_LEGEND_ENTRIES else f" · {len(ordered)} values"
        self.count_label.setText(
            f"<b>{len(coords):,}</b> of {len(rows):,} points · {label}{extra}"
        )

    # -- interaction ------------------------------------------------------

    def _describe(self, row_index: int) -> str:
        row = self.frame.iloc[row_index]
        parts: list[str] = []
        condition = row.get("condition", "")
        if condition:
            parts.append(f"<b>{condition}</b>")
        location = " ".join(x for x in (row.get(PLATE, ""), row.get(WELL, "")) if x)
        if location:
            parts.append(location)
        parts.append("")
        for column in (GENE, "guide", DRUG, CONCENTRATION, MOA, PATHWAY, ROLE):
            value = str(row.get(column, "") or "")
            if not value or (column in (MOA, PATHWAY) and value == UNANNOTATED):
                continue
            parts.append(f"{field_label(column)}: <b>{value}</b>")
        name = str(row.get(IMAGE_NAME, "") or "")
        if name:
            parts.extend(["", f"<i style='font-size:10px'>{name}</i>"])
        return "<br>".join(parts)

    def _path_for(self, row_index: int) -> Path | None:
        if self.resolver is None or self.frame is None:
            return None
        return self.resolver.path_for(self.frame.iloc[row_index])

    def _on_hover(self, row_index: int) -> None:
        if row_index < 0 or self.frame is None:
            self.preview.clear()
            return
        self.preview.show_row(
            row_index, self._describe(row_index), self._path_for(row_index)
        )

    def _on_click(self, row_index: int) -> None:
        if self.frame is None:
            return
        path = self._path_for(row_index)
        if path is None:
            self.status.emit("no original image found for this point")
            return

        # Build ImageRows for everything currently visible, so left/right in
        # the viewer steps through the filtered selection -- the same contract
        # the browser's viewer has.
        rows: list[ImageRow] = []
        start = 0
        for position, index in enumerate(self._visible_rows):
            candidate = self._path_for(int(index))
            if candidate is None:
                continue
            if int(index) == row_index:
                start = len(rows)
            record = self.frame.iloc[int(index)]
            rows.append(
                ImageRow(
                    image_id=str(record.get(IMAGE_NAME, "")),
                    path=str(candidate),
                    plate=str(record.get(PLATE, "")),
                    well=str(record.get(WELL, "")),
                    field=None,
                    channel=None,
                    metadata={
                        field_label(c): record.get(c, "")
                        for c in (GENE, "guide", DRUG, CONCENTRATION, MOA, PATHWAY, ROLE)
                        if str(record.get(c, "") or "")
                    },
                    rating=None,
                    flagged=False,
                )
            )
        if not rows:
            return
        window = ImageWindow(self.session, rows, start, {}, parent=self)
        window.setWindowFlag(Qt.WindowType.Window, True)
        window.show()
        self._windows.append(window)

    # -- export -----------------------------------------------------------

    def export(self, fmt: str) -> None:
        if self.result is None:
            QMessageBox.information(self, "Export", "Compute a projection first.")
            return
        method = self.result.params.method.replace("-", "")
        column = self.colour_box.currentData() or "plot"
        suggested = f"{method}_by_{column}.{fmt}"
        filters = "PNG image (*.png)" if fmt == "png" else "SVG vector (*.svg)"
        target, _ = QFileDialog.getSaveFileName(self, f"Export {fmt.upper()}", suggested, filters)
        if not target:
            return
        if not target.lower().endswith(f".{fmt}"):
            target = f"{target}.{fmt}"
        try:
            self.scatter.export(target)
        except Exception as exc:  # noqa: BLE001 - report rather than crash
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        self.status.emit(f"exported {target}")
