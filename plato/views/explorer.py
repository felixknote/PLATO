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

import os
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
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
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
from ..data.locations import (
    DATA_LIBRARY,
    IMAGE_ROOT,
    get_root,
    guess_data_library,
    set_root,
)
from ..data.projection import (
    METHODS,
    TSNE,
    UMAP,
    ProjectionCache,
    ProjectionParams,
    project,
    suggest,
)
from ..gui.theme import BORDER, SURFACE, TEXT, TEXT_FAINT, TEXT_MUTED
from ..gui.viewer import ImageWindow
from .palette import UNKNOWN_COLOUR, colours_for, is_unknown, sort_values
from .gallery import SelectionGallery
from .preview import PreviewPane
from .scatter import EmbeddingScatter

# Legend gets unreadable long before this; past it, colour still encodes the
# grouping but the legend is replaced by a count.
MAX_LEGEND_ENTRIES = 24

COMPUTED_FROM_PLATES = "<loaded-plates>"

# Subsample choices, as a share of the dataset.
FULL_SAMPLE = "100% (all)"
SUBSAMPLE_CHOICES = (FULL_SAMPLE, "50%", "25%", "10%", "5%", "1%")


def _percent_of(text: str) -> float | None:
    """The fraction a subsample choice stands for, or None for all of it."""
    cleaned = text.strip().rstrip("%").split(" ")[0]
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return None if value >= 100 else max(0.01, value) / 100.0


def _cap_for(text: str, n_points: int) -> int | None:
    """``max_points`` for a choice, or None to project everything."""
    fraction = _percent_of(text)
    if fraction is None:
        return None
    # At least a handful of points, or the projection has nothing to lay out
    # -- but never more than exist, which matters for a tiny dataset where the
    # floor would otherwise exceed the whole thing.
    return min(n_points, max(10, int(round(n_points * fraction))))


def _choice_for(max_points: int | None, n_points: int) -> str:
    """The nearest offered choice to a stored ``max_points``."""
    if max_points is None or n_points <= 0 or max_points >= n_points:
        return FULL_SAMPLE
    wanted = max_points / n_points * 100
    offered = [c for c in SUBSAMPLE_CHOICES if _percent_of(c) is not None]
    return min(offered, key=lambda c: abs((_percent_of(c) or 1) * 100 - wanted))

# Filter lists longer than this are searchable rather than fully listed.
MAX_FILTER_VALUES = 120

# Where the original micrographs live is configuration, not a constant: every
# installation keeps them somewhere different, and a baked-in drive letter
# works on exactly one machine. See plato.data.locations.


class _WorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)
    # (fraction 0..1, phase) from the backend's own reporting.
    advanced = Signal(float, str)


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
                on_progress=self._signals.advanced.emit,
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


class _WarmUpTask(QRunnable):
    """Compiles the projection backend's kernels off the GUI thread."""

    def run(self) -> None:  # pragma: no cover - worker thread
        from ..data.projection import warm_up

        warm_up()


class _ResolveSignals(QObject):
    found = Signal(str, object)  # dataset key, ImageResolver | None
    ambiguous = Signal(str, list)


class _ResolveTask(QRunnable):
    """Hunts for a dataset's image root without blocking the window.

    Probing one candidate root costs a directory stat per sampled row, which
    on a network share is ~0.3 s. With ~30 screens in the data library that is
    9-12 s -- and it used to run on the GUI thread while the explorer opened,
    so the window simply froze. The plot does not need the answer: only hover
    previews do, and those come later.
    """

    def __init__(self, key, frame, candidates, signals) -> None:
        super().__init__()
        self._key = key
        self._frame = frame
        self._candidates = candidates
        self._signals = signals

    def run(self) -> None:  # pragma: no cover - worker thread
        matches = []
        try:
            for candidate in self._candidates:
                resolver = ImageResolver.detect(candidate, self._frame)
                if resolver is not None:
                    matches.append(resolver)
                    if len(matches) >= 2:
                        break
        except Exception:  # noqa: BLE001 - a bad share must not kill the task
            matches = []
        try:
            self._signals.ambiguous.emit(self._key, [str(r.root) for r in matches[1:]])
            self._signals.found.emit(self._key, matches[0] if matches else None)
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
        # dataset directory -> resolver (or None). Scanning the data
        # library is a directory walk; do it once per dataset.
        self._resolver_cache: dict[str, tuple[ImageResolver | None, list[str]]] = {}
        # Other roots that matched this dataset equally well, if any.
        self.ambiguous_roots: list[str] = []
        # True while the background root hunt is running.
        self._resolving = False
        # Parameters suggested for the loaded dataset; the controls start here.
        self._suggested: ProjectionParams | None = None
        self._windows: list[ImageWindow] = []
        self._busy = False
        self._cancelled = False

        # Redraws are coalesced. Dragging a slider emits a value per pixel of
        # travel, and a full regroup + repaint of 32k points per tick turns a
        # smooth drag into a slideshow. One redraw after the control settles
        # looks identical and costs a fraction as much.
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(60)
        self._redraw_timer.timeout.connect(self._redraw)
        self._pending_reset = False

        self._resolve_signals = _ResolveSignals()
        self._resolve_signals.found.connect(self._on_resolver_found)
        self._resolve_signals.ambiguous.connect(self._on_resolver_ambiguous)

        search_roots = [Path(__file__).resolve().parents[2]]
        self.moa_table = load_or_empty(find_default_moa(search_roots))
        self.pathway_table = load_or_empty(find_default_pathway(search_roots))

        self._build_ui()

        # Compile UMAP's numba kernels now, on a worker, so the first real
        # projection is not also paying for the compiler.
        QThreadPool.globalInstance().start(_WarmUpTask())

    # -- construction -----------------------------------------------------

    def _build_ui(self) -> None:
        # --- left: data + projection controls
        self.dataset_box = QComboBox()
        # A long dataset name must not widen the sidebar; elide it instead.
        self.dataset_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.dataset_box.setMinimumContentsLength(12)
        self.dataset_box.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self.dataset_box.currentIndexChanged.connect(self._on_dataset_changed)

        browse = QPushButton("Browse…")
        browse.setToolTip("Choose a folder of embedding exports.")
        browse.clicked.connect(self._browse_for_dataset)

        # Stacked, not side by side: the two together are the widest row in
        # the panel and would force the whole sidebar wider than the plot can
        # spare.
        source_row = QVBoxLayout()
        source_row.setContentsMargins(0, 0, 0, 0)
        source_row.setSpacing(4)
        source_row.addWidget(self.dataset_box)
        source_row.addWidget(browse)
        source_widget = QWidget()
        source_widget.setLayout(source_row)

        # Every export came from a different screen, so the images are found
        # by search -- and a search can miss (a screen on another drive, a
        # renamed folder). This is the manual answer, and the label says
        # whether the automatic one worked.
        self.source_button = QPushButton("Locate…")
        self.source_button.setToolTip(
            "Choose the folder holding this dataset's original images.\n"
            "PLATO searches for it automatically; use this when it cannot "
            "find them, or to point at a different copy."
        )
        self.source_button.clicked.connect(self._browse_for_source_data)

        self.source_label = QLabel("—")
        self.source_label.setWordWrap(True)
        self.source_label.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 10px;")

        source_data_row = QVBoxLayout()
        source_data_row.setContentsMargins(0, 0, 0, 0)
        source_data_row.setSpacing(4)
        source_data_row.addWidget(self.source_button)
        source_data_row.addWidget(self.source_label)
        source_data_widget = QWidget()
        source_data_widget.setLayout(source_data_row)

        self.method_box = QComboBox()
        self.method_box.addItems(METHODS)

        # Editable, because the suggested value scales with the dataset and
        # will usually not be one of the presets.
        self.neighbours_box = QComboBox()
        self.neighbours_box.setEditable(True)
        self.neighbours_box.addItems(["15", "50", "100", "200", "500"])
        self.neighbours_box.setToolTip(
            "How much of the neighbourhood UMAP preserves.\n"
            "Low values emphasise local structure, high values global shape.\n"
            "Set from the dataset size when one is loaded."
        )

        self.perplexity_box = QComboBox()
        self.perplexity_box.setEditable(True)
        self.perplexity_box.addItems(["10", "30", "50", "100"])
        self.perplexity_box.setToolTip(
            "t-SNE's effective neighbourhood size.\n"
            "Must stay below a third of the point count, which is enforced."
        )

        # Subsampling. A UMAP of 30k x 1024 is minutes; 5k is seconds, and for
        # judging whether two conditions overlap a random subsample answers the
        # same question far faster. It is a property of the projection, so it
        # is part of the cache key -- a 5k run and a full run are two separate
        # cached results, not one overwriting the other.
        # A share of the dataset, not a count. Datasets here run from a few
        # hundred computed descriptors to 36k embeddings, and "10000" means
        # "all of it" for one and "a third" for another; a percentage means
        # the same thing to both.
        self.max_points_box = QComboBox()
        self.max_points_box.addItems(SUBSAMPLE_CHOICES)
        self.max_points_box.setCurrentText(FULL_SAMPLE)
        self.max_points_box.setToolTip(
            "Project a share of the points instead of all of them.\n"
            "Sampling is seeded, so the same setting always draws the same "
            "points, and each share is cached separately."
        )
        self.max_points_box.currentTextChanged.connect(self._update_points_hint)

        self.points_hint = QLabel("")
        self.points_hint.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 10px;")

        self.deterministic_box = QCheckBox("Reproducible (slower)")
        self.deterministic_box.setToolTip(
            "UMAP cannot use more than one core once its seed is fixed, so "
            "this trades ~10x speed for an identical layout every run.\n"
            "The cluster structure is the same either way — only the "
            "orientation of the plot changes — so leave it off while "
            "exploring and turn it on for a figure you need to regenerate "
            "exactly. t-SNE is reproducible and threaded regardless."
        )

        self.run_button = QPushButton("Compute projection")
        self.run_button.setDefault(True)
        self.run_button.clicked.connect(self.compute)

        # A determinate bar: UMAP and t-SNE both report their phases and their
        # optimiser's iterations, so this shows real progress rather than a
        # barber's pole. Hidden until a run starts.
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.hide()

        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        self.progress_label.setStyleSheet(f"color: {TEXT_FAINT}; font-size: 10px;")
        self.progress_label.hide()

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel_projection)
        self.cancel_button.hide()

        projection_form = QFormLayout()
        projection_form.setContentsMargins(6, 4, 6, 4)
        projection_form.addRow("Dataset", source_widget)
        projection_form.addRow("Source data", source_data_widget)
        projection_form.addRow("Method", self.method_box)
        projection_form.addRow("Neighbours", self.neighbours_box)
        projection_form.addRow("Perplexity", self.perplexity_box)
        points_column = QVBoxLayout()
        points_column.setContentsMargins(0, 0, 0, 0)
        points_column.setSpacing(4)
        points_column.addWidget(self.max_points_box)
        points_column.addWidget(self.points_hint)
        points_widget = QWidget()
        points_widget.setLayout(points_column)
        projection_form.addRow("Subsample", points_widget)
        projection_form.addRow(self.deterministic_box)
        projection_form.addRow(self.run_button)
        projection_form.addRow(self.progress_bar)
        projection_form.addRow(self.progress_label)
        projection_form.addRow(self.cancel_button)
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

        self.density_box = QCheckBox("Density")
        self.density_box.setToolTip(
            "Draw where points are concentrated instead of every mark.\n"
            "At tens of thousands of points the marks overlap into a solid "
            "blob; density shows how many, not merely 'at least one'."
        )
        self.density_box.toggled.connect(self._on_density_toggled)

        self.lasso_box = QCheckBox("Lasso select")
        self.lasso_box.setToolTip(
            "Click to start an outline, move to trace it, click again to "
            "close.\nThe images inside appear below the plot."
        )
        self.lasso_box.toggled.connect(self._on_lasso_toggled)

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
        display_form.addRow(self.density_box)
        display_form.addRow(self.lasso_box)
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
        left_scroll.setMinimumWidth(290)
        left_scroll.setMaximumWidth(340)
        # Never a horizontal scrollbar: the controls should wrap into the
        # width they are given, not slide out of reach under one.
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # --- centre: the plot
        self.scatter = EmbeddingScatter()
        self.scatter.point_hovered.connect(self._on_hover)
        self.scatter.point_clicked.connect(self._on_click)
        self.scatter.points_selected.connect(self._on_selection)

        self.gallery = SelectionGallery()
        self.gallery.row_activated.connect(self._on_click)
        self.gallery.hide()

        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setStyleSheet(f"color: {TEXT_MUTED}; padding: 18px;")

        self.compute_features_button = QPushButton("Compute features from images")
        self.compute_features_button.setToolTip(
            "Describe the images this export refers to, using its own "
            "metadata. Simple descriptors, not learned embeddings."
        )
        self.compute_features_button.clicked.connect(self._compute_missing_features)
        self.compute_features_button.hide()
        self._missing_vectors_dir: Path | None = None

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
        centre.addWidget(self.message, 1)
        centre.addWidget(self.compute_features_button, 0, Qt.AlignmentFlag.AlignCenter)
        centre.addWidget(self.scatter, 1)
        centre.addWidget(self.gallery)
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
        splitter.setSizes([300, 1060, 300])
        splitter.setChildrenCollapsible(False)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self.setLayout(layout)

        self._sync_method_options()
        self._set_message("Choose an embedding dataset to begin.")

    def _update_points_hint(self, *_args) -> None:
        """Spell the percentage out as a count, which is what costs the time."""
        if self.dataset is None:
            self.points_hint.setText("")
            return
        total = self.dataset.n_points
        cap = _cap_for(self.max_points_box.currentText(), total)
        if cap is None or cap >= total:
            self.points_hint.setText(f"all {total:,} points")
        else:
            self.points_hint.setText(f"{cap:,} of {total:,} points")

    def _sync_method_options(self) -> None:
        is_umap = self.method_box.currentText() == UMAP
        self.neighbours_box.setEnabled(is_umap)
        self.perplexity_box.setEnabled(not is_umap)

    def _set_message(self, text: str) -> None:
        self.message.setText(text)
        self.message.setVisible(bool(text))
        self.scatter.setVisible(not text)

    # -- dataset loading --------------------------------------------------

    def set_dataset_root(self, root: Path | None) -> None:
        """Populate the dataset dropdown from a directory of exports.

        The loaded plates are always offered as a dataset in their own right,
        so the tab is usable before anybody has run an embedding export --
        which is exactly when a new plate most wants looking at.
        """
        directories = discover_datasets(Path(root)) if root is not None else []
        self.dataset_box.blockSignals(True)
        self.dataset_box.clear()
        for directory in directories:
            self.dataset_box.addItem(directory.name, str(directory))
        if getattr(self.session, "plates", []):
            from ..data.plate_features import LOADED_PLATES

            self.dataset_box.addItem(LOADED_PLATES, COMPUTED_FROM_PLATES)
        self.dataset_box.blockSignals(False)

        if self.dataset_box.count():
            self._on_dataset_changed()
        elif root is None:
            self._set_message(
                "No embedding dataset chosen.\n\n"
                "Use Browse… to pick a folder of exports, or load a plate in "
                "the Plate Browser to explore its images directly."
            )
        else:
            self._set_message(
                f"No embedding datasets found under\n{root}\n\n"
                "Use Browse… to pick a different folder, or load a plate in "
                "the Plate Browser to explore its images directly."
            )

    def plates_changed(self) -> None:
        """Re-offer the computed dataset after plates are added or removed."""
        from ..data.locations import EMBEDDING_ROOT, get_root

        current = self.dataset_box.currentData()
        self.dataset_box.blockSignals(True)
        self.set_dataset_root(get_root(EMBEDDING_ROOT))
        # Keep the user on whatever they were looking at, if it still exists.
        index = self.dataset_box.findData(current)
        if index >= 0:
            self.dataset_box.setCurrentIndex(index)
        self.dataset_box.blockSignals(False)

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

    def _browse_for_source_data(self) -> None:
        """Point this dataset at its images, with the dialog doing the work.

        The dialog indexes whatever folder is chosen and reports what it found
        -- how many images, how many rows they cover, and which neighbouring
        folders to try when the pick is close but wrong. The old flow was a
        bare picker followed by "none of this dataset's images were found",
        which said neither what it saw nor what to do next.
        """
        if self.frame is None:
            return
        from .locate_dialog import LocateDataDialog

        dialog = LocateDataDialog(
            self.frame,
            start=self.resolver.root if self.resolver is not None else None,
            parent=self,
        )
        if dialog.exec() != LocateDataDialog.DialogCode.Accepted:
            return
        if dialog.resolver is None:
            return

        self.resolver = dialog.resolver
        self.ambiguous_roots = []
        self._resolving = False
        root = self.resolver.root
        # Remember it: this is the answer for every dataset from this screen,
        # and its parent is where the sibling screens live.
        set_root(IMAGE_ROOT, root)
        if get_root(DATA_LIBRARY) is None:
            library = guess_data_library(root)
            if library is not None:
                set_root(DATA_LIBRARY, library)
        if self.dataset is not None:
            self._resolver_cache[str(self.dataset.directory)] = (self.resolver, [])
        self._update_source_label()
        self.preview.clear()
        self.status.emit(f"source data: {root}")

    def _update_source_label(self) -> None:
        """Say where previews come from, and whether that was a guess."""
        if self.resolver is None:
            self.source_label.setText(
                "looking…" if self._resolving else "not found — click Locate…"
            )
            self.source_label.setToolTip("")
            return

        root = str(self.resolver.root)
        # Keep the tail, which is the part that identifies the screen.
        shown = root if len(root) <= 34 else "…" + root[-33:]

        if self.ambiguous_roots:
            # Several screens matched. Naming one as if it were certain is how
            # a preview from the wrong experiment gets believed.
            self.source_label.setText(f"{shown}\n(guessed — other folders also match)")
            self.source_label.setToolTip(
                "Using:\n"
                f"    {root}\n\n"
                "These also matched, because screens share a filename "
                "convention:\n    "
                + "\n    ".join(self.ambiguous_roots)
                + "\n\nUse Locate… if this is the wrong one."
            )
        else:
            self.source_label.setText(shown)
            self.source_label.setToolTip(root)

    def _on_dataset_changed(self) -> None:
        directory = self.dataset_box.currentData()
        if not directory:
            return
        if directory == COMPUTED_FROM_PLATES:
            self._load_computed_dataset()
            return
        try:
            self.dataset = load_dataset(Path(directory))
            self.compute_features_button.setVisible(False)
        except EmbeddingError as exc:
            self.dataset = None
            self.frame = None
            self._clear_filters()
            # An export whose metadata exists but whose vectors do not is not
            # a dead end: the images it describes can be described directly,
            # keeping all of its own annotation.
            self._missing_vectors_dir = Path(directory)
            self._set_message(
                str(exc)
                + "\n\nPLATO can describe the images themselves instead — "
                "the export's own metadata is kept, so colouring by gene, "
                "drug and MoA still works."
            )
            self.compute_features_button.setVisible(True)
            self.status.emit("embeddings unavailable")
            return
        except (OSError, ValueError) as exc:
            self.dataset = None
            self._set_message(f"Could not read this dataset:\n{exc}")
            return

        self.frame, _ = build_frame(
            self.dataset,
            moa_table=self.moa_table,
            pathway_table=self.pathway_table,
        )
        self.resolver = self._start_resolving_images(self.frame)
        self._update_source_label()
        self.result = None
        self.apply_suggested_params()
        self._rebuild_colour_options()
        self._rebuild_filters()
        self._update_points_hint()
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

    def _compute_missing_features(self) -> None:
        """Describe an export's images when its vector file is missing."""
        directory = self._missing_vectors_dir
        if directory is None:
            return

        from ..data.export_features import build as build_from_export
        from ..gui.progress import run_with_progress

        # Locating the images is a prerequisite, and this export has no frame
        # yet -- so build one from its metadata to search with.
        import pandas as pd

        from ..data.embeddings import METADATA_FILENAME

        try:
            metadata = pd.read_csv(directory / METADATA_FILENAME, dtype=str).fillna("")
        except (OSError, ValueError) as exc:
            self._set_message(f"Could not read this export's metadata:\n{exc}")
            return

        resolver = self.resolver
        if resolver is None:
            resolver = self._resolve_images_now(metadata)
        if resolver is None:
            QMessageBox.information(
                self,
                "Source data",
                "The original images have not been found yet.\n\n"
                "Use Locate… to point PLATO at them, then compute again.",
            )
            return

        def job(report):
            return build_from_export(directory, resolver, progress=report)

        dataset, error, cancelled = run_with_progress(
            job, f"Describing {directory.name}", self
        )
        if cancelled:
            return
        if error is not None:
            self._set_message(str(error))
            return

        self.compute_features_button.setVisible(False)
        self.dataset = dataset
        self.frame, _ = build_frame(
            dataset, moa_table=self.moa_table, pathway_table=self.pathway_table
        )
        self.resolver = resolver
        self._update_source_label()
        self.result = None
        self.apply_suggested_params()
        self._rebuild_colour_options()
        self._rebuild_filters()
        self._update_points_hint()
        self._set_message(
            f"{dataset.n_points:,} images described by "
            f"{dataset.n_dimensions} image features, using "
            f"{directory.name}'s own metadata.\n\n"
            "These are simple descriptors computed from the images, not "
            "learned embeddings — good for plate effects, outliers and gross "
            "phenotypes, but not a substitute for the real export when making "
            "a phenotype claim.\n\n"
            "Choose a method and press Compute projection."
        )
        self.status.emit(f"{dataset.n_points:,} images described")

    def _resolve_images_now(self, frame):
        """Synchronous root hunt, for when the answer is needed immediately."""
        for candidate in self._image_root_candidates():
            resolver = ImageResolver.detect(candidate, frame)
            if resolver is not None:
                return resolver
        return None

    def _load_computed_dataset(self) -> None:
        """Describe the loaded plates' thumbnails and use that as the dataset."""
        from ..data.plate_features import build
        from ..gui.progress import run_with_progress

        def job(report):
            return build(self.session, progress=report)

        dataset, error, cancelled = run_with_progress(
            job, "Describing loaded images", self
        )
        if cancelled:
            return
        if error is not None:
            self.dataset = None
            self.frame = None
            self._clear_filters()
            self._set_message(error)
            return

        self.dataset = dataset
        self.frame, _ = build_frame(
            dataset, moa_table=self.moa_table, pathway_table=self.pathway_table
        )
        # The images are the ones already loaded, so the browser's own paths
        # answer this without a search.
        self.resolver = self._start_resolving_images(self.frame)
        self._update_source_label()
        self.result = None
        self.apply_suggested_params()
        self._rebuild_colour_options()
        self._rebuild_filters()
        self._update_points_hint()
        self._set_message(
            f"{dataset.n_points:,} images described by "
            f"{dataset.n_dimensions} image features.\n\n"
            "These are simple descriptors computed from the thumbnails, not "
            "learned embeddings — good for spotting plate effects, outliers "
            "and gross phenotypes, but not a substitute for a DINO export "
            "when making a phenotype claim.\n\n"
            "Choose a method and press Compute projection."
        )
        self.status.emit(f"{dataset.n_points:,} images described")

    def _image_root_candidates(self) -> list[Path]:
        """Directories that might hold this dataset's micrographs, best first.

        A loaded plate's image directory is the strongest signal -- the user
        confirmed that path by loading it -- and its PARENT matters just as
        much, because an export is organised one folder per plate and the
        metadata's ``plate`` column supplies that folder name.

        After those comes the data library: every DINO export was extracted
        from a DIFFERENT screen, so there is no single correct image root.
        The library's immediate subfolders are offered as candidates and the
        one that actually resolves this dataset's rows wins.
        """
        candidates: list[Path] = []

        def offer(path: Path) -> None:
            if path.is_dir() and path not in candidates:
                candidates.append(path)

        for plate in getattr(self.session, "plates", []):
            directory = Path(plate.cfg.images.dir)
            offer(directory)
            offer(directory.parent)
            # An images/ subfolder means the plate folder is one level up again.
            if directory.name.lower() == "images":
                offer(directory.parent.parent)

        configured = get_root(IMAGE_ROOT)
        if configured is not None:
            offer(configured)

        # The screens themselves. Sorted newest first: a dataset being
        # explored is usually a recent one, and this is a linear scan whose
        # cost is one directory listing per candidate until a hit.
        library = get_root(DATA_LIBRARY) or guess_data_library(configured)
        if library is not None and library.is_dir():
            try:
                screens = [d for d in library.iterdir() if d.is_dir()]
            except OSError:
                screens = []
            for screen in sorted(screens, key=lambda d: d.name, reverse=True):
                offer(screen)
        return candidates

    def _start_resolving_images(self, frame) -> ImageResolver | None:
        """Return a cached resolver, or start hunting for one in the background.

        Returns immediately either way; when the hunt finishes the resolver
        arrives on ``_on_resolver_found`` and the label updates. Until then
        previews say they are still looking, which is honest and costs the
        window nothing.
        """
        key = str(self.dataset.directory) if self.dataset is not None else ""
        if key and key in self._resolver_cache:
            resolver, self.ambiguous_roots = self._resolver_cache[key]
            return resolver

        self.ambiguous_roots = []
        self._resolving = True
        QThreadPool.globalInstance().start(
            _ResolveTask(key, frame, self._image_root_candidates(), self._resolve_signals)
        )
        return None

    def _on_resolver_ambiguous(self, key: str, others: list) -> None:
        if self.dataset is not None and key == str(self.dataset.directory):
            self.ambiguous_roots = others

    def _on_resolver_found(self, key: str, resolver) -> None:
        """The background hunt finished."""
        if key:
            self._resolver_cache[key] = (resolver, self.ambiguous_roots)
        # A different dataset may have been selected while we were looking.
        if self.dataset is None or key != str(self.dataset.directory):
            return
        self._resolving = False
        self.resolver = resolver
        self._update_source_label()

    def _resolve_images(self, frame) -> ImageResolver | None:
        """Find a root under which THIS dataset's rows resolve.

        Detection is run against the built frame, not guessed from paths: a
        directory existing proves nothing, and every dataset comes from a
        different screen.

        Screens often share a filename convention -- NIS writes
        ``WellA01_PointA01_0000_Channel....tiff`` for every plate of every
        experiment -- so several roots can match one dataset equally well and
        no heuristic can tell which is the right one. When that happens the
        first is used and ``ambiguous_roots`` records the rest, so the UI can
        say the choice was a guess instead of quietly showing images from the
        wrong screen.
        """
        key = str(self.dataset.directory) if self.dataset is not None else ""
        if key and key in self._resolver_cache:
            resolver, self.ambiguous_roots = self._resolver_cache[key]
            return resolver

        matches: list[ImageResolver] = []
        for candidate in self._image_root_candidates():
            resolver = ImageResolver.detect(candidate, frame)
            if resolver is not None:
                matches.append(resolver)
                # Two is enough to know it is ambiguous; scanning the whole
                # library to count them all costs more than the answer is worth.
                if len(matches) >= 2:
                    break

        chosen = matches[0] if matches else None
        self.ambiguous_roots = [str(r.root) for r in matches[1:]]
        if key:
            self._resolver_cache[key] = (chosen, self.ambiguous_roots)
        return chosen

    # -- controls ---------------------------------------------------------

    def apply_suggested_params(self) -> None:
        """Preset the projection controls for the dataset just loaded.

        Defaults that ignore the data are wrong for most of it: the tuned
        n_neighbors=500 suits a 30k-row DINO export and exceeds the sample
        entirely for a few hundred computed descriptors. See
        ``plato.data.projection.suggest``.
        """
        if self.dataset is None:
            return
        learned = not bool(self.dataset.run_info.get("computed"))
        params = suggest(self.dataset.n_points, learned=learned)
        self._suggested = params

        for box, value in (
            (self.neighbours_box, str(params.n_neighbors)),
            (self.perplexity_box, str(int(params.perplexity))),
        ):
            box.blockSignals(True)
            box.setCurrentText(value)
            box.blockSignals(False)

        self.max_points_box.blockSignals(True)
        self.max_points_box.setCurrentText(
            _choice_for(params.max_points, self.dataset.n_points)
        )
        self.max_points_box.blockSignals(False)
        self._update_points_hint()

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
        """The controls, over the suggestion's non-tunable parts.

        Metric, normalisation and PCA depth are not exposed: they follow from
        what the vectors ARE (learned embeddings want cosine geometry after
        an L2 normalise; hand-computed descriptors are already standardised
        and live in a Euclidean space), not from a preference.
        """
        base = self._suggested or ProjectionParams()

        def _number(box, fallback):
            try:
                return type(fallback)(box.currentText().strip())
            except (TypeError, ValueError):
                # An editable combo can hold anything the user typed.
                return fallback

        total = self.dataset.n_points if self.dataset is not None else 0
        max_points = _cap_for(self.max_points_box.currentText(), total)

        return ProjectionParams(
            method=self.method_box.currentText(),
            normalize=base.normalize,
            pca_components=base.pca_components,
            metric=base.metric,
            n_neighbors=max(2, _number(self.neighbours_box, base.n_neighbors)),
            min_dist=base.min_dist,
            perplexity=max(5.0, _number(self.perplexity_box, base.perplexity)),
            max_points=max_points,
            deterministic=self.deterministic_box.isChecked(),
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
        self._cancelled = False
        self._set_running(True)
        self._set_message(
            f"Computing {params.method} over {self.dataset.n_points:,} points…\n"
            "This runs once per parameter set and is then cached."
        )
        signals = _WorkerSignals()
        signals.finished.connect(self._on_projection)
        signals.failed.connect(self._on_projection_failed)
        signals.progress.connect(lambda text: self.status.emit(text))
        signals.advanced.connect(self._on_advanced)
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

    def _on_advanced(self, fraction: float, message: str) -> None:
        """One step of the backend's own progress reporting."""
        self.progress_bar.setValue(int(max(0.0, min(1.0, fraction)) * 1000))
        self.progress_label.setText(f"{message} — {fraction * 100:.0f}%")

    def _set_running(self, running: bool) -> None:
        for widget in (self.progress_bar, self.progress_label, self.cancel_button):
            widget.setVisible(running)
        self.run_button.setEnabled(not running)
        if running:
            self.progress_bar.setValue(0)
            self.progress_label.setText("starting…")

    def _cancel_projection(self) -> None:
        """Abandon the run.

        The fit itself cannot be interrupted -- neither library takes a stop
        flag -- so this detaches from the result rather than killing the work:
        the worker finishes into a cache entry that a later run will reuse,
        and the UI stops waiting. Saying so plainly beats a Cancel that
        appears to hang.
        """
        self._cancelled = True
        self._set_running(False)
        self._set_message(
            "Cancelled. The projection is still finishing in the background "
            "and will be cached, so computing it again will be instant."
        )
        self.status.emit("projection cancelled")

    def _on_projection(self, result) -> None:
        self._busy = False
        self._set_running(False)
        if self._cancelled:
            # The user walked away from this run; it is cached, not shown.
            return
        self.result = result
        self._set_message("")
        origin = "cached" if result.from_cache else f"{result.seconds:.1f}s"
        self.status.emit(
            f"{result.params.method}: {len(result.coords):,} points ({origin})"
        )
        self._redraw(reset_view=True)

    def _on_projection_failed(self, message: str) -> None:
        self._busy = False
        self._set_running(False)
        if self._cancelled:
            return
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

        if self.density_box.isChecked():
            self.scatter.set_density(True)

        label = field_label(column) if column else ""
        extra = "" if len(ordered) <= MAX_LEGEND_ENTRIES else f" · {len(ordered)} values"
        self.count_label.setText(
            f"<b>{len(coords):,}</b> of {len(rows):,} points · {label}{extra}"
        )

    # -- selection --------------------------------------------------------

    def _on_density_toggled(self, enabled: bool) -> None:
        self.scatter.set_density(enabled)
        # A legend of colours means nothing when colour encodes density.
        self.legend_box.setEnabled(not enabled)
        self.colour_box.setEnabled(not enabled)
        self.status.emit("density" if enabled else "points")

    def _on_lasso_toggled(self, enabled: bool) -> None:
        self.scatter.set_lasso(enabled)
        self.gallery.setVisible(enabled)
        if not enabled:
            self.scatter.clear_selection()
        else:
            self.status.emit("click to start an outline, click again to close it")

    def _on_selection(self, rows) -> None:
        """Show the images behind a lasso'd region."""
        if self.frame is None or rows is None or len(rows) == 0:
            self.gallery.show_selection([], 0)
            return

        entries: list[tuple[int, Path]] = []
        # Resolving a path is a filesystem stat per row, so stop once there
        # are enough tiles to fill the grid rather than checking thousands.
        from .gallery import MAX_TILES

        for row_index in rows:
            path = self._path_for(int(row_index))
            if path is not None:
                entries.append((int(row_index), path))
                if len(entries) >= MAX_TILES:
                    break
        self.gallery.show_selection(entries, len(rows))
        self.status.emit(f"{len(rows):,} points selected")

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
