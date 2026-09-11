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
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
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
from ..data.workspace import (
    DATASET_COLUMN,
    SOURCE_COMPUTED,
    SOURCE_EXPORT,
    SOURCE_JOINT,
    EmbeddingEntry,
    Workspace,
)
from ..data.explorer_model import (
    CONCENTRATION,
    CONDITION,
    DRUG,
    GENE,
    GUIDE,
    IMAGE_NAME,
    MOA,
    PATHWAY,
    PLATE,
    ROLE,
    WELL,
    ImageResolver,
    build_frame,
    categorical_fields,
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
from . import palettes, shapes
from .palette import (
    UNKNOWN_COLOUR,
    colours_for,
    is_unknown,
    set_active_palette,
)
from .cluster_panel import ClusterPanel
from .tsne_panel import TsnePanel
from .grid_view import BY_NAME, BY_SIZE, SORT_LABELS, GridView, build_groups
from .metadata_panel import MetadataPanel
from .open_embeddings_list import OpenEmbeddingsList
from .stats_panel import StatsPanel
from .selection_panel import SelectionPanel
from .scatter import BACKGROUND_CHOICES, EmbeddingScatter
from .section import Accordion

# Legend gets unreadable long before this; past it, colour still encodes the
# grouping but the legend is replaced by a count.
MAX_LEGEND_ENTRIES = 24

COMPUTED_FROM_PLATES = "<loaded-plates>"

# Cache sentinel: None is a real answer ("looked, not found"), so it cannot
# double as "not looked up yet".
_MISSING = object()

# Most columns a single comparison window will open. Each column decodes and
# holds its own image, so an unbounded comparison of a 5,000-point lasso would
# exhaust memory long before it became something anyone could look at.
MAX_COMPARE_COLUMNS = 24

# Most groups a facet variable may have before it is not offered. Well (96+)
# is past the point where a grid tells you anything a single plot does not.
MAX_FACET_GROUPS = 60

# Subsample choices, as a share of the dataset.
FULL_SAMPLE = "100% (all)"
SUBSAMPLE_CHOICES = (FULL_SAMPLE, "50%", "25%", "10%", "5%", "1%")


def _filename_slug(text: str | None) -> str:
    """A safe, compact filename fragment -- alnum runs joined by underscores."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(text or "")).strip("_")
    return cleaned


def _suggested_export_name(
    *,
    dataset: str | None,
    method: str,
    column: str | None,
    group_column: str | None,
    fmt: str,
    stamp: str,
) -> str:
    """Build an export filename identifying dataset, method, encoding and time.

    Distinct exports must not collide or look interchangeable -- a bare
    ``tsne_by_plot.png`` gave no way to tell two exports apart later.
    """
    faceting = bool(group_column)
    parts = [p for p in (_filename_slug(dataset), _filename_slug(method)) if p]
    if faceting:
        parts.append(f"grid_by_{_filename_slug(group_column)}")
        if column and column != group_column:
            parts.append(f"colour_{_filename_slug(column)}")
    else:
        column_slug = _filename_slug(column)
        parts.append(f"by_{column_slug}" if column_slug else "plot")
    parts.append(stamp)
    return "_".join(parts) + f".{fmt}"


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

    def __init__(self, vectors, params, fingerprint, cache, signals, is_cancelled=None) -> None:
        super().__init__()
        self._vectors = vectors
        self._params = params
        self._fingerprint = fingerprint
        self._cache = cache
        self._signals = signals
        # A callable rather than a captured bool: read fresh from the GUI
        # thread's flag each time it is checked, so a cancel that arrives
        # after the task started is still seen when the fit finishes.
        self._is_cancelled = is_cancelled

    def run(self) -> None:  # pragma: no cover - worker thread
        try:
            result = project(
                self._vectors,
                self._params,
                fingerprint=self._fingerprint,
                cache=self._cache,
                progress=self._signals.progress.emit,
                on_progress=self._signals.advanced.emit,
                is_cancelled=self._is_cancelled,
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

        # Every embedding open in this session. `self.dataset`/`self.frame`
        # remain as the CURRENT entry's view of it, so the rest of the panel
        # is unchanged; switching is a pointer move rather than a reload.
        self.workspace = Workspace()
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
        # Per-image statistics, once computed. None until the user asks.
        self._image_stats: dict | None = None
        self._stats_run = None
        self._stats_signals = None
        # Which statistic is colouring the plot, or None for metadata.
        self._stat_column: str | None = None
        # Whether points are traced back to micrographs at all. When off, no
        # resolver is hunted, no path is resolved and no pixel is read -- see
        # set_image_mode. Selection, lasso analysis and filtering are
        # unaffected, because none of them touch an image.
        self._image_mode = True
        # Facet variable, or None for a single plot.
        self._group_column: str | None = None
        # Set once the user picks a categorical palette themselves, so a
        # background change (which otherwise suggests the palette suited to
        # the new ground -- see _on_background_changed) never overrides a
        # deliberate choice. Same "explicit wins" precedent as the app's
        # saved theme overriding OS theme detection.
        self._palette_chosen_by_user = False
        # Which workspace entry the panel is currently showing.
        self._active_key: str | None = None
        # row -> resolved path, memoising the filesystem stat behind it.
        self._path_cache: dict[int, Path | None] = {}
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
        # Datasets available on disk vs embeddings actually OPEN are
        # different lists: the first is a directory listing, the second is
        # session state, and only the second can be switched between without
        # a load. Every open one gets its own row -- see
        # open_embeddings_list.py for why this is not a checkbox-per-row
        # "show several at once", which the single-scatter architecture does
        # not support; Combine below is the real answer to that.
        self.open_list = OpenEmbeddingsList()
        self.open_list.activated.connect(self._switch_to)
        self.open_list.close_requested.connect(self._close_embedding)

        # Projecting several open embeddings together as one fit, so position
        # is actually comparable between them -- see joint_projection.py for
        # why colouring by dataset over separately-fit layouts is not that.
        self.combine_button = QPushButton("Combine…")
        self.combine_button.setToolTip(
            "Project two or more open embeddings together as one UMAP/t-SNE "
            "fit, so their positions are directly comparable.\n"
            "Only embeddings with matching dimensionality can be combined."
        )
        self.combine_button.clicked.connect(self._combine_embeddings)
        self.combine_button.hide()

        source_row.addWidget(self.dataset_box)
        source_row.addWidget(browse)
        source_row.addWidget(self.open_list)
        source_row.addWidget(self.combine_button)
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
        self.source_label.setObjectName("hintSmall")

        source_data_row = QVBoxLayout()
        source_data_row.setContentsMargins(0, 0, 0, 0)
        source_data_row.setSpacing(4)
        source_data_row.addWidget(self.source_button)
        source_data_row.addWidget(self.source_label)
        source_data_widget = QWidget()
        source_data_widget.setLayout(source_data_row)

        self.method_box = QComboBox()
        self.method_box.addItems(METHODS)
        # t-SNE, not UMAP, is the standard starting method: for this kind of
        # perturbation screen (tens of expected conditions, not a continuum)
        # its cluster-preserving behaviour at a moderate perplexity reads
        # more directly as "these images are alike" than UMAP's global-shape
        # emphasis does, and it is the one already captioned with the
        # how-to-read-this-safely caveats (see TsnePanel).
        self.method_box.setCurrentText(TSNE)

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
        self.points_hint.setObjectName("hintSmall")

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
        self.progress_label.setObjectName("hintSmall")
        self.progress_label.hide()

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel_projection)
        self.cancel_button.hide()

        # Split into what data is being looked at vs how it is projected --
        # these used to share one "Projection" group, which mixed "which
        # embedding" (Dataset, Source data, Open, Combine) with "how is it
        # laid out" (Method, Neighbours, Subsample, Compute) in one flat
        # 11-row form.
        data_form = QFormLayout()
        data_form.setContentsMargins(6, 4, 6, 4)
        data_form.addRow("Dataset", source_widget)
        data_form.addRow("Source data", source_data_widget)
        data_group = QGroupBox()
        data_group.setLayout(data_form)

        embedding_form = QFormLayout()
        embedding_form.setContentsMargins(6, 4, 6, 4)
        embedding_form.addRow("Method", self.method_box)
        embedding_form.addRow("Neighbours", self.neighbours_box)
        points_column = QVBoxLayout()
        points_column.setContentsMargins(0, 0, 0, 0)
        points_column.setSpacing(4)
        points_column.addWidget(self.max_points_box)
        points_column.addWidget(self.points_hint)
        points_widget = QWidget()
        points_widget.setLayout(points_column)
        embedding_form.addRow("Subsample", points_widget)
        embedding_form.addRow(self.deterministic_box)
        embedding_form.addRow(self.run_button)
        embedding_form.addRow(self.progress_bar)
        embedding_form.addRow(self.progress_label)
        embedding_form.addRow(self.cancel_button)
        projection_group = QGroupBox()
        projection_group.setLayout(embedding_form)

        # t-SNE has far more parameters that matter than UMAP does, and they
        # matter in ways that are easy to misread, so they get their own
        # grouped panel rather than more rows in the Projection form.
        self.tsne_panel = TsnePanel()
        self.tsne_panel.changed.connect(self._on_tsne_changed)
        self.tsne_group = QGroupBox("t-SNE settings")
        tsne_layout = QVBoxLayout()
        tsne_layout.setContentsMargins(6, 4, 6, 4)
        tsne_layout.addWidget(self.tsne_panel)
        self.tsne_group.setLayout(tsne_layout)
        self.tsne_group.hide()
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

        # --- grouping / faceting
        self.group_box = QComboBox()
        self.group_box.setToolTip(
            "Split the plot into one panel per value of a variable.\n"
            "Answers 'does this cluster exist in every plate?', which colour "
            "cannot at these point counts because the groups overlap."
        )
        self.group_box.currentIndexChanged.connect(self._on_group_changed)

        self.shared_axes_box = QCheckBox("Shared axes")
        self.shared_axes_box.setChecked(True)
        self.shared_axes_box.setToolTip(
            "On (default): every panel shows the same region, so position and "
            "spread are comparable between groups.\n\n"
            "Off: each panel autoscales to its own points. Shows a group's "
            "internal structure, but makes groups look alike even when they "
            "occupy completely different parts of the embedding."
        )
        self.shared_axes_box.toggled.connect(self._redraw)

        self.grid_columns_box = QComboBox()
        self.grid_columns_box.addItem("Automatic", 0)
        for count in (1, 2, 3, 4, 5, 6):
            self.grid_columns_box.addItem(f"{count} columns", count)
        self.grid_columns_box.currentIndexChanged.connect(self._redraw)

        self.grid_sort_box = QComboBox()
        for key in (BY_SIZE, BY_NAME):
            self.grid_sort_box.addItem(SORT_LABELS[key], key)
        self.grid_sort_box.currentIndexChanged.connect(self._redraw)

        self.grid_page_label = QLabel("")
        self.grid_prev = QPushButton("◀")
        self.grid_prev.setFixedWidth(30)
        self.grid_prev.clicked.connect(lambda: self._step_page(-1))
        self.grid_next = QPushButton("▶")
        self.grid_next.setFixedWidth(30)
        self.grid_next.clicked.connect(lambda: self._step_page(1))
        page_row = QHBoxLayout()
        page_row.setContentsMargins(0, 0, 0, 0)
        page_row.addWidget(self.grid_prev)
        page_row.addWidget(self.grid_page_label, 1)
        page_row.addWidget(self.grid_next)
        self.grid_page_widget = QWidget()
        self.grid_page_widget.setLayout(page_row)

        # --- a second encoding: shape, independent of colour
        self.shape_box = QComboBox()
        self.shape_box.setToolTip(
            "Map a second variable to point shape, so colour and shape carry "
            "different fields at once — colour by antibiotic, shape by "
            "dataset.\n\n"
            "Only a handful of shapes are distinguishable at plot size, so "
            "this suits fields with few categories."
        )
        self.shape_box.currentIndexChanged.connect(self.schedule_redraw)

        self.shape_hint = QLabel("")
        self.shape_hint.setWordWrap(True)
        self.shape_hint.hide()

        self.palette_box = QComboBox()
        self.palette_box.setToolTip(
            "Which colours the categories or the numeric ramp use."
        )
        self.palette_box.currentIndexChanged.connect(self._on_palette_changed)

        self.background_box = QComboBox()
        for key, label in BACKGROUND_CHOICES:
            self.background_box.addItem(label, key)
        self.background_box.setToolTip(
            "The plot's own ground, independent of the application theme.\n"
            "Transparent exports with a real alpha channel, for dropping into "
            "a figure that has its own background."
        )
        self.background_box.currentIndexChanged.connect(self._on_background_changed)

        self.grey_out_box = QCheckBox("Grey out unselected")
        self.grey_out_box.setToolTip(
            "Draw everything except the current selection in grey, so the "
            "selection stands out. Works for clicks, shift-clicks and lassos."
        )
        self.grey_out_box.toggled.connect(self._on_grey_out_toggled)

        self.density_box = QCheckBox("Density")
        self.density_box.setToolTip(
            "Draw where points are concentrated instead of every mark.\n"
            "At tens of thousands of points the marks overlap into a solid "
            "blob; density shows how many, not merely 'at least one'."
        )
        self.density_box.toggled.connect(self._on_density_toggled)

        self.dim_others_box = QCheckBox("Grey out unannotated")
        self.dim_others_box.setChecked(True)
        self.dim_others_box.setToolTip(
            "Draw points with no value for the chosen field in grey, so an "
            "unannotated condition never reads as a category of its own."
        )
        self.dim_others_box.toggled.connect(self._redraw)

        # Split into HOW EACH POINT LOOKS (encoding: colour, shape, size,
        # background -- properties of one point) vs HOW POINTS ARE ARRANGED
        # ON SCREEN (grouping: one plot or a grid of them, faceted by which
        # variable). These used to share one 16-row "Display" group, where
        # "Display by" (a facet variable) sat two rows from "Colour by" and
        # "Shape by" (both point properties) with no visual distinction --
        # exactly the kind of name collision that motivated this split.
        encoding_form = QFormLayout()
        encoding_form.setContentsMargins(6, 4, 6, 4)
        encoding_form.addRow("Colour by", self.colour_box)
        encoding_form.addRow("Palette", self.palette_box)
        encoding_form.addRow("Shape by", self.shape_box)
        encoding_form.addRow(self.shape_hint)
        encoding_form.addRow("Background", self.background_box)
        encoding_form.addRow("Point size", self.size_slider)
        encoding_form.addRow("Opacity", self.opacity_slider)
        encoding_form.addRow(self.legend_box)
        encoding_form.addRow(self.dim_others_box)
        encoding_form.addRow(self.grey_out_box)
        encoding_form.addRow(self.density_box)
        encoding_group = QGroupBox()
        encoding_group.setLayout(encoding_form)

        grouping_form = QFormLayout()
        grouping_form.setContentsMargins(6, 4, 6, 4)
        grouping_form.addRow("Display by", self.group_box)
        grouping_form.addRow(self.shared_axes_box)
        grouping_form.addRow("Grid", self.grid_columns_box)
        grouping_form.addRow("Order", self.grid_sort_box)
        grouping_form.addRow(self.grid_page_widget)
        grouping_group = QGroupBox()
        grouping_group.setLayout(grouping_form)

        # --- lasso analysis
        #
        # Its own group rather than another checkbox in Display, because it is
        # not a display option: it is a mode plus the readout that mode
        # produces, and the readout is the larger half.
        self.cluster_panel = ClusterPanel()
        self.cluster_panel.lasso_toggled.connect(self._on_lasso_toggled)
        self.cluster_panel.additive_toggled.connect(self._on_lasso_additive)
        self.cluster_panel.show_images_requested.connect(self._show_selection_images)
        # clear_requested is connected after the scatter exists; the display
        # controls are built before it.
        cluster_group = QGroupBox()
        cluster_layout = QVBoxLayout()
        cluster_layout.setContentsMargins(6, 4, 6, 4)
        cluster_layout.addWidget(self.cluster_panel)
        cluster_group.setLayout(cluster_layout)

        # --- image statistics
        #
        # Its own group, and off by default. Every other control here works
        # from data already in memory; this one reads every image in the
        # dataset, so it is opt-in rather than something the user can switch
        # on without realising what it costs.
        self.stats_panel = StatsPanel()
        self.stats_panel.compute_requested.connect(self._compute_image_stats)
        self.stats_panel.cancel_requested.connect(self._cancel_image_stats)
        self.stats_panel.stat_selected.connect(self._on_stat_selected)
        stats_group = QGroupBox()
        stats_layout = QVBoxLayout()
        stats_layout.setContentsMargins(6, 4, 6, 4)
        stats_layout.addWidget(self.stats_panel)
        stats_group.setLayout(stats_layout)

        # --- filters
        self.filter_layout = QVBoxLayout()
        self.filter_layout.setContentsMargins(0, 0, 0, 0)
        self.filter_layout.setSpacing(8)
        self.filters: list[FilterList] = []

        clear_filters = QPushButton("Clear filters")
        clear_filters.clicked.connect(self.clear_filters)

        # --- the sidebar, as an accordion
        #
        # These seven used to be seven permanently-expanded QGroupBoxes in one
        # scrolling column, and the t-SNE panel added four nested boxes of its
        # own. See views/section.py for why one-open-at-a-time is the fix and
        # why the closed summaries are what make it safe.
        #
        # Each section body is a plain QWidget named "sectionBody", which is
        # what the stylesheet targets to strip the now-redundant group-box
        # chrome from anything inside.

        def _body(*widgets: QWidget) -> QWidget:
            body = QWidget()
            body.setObjectName("sectionBody")
            layout = QVBoxLayout()
            layout.setContentsMargins(10, 6, 10, 10)
            layout.setSpacing(8)
            for widget in widgets:
                layout.addWidget(widget)
            body.setLayout(layout)
            return body

        filter_body = QWidget()
        filter_body.setLayout(self.filter_layout)

        self.accordion = Accordion()
        # Order follows the workflow: choose data, project it, encode it,
        # group it, analyse a region, filter. Image Statistics goes LAST
        # because it is the only optional, expensive step -- it belongs where
        # you arrive after everything else, not in the middle of styling.
        self.accordion.add("data", "Data", _body(data_group))
        # t-SNE settings live INSIDE Embedding rather than beside it: they are
        # that section's parameters, not a peer of it, and as a sibling they
        # appeared and vanished from the column depending on the method.
        self.accordion.add(
            "embedding", "Embedding", _body(projection_group, self.tsne_group)
        )
        self.accordion.add("encoding", "Encoding", _body(encoding_group))
        self.accordion.add("grouping", "Grouping", _body(grouping_group))
        self.accordion.add("lasso", "Lasso analysis", _body(cluster_group))
        self.accordion.add("filters", "Filters", _body(filter_body, clear_filters))
        self.accordion.add("stats", "Image statistics", _body(stats_group))
        self.accordion.add_stretch()
        # Data is where the workflow starts and the only section that is
        # useless closed on an empty app.
        self.accordion.open_section("data")
        # Seed the closed-state summaries. Without this the headers stay blank
        # until the first control change, which is exactly the state the
        # summaries exist to explain.
        self._update_section_summaries()

        left = QVBoxLayout()
        left.setContentsMargins(10, 10, 10, 10)
        left.setSpacing(10)
        left.addWidget(self.accordion)

        left_container = QWidget()
        left_container.setLayout(left)
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_container)
        left_scroll.setWidgetResizable(True)
        # Resizable, not pinned. The controls grew past what 340px can hold --
        # the t-SNE panel's nested groups clipped their own labels off the
        # left edge at that width -- and a hard maximum meant the only way to
        # read them was to not have them. The splitter now decides, within
        # bounds that keep the plot usable.
        left_scroll.setMinimumWidth(300)
        left_scroll.setMaximumWidth(560)
        # Never a horizontal scrollbar: the controls should wrap into the
        # width they are given, not slide out of reach under one.
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # --- centre: the plot
        self.scatter = EmbeddingScatter()
        self.scatter.point_hovered.connect(self._on_hover)
        self.scatter.point_clicked.connect(self._on_point_clicked)
        self.scatter.point_activated.connect(self._on_click)
        self.scatter.points_selected.connect(self._on_selection)
        self.cluster_panel.clear_requested.connect(self.scatter.clear_selection)

        # Facets report into exactly the same handlers as the single plot, so
        # there is one selection however it was made.
        self.grid = GridView()
        self.grid.point_clicked.connect(self._on_point_clicked)
        self.grid.point_activated.connect(self._on_click)
        self.grid.points_selected.connect(self._on_selection)
        self.grid.hide()

        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setObjectName("message")

        self.compute_features_button = QPushButton("Compute features from images")
        self.compute_features_button.setToolTip(
            "Describe the images this export refers to, using its own "
            "metadata. Simple descriptors, not learned embeddings."
        )
        self.compute_features_button.clicked.connect(self._compute_missing_features)
        self.compute_features_button.hide()
        self._missing_vectors_dir: Path | None = None

        self.count_label = QLabel("—")
        self.count_label.setObjectName("muted")

        # A global switch, in the toolbar rather than buried in a group, since
        # it changes what the whole right-hand half of the window is for.
        self.image_mode_box = QCheckBox("Image viewer")
        self.image_mode_box.setChecked(True)
        self.image_mode_box.setToolTip(
            "On: points trace back to their micrographs — previews, the "
            "detail viewer and comparison all work.\n\n"
            "Off: the explorer is a pure embedding/metadata tool. No image "
            "root is searched, no path is resolved and no pixel is read, "
            "which matters most on a network share. Selection, lasso "
            "analysis and filtering are unaffected."
        )
        self.image_mode_box.toggled.connect(self.set_image_mode)

        reset_view = QPushButton("Reset view")
        reset_view.clicked.connect(self.scatter.reset_view)
        export_png = QPushButton("Export PNG…")
        export_png.clicked.connect(lambda: self.export("png"))
        export_svg = QPushButton("Export SVG…")
        export_svg.clicked.connect(lambda: self.export("svg"))

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(10, 6, 10, 6)
        toolbar.addWidget(self.count_label, 1)
        toolbar.addWidget(self.image_mode_box)
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
        centre.addWidget(self.grid, 1)
        centre_widget = QWidget()
        centre_widget.setLayout(centre)

        # --- right: the selection, one cropped preview per point
        self.selection_panel = SelectionPanel()
        self.selection_panel.set_providers(self._describe, self._path_for)
        self.selection_panel.row_clicked.connect(self._on_entry_clicked)
        self.selection_panel.row_activated.connect(self._on_click)
        self.selection_panel.selection_changed.connect(self._on_panel_edited)
        self.selection_panel.compare_requested.connect(self.compare_selection)
        self.selection_panel.reveal_failed.connect(self.status.emit)

        # With images off the right column shows the selection's metadata
        # instead of its pixels, so the panel still answers "what did I just
        # select?" rather than vanishing and leaving dead space.
        self.metadata_panel = MetadataPanel()
        self.metadata_panel.compare_requested.connect(self.compare_selection)
        self.metadata_panel.selection_changed.connect(self._on_panel_edited)
        self.metadata_panel.hide()

        right_container = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.selection_panel)
        right_layout.addWidget(self.metadata_panel)
        right_container.setLayout(right_layout)
        # No maximum: the column is meant to be dragged wide when you want to
        # look closely, which is the point of making it resizable at all.
        right_container.setMinimumWidth(200)
        self._right_container = right_container

        self.splitter = QSplitter()
        self.splitter.addWidget(left_scroll)
        self.splitter.addWidget(centre_widget)
        self.splitter.addWidget(right_container)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        # 340, not 300: the sidebar opens at its minimum, and at 300 the
        # section forms (a label column plus a combo) are a few pixels wider
        # than the pane, so every combo clipped its right edge on first show.
        # Measured against the widest section body rather than guessed.
        self.splitter.setSizes([340, 1000, 360])
        self.splitter.setChildrenCollapsible(False)
        # Wider column, bigger previews. Debounced: a drag emits a signal per
        # pixel, and re-decoding images on each would make the drag crawl.
        self._preview_resize_timer = QTimer(self)
        self._preview_resize_timer.setSingleShot(True)
        self._preview_resize_timer.setInterval(120)
        self._preview_resize_timer.timeout.connect(self._apply_preview_width)
        self.splitter.splitterMoved.connect(
            lambda *_: self._preview_resize_timer.start()
        )
        splitter = self.splitter

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
        self.tsne_group.setVisible(not is_umap)

    def _set_message(self, text: str) -> None:
        self.message.setText(text)
        self.message.setVisible(bool(text))
        # Whichever plot is live must yield to a message and come back after.
        # Asking for the scatter unconditionally would show it over the grid
        # when faceting, so the choice is left to _draw.
        showing = not text
        faceting = bool(self._group_column)
        self.scatter.setVisible(showing and not faceting)
        self.grid.setVisible(showing and faceting)

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
        """Add one or more folders of embeddings/plates to this session.

        Recursive and multi-root, unlike a bare folder picker: a lab drive is
        rarely organised as one folder holding exactly the datasets wanted,
        and a user should not have to know the exact level in advance. See
        plato.views.load_data_dialog and plato.data.discovery.
        """
        from ..data.locations import EMBEDDING_ROOT
        from .load_data_dialog import LoadDataDialog

        start = get_root(EMBEDDING_ROOT)
        dialog = LoadDataDialog(self, start=Path(start) if start else None)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        embeddings, plates = dialog.selected()
        if not embeddings and not plates:
            return

        loaded, failed = 0, []
        for directory in embeddings:
            if self._load_embedding_directory(directory, announce=False):
                loaded += 1
            else:
                failed.append(directory.name)

        self._sync_open_box()
        if loaded and self.workspace.current is not None:
            # Land on the last one loaded rather than whatever was current
            # before -- that is what "just loaded" means to the user.
            # _load_embedding_directory already left _active_key on the last
            # entry it loaded, so this is usually a no-op; calling it
            # explicitly is what stays correct if that ever changes.
            self._switch_to(self.workspace.current.key)

        summary = f"loaded {loaded} embedding{'s' if loaded != 1 else ''}"
        if failed:
            summary += f" — {len(failed)} failed: {', '.join(failed[:3])}"
            if len(failed) > 3:
                summary += f" (+{len(failed) - 3} more)"
        self.status.emit(summary)

        if plates:
            # A plate needs a plate map, which discovery cannot guess -- an
            # image folder alone does not say which file is the map or how
            # its columns are laid out. Point at the folder found and hand
            # off to the dialog that actually asks for the map, rather than
            # guessing and silently loading the wrong thing.
            names = "\n".join(f"  {p}" for p in plates[:8])
            more = f"\n  (+{len(plates) - 8} more)" if len(plates) > 8 else ""
            QMessageBox.information(
                self,
                "Plate image folders found",
                f"{len(plates)} folder(s) look like plate images, but "
                "loading a plate also needs its plate map, which a scan "
                "cannot determine on its own:\n\n"
                f"{names}{more}\n\n"
                "Use Data → Load Data in the Plate Browser tab to add "
                "one, pointing it at the folder above.",
            )
        elif not loaded:
            QMessageBox.information(
                self, "Load data", "Nothing was loaded — check the selection."
            )

    def _browse_for_source_data(self) -> None:
        """Locate images for every open embedding, in one popup.

        Mirrors "Browse embeddings": one list, one row per embedding
        currently open in the workspace, each independently scanned and
        accepted. With one embedding open this is exactly the old single-
        dataset flow; with several, locating them all no longer means
        reopening the dialog once per entry with no memory of what was
        already resolved.

        The dialog indexes whatever folder is chosen per row and reports what
        it found -- how many images, how many rows they cover, and which
        neighbouring folders to try when the pick is close but wrong.
        """
        entries = self.workspace.entries
        if not entries:
            return
        from .locate_all_dialog import LocateAllDialog

        dialog = LocateAllDialog(entries, parent=self)
        if dialog.exec() != LocateAllDialog.DialogCode.Accepted:
            return

        results = dialog.results()
        if not results:
            return

        remembered_root: Path | None = None
        for entry in entries:
            resolver = results.get(entry.key)
            if resolver is None:
                continue
            entry.resolver = resolver
            entry.resolver_searched = True
            # Feed the same cache the automatic background hunt reads from,
            # keyed by dataset directory, so switching to this entry again
            # (or opening the same directory as a second embedding) does not
            # re-hunt for a root that was just found by hand.
            self._resolver_cache[str(entry.directory)] = (resolver, [])
            if remembered_root is None:
                remembered_root = resolver.root
            if entry.key == self._active_key:
                self.resolver = resolver
                self._invalidate_paths()
                self._update_source_label()
                # Paths just changed, so any "image not found" entries can
                # now resolve.
                self.selection_panel.set_rows(list(self.selection_panel.rows))

        # Remember one of them: this is the answer for every dataset from
        # this screen, and its parent is where the sibling screens live.
        if remembered_root is not None:
            set_root(IMAGE_ROOT, remembered_root)
            if get_root(DATA_LIBRARY) is None:
                library = guess_data_library(remembered_root)
                if library is not None:
                    set_root(DATA_LIBRARY, library)

        self.status.emit(
            f"source data located for {len(results)} of {len(entries)} embedding(s)"
        )

    def _update_source_label(self) -> None:
        """Say where previews come from, and whether that was a guess."""
        if self.resolver is None:
            self.source_label.setText(
                "looking…" if self._resolving else "not found — click Locate…"
            )
            self.source_label.setToolTip("")
            return

        roots = self.resolver.roots
        root = str(self.resolver.root)
        if len(roots) > 1:
            # Images live under more than one folder for this dataset -- say
            # so, rather than naming only the first and implying it is the
            # whole story.
            shown = f"{len(roots)} folders"
            tooltip_root = "\n    ".join(str(r) for r in roots)
        else:
            # Keep the tail, which is the part that identifies the screen.
            shown = root if len(root) <= 34 else "…" + root[-33:]
            tooltip_root = root

        if self.ambiguous_roots:
            # Several screens matched. Naming one as if it were certain is how
            # a preview from the wrong experiment gets believed.
            self.source_label.setText(f"{shown}\n(guessed — other folders also match)")
            self.source_label.setToolTip(
                "Using:\n"
                f"    {tooltip_root}\n\n"
                "These also matched, because screens share a filename "
                "convention:\n    "
                + "\n    ".join(self.ambiguous_roots)
                + "\n\nUse Locate… if this is the wrong one."
            )
        else:
            self.source_label.setText(shown)
            self.source_label.setToolTip(tooltip_root)

    def _on_dataset_changed(self) -> None:
        directory = self.dataset_box.currentData()
        if not directory:
            return
        if directory == COMPUTED_FROM_PLATES:
            self._load_computed_dataset()
            return

        # Already open in this session (loaded via this dropdown before, or
        # via the multi-folder loader): switch to it instead of reloading
        # and creating a duplicate entry for the same directory.
        existing = self.workspace.find_by_directory(Path(directory))
        if existing:
            self._switch_to(existing[0].key)
            return

        self._load_embedding_directory(Path(directory), announce=True)

    def _load_embedding_directory(self, directory: Path, *, announce: bool) -> bool:
        """Load one embedding export and register it with the workspace.

        Shared by the single-select dropdown and the multi-folder loader, so
        both paths register entries, resolve images and rebuild the controls
        identically. Returns whether the load succeeded; a failure updates
        the message panel only when ``announce`` is set, since a bulk load
        reports its own failures rather than overwriting its progress message
        for every file that does not load.
        """
        try:
            dataset = load_dataset(directory)
            self.compute_features_button.setVisible(False)
        except EmbeddingError as exc:
            if announce:
                self.dataset = None
                self.frame = None
                self._clear_filters()
                # An export whose metadata exists but whose vectors do not is
                # not a dead end: the images it describes can be described
                # directly, keeping all of its own annotation.
                self._missing_vectors_dir = directory
                self._set_message(
                    str(exc)
                    + "\n\nPLATO can describe the images themselves instead — "
                    "the export's own metadata is kept, so colouring by gene, "
                    "drug and MoA still works."
                )
                self.compute_features_button.setVisible(True)
                self.status.emit("embeddings unavailable")
            return False
        except (OSError, ValueError) as exc:
            if announce:
                self.dataset = None
                self._set_message(f"Could not read this dataset:\n{exc}")
            return False

        frame, _ = build_frame(
            dataset, moa_table=self.moa_table, pathway_table=self.pathway_table
        )
        self.dataset = dataset
        self.frame = frame
        self._invalidate_paths()
        self.resolver = self._start_resolving_images(frame)
        self._register_entry(dataset, frame)
        self._active_key = self.workspace.current_key
        self._update_source_label()
        self.result = None
        self.apply_suggested_params()
        self.metadata_panel.set_frame(frame)
        self._rebuild_group_options()
        self._rebuild_shape_options()
        self._rebuild_colour_options()
        self._rebuild_palette_options()
        self._rebuild_filters()
        self._update_points_hint()

        if announce:
            run = dataset.run_info.get("model", "")
            self._set_message(
                f"{dataset.name}: {dataset.n_points:,} points × "
                f"{dataset.n_dimensions} dimensions"
                f"{' · ' + str(run) if run else ''}\n\n"
                "Choose a method and press Compute projection."
            )
            self.status.emit(
                f"{dataset.name}: {dataset.n_points:,} embeddings"
                + ("" if self.resolver else " · original images not found")
            )
        return True

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

        # Describing images requires reading them, so the image viewer being
        # off is a hard stop rather than something to work around: turning it
        # off is a statement that no pixel should be read.
        if not self._image_mode:
            QMessageBox.information(
                self,
                "Image statistics",
                "Computing features reads every image, and the image viewer "
                "is off.\n\nTurn it on first, then compute again.",
            )
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
        self._invalidate_paths()
        # Register it like any other load, or the workspace and the panel
        # disagree about what is current and switching embeddings restores
        # the wrong frame.
        self._register_entry(dataset, self.frame, source=SOURCE_COMPUTED)
        self._active_key = self.workspace.current_key
        self._update_source_label()
        self.result = None
        self.apply_suggested_params()
        self.metadata_panel.set_frame(self.frame)
        self._rebuild_group_options()
        self._rebuild_shape_options()
        self._rebuild_colour_options()
        self._rebuild_palette_options()
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
        self._invalidate_paths()
        self.resolver = self._start_resolving_images(self.frame)
        self._register_entry(self.dataset, self.frame)
        self._active_key = self.workspace.current_key
        self._update_source_label()
        self.result = None
        self.apply_suggested_params()
        self.metadata_panel.set_frame(self.frame)
        self._rebuild_group_options()
        self._rebuild_shape_options()
        self._rebuild_colour_options()
        self._rebuild_palette_options()
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
        # With images off there is nothing to resolve for, and the hunt is the
        # single most expensive thing that happens when a dataset loads.
        if not self._image_mode:
            return None

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
        self._invalidate_paths()
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

        self.neighbours_box.blockSignals(True)
        self.neighbours_box.setCurrentText(str(params.n_neighbors))
        self.neighbours_box.blockSignals(False)

        # t-SNE's own panel owns perplexity (and every other t-SNE knob) once
        # it exists -- load_from applies the whole suggested ProjectionParams
        # at once rather than perplexity alone, so a dataset that scales past
        # the panel's hardcoded 500/250-iteration "Standard" preset gets a
        # correspondingly larger iteration count too, not just a bigger
        # perplexity with too few steps to converge on it.
        self.tsne_panel.load_from(params)

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
        # "Dataset" only exists on a joint entry's combined frame (see
        # joint_projection.py); explorer_model.COLOUR_FIELDS does not know
        # about it, since that module is deliberately workspace-agnostic.
        # Listed first when present -- for a joint embedding it is usually
        # the first thing worth colouring by.
        if DATASET_COLUMN in self.frame.columns:
            self.colour_box.addItem("Dataset", DATASET_COLUMN)
        for column in colour_fields(self.frame):
            self.colour_box.addItem(field_label(column), column)
        # Prefer a field that says something biological on first open.
        if current is not None:
            index = self.colour_box.findData(current)
            if index >= 0:
                self.colour_box.setCurrentIndex(index)
        elif DATASET_COLUMN in self.frame.columns:
            self.colour_box.setCurrentIndex(0)
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
        self._update_section_summaries()

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

        params = ProjectionParams(
            method=self.method_box.currentText(),
            normalize=base.normalize,
            pca_components=base.pca_components,
            metric=base.metric,
            n_neighbors=max(2, _number(self.neighbours_box, base.n_neighbors)),
            min_dist=base.min_dist,
            max_points=max_points,
            deterministic=self.deterministic_box.isChecked(),
        )
        # The t-SNE panel is the sole authority on every t-SNE parameter,
        # perplexity included -- there is no other control for it now that
        # the old standalone perplexity box is gone (superseded by this
        # panel; see apply_suggested_params).
        if params.method == TSNE:
            self.tsne_panel.apply_to(params)
        return params

    def compute(self) -> None:
        if self.dataset is None:
            QMessageBox.information(
                self, "Embeddings", "Load an embedding dataset first."
            )
            return
        if self._busy:
            return
        params = self._params()

        # In-memory first: switching back to an embedding already projected
        # with these exact parameters should be instant, not a disk read.
        # Keyed by params, not just by entry, because the same entry can
        # legitimately hold both a UMAP and a t-SNE layout (or two parameter
        # sets) side by side without either evicting the other.
        entry = self.workspace.current
        if entry is not None:
            in_memory = entry.projections.get(params.key())
            if in_memory is not None:
                self._on_projection(in_memory)
                return

        cached = self.cache.load(self.dataset.fingerprint(), params)
        if cached is not None:
            if entry is not None:
                entry.projections[params.key()] = cached
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
                is_cancelled=lambda: self._cancelled,
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
        flag -- so the worker keeps running until it naturally finishes; this
        detaches the UI from waiting on it. What Cancel DOES guarantee: the
        result is discarded rather than cached (see project()'s
        ``is_cancelled``), so it cannot silently reappear as if it had
        finished successfully the next time these parameters are used --
        that would look exactly like a Cancel that did not work.
        """
        self._cancelled = True
        self._set_running(False)
        self._set_message(
            "Cancelled. The result will not be kept or cached, even though "
            "the computation finishes in the background."
        )
        self.status.emit("projection cancelled")

    def _on_projection(self, result) -> None:
        self._busy = False
        self._set_running(False)
        if self._cancelled:
            # The user walked away from this run; it is cached, not shown.
            return
        self.result = result
        entry = self.workspace.current
        if entry is not None:
            # Keyed by parameters, so switching back to this embedding later
            # (see _on_open_changed) finds it without recomputing, and a
            # second method/parameter set on the same entry does not evict
            # the first.
            entry.projections[result.params.key()] = result
            entry.last_result_key = result.params.key()
        self._set_message("")
        # A new projection is new coordinates: points that sat together no
        # longer do, so a selection made on the old layout describes nothing
        # in this one. Filtering and recolouring keep the selection (they only
        # change what is drawn); reprojecting cannot.
        self.scatter.clear_selection()
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
        self._update_section_summaries()

    def _update_section_summaries(self) -> None:
        """Put each section's current value on its header.

        This is what makes one-open-at-a-time safe: the whole configuration
        stays readable with every section shut, so collapsing a control never
        conceals its effect. Sections whose state is actively changing what is
        drawn (a live filter, a held selection, a facet) also get a badge, in
        the accent -- the interface's one colour for "this is live".

        Cheap and defensive by design: it runs on every control change, and a
        missing widget or an empty frame must degrade to a blank summary
        rather than raise into a redraw path.
        """
        accordion = getattr(self, "accordion", None)
        if accordion is None:
            return

        def _label(box) -> str:
            data = box.currentData()
            if not data:
                return ""
            return field_label(data) if isinstance(data, str) else str(data)

        # Data: which export is open. From the workspace, not dataset_box --
        # that combo picks a DIRECTORY to search and is empty whenever the
        # dataset arrived by any other route (a restored workspace, a
        # combine, computed features).
        entry = self.workspace.current
        name = entry.name if entry is not None else ""
        extra = len(self.workspace.entries) - 1
        if name and extra > 0:
            name = f"{name} (+{extra})"
        accordion.set_summary("data", name)

        # Embedding: method and how many points it was fitted on.
        method = self.method_box.currentText()
        if self.result is not None:
            points = f"{len(self.result.row_indices):,} pts"
            accordion.set_summary("embedding", f"{method} · {points}")
        else:
            accordion.set_summary("embedding", f"{method} · not run")

        # Encoding: what colour means, since that is the one a reader of the
        # plot has to know to interpret it at all.
        colour = _label(self.colour_box)
        accordion.set_summary("encoding", f"Colour: {colour}" if colour else "")

        # Grouping: faceting is a structural change to the plot, so it badges.
        group = _label(self.group_box)
        accordion.set_summary("grouping", group or "off")
        accordion.set_badge("grouping", bool(self._group_column))

        # Lasso: a held selection changes what the right-hand panels show.
        scatter = getattr(self, "scatter", None)
        selected = len(scatter.selected_rows) if scatter is not None else 0
        accordion.set_summary("lasso", f"{selected:,} selected" if selected else "")
        accordion.set_badge("lasso", bool(selected))

        # Filters: the one that genuinely hides something. An active filter
        # silently removes points, so it always badges and always counts.
        active = len(self.current_filters())
        accordion.set_summary("filters", f"{active} active" if active else "")
        accordion.set_badge("filters", bool(active))

    def _colour_universe(self, column: str | None) -> list[str]:
        """Every value ``column`` takes across the whole frame, unknowns last.

        The basis for colour/shape assignment -- see ``_redraw`` and
        ``_symbols_for``. ``distinct_values`` already excludes blanks (it
        exists to feed filter checklists, where an empty-string row is not a
        choice worth offering), but here that would make blank/unannotated
        rows vanish from the legend and the draw loop entirely rather than
        drawing grey, so any unknown-marker values actually present in the
        column are appended, sorted after the known ones.
        """
        if not column or self.frame is None or column not in self.frame.columns:
            return []
        raw = self.frame[column].astype(str)
        # distinct_values sorts plain-alphabetically and only excludes "" --
        # other unknown markers ("nan", "unknown", the MoA/pathway sentinel)
        # are real strings to it and would land wherever they alphabetise to.
        # Pulled out here and appended last instead, so every unknown-marker
        # value shares one grey slot at the end rather than scattering through
        # the real categories.
        known = [v for v in distinct_values(self.frame, column) if not is_unknown(v)]
        unknowns = sorted({v for v in raw if is_unknown(v)})
        return known + unknowns

    def _redraw(self, *_args, reset_view: bool = False) -> None:
        # A queued redraw may still be pending when an immediate one runs
        # (e.g. a fresh projection); dropping it avoids drawing twice.
        self._redraw_timer.stop()
        self._update_section_summaries()
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

        # Colouring by a measured statistic short-circuits the categorical
        # path entirely: the values are continuous, so they get a ramp and a
        # range readout rather than a palette and a legend.
        stat_values = self._stat_values(visible)
        if stat_values is not None:
            self._draw_continuous(coords, visible, stat_values, rows, reset_view)
            return

        column = self.colour_box.currentData()
        values = subset[column].astype(str).to_numpy()[mask] if column else np.array([""] * len(coords))
        # The colour->value mapping is built from every value the FULL frame
        # carries for this column, not just what happens to be on screen after
        # filtering. Deriving it from the visible subset would reassign
        # colours every time a filter or facet changed which categories are
        # present -- e.g. filtering out "Amikacin" would shift "Ampicillin"
        # into its palette slot. Filtering must only ever hide points, never
        # repaint the ones that remain.
        universe = self._colour_universe(column)
        mapping, continuous = colours_for(universe)
        if self.dim_others_box.isChecked():
            for value in universe:
                if is_unknown(value):
                    mapping[value] = UNKNOWN_COLOUR

        # What is actually on screen right now -- a subset of the universe
        # above once filters or a facet's page are in play. The mapping stays
        # fixed to the universe; only which entries get drawn/listed narrows.
        ordered = [v for v in universe if v in set(values.tolist())]

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

        # Per-point colours as well as the grouped masks: the single plot uses
        # the groups (one draw call each), while a facet needs to pick out the
        # colours of an arbitrary subset, which only a per-point list allows.
        point_colours = [mapping.get(v, UNKNOWN_COLOUR) for v in values]

        self._draw(
            coords,
            visible,
            point_colours,
            groups=groups,
            legend_entries=legend_entries,
            reset_view=reset_view,
        )

        label = field_label(column) if column else ""
        extra = "" if len(ordered) <= MAX_LEGEND_ENTRIES else f" · {len(ordered)} values"
        self.count_label.setText(
            f"<b>{len(coords):,}</b> of {len(rows):,} points · {label}{extra}"
        )

    def _draw_continuous(self, coords, visible, values, rows, reset_view: bool) -> None:
        """Colour points by a measured statistic, on a continuous ramp.

        Separate from the categorical path because the questions differ. A
        category needs distinguishable colours and a legend; a measurement
        needs an ordered ramp and its range, so that "brighter is further
        along the scale" is readable without looking anything up.

        Percentile limits, not min/max: one saturated field would otherwise
        compress every other point into the bottom of the scale.
        """
        from .palette import ramp_over_array

        values = np.asarray(values, dtype=np.float32)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            # Measured nothing that is on screen: draw the points plainly
            # rather than an all-grey plot with a meaningless scale.
            self.scatter.set_points(
                coords,
                visible,
                [UNKNOWN_COLOUR] * len(coords),
                point_size=self.size_slider.value(),
                opacity=self.opacity_slider.value() / 100.0,
                legend_entries=None,
                reset_view=reset_view,
            )
            self.count_label.setText(
                f"<b>{len(coords):,}</b> of {len(rows):,} points · "
                f"no measurements for these"
            )
            return

        low, high = (float(v) for v in np.percentile(finite, [2, 98]))
        if high <= low:
            high = low + 1e-6
        colours, groups = ramp_over_array(values, low, high)

        self._draw(
            coords,
            visible,
            colours,
            groups=groups,
            legend_entries=None,
            reset_view=reset_view,
        )

        from ..data.image_stats import STAT_LABELS

        label = STAT_LABELS.get(self._stat_column, self._stat_column or "")
        unmeasured = int((~np.isfinite(values)).sum())
        note = f" · {unmeasured:,} unmeasured" if unmeasured else ""
        self.count_label.setText(
            f"<b>{len(coords):,}</b> of {len(rows):,} points · {label} "
            f"{low:,.0f}–{high:,.0f}{note}"
        )

    def _draw(
        self,
        coords,
        visible,
        point_colours,
        *,
        groups=None,
        legend_entries=None,
        reset_view: bool = False,
    ) -> None:
        """Send one set of styled points to the single plot or to the facets.

        Both colouring paths end here, so faceting, shapes, grey-out and the
        background are applied once rather than duplicated per path -- and a
        facet can never disagree with the single plot about how a point looks.
        """
        symbols = self._symbols_for(visible)
        faceting = bool(self._group_column) and self.frame is not None

        self.scatter.setVisible(not faceting)
        self.grid.setVisible(faceting)

        if not faceting:
            # Drop the facets rather than hiding them: each owns a pyqtgraph
            # scene, and a dozen of those behind a hidden widget is memory
            # that accumulates on every switch.
            self.grid.clear()
            self._update_page_label()
            self.scatter.set_points(
                coords,
                visible,
                point_colours,
                groups=groups,
                point_size=self.size_slider.value(),
                opacity=self.opacity_slider.value() / 100.0,
                symbols=symbols,
                legend_entries=legend_entries,
                reset_view=reset_view,
            )
            if self.density_box.isChecked():
                self.scatter.set_density(True)
            return

        # Faceted: group by the chosen column over the points on screen.
        values = self.frame.iloc[visible][self._group_column].astype(str).to_numpy()
        groups_list, skipped = build_groups(values, max_groups=MAX_FACET_GROUPS)
        self.grid.shared_axes = self.shared_axes_box.isChecked()
        self.grid.columns = self.grid_columns_box.currentData() or 0
        page = self.grid.page
        self.grid.set_groups(groups_list, sort=self.grid_sort_box.currentData())
        self.grid.set_page(page)
        # Colouring by the same field the grid is split on makes every facet
        # monochrome -- the facet's own title already says which category it
        # is, so a legend here would repeat that rather than add anything.
        # Any other colour-by field is exactly the case the legend exists
        # for: colour is carrying information the facet titles do not.
        strip_entries = (
            legend_entries if self.colour_box.currentData() != self._group_column else None
        )
        self.grid.render(
            np.asarray(coords),
            np.asarray(visible),
            list(point_colours),
            point_size=self.size_slider.value(),
            opacity=self.opacity_slider.value() / 100.0,
            symbols=symbols,
            selected=self.scatter.selected_rows,
            background=self.background_box.currentData(),
            legend_entries=strip_entries,
        )
        self._update_page_label()
        if skipped:
            self.status.emit(
                f"showing the {MAX_FACET_GROUPS} largest groups; {skipped} smaller ones hidden"
            )

    def _symbols_for(self, visible) -> list[str] | None:
        """Per-point symbols for the shape-by field, or None."""
        column = self.shape_box.currentData()
        if not column or self.frame is None or column not in self.frame.columns:
            return None
        values = self.frame.iloc[visible][column].astype(str).to_numpy()
        # Same stability requirement as colour: assign from the full column,
        # not from whatever happens to be visible, so a shape never shifts
        # when a filter or facet page changes what is on screen.
        universe = self._colour_universe(column)
        mapping, overflow = shapes.assign(universe)
        if overflow:
            self.shape_hint.setText(shapes.describe_overflow(overflow, len(universe)))
            self.shape_hint.show()
        else:
            self.shape_hint.hide()
        return [mapping[v] for v in values]

    # -- selection --------------------------------------------------------

    def _on_density_toggled(self, enabled: bool) -> None:
        self.scatter.set_density(enabled)
        # A legend of colours means nothing when colour encodes density.
        self.legend_box.setEnabled(not enabled)
        self.colour_box.setEnabled(not enabled)
        self.status.emit("density" if enabled else "points")

    def _on_lasso_toggled(self, enabled: bool) -> None:
        self.scatter.set_lasso(enabled)
        self.cluster_panel.set_lasso_active(enabled)
        if enabled:
            self.status.emit("click to start an outline, click again to close it")

    def _register_entry(self, dataset, frame, *, source=SOURCE_EXPORT) -> None:
        """Add the just-loaded dataset to the workspace and make it current.

        Loading the same directory twice is allowed on purpose -- an export's
        own vectors and descriptors computed from the same images are two
        embeddings of one dataset, and comparing them is exactly the kind of
        question the workspace exists for.
        """
        info = dict(dataset.run_info) if isinstance(dataset.run_info, dict) else {}
        entry = EmbeddingEntry(
            name=dataset.name,
            dataset=dataset,
            frame=frame,
            source=source,
            info=info,
        )
        entry.resolver = self.resolver
        self._add_entry(entry)

    def _add_entry(self, entry: EmbeddingEntry) -> None:
        """Add an already-built entry to the workspace and make it current.

        The lower-level primitive _register_entry and the joint-projection
        path both fall through to this, so there is exactly one place an
        entry actually joins the workspace.

        Workspace.add() makes the new entry current in the WORKSPACE, but
        self._active_key is the explorer's own mirror of that and does not
        follow automatically. Some callers (_load_embedding_directory) go on
        to set self.dataset/self.frame/self._active_key themselves right
        after this returns, in which case this is a harmless no-op via
        _switch_to's own "already active" guard; callers that do NOT
        (_combine_embeddings used to be one, and a plain _add_entry call from
        anywhere else would silently be another) previously left the mirror
        pointing at the PREVIOUS entry -- verified: adding a third embedding
        while a different one was being viewed left self._active_key on the
        one being viewed instead of the one just added, so closing what
        LOOKED like a non-active row actually closed the one truly active.
        """
        self.workspace.add(entry)
        self._sync_open_box()
        if self.workspace.current_key == entry.key and self._active_key != entry.key:
            self._switch_to(entry.key)
        # Which export is open is the Data section's whole summary, and this
        # is the one place that changes.
        self._update_section_summaries()

    def _sync_open_box(self) -> None:
        """Refresh the open-embeddings list from the workspace.

        Rebuilds the whole list rather than diffing -- see
        OpenEmbeddingsList.set_entries. Combining needs two REAL sources; a
        lone joint entry (its sources since closed) cannot be combined with
        anything new until another ordinary embedding is loaded alongside it,
        which is the same "more than one" condition the list itself uses to
        decide whether it is worth showing at all.
        """
        entries = self.workspace.entries
        self.open_list.set_entries(entries, self.workspace.current_key)
        self.combine_button.setVisible(len(entries) > 1)

    def _close_embedding(self, key: str) -> None:
        """Close one open embedding by key, from its row's own close button.

        Unlike the old single "close the current one" button, any row can be
        closed directly -- including one that is not currently active.
        """
        if len(self.workspace) <= 1:
            return
        closing_active = key == self.workspace.current_key
        self.workspace.remove(key)
        if closing_active:
            self._active_key = None
        self._sync_open_box()
        if self.workspace.current is not None:
            self._switch_to(self.workspace.current.key)

    def _switch_to(self, key: str) -> None:
        """Switch to another open embedding without reloading anything."""
        if key == self._active_key:
            return
        entry = self.workspace.get(key)
        if entry is None:
            return

        # Park the current selection with the entry it belongs to, so coming
        # back finds it intact and it never indexes into other points.
        if self._active_key:
            self.workspace.set_selection(
                self.scatter.selected_rows, self._active_key
            )

        self.workspace.set_current(key)
        self._active_key = key
        self.dataset = entry.dataset
        self.frame = entry.frame
        self.resolver = entry.resolver
        self._invalidate_paths()
        # Move the highlighted row, so the list's own visual state can never
        # lag behind whichever entry is actually active -- every caller of
        # _switch_to gets this for free rather than having to remember it.
        self.open_list.set_entries(self.workspace.entries, key)
        # Restore whatever was last on screen for this entry, so switching
        # back and forth between embeddings that have already been projected
        # is instant rather than forcing a recompute every time. Falls back
        # to None only for an entry that has never been projected at all.
        self.result = (
            entry.projections.get(entry.last_result_key)
            if entry.last_result_key
            else None
        )

        self.metadata_panel.set_frame(self.frame)
        self._rebuild_group_options()
        self._rebuild_shape_options()
        self._rebuild_colour_options()
        self._rebuild_palette_options()
        self._rebuild_filters()
        self.apply_suggested_params()
        self._update_points_hint()
        self._update_source_label()

        # A projection already computed for this entry is reused as it is;
        # otherwise the cache is consulted before anything is recomputed.
        self.scatter.set_selection(
            self.workspace.selection(key), notify=False
        )
        if self.result is not None:
            # Restored, not recomputed: draw it straight away rather than
            # leaving the plot hidden behind a "press Compute" message that
            # would be actively wrong -- the layout is already sitting there.
            self._set_message("")
            self._redraw(reset_view=True)
            self.status.emit(f"switched to {entry.label()} — restored last view")
        else:
            self._set_message(
                f"{entry.describe()}\n\nPress Compute projection to lay it out."
            )
            self.status.emit(f"switched to {entry.label()}")

    def _close_current_embedding(self) -> None:
        """Close whichever embedding is currently active.

        Kept as a convenience wrapper over _close_embedding(key) -- the row-
        level close button in OpenEmbeddingsList closes ANY row directly, not
        only the active one, which is the more general operation this now
        delegates to.
        """
        key = self.workspace.current_key
        if key:
            self._close_embedding(key)

    def _combine_embeddings(self) -> None:
        """Project several open embeddings together as one fit.

        Building the joint entry is cheap (concatenation, not a fit --
        measured ~76ms for two real 24k/36k x 1024 exports) and runs here on
        the GUI thread; the expensive part is the projection itself, which
        goes through the ordinary compute()/_ProjectionTask path exactly as
        for any other entry, so it is threaded and cancellable the same way.
        """
        # Only real, single-source entries can be combined -- combining a
        # joint entry again would silently duplicate its source vectors
        # (concatenating an entry that already contains A+B with a fresh A
        # would count A's points twice), which is not what "combine" means.
        candidates = [e for e in self.workspace.entries if e.source != SOURCE_JOINT]
        if len(candidates) < 2:
            QMessageBox.information(
                self,
                "Combine embeddings",
                "Need at least two non-combined embeddings open to combine. "
                "A joint embedding cannot itself be combined again.",
            )
            return

        from .combine_dialog import CombineEmbeddingsDialog

        dialog = CombineEmbeddingsDialog(candidates, parent=self)
        if dialog.exec() != CombineEmbeddingsDialog.DialogCode.Accepted:
            return
        chosen = dialog.selected_entries()
        if len(chosen) < 2:
            return

        from ..data.joint_projection import IncompatibleEmbeddings, make_joint_entry

        try:
            entry = make_joint_entry(chosen)
        except IncompatibleEmbeddings as exc:
            QMessageBox.warning(self, "Combine embeddings", str(exc))
            return
        except MemoryError:
            # A real possibility: several 30k x 1024 float32 exports
            # concatenated is a genuine amount of memory, and failing loudly
            # here is far better than a cryptic crash mid-fit.
            QMessageBox.warning(
                self,
                "Combine embeddings",
                "Not enough memory to combine these embeddings. Try "
                "combining fewer at once, or subsample after switching to "
                "the joint embedding.",
            )
            return

        self._add_entry(entry)
        # _switch_to directly, not setCurrentIndex: _add_entry -> Workspace.add
        # already made the joint entry current and _sync_open_box already
        # moved the combobox there WHILE SIGNALS WERE BLOCKED, so
        # setCurrentIndex would be asking for the index already showing and
        # Qt would not emit currentIndexChanged at all -- self.frame/
        # self.dataset would then silently keep pointing at the PREVIOUS
        # entry. Measured: this exact sequence left ex.frame at one source
        # entry's row count instead of the joint total.
        self._switch_to(entry.key)
        # t-SNE by default for a joint embedding: its emphasis on local
        # structure is usually what a direct cross-dataset comparison wants,
        # and it is not what apply_suggested_params (called by the switch
        # above) chooses on its own.
        self.method_box.setCurrentText(TSNE)
        # Dataset by default too, overriding whatever _switch_to kept from
        # the previously-viewed entry: the natural first question for a
        # BRAND NEW joint embedding is "where does each dataset land", not
        # whichever field happened to be selected before combining.
        dataset_index = self.colour_box.findData(DATASET_COLUMN)
        if dataset_index >= 0:
            self.colour_box.setCurrentIndex(dataset_index)
        self.status.emit(
            f"combined {len(chosen)} embeddings into {entry.n_points:,} points "
            f"-- press Compute projection"
        )

    def _on_tsne_changed(self) -> None:
        """A t-SNE parameter changed: the current layout is now stale.

        Deliberately does NOT recompute. A projection is seconds to minutes of
        work and the user may be setting several parameters; recomputing on
        each keystroke would make the panel unusable. The button says the
        result is out of date instead.
        """
        if self.method_box.currentText() == TSNE and self.result is not None:
            self.run_button.setText("Recompute projection")

    def _on_background_changed(self) -> None:
        mode = self.background_box.currentData()
        self.scatter.set_background(mode)
        for facet in self.grid.facets:
            facet.scatter.set_background(mode)
        # Suggest the categorical palette suited to the new ground -- most of
        # PLATO's default palette falls below readable contrast on white (see
        # palettes.categorical_for_background) -- but only while the user has
        # not picked a palette themselves. A deliberate choice must survive a
        # background change exactly as it survives everything else. Applied
        # with signals blocked and set_active_palette called directly, same
        # as _rebuild_palette_options's own programmatic updates: routing it
        # through _on_palette_changed would mark this suggestion itself as
        # the user's "explicit" choice and defeat the whole guard.
        if not self._palette_chosen_by_user and self.palette_box.currentData() in (
            None,
            *(p.key for p in palettes.of_kind(palettes.CATEGORICAL)),
        ):
            from ..gui import themes as _themes

            suggested = palettes.categorical_for_background(
                mode, theme_is_dark=_themes.is_dark()
            )
            index = self.palette_box.findData(suggested.key)
            if index >= 0 and index != self.palette_box.currentIndex():
                self.palette_box.blockSignals(True)
                self.palette_box.setCurrentIndex(index)
                self.palette_box.blockSignals(False)
                set_active_palette(suggested.key)
        self._redraw()

    def _on_palette_changed(self) -> None:
        """Switch the active palette and repaint.

        Only reached by a real signal emission -- a user pick, since every
        programmatic change to palette_box's current index in this file goes
        through blockSignals. Safe to treat unconditionally as "the user
        chose this" and stop overriding it on background changes.
        """
        key = self.palette_box.currentData()
        if key:
            set_active_palette(key)
        self._palette_chosen_by_user = True
        self._redraw()

    def _rebuild_palette_options(self) -> None:
        """Offer the palettes that suit the current colour variable.

        Categorical and numeric colouring want different scales, and offering
        viridis for a gene list (or Okabe-Ito for entropy) is offering the
        wrong tool. The list follows what colour is currently encoding.
        """
        numeric = self._stat_column is not None
        kinds = (
            (palettes.SEQUENTIAL, palettes.DIVERGING)
            if numeric
            else (palettes.CATEGORICAL,)
        )
        previous = self.palette_box.currentData()
        self.palette_box.blockSignals(True)
        self.palette_box.clear()
        for kind in kinds:
            for palette in palettes.of_kind(kind):
                self.palette_box.addItem(palette.name, palette.key)
                index = self.palette_box.count() - 1
                self.palette_box.setItemData(
                    index, palette.description, Qt.ItemDataRole.ToolTipRole
                )
        index = self.palette_box.findData(previous)
        if index < 0:
            if numeric or self._palette_chosen_by_user:
                default_key = palettes.default_for(kinds[0]).key
            else:
                # No remembered choice and the user has never overridden the
                # palette: pick the one suited to the current background
                # rather than unconditionally falling back to the dark-tuned
                # default -- same reasoning as _on_background_changed.
                from ..gui import themes as _themes

                default_key = palettes.categorical_for_background(
                    self.background_box.currentData(), theme_is_dark=_themes.is_dark()
                ).key
            index = self.palette_box.findData(default_key)
        self.palette_box.setCurrentIndex(max(0, index))
        self.palette_box.blockSignals(False)
        key = self.palette_box.currentData()
        if key:
            set_active_palette(key)

    def _rebuild_shape_options(self) -> None:
        """Offer low-cardinality categoricals for shape."""
        previous = self.shape_box.currentData()
        self.shape_box.blockSignals(True)
        self.shape_box.clear()
        self.shape_box.addItem("None (all circles)", None)
        if self.frame is not None:
            for column in categorical_fields(
                self.frame, max_values=shapes.MAX_SHAPES * 3
            ):
                self.shape_box.addItem(field_label(column), column)
        index = self.shape_box.findData(previous)
        self.shape_box.setCurrentIndex(max(0, index))
        self.shape_box.blockSignals(False)

    def _on_group_changed(self) -> None:
        """Switch between one plot and a facet per group."""
        self._group_column = self.group_box.currentData()
        faceting = bool(self._group_column)
        for widget in (
            self.shared_axes_box,
            self.grid_columns_box,
            self.grid_sort_box,
            self.grid_page_widget,
        ):
            widget.setEnabled(faceting)
        self._redraw(reset_view=True)

    def _on_grey_out_toggled(self, enabled: bool) -> None:
        self.scatter.set_grey_out(enabled)
        for facet in self.grid.facets:
            facet.scatter.set_grey_out(enabled)
        self._redraw()

    def _step_page(self, delta: int) -> None:
        self.grid.set_page(self.grid.page + delta)
        self._redraw()

    def _update_page_label(self) -> None:
        pages = self.grid.n_pages
        if pages <= 1:
            self.grid_page_label.setText(
                f"{self.grid.n_groups} group{'s' if self.grid.n_groups != 1 else ''}"
            )
            self.grid_prev.setEnabled(False)
            self.grid_next.setEnabled(False)
            return
        self.grid_page_label.setText(
            f"page {self.grid.page + 1} / {pages} · {self.grid.n_groups} groups"
        )
        self.grid_prev.setEnabled(self.grid.page > 0)
        self.grid_next.setEnabled(self.grid.page < pages - 1)

    def _rebuild_group_options(self) -> None:
        """Offer every categorical column the frame actually has."""
        previous = self.group_box.currentData()
        self.group_box.blockSignals(True)
        self.group_box.clear()
        self.group_box.addItem("Off (single plot)", None)
        if self.frame is not None:
            for column in categorical_fields(self.frame, max_values=MAX_FACET_GROUPS):
                self.group_box.addItem(field_label(column), column)
        index = self.group_box.findData(previous)
        self.group_box.setCurrentIndex(max(0, index))
        self.group_box.blockSignals(False)
        self._group_column = self.group_box.currentData()

    def set_image_mode(self, enabled: bool) -> None:
        """Turn image tracing on or off for the whole explorer.

        Off is a real optimisation, not a hidden panel: ``_path_for`` returns
        None immediately, so nothing resolves a path or reads a file, and no
        background hunt for an image root is started when a dataset loads.
        On a network share that hunt is ~0.3 s per candidate directory and the
        reads are hundreds of megabytes, so this is the difference between a
        metadata session that opens instantly and one that does not.
        """
        enabled = bool(enabled)
        if enabled == self._image_mode:
            return
        self._image_mode = enabled

        if self.image_mode_box.isChecked() != enabled:
            self.image_mode_box.blockSignals(True)
            self.image_mode_box.setChecked(enabled)
            self.image_mode_box.blockSignals(False)

        self.selection_panel.setVisible(enabled)
        self.metadata_panel.setVisible(not enabled)
        self.source_button.setEnabled(enabled)
        self.stats_panel.setEnabled(enabled)

        rows = self.scatter.selected_rows
        if enabled:
            # Coming back on: the root was never hunted while off, so start
            # now rather than leaving previews permanently "not found".
            if self.frame is not None and self.resolver is None:
                self._invalidate_paths()
                self.resolver = self._start_resolving_images(self.frame)
            self.selection_panel.set_rows(rows)
            self.status.emit("image viewer on")
        else:
            self.metadata_panel.set_rows(rows)
            self.status.emit("image viewer off — embedding and metadata only")
        self._update_source_label()

    @property
    def image_mode(self) -> bool:
        return self._image_mode

    def _compute_image_stats(self) -> None:
        """Measure every image in the dataset, on the thread pool."""
        if self.frame is None or self.dataset is None:
            return
        if self.resolver is None:
            QMessageBox.information(
                self,
                "Image statistics",
                "The images for this dataset have not been located yet.\n\n"
                "Use Locate… to point PLATO at the folder holding them, then "
                "try again.",
            )
            return

        from ..data.image_stats import StatsCache
        from . import stats_worker

        n_rows = len(self.frame)
        cache = StatsCache(self.work_dir / "projections")
        fingerprint = self.dataset.fingerprint()
        cached = cache.load(fingerprint, n_rows)
        if cached is not None:
            self._apply_image_stats(cached, from_cache=True)
            return

        # Resolving a path is a filesystem stat per row; done here on the GUI
        # thread it would freeze for as long as the walk takes, so it happens
        # inside the same pass as the reads.
        self.stats_panel.set_running(True)
        self.status.emit(f"measuring {n_rows:,} images…")

        rows = list(range(n_rows))
        paths = [self._path_for(row) for row in rows]

        signals = stats_worker.StatsSignals()
        signals.progress.connect(self.stats_panel.set_progress)
        signals.finished.connect(self._on_stats_finished)
        signals.failed.connect(self._on_stats_failed)
        self._stats_signals = signals  # keep alive for the run
        self._stats_run = stats_worker.start(
            QThreadPool.globalInstance(), rows, paths, n_rows, signals
        )

    def _cancel_image_stats(self) -> None:
        if self._stats_run is not None:
            self._stats_run.cancel()
        self.stats_panel.set_running(False)
        self.status.emit("image statistics cancelled")

    def _on_stats_finished(self, values) -> None:
        self.stats_panel.set_running(False)
        if self.dataset is not None:
            from ..data.image_stats import StatsCache

            try:
                StatsCache(self.work_dir / "projections").save(
                    self.dataset.fingerprint(), values
                )
            except OSError:
                # A cache that cannot be written costs a recompute, nothing more.
                pass
        self._apply_image_stats(values, from_cache=False)

    def _on_stats_failed(self, message: str) -> None:
        self.stats_panel.set_running(False)
        self.status.emit(f"image statistics failed: {message}")

    def _apply_image_stats(self, values, *, from_cache: bool) -> None:
        """Attach measured statistics to the frame and offer them for colour."""
        self._image_stats = values
        first = next(iter(values.values())) if values else np.empty(0)
        measured = int(np.isfinite(first).sum()) if len(first) else 0
        total = len(self.frame) if self.frame is not None else 0
        self.stats_panel.set_available(measured, total)
        origin = "cached" if from_cache else "measured"
        self.status.emit(f"image statistics {origin}: {measured:,} of {total:,} images")

    def _on_stat_selected(self, name) -> None:
        """Colour by a statistic, or hand colouring back to the metadata."""
        self._stat_column = name
        # Colouring by a continuous statistic and by a category are mutually
        # exclusive; disable the other control rather than letting one
        # silently win.
        self.colour_box.setEnabled(name is None)
        self._redraw()

    def _stat_values(self, rows) -> np.ndarray | None:
        """The selected statistic for ``rows``, or None if not applicable."""
        if self._stat_column is None or self._image_stats is None:
            return None
        array = self._image_stats.get(self._stat_column)
        if array is None:
            return None
        return np.asarray(array)[rows]

    def _on_lasso_additive(self, additive: bool) -> None:
        """Whether the next lasso adds to the selection or replaces it."""
        self.scatter.set_lasso(self.scatter.lasso_enabled, additive=additive)

    def _on_selection(self, rows) -> None:
        """One place where a selection change lands, whatever caused it.

        Click, shift-click, lasso and removing an entry all arrive here, so
        the plot rings, the preview column and the cluster breakdown can never
        disagree about what is selected.
        """
        rows = np.asarray(rows, dtype=np.int64).ravel() if rows is not None else np.empty(0, np.int64)
        if self._image_mode:
            self.selection_panel.set_rows(rows)
        else:
            self.metadata_panel.set_rows(rows)
        self._update_composition(rows)
        self._update_section_summaries()
        if len(rows):
            self.status.emit(f"{len(rows):,} point{'s' if len(rows) != 1 else ''} selected")

    def _update_composition(self, rows) -> None:
        """Break the selection down for the Lasso Analysis panel.

        Only while the lasso tool is active. points_selected fires for a
        plain click and a shift-click too, and running the full composition
        breakdown on a 1-5 point selection produced nonsense like "60% of
        the selection" for 3 points -- a comparison the panel exists to make
        about a REGION, not about whichever points happen to be clicked.
        """
        if self.frame is None:
            return
        if not self.scatter.lasso_enabled:
            return
        from ..data.cluster_stats import compose

        self.cluster_panel.show_composition(compose(self.frame, rows))

    def _on_point_clicked(self, row_index: int) -> None:
        """A single click on the plot: show that point, do NOT open it."""
        self.selection_panel.set_current(row_index)

    def _on_entry_clicked(self, row_index: int) -> None:
        """A click in the preview column: highlight that point in the plot."""
        self.scatter.set_hover_marker(row_index)

    def _on_panel_edited(self, rows) -> None:
        """The panel removed a point, or cleared the selection."""
        self.scatter.set_selection(np.asarray(list(rows), dtype=np.int64))

    def _show_selection_images(self) -> None:
        """Bring the lasso's points into the preview column."""
        rows = self.scatter.selected_rows
        if len(rows) == 0:
            return
        self.selection_panel.set_rows(rows)
        # A lasso can hold thousands of points; the panel builds them a page
        # at a time, so say what is actually being shown.
        self.status.emit(f"listing {len(rows):,} selected images")

    def _apply_preview_width(self) -> None:
        """Scale the preview images to the column's current width."""
        sizes = self.splitter.sizes()
        if len(sizes) >= 3:
            self.selection_panel.set_image_width(sizes[2])

    def showEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt override
        """Size the previews to the column the first time it is on screen.

        Splitter sizes are not final until the widget has been laid out, so a
        width read during construction is the placeholder rather than the real
        one; without this the first previews render at the default size no
        matter how wide the column actually is.
        """
        super().showEvent(event)
        self._preview_resize_timer.start()

    def compare_selection(self) -> None:
        """Open every selected point side by side in the comparison view.

        Reuses the browser's ComparisonView rather than a second
        implementation: one column per selected point, each holding that one
        image, sharing the locked zoom/pan viewport that makes columns
        genuinely comparable.
        """
        rows = [int(r) for r in self.selection_panel.rows]
        if len(rows) < 2:
            QMessageBox.information(
                self,
                "Compare",
                "Select at least two points to compare.\n\n"
                "Click a point, then shift-click others — or lasso a region "
                "and press Show images.",
            )
            return

        from .compare_view import ComparisonColumn, ComparisonView

        columns = []
        missing = 0
        for row_index in rows[:MAX_COMPARE_COLUMNS]:
            image_row = self._image_row(row_index)
            if image_row is None:
                missing += 1
                continue
            columns.append(
                ComparisonColumn(
                    self._compare_title(row_index),
                    [image_row],
                    caption=self._compare_caption(row_index),
                )
            )
        if not columns:
            QMessageBox.warning(
                self, "Compare", "None of the selected points have an image on disk."
            )
            return

        window = QWidget(self)
        window.setWindowFlag(Qt.WindowType.Window, True)
        window.setWindowTitle(f"Compare — {len(columns)} images")
        window.resize(min(1800, 380 * len(columns) + 120), 900)

        view = ComparisonView(window)
        view.status.connect(self.status.emit)
        view.set_columns(columns)
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(view)
        window.setLayout(layout)
        window.show()
        # Held so Python does not garbage-collect the window immediately.
        self._windows.append(window)

        note = f"comparing {len(columns)} images"
        if missing:
            note += f" ({missing} had no image on disk)"
        if len(rows) > MAX_COMPARE_COLUMNS:
            note += f" — first {MAX_COMPARE_COLUMNS} of {len(rows):,}"
        self.status.emit(note)

    def _compare_title(self, row_index: int) -> str:
        """A short heading for one comparison column."""
        if self.frame is None:
            return str(row_index)
        record = self.frame.iloc[row_index]
        condition = str(record.get(CONDITION, "") or "")
        return condition or str(record.get(IMAGE_NAME, "") or f"row {row_index}")

    def _compare_caption(self, row_index: int) -> str:
        """Where this image came from, under its column.

        Built from whichever identifying fields the dataset has rather than a
        fixed set, so an antibiotic plate and a CRISPRi plate each get a
        caption made of what they actually carry.
        """
        if self.frame is None:
            return ""
        record = self.frame.iloc[row_index]
        parts = [
            str(record.get(column, "") or "")
            for column in (PLATE, WELL, DRUG, CONCENTRATION, GENE, GUIDE)
        ]
        return " · ".join(p for p in parts if p)

    def _image_row(self, row_index: int) -> ImageRow | None:
        """One frame row as an ImageRow, or None if its file is missing."""
        path = self._path_for(row_index)
        if path is None or self.frame is None:
            return None
        record = self.frame.iloc[row_index]
        return ImageRow(
            image_id=str(record.get(IMAGE_NAME, "")),
            path=str(path),
            plate=str(record.get(PLATE, "")),
            well=str(record.get(WELL, "")),
            field=None,
            channel=None,
            metadata={
                field_label(c): record.get(c, "")
                for c in (GENE, GUIDE, DRUG, CONCENTRATION, MOA, PATHWAY, ROLE)
                if str(record.get(c, "") or "")
            },
            rating=None,
            flagged=False,
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
        """Where this point's image is, or None.

        The single chokepoint for image resolution, so the mode switch is
        enforced in exactly one place: with the image viewer off nothing below
        this line runs, however it was reached.

        Memoised, because resolving is a filesystem stat and the same row is
        asked for by several panels at once -- measured at two lookups per
        selected point before this cache. Cleared whenever the resolver or the
        frame changes, since both change what the answer is.
        """
        if not self._image_mode or self.resolver is None or self.frame is None:
            return None
        row_index = int(row_index)
        cached = self._path_cache.get(row_index, _MISSING)
        if cached is not _MISSING:
            return cached
        path = self.resolver.path_for(self.frame.iloc[row_index])
        self._path_cache[row_index] = path
        return path

    def _invalidate_paths(self) -> None:
        """Forget resolved paths. Call when the resolver or frame changes."""
        self._path_cache.clear()

    def _on_hover(self, row_index: int) -> None:
        """Hovering names the point in the status bar; it no longer loads it.

        The right-hand column belongs to the SELECTION now. Letting the cursor
        overwrite it would mean the images you deliberately chose disappear
        the moment you move the mouse across the plot -- which is exactly what
        makes a preview-on-hover panel useless for comparing.
        """
        if row_index < 0 or self.frame is None:
            return
        condition = str(self.frame.iloc[row_index].get(CONDITION, "") or "")
        if condition:
            self.status.emit(condition)

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
        faceting = bool(self._group_column)
        suggested = _suggested_export_name(
            dataset=self.dataset_box.currentText(),
            method=method,
            column=column,
            group_column=self._group_column,
            fmt=fmt,
            stamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        filters = "PNG image (*.png)" if fmt == "png" else "SVG vector (*.svg)"
        target, _ = QFileDialog.getSaveFileName(self, f"Export {fmt.upper()}", suggested, filters)
        if not target:
            return
        if not target.lower().endswith(f".{fmt}"):
            target = f"{target}.{fmt}"
        try:
            # Whichever view is actually on screen -- the single plot has its
            # own scene, the grid is several, composed differently. Exporting
            # the wrong one is what makes a saved file not match what a user
            # was just looking at.
            if faceting:
                self.grid.export(target)
            else:
                self.scatter.export(target)
        except Exception as exc:  # noqa: BLE001 - report rather than crash
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        self.status.emit(f"exported {target}")

    # -- cache --------------------------------------------------------------

    def clear_projection_cache(self) -> None:
        """Delete every cached t-SNE/UMAP layout, on disk and in memory.

        A projection of 30k points can take minutes, so this asks first --
        the cache exists specifically to avoid paying that cost twice, and
        clearing it is only useful after a code or data change makes the old
        layouts suspect, not something to do by accident. Reached from the
        Help menu because it is a whole-cache operation, not a per-embedding
        one: one ProjectionCache backs every entry in this workspace (see
        __init__), so there is nothing finer to target.
        """
        on_disk = self.cache.entries()
        in_memory = sum(len(e.projections) for e in self.workspace.entries)
        if not on_disk and not in_memory:
            QMessageBox.information(
                self, "Clear cached projections", "No cached projections to clear."
            )
            return

        reply = QMessageBox.question(
            self,
            "Clear cached projections",
            f"Delete {len(on_disk)} cached projection(s) from disk?\n\n"
            "Any embedding shown from cache right now will need to be "
            "recomputed the next time you view it, which can take minutes "
            "for a large dataset.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        removed = self.cache.clear()
        for entry in self.workspace.entries:
            entry.projections.clear()
            entry.last_result_key = None
        # The plot on screen right now came from one of those projections --
        # leaving it up would show a result that no longer exists anywhere,
        # silently surviving until the next redraw picks a different one.
        self.result = None
        self._set_message("Cache cleared. Press Compute projection to lay it out again.")
        self.status.emit(f"cleared {removed} cached projection(s)")
