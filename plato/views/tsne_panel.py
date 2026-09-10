"""t-SNE's parameters, grouped, with the caveats attached to the controls.

t-SNE is unusually easy to misread, and the standard failures are all
parameter failures: a single perplexity read as if it were the answer,
a run stopped before convergence and its filaments read as structure, cluster
sizes and inter-cluster distances read as meaningful when neither is.

So the panel does two things beyond exposing values. It groups them by what
they do, and each control's tooltip says what going wrong looks like -- which
is the part a number alone cannot convey.

The three facts worth internalising, from the Distill article
(distill.pub/2016/misread-tsne):

* **Cluster sizes mean nothing.** t-SNE expands dense regions and contracts
  sparse ones, so a big blob is not a more variable group.
* **Distances between clusters mostly mean nothing** -- particularly at low
  perplexity. Well-separated groups can land anywhere relative to each other.
* **Perplexity changes the answer**, so looking at one value is not a result.
  Two or three is the minimum honest reading.

None of that is fixable by better defaults, which is why it is written into
the UI instead.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..data.projection import TSNE_PRESETS, tsne_preset
from ..gui import themes

CUSTOM = "Custom"

# Short labels: the panel lives in a sidebar, and a combo wide enough for a
# sentence forces the whole column wider than the plot can spare. The full
# explanation is on the control's tooltip.
INITIALIZATIONS = (
    ("pca", "PCA"),
    ("random", "Random"),
    ("spectral", "Spectral"),
)

METRICS = ("cosine", "euclidean", "manhattan", "correlation", "chebyshev")


class TsnePanel(QWidget):
    """Algorithm parameters for t-SNE, grouped by what they control."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # -- quality / iterations
        self.preset_box = QComboBox()
        for name, _iterations, _early in TSNE_PRESETS:
            self.preset_box.addItem(name)
        self.preset_box.addItem(CUSTOM)
        self.preset_box.setCurrentText("Standard")
        self.preset_box.setToolTip(
            "How long to run. A fast look is not a converged layout: an "
            "unconverged t-SNE shows pinched, filament-shaped clusters that "
            "are an artefact of where it stopped, not structure in the data."
        )
        self.preset_box.currentTextChanged.connect(self._on_preset)

        self.iterations = QSpinBox()
        # No arbitrary ceiling: openTSNE has none, and a large dataset can
        # legitimately want several thousand iterations.
        self.iterations.setRange(50, 100_000)
        self.iterations.setSingleStep(250)
        self.iterations.setValue(500)
        self.iterations.setToolTip(
            "Gradient-descent steps after early exaggeration.\n"
            "Too few and the layout has not settled. There is no upper limit "
            "here beyond what you are willing to wait for."
        )
        self.iterations.valueChanged.connect(self._on_custom)

        self.early_iterations = QSpinBox()
        self.early_iterations.setRange(0, 10_000)
        self.early_iterations.setSingleStep(50)
        self.early_iterations.setValue(250)
        self.early_iterations.setToolTip(
            "Steps spent in the early-exaggeration phase, which lets clusters "
            "separate before fine structure is fitted."
        )
        self.early_iterations.valueChanged.connect(self._on_custom)

        # -- neighbourhood
        self.perplexity = QDoubleSpinBox()
        self.perplexity.setRange(2.0, 5000.0)
        self.perplexity.setDecimals(1)
        self.perplexity.setValue(30.0)
        self.perplexity.setToolTip(
            "Roughly, how many neighbours each point is fitted against.\n\n"
            "This changes the answer, so one value is not a result. Low "
            "perplexity fragments real clusters into sub-blobs; high "
            "perplexity merges distinct ones. Compare two or three values "
            "before believing a cluster.\n\n"
            "Automatically capped below n/3, which openTSNE requires."
        )
        self.perplexity.valueChanged.connect(self.changed.emit)

        self.metric_box = QComboBox()
        for metric in METRICS:
            self.metric_box.addItem(metric)
        self.metric_box.setToolTip(
            "Distance used for the neighbour search.\n"
            "Cosine suits learned embeddings; after L2-normalisation it is "
            "computed as Euclidean, which is equivalent and faster."
        )
        self.metric_box.currentTextChanged.connect(self.changed.emit)

        # -- optimisation
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.0, 100_000.0)
        self.learning_rate.setDecimals(1)
        self.learning_rate.setValue(0.0)
        self.learning_rate.setSpecialValueText("auto")
        self.learning_rate.setToolTip(
            "0 = auto (n / early exaggeration), which is almost always right.\n"
            "Too low and the layout barely moves; too high and it scatters."
        )
        self.learning_rate.valueChanged.connect(self.changed.emit)

        self.early_exaggeration = QDoubleSpinBox()
        self.early_exaggeration.setRange(1.0, 100.0)
        self.early_exaggeration.setDecimals(1)
        self.early_exaggeration.setValue(12.0)
        self.early_exaggeration.setToolTip(
            "How strongly attraction is inflated during the early phase.\n"
            "Too little and clusters stay entangled; too much and they "
            "collapse to points."
        )
        self.early_exaggeration.valueChanged.connect(self.changed.emit)

        self.late_exaggeration = QDoubleSpinBox()
        self.late_exaggeration.setRange(0.0, 100.0)
        self.late_exaggeration.setDecimals(1)
        self.late_exaggeration.setValue(0.0)
        self.late_exaggeration.setSpecialValueText("off")
        self.late_exaggeration.setToolTip(
            "Exaggeration during the main phase. Off by default.\n"
            "Above 1 it tightens clusters and shrinks the whole layout — "
            "measured here: the cluster structure is preserved (ARI 0.94 "
            "against the truth) while the extent falls roughly 10x. It "
            "changes the picture, not the grouping."
        )
        self.late_exaggeration.valueChanged.connect(self.changed.emit)

        self.initialization = QComboBox()
        for key, label in INITIALIZATIONS:
            self.initialization.addItem(label, key)
        self.initialization.setToolTip(
            "Where the layout starts.\n"
            "PCA is reproducible and preserves some global structure, so "
            "distances between well-separated clusters carry a little "
            "meaning. Random is the classic default and shows local "
            "structure only."
        )
        self.initialization.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.initialization.setMinimumContentsLength(8)
        self.initialization.currentIndexChanged.connect(self.changed.emit)

        # -- reproducibility
        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.seed.setValue(1)
        self.seed.setToolTip(
            "Same seed and same parameters give exactly the same layout.\n"
            "Change it to check whether a cluster is stable or an artefact of "
            "one initialisation."
        )
        self.seed.valueChanged.connect(self.changed.emit)

        self.reseed_button = QPushButton("New seed")
        self.reseed_button.setToolTip(
            "Draw a different seed, to see whether the structure survives."
        )
        self.reseed_button.clicked.connect(self._reseed)

        seed_row = QHBoxLayout()
        seed_row.setContentsMargins(0, 0, 0, 0)
        seed_row.addWidget(self.seed, 1)
        seed_row.addWidget(self.reseed_button)
        seed_widget = QWidget()
        seed_widget.setLayout(seed_row)

        # Labels ABOVE their fields, not beside them. This panel sits inside a
        # group box, inside another group box, inside a scrolling sidebar, and
        # each level takes margin -- at which point a side label has so little
        # width left that Qt clips it off the left edge entirely. Wrapping is
        # what makes the panel fit any sidebar width.
        quality = QFormLayout()
        quality.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        quality.setContentsMargins(6, 4, 6, 4)
        quality.addRow("Quality", self.preset_box)
        quality.addRow("Iterations", self.iterations)
        quality.addRow("Early phase", self.early_iterations)
        quality_group = QGroupBox("Run length")
        quality_group.setLayout(quality)

        neighbourhood = QFormLayout()
        neighbourhood.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        neighbourhood.setContentsMargins(6, 4, 6, 4)
        neighbourhood.addRow("Perplexity", self.perplexity)
        neighbourhood.addRow("Metric", self.metric_box)
        neighbourhood_group = QGroupBox("Neighbourhood")
        neighbourhood_group.setLayout(neighbourhood)

        optimisation = QFormLayout()
        optimisation.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        optimisation.setContentsMargins(6, 4, 6, 4)
        optimisation.addRow("Learning rate", self.learning_rate)
        optimisation.addRow("Early exagg.", self.early_exaggeration)
        optimisation.addRow("Late exagg.", self.late_exaggeration)
        optimisation.addRow("Initialisation", self.initialization)
        optimisation_group = QGroupBox("Optimisation")
        optimisation_group.setLayout(optimisation)

        reproducibility = QFormLayout()
        reproducibility.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        reproducibility.setContentsMargins(6, 4, 6, 4)
        reproducibility.addRow("Seed", seed_widget)
        reproducibility_group = QGroupBox("Reproducibility")
        reproducibility_group.setLayout(reproducibility)

        self.caveat = QLabel(
            "In a t-SNE plot, cluster <b>sizes</b> and the <b>distances "
            "between</b> clusters are mostly artefacts. Only which points sit "
            "together is meaningful — and that changes with perplexity, so "
            "compare a couple of values."
        )
        self.caveat.setWordWrap(True)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(quality_group)
        layout.addWidget(neighbourhood_group)
        layout.addWidget(optimisation_group)
        layout.addWidget(reproducibility_group)
        layout.addWidget(self.caveat)
        self.setLayout(layout)
        self.restyle()

    def restyle(self) -> None:
        colours = themes.current()
        self.caveat.setStyleSheet(
            f"color: {colours.text_faint}; font-size: 10px; padding: 2px;"
        )

    # -- presets -----------------------------------------------------------

    def _on_preset(self, name: str) -> None:
        values = tsne_preset(name)
        if values is None:
            return
        iterations, early = values
        for box, value in (
            (self.iterations, iterations),
            (self.early_iterations, early),
        ):
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)
        self.changed.emit()

    def _on_custom(self) -> None:
        """Typing an iteration count means the preset no longer describes it."""
        if tsne_preset(self.preset_box.currentText()) is not None:
            current = (self.iterations.value(), self.early_iterations.value())
            if current != tsne_preset(self.preset_box.currentText()):
                self.preset_box.blockSignals(True)
                self.preset_box.setCurrentText(CUSTOM)
                self.preset_box.blockSignals(False)
        self.changed.emit()

    def _reseed(self) -> None:
        import random

        self.seed.setValue(random.randint(1, 2**31 - 2))

    # -- values ------------------------------------------------------------

    def apply_to(self, params):
        """Copy the panel's values onto a ProjectionParams."""
        params.perplexity = float(self.perplexity.value())
        params.n_iter = int(self.iterations.value())
        params.early_exaggeration_iter = int(self.early_iterations.value())
        params.early_exaggeration = float(self.early_exaggeration.value())
        params.late_exaggeration = float(self.late_exaggeration.value())
        params.learning_rate = float(self.learning_rate.value())
        params.initialization = str(self.initialization.currentData())
        params.seed = int(self.seed.value())
        params.metric = str(self.metric_box.currentText())
        return params

    def load_from(self, params) -> None:
        """Show a ProjectionParams' values, without emitting changes."""
        pairs = (
            (self.perplexity, float(params.perplexity)),
            (self.iterations, int(params.n_iter)),
            (self.early_iterations, int(params.early_exaggeration_iter)),
            (self.early_exaggeration, float(params.early_exaggeration)),
            (self.late_exaggeration, float(params.late_exaggeration)),
            (self.learning_rate, float(params.learning_rate)),
            (self.seed, int(params.seed)),
        )
        for widget, value in pairs:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

        for widget, setter, value in (
            (self.initialization, "setCurrentIndex", params.initialization),
            (self.metric_box, "setCurrentText", params.metric),
        ):
            widget.blockSignals(True)
            if setter == "setCurrentIndex":
                index = widget.findData(value)
                widget.setCurrentIndex(max(0, index))
            else:
                widget.setCurrentText(value)
            widget.blockSignals(False)
