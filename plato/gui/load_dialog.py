"""Dialog used by both "Load Data" (first plate) and "Add Plate" (Nth plate).

Picks an image folder and a plate map file, tries the built-in filename
patterns from :mod:`detect`, falls back to an editable regex field if none of
them parse anything, then runs the real indexing/thumbnail pipeline
(:func:`build_index`, :func:`build_thumbnails`) exactly as the CLI does.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..cache import build_thumbnails
from ..config import (
    Config,
    GuiConfig,
    ImagesConfig,
    PlatemapConfig,
    ProjectConfig,
    ThumbnailsConfig,
)
from ..index import build_index
from ..index.platemap import plate_from_filename
from .detect import WELL_ONLY_PATTERN, compile_or_none, detect_pattern, detect_platemap_layout

FALLBACK_PATTERN = WELL_ONLY_PATTERN


class LoadPlateDialog(QDialog):
    """Collects an image folder + plate map, builds an in-memory Config."""

    def __init__(self, parent: QWidget | None = None, *, title: str = "Load data") -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(520)
        self.result_config: Config | None = None

        self.image_dir: Path | None = None
        self.platemap_path: Path | None = None

        self.image_dir_label = QLabel("(not selected)")
        pick_images = QPushButton("Choose image folder…")
        pick_images.clicked.connect(self._pick_image_dir)

        self.platemap_label = QLabel("(not selected)")
        pick_platemap = QPushButton("Choose plate map…")
        pick_platemap.clicked.connect(self._pick_platemap)

        self.pattern_edit = QLineEdit()
        self.pattern_edit.setPlaceholderText(
            r"e.g. ^(?P<well>[A-P]\d{1,2})\.tif$   (must capture 'well')"
        )
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Image folder", self.image_dir_label)
        form.addRow("", pick_images)
        form.addRow("Plate map", self.platemap_label)
        form.addRow("", pick_platemap)
        form.addRow("Filename pattern", self.pattern_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self.status_label)
        layout.addWidget(buttons)
        self.setLayout(layout)

    # -- pickers --------------------------------------------------------------

    def _pick_image_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not chosen:
            return
        self.image_dir = Path(chosen)
        self.image_dir_label.setText(str(self.image_dir))
        self._suggest_pattern()

    def _pick_platemap(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Select plate map", "", "Plate maps (*.csv *.tsv *.xlsx *.xls)"
        )
        if not chosen:
            return
        self.platemap_path = Path(chosen)
        self.platemap_label.setText(str(self.platemap_path))

    def _suggest_pattern(self) -> None:
        if self.image_dir is None:
            return
        guess = detect_pattern(self.image_dir)
        if guess is not None:
            self.pattern_edit.setText(guess.pattern)
            self.status_label.setText(f"detected pattern, matched {guess.n_parsed} files")
        else:
            self.pattern_edit.setText(FALLBACK_PATTERN)
            self.status_label.setText(
                "could not auto-detect a filename pattern — edit it above before continuing"
            )

    # -- accept -----------------------------------------------------------

    def _accept(self) -> None:
        if self.image_dir is None or self.platemap_path is None:
            QMessageBox.warning(self, "Load data", "Choose both an image folder and a plate map.")
            return
        pattern = self.pattern_edit.text().strip()
        if not compile_or_none(pattern):
            QMessageBox.warning(
                self, "Load data", "The filename pattern is invalid or missing the 'well' group."
            )
            return

        layout_guess = detect_platemap_layout(self.platemap_path)
        if layout_guess is None:
            QMessageBox.warning(
                self,
                "Load data",
                f"Could not read {self.platemap_path.name} as a plate map "
                "(tried a 'Well' column and a headerless grid). Check the file.",
            )
            return

        cfg = self._build_config(pattern, layout_guess)
        try:
            report = build_index(cfg, verbose=False)
            build_thumbnails(
                cfg.db_path,
                cfg.thumb_db_path,
                size=cfg.thumbnails.size,
                percentiles=cfg.thumbnails.percentiles,
                sample_size=cfg.thumbnails.sample_size,
                verbose=False,
            )
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            QMessageBox.critical(self, "Load data", f"Indexing failed:\n{exc}")
            return

        if not report.ok:
            proceed = QMessageBox.question(
                self,
                "Load data",
                f"{self.platemap_path.name} indexed with some mismatches "
                f"(see {cfg.report_path.name} for details). Load anyway?",
            )
            if proceed != QMessageBox.StandardButton.Yes:
                return

        self.result_config = cfg
        self.accept()

    def _build_config(self, pattern: str, layout_guess) -> Config:
        assert self.image_dir is not None and self.platemap_path is not None
        work_dir = self.image_dir / ".plato"
        plate_pattern = r"_(?P<plate>P\d+)\."
        default_plate = (
            plate_from_filename(self.platemap_path, plate_pattern) or self.image_dir.name
        )
        return Config(
            project=ProjectConfig(name=default_plate, work_dir=work_dir, plate_format=96),
            images=ImagesConfig(dir=self.image_dir, glob="**/*.tif*", pattern=pattern),
            platemap=PlatemapConfig(
                path=self.platemap_path,
                layout=layout_guess.layout,
                well_column=layout_guess.well_column,
                header_row=layout_guess.header_row,
                index_col=layout_guess.index_col,
                split_pattern=layout_guess.split_pattern,
                plate_pattern=plate_pattern,
                default_plate=default_plate,
            ),
            thumbnails=ThumbnailsConfig(),
            gui=GuiConfig(),
            source=None,
        )
