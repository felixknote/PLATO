"""Finding the image behind an embedding row, whatever the folder layout.

The previous approach was a list of guessed path templates -- ``{plate}/{name}``,
``{arm}/{plate_tail}/{name}`` and so on -- checked against fixed column names.
It broke every time a new export appeared, because every export is shaped
differently:

===========================  =========================================
export                       columns
===========================  =========================================
Apr26 CRISPRi & ABx          source, plate, well, label
Aug26 CRISPRi & ABx          experiment, plate ("ABx_P1"), well, label
Jul26 ABx / Jul26CRISPRi     plate, timepoint, plate_timepoint ("P1_T1")
===========================  =========================================

and every screen on disk is organised differently too: ``CRISPRi_P1/``,
``CRISPRi/P1/``, ``P1_T1/``, or everything in one folder.

So this module stops guessing. It indexes the file names actually present
under a root, once, and then matches rows to files by **file name** -- which
is the one thing every export does record (``image_name``) and the one thing
that does not depend on how anybody arranged their folders.

Where a name is ambiguous (the same ``WellA01_...Seq0000.tiff`` exists under
twenty plate folders, which is exactly what the timepoint exports look like),
the row's other metadata columns are used to choose between the candidates:
whichever candidate's path contains the row's plate, timepoint, or whatever
else that export happens to carry. No column names are assumed; every string
column is tried.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# Directories that never hold micrographs; skipping them keeps the scan quick
# and stops a cache or an export folder polluting the index.
SKIP_DIRECTORIES = {
    ".plato",
    "__pycache__",
    ".git",
    "_original",
    "Plate_Maps",
    "$RECYCLE.BIN",
    "System Volume Information",
    "analysis",
    "results",
}

# How deep to walk below the chosen root. A screen is usually root/plate/file
# or root/arm/plate/file; four levels covers those with room to spare without
# descending into a whole drive.
MAX_DEPTH = 4

# Stop after this many files. A screen is tens of thousands of images; a
# runaway scan of an entire drive is what this guards against.
MAX_FILES = 400_000


def _tokens(value: str) -> list[str]:
    """Split a metadata value into comparable tokens.

    "ABx_P1" -> ["abx", "p1"], so it matches a folder called either.
    """
    return [t for t in re.split(r"[^A-Za-z0-9]+", str(value).lower()) if t]


def _walk_into(
    root: Path, by_stem: dict[str, list[Path]], *, seen: int, max_files: int
) -> tuple[int, bool]:
    """Add every image file under ``root`` to ``by_stem``. Returns (seen, truncated)."""
    root_depth = len(root.parts)
    for current, directories, files in os.walk(root):
        here = Path(current)
        if len(here.parts) - root_depth >= MAX_DEPTH:
            directories[:] = []
        directories[:] = [
            d for d in directories if d not in SKIP_DIRECTORIES and not d.startswith(".")
        ]
        for name in files:
            stem, suffix = os.path.splitext(name)
            if suffix.lower() not in IMAGE_SUFFIXES:
                continue
            by_stem[stem].append(here / name)
            seen += 1
            if seen >= max_files:
                return seen, True
    return seen, False


@dataclass
class ImageIndex:
    """Every image file under a root, keyed by file name stem.

    Built once per root and reused: the scan is the expensive part on a
    network share, and the answer does not change while the app is open.

    Some exports split their images across more than one folder -- an
    experiment arm per folder, a plate per drive, an incremental delivery
    kept separate from the original -- so a single dataset's images need not
    live under one root. ``merged_with`` combines two indexes built this way
    into one; ``roots`` then lists every folder actually scanned, ``root``
    staying the first of them for callers that only show one path.
    """

    root: Path
    by_stem: dict[str, list[Path]] = field(default_factory=dict)
    truncated: bool = False
    roots: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.roots:
            self.roots = [self.root]

    @property
    def n_files(self) -> int:
        return sum(len(v) for v in self.by_stem.values())

    @classmethod
    def build(cls, root: Path, *, max_files: int = MAX_FILES) -> ImageIndex:
        root = Path(root)
        by_stem: dict[str, list[Path]] = defaultdict(list)
        seen, truncated = _walk_into(root, by_stem, seen=0, max_files=max_files)
        return cls(root=root, by_stem=dict(by_stem), truncated=truncated)

    def merged_with(self, other: ImageIndex) -> ImageIndex:
        """A new index covering this one's roots plus ``other``'s.

        Files are re-unioned by stem rather than concatenating lists blindly,
        so a name present under both (the same file copied, or a genuine
        clash) still ends up de-duplicated by path.
        """
        by_stem: dict[str, list[Path]] = defaultdict(list, {k: list(v) for k, v in self.by_stem.items()})
        for stem, paths in other.by_stem.items():
            existing = by_stem[stem]
            for path in paths:
                if path not in existing:
                    existing.append(path)
        roots = self.roots + [r for r in other.roots if r not in self.roots]
        return ImageIndex(
            root=self.root,
            by_stem=dict(by_stem),
            truncated=self.truncated or other.truncated,
            roots=roots,
        )

    # -- lookup -----------------------------------------------------------

    def candidates(self, stem: str) -> list[Path]:
        return self.by_stem.get(stem, [])

    def resolve(self, stem: str, hints: list[str]) -> Path | None:
        """The file for ``stem``, disambiguated by ``hints`` when needed.

        ``hints`` are the row's other metadata values (plate, timepoint,
        whatever the export carries). A name that occurs once needs none of
        them; a name that occurs under twenty plate folders is decided by
        which candidate's path contains the most of them.
        """
        found = self.by_stem.get(stem)
        if not found:
            return None
        if len(found) == 1:
            return found[0]

        wanted: list[str] = []
        for hint in hints:
            wanted.extend(_tokens(hint))
        if not wanted:
            return found[0]

        best: Path | None = None
        best_score = -1
        for path in found:
            # Only the part below the root varies; the root is common to all.
            try:
                relative = path.relative_to(self.root)
            except ValueError:
                relative = path
            haystack = set(_tokens(str(relative.parent)))
            score = sum(1 for token in wanted if token in haystack)
            if score > best_score:
                best, best_score = path, score
        # A tie at zero means no hint matched anything; that is not a match,
        # it is a coincidence of file names, and returning one at random would
        # show an image from the wrong experiment.
        return best if best_score > 0 else None


@dataclass
class LookupReport:
    """What happened when a root was tried against a frame."""

    root: Path
    n_files: int
    matched: int
    sampled: int
    truncated: bool = False

    @property
    def fraction(self) -> float:
        return self.matched / self.sampled if self.sampled else 0.0

    @property
    def ok(self) -> bool:
        return self.matched > 0

    def describe(self) -> str:
        if self.n_files == 0:
            return f"no image files found under {self.root}"
        if self.matched == 0:
            return (
                f"{self.n_files:,} images found under {self.root}, but none of "
                f"their names match this dataset"
            )
        return f"{self.matched}/{self.sampled} sampled rows resolved"


def hint_columns(frame: pd.DataFrame, *, exclude: tuple[str, ...] = ()) -> list[str]:
    """Columns worth using to disambiguate a repeated file name.

    Every string column except the file name itself and anything the caller
    excludes. No column names are assumed, because every export names things
    differently.
    """
    out = []
    for column in frame.columns:
        if column in exclude:
            continue
        values = frame[column].astype(str)
        distinct = values[values.ne("")].nunique()
        # A column with one value cannot separate anything; one with a value
        # per row (the file name) is not a folder hint.
        if 1 < distinct < max(2, len(frame) // 2):
            out.append(str(column))
    return out


def probe(
    index: ImageIndex,
    frame: pd.DataFrame,
    *,
    name_column: str,
    hints: list[str],
    samples: int = 24,
) -> LookupReport:
    """How well ``index`` resolves a sample of ``frame``'s rows."""
    if frame.empty:
        return LookupReport(index.root, index.n_files, 0, 0, index.truncated)
    sample = frame.sample(n=min(samples, len(frame)), random_state=0)
    matched = 0
    for _, row in sample.iterrows():
        stem = str(row.get(name_column) or "").strip()
        if not stem:
            continue
        if index.resolve(stem, [str(row.get(c, "")) for c in hints]) is not None:
            matched += 1
    return LookupReport(index.root, index.n_files, matched, len(sample), index.truncated)
