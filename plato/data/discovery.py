"""Find embedding exports and plate image folders under one or more roots.

The Embedding Explorer's "Browse..." and the browser's "Load Data" have
always required picking one exact folder. Real screens are not organised that
way: a lab drive holds many screens under one parent, an embeddings folder and
its source images are usually siblings a level or two apart, and a user
pointed at the wrong level should not have to know the right one in advance.

This module answers "what is under here" without loading anything. It is
deliberately cheap and deliberately bounded, because the drives it runs
against are often large network shares or mechanical disks: a directory tree
walk with no limit is exactly the kind of load that queues up a spinning disk
behind a stalled trainer (measured on this project: a 6-worker scan job with
zero throughput while the queue sat at 57, on a 12 TB HDD).

Two things keep it cheap:

* **Pruning.** The moment a directory is recognised as an embedding export or
  a plate's image folder, its own subtree is not descended into. A plate can
  hold tens of thousands of images; walking into one to look for MORE
  datasets inside it is pointless and is exactly the cost this module exists
  to avoid.
* **A budget.** ``max_entries`` and ``max_seconds`` both stop the walk, and
  the result says so (``truncated=True``) rather than silently returning a
  partial answer that looks complete.

Detection here is deliberately shallow -- a metadata filename, an image
extension count -- not the full pattern-matching in ``plato.gui.detect``.
That runs once a folder is actually chosen to load, where its cost is paid
once instead of once per candidate directory in a tree walk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .embeddings import METADATA_FILENAME, is_dataset_dir

# Extensions a plate's image folder is expected to contain. Matched
# case-insensitively; sampling a handful of files is enough to tell "this is
# a folder of micrographs" from "this is a folder of something else" without
# counting all of them, which matters when there are tens of thousands.
IMAGE_EXTENSIONS = frozenset({".tif", ".tiff", ".png", ".jpg", ".jpeg"})

# Files sampled from a directory before deciding whether it looks like a
# plate's image folder. Cheap: a single scandir pass, stopped early.
IMAGE_SAMPLE_SIZE = 12
IMAGE_SAMPLE_MIN_HITS = 3

EMBEDDING = "embedding"
PLATE = "plate"

DEFAULT_MAX_DEPTH = 4
DEFAULT_MAX_ENTRIES = 20_000
DEFAULT_MAX_SECONDS = 20.0


@dataclass(slots=True)
class Found:
    """One thing discovered under a scan root."""

    path: Path
    kind: str  # EMBEDDING or PLATE
    # A cheap, human-readable hint about what is inside -- row count for an
    # embedding, sampled image count for a plate -- so a chooser can show
    # something more useful than a bare path before anything is loaded.
    detail: str = ""


@dataclass(slots=True)
class ScanResult:
    found: list[Found] = field(default_factory=list)
    truncated: bool = False
    seconds: float = 0.0
    directories_visited: int = 0


def looks_like_image_folder(directory: Path) -> bool:
    """Cheap heuristic: does this directory hold a handful of image files.

    Not the real filename-pattern detection in ``plato.gui.detect`` -- this
    only has to be right enough to prune a scan, and running the full
    pattern match against every directory in a tree would be the expensive
    thing this module exists to avoid. The precise check happens once, when
    a folder is actually chosen to load.
    """
    hits = 0
    seen = 0
    try:
        with __import__("os").scandir(directory) as it:
            for entry in it:
                if seen >= IMAGE_SAMPLE_SIZE:
                    break
                if not entry.is_file():
                    continue
                seen += 1
                if Path(entry.name).suffix.lower() in IMAGE_EXTENSIONS:
                    hits += 1
    except OSError:
        return False
    return hits >= IMAGE_SAMPLE_MIN_HITS


def _row_count_hint(directory: Path) -> str:
    """A cheap row count for an embedding export, without loading vectors."""
    try:
        with (directory / METADATA_FILENAME).open("r", encoding="utf-8", errors="ignore") as handle:
            # Header + data rows; off by the header line, stated as such.
            n = sum(1 for _ in handle) - 1
        return f"{max(0, n):,} rows" if n >= 0 else ""
    except OSError:
        return ""


def _image_count_hint(directory: Path) -> str:
    count = 0
    try:
        with __import__("os").scandir(directory) as it:
            for entry in it:
                if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_EXTENSIONS:
                    count += 1
                    if count >= 999:
                        return "1000+ images"
    except OSError:
        return ""
    return f"{count:,} image{'s' if count != 1 else ''}" if count else ""


class _Budget:
    def __init__(self, max_entries: int, max_seconds: float) -> None:
        self.max_entries = max_entries
        self.max_seconds = max_seconds
        self.started = time.monotonic()
        self.entries = 0
        self.directories = 0
        self.truncated = False

    def exhausted(self) -> bool:
        if self.entries >= self.max_entries:
            self.truncated = True
        elif time.monotonic() - self.started >= self.max_seconds:
            self.truncated = True
        return self.truncated


def _scan_one(root: Path, depth: int, budget: _Budget, out: list[Found]) -> None:
    if budget.exhausted() or not root.is_dir():
        return
    budget.directories += 1

    if is_dataset_dir(root):
        out.append(Found(root, EMBEDDING, _row_count_hint(root)))
        # Do not descend: an export's own directory holds only its metadata
        # and vectors, never a nested dataset worth finding.
        return

    if looks_like_image_folder(root):
        out.append(Found(root, PLATE, _image_count_hint(root)))
        # Do not descend into a plate's own images looking for more datasets.
        return

    if depth <= 0:
        return

    try:
        children = [p for p in root.iterdir() if p.is_dir()]
    except OSError:
        return

    for child in children:
        budget.entries += 1
        if budget.exhausted():
            return
        _scan_one(child, depth - 1, budget, out)


def scan(
    roots: list[Path],
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> ScanResult:
    """Find every embedding export and plate image folder under ``roots``.

    Each root is scanned to ``max_depth`` directory levels, pruning as soon
    as a directory is recognised (see module docstring). Duplicate paths
    across overlapping roots are collapsed, keeping the first hit.

    Bounded by both an entry count and a wall clock, because the directories
    this runs against are sometimes large network shares or mechanical disks
    where an unbounded walk is a real cost, not just a slow function call.
    Hitting either bound sets ``truncated=True`` on the result rather than
    silently returning a partial answer that looks complete.
    """
    budget = _Budget(max_entries, max_seconds)
    out: list[Found] = []
    for root in roots:
        _scan_one(Path(root), max_depth, budget, out)

    seen: set[str] = set()
    deduped: list[Found] = []
    for item in out:
        key = str(item.path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    deduped.sort(key=lambda f: (f.kind, str(f.path)))

    return ScanResult(
        found=deduped,
        truncated=budget.truncated,
        seconds=time.monotonic() - budget.started,
        directories_visited=budget.directories,
    )
