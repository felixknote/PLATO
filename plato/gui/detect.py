"""Best-effort guesses for the filename pattern and plate map layout.

Detection never invents a new parsing path: it just tries a short list of
patterns/layouts already known to the indexing layer (``compile_pattern``,
``read_platemap``) against the real files and keeps the first one that
actually parses something. If nothing matches, the caller falls back to
asking the user — this module only saves that step when it can.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..index.filenames import compile_pattern, parse_files
from ..index.platemap import read_platemap

NIS_PATTERN = (
    r"^Well(?P<well>[A-P]\d{1,2})_Point(?P<point>[^_]+)_(?P<field>\d+)"
    r"_Channel(?P<channel>.+?)_Seq(?P<seq>\d+)\.tiff?$"
)
SIMPLE_PATTERN = r"^(?P<plate>[^_]+)_(?P<well>[A-P]\d{1,2})_f(?P<field>\d+)_(?P<channel>[A-Za-z0-9]+)\.tif$"
WELL_ONLY_PATTERN = r"^(?P<well>[A-P]\d{1,2})\.tif{1,2}$"

KNOWN_FILENAME_PATTERNS: list[str] = [NIS_PATTERN, SIMPLE_PATTERN, WELL_ONLY_PATTERN]

COMMON_GLOBS = ["**/*.tif*"]
COMMON_WELL_COLUMNS = ["Well", "well", "WELL"]


@dataclass(slots=True)
class PatternGuess:
    pattern: str
    glob: str
    n_parsed: int


def detect_pattern(image_dir: Path, plate_format: int | None = 96) -> PatternGuess | None:
    """Try each known filename pattern against the folder; keep the best match."""
    best: PatternGuess | None = None
    for glob in COMMON_GLOBS:
        for pattern in KNOWN_FILENAME_PATTERNS:
            try:
                records, _ = parse_files(
                    image_dir, glob, pattern, plate_format=plate_format
                )
            except (ValueError, FileNotFoundError):
                continue
            if records and (best is None or len(records) > best.n_parsed):
                best = PatternGuess(pattern=pattern, glob=glob, n_parsed=len(records))
    return best


@dataclass(slots=True)
class PlatemapGuess:
    layout: str
    well_column: str = "Well"
    header_row: bool = False
    index_col: bool = False


def detect_platemap_layout(
    path: Path, *, plate_format: int | None = 96
) -> PlatemapGuess | None:
    """Guess "long" (a Well column) vs "matrix" (a headerless plate grid)."""
    for well_column in COMMON_WELL_COLUMNS:
        try:
            result = read_platemap(
                path,
                layout="long",
                well_column=well_column,
                plate_format=plate_format,
            )
        except (KeyError, ValueError, FileNotFoundError):
            continue
        if len(result.frame) > 0:
            return PlatemapGuess(layout="long", well_column=well_column)

    try:
        result = read_platemap(
            path,
            layout="matrix",
            header_row=False,
            index_col=False,
            plate_format=plate_format,
        )
    except (ValueError, FileNotFoundError):
        return None
    if len(result.frame) > 0:
        return PlatemapGuess(layout="matrix", header_row=False, index_col=False)
    return None


def compile_or_none(pattern: str) -> bool:
    try:
        compile_pattern(pattern)
    except ValueError:
        return False
    return True
