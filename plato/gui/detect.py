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
# Same NIS-Elements format, but prefixed with an acquisition timestamp, e.g.
# "20260728_143951_318__WellA01_PointA01_0000_Channel..._Seq0000.tiff".
NIS_TIMESTAMPED_PATTERN = (
    r"^\d{8}_\d{6}_\d+__Well(?P<well>[A-P]\d{1,2})_Point(?P<point>[^_]+)_(?P<field>\d+)"
    r"_Channel(?P<channel>.+?)_Seq(?P<seq>\d+)\.tiff?$"
)
SIMPLE_PATTERN = r"^(?P<plate>[^_]+)_(?P<well>[A-P]\d{1,2})_f(?P<field>\d+)_(?P<channel>[A-Za-z0-9]+)\.tif$"
WELL_ONLY_PATTERN = r"^(?P<well>[A-P]\d{1,2})\.tif{1,2}$"

KNOWN_FILENAME_PATTERNS: list[str] = [
    NIS_PATTERN,
    NIS_TIMESTAMPED_PATTERN,
    SIMPLE_PATTERN,
    WELL_ONLY_PATTERN,
]

COMMON_GLOBS = ["**/*.tif*"]
COMMON_WELL_COLUMNS = ["Well", "well", "WELL"]

# Matrix cells often pack a condition and a replicate index into one string,
# e.g. "ftsZ_2" or "ACE-1 NC_6" — this is the split used throughout the docs
# and example configs, so it is the first (and currently only) guess tried.
GENE_REPLICATE_SPLIT = r"^(?P<gene>.+)_(?P<replicate>\d+)$"


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
    split_pattern: str = ""


def detect_platemap_layout(
    path: Path, *, plate_format: int | None = 96
) -> PlatemapGuess | None:
    """Guess "long" (a Well column) vs "matrix" (a headerless plate grid).

    For a matrix layout, also try splitting each cell as "<gene>_<replicate>"
    (the packed-condition format used throughout this project) — most cells
    matching means real metadata columns (gene, replicate, ...) get indexed
    instead of one opaque "condition" column, which is what the sidebar
    filters are built from.
    """
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
        plain = read_platemap(
            path,
            layout="matrix",
            header_row=False,
            index_col=False,
            plate_format=plate_format,
        )
    except (ValueError, FileNotFoundError):
        return None
    if len(plain.frame) == 0:
        return None

    try:
        split = read_platemap(
            path,
            layout="matrix",
            header_row=False,
            index_col=False,
            split_pattern=GENE_REPLICATE_SPLIT,
            plate_format=plate_format,
        )
    except ValueError:
        split = None
    if split is not None and len(split.frame) > 0 and len(split.unsplit_values) < len(split.frame) / 2:
        return PlatemapGuess(
            layout="matrix", header_row=False, index_col=False, split_pattern=GENE_REPLICATE_SPLIT
        )

    return PlatemapGuess(layout="matrix", header_row=False, index_col=False)


def compile_or_none(pattern: str) -> bool:
    try:
        compile_pattern(pattern)
    except ValueError:
        return False
    return True
