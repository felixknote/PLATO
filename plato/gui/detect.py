"""Best-effort guesses for the filename pattern and plate map layout.

Detection never invents a new parsing path: it just tries a short list of
patterns/layouts already known to the indexing layer (``compile_pattern``,
``read_platemap``) against the real files and keeps the first one that
actually parses something. If nothing matches, the caller falls back to
asking the user — this module only saves that step when it can.

To support a new screen naming convention, add one entry to a list below and
nothing else needs to change:

* A new filename layout (e.g. a different microscope's export format) ->
  add a regex to ``KNOWN_FILENAME_PATTERNS``. It must capture ``well`` and
  may capture any of ``plate, field, channel, z, seq, point``.
* A new way a plate map cell packs several fields into one string (e.g.
  "<gene>_<replicate>", "<antibiotic>\\n<concentration>") -> add a named-group
  regex to ``KNOWN_SPLIT_PATTERNS``. Whichever pattern splits the most cells
  on the real file wins, so this is what makes the sidebar filters (gene,
  antibiotic, concentration, ...) adapt automatically to whichever dataset
  was loaded — no per-dataset configuration needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data.index.filenames import compile_pattern, parse_files
from ..data.index.platemap import read_platemap

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

# Matrix cells pack more than one field into a single string, in more than
# one convention depending on the screen. Each entry here is tried against
# the real plate map; whichever splits the most cells wins, so the sidebar
# filters (gene/replicate for a mutant screen, antibiotic/concentration for
# a compound screen, ...) adapt to whichever dataset was actually loaded.
GENE_REPLICATE_SPLIT = r"^(?P<gene>.+)_(?P<replicate>\d+)$"
# "Ciprofloxacin\n1/2x", "Cefepime\n1x" — antibiotic name and a fractional-MIC
# concentration on the next line. Control wells ("WT", "WT NC") have no
# concentration line and are correctly left unsplit, not misparsed.
ANTIBIOTIC_CONCENTRATION_SPLIT = r"^(?P<antibiotic>[^\n]+)\n(?P<concentration>.+)$"

KNOWN_SPLIT_PATTERNS: list[str] = [
    GENE_REPLICATE_SPLIT,
    ANTIBIOTIC_CONCENTRATION_SPLIT,
]


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

    For a matrix layout, also try every pattern in KNOWN_SPLIT_PATTERNS and
    keep whichever splits the most cells (control wells like "WT" are allowed
    to stay unsplit and don't count against a pattern). This is what makes
    the sidebar filters adapt per dataset: gene/replicate for a mutant screen
    that packs "ftsZ_2" into a cell, antibiotic/concentration for a compound
    screen that packs "Ciprofloxacin\\n1/2x" into a cell, and so on — instead
    of everything collapsing into one opaque "condition" column.
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

    best_pattern: str | None = None
    best_split_count = 0
    for pattern in KNOWN_SPLIT_PATTERNS:
        try:
            split = read_platemap(
                path,
                layout="matrix",
                header_row=False,
                index_col=False,
                split_pattern=pattern,
                plate_format=plate_format,
            )
        except ValueError:
            continue
        if len(split.frame) == 0:
            continue
        split_count = len(split.frame) - len(split.unsplit_values)
        # Require most cells to actually split; a pattern that "matches" by
        # accident on a handful of cells isn't worth adopting over no split.
        if split_count > best_split_count and split_count >= len(split.frame) / 2:
            best_pattern = pattern
            best_split_count = split_count

    if best_pattern is not None:
        return PlatemapGuess(
            layout="matrix", header_row=False, index_col=False, split_pattern=best_pattern
        )

    return PlatemapGuess(layout="matrix", header_row=False, index_col=False)


def compile_or_none(pattern: str) -> bool:
    try:
        compile_pattern(pattern)
    except ValueError:
        return False
    return True
