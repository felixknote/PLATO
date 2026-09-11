"""Colouring by where a point sat on the plate, rather than by what was in it.

Every other colour-by answers "what was this well treated with". These answer
"where was it", which is what you reach for when you suspect the plate itself
is in the data: edge effects from evaporation, a row-wise dispensing error, a
column-wise gradient from the order wells were filled, a plate-reader or
autofocus drift that tracks position.

That makes these **diagnostic** encodings. Structure that lines up with a
biological label is a finding; structure that lines up with plate geometry is
usually an artefact, and seeing the second is what stops you reporting it as
the first.

Four derivations, because they answer different questions:

``well``      the well itself -- 96 categories, so this is a lookup, not a
              pattern you read by eye
``row``       A-H. Separates a dispensing or pipetting error that runs along
              rows
``column``    1-12. The same for the other axis, and the one that usually
              carries a fill-order gradient
``edge``      rings inward from the plate boundary: 0 for the outer ring, 1
              for the next, and so on. Evaporation and thermal edge effects
              are radial, not row- or column-shaped, so neither of the two
              axis encodings shows them cleanly and this one does

Nothing here is stored. The columns are derived from ``well`` on demand,
because the plate geometry is a property of the dataset, not of the export,
and a dataset whose wells do not parse must simply not offer these options
rather than carry empty columns around.
"""

from __future__ import annotations

import pandas as pd

from ..wells import PLATE_GEOMETRIES, WellParseError, parse_well

# Derived column names. Prefixed so they cannot collide with a real metadata
# column named "row" or "column", which is not far-fetched in a plate map.
WELL_ROW = "plate_row"
WELL_COL = "plate_column"
EDGE_DISTANCE = "plate_edge_distance"

LOCATION_FIELDS = (WELL_ROW, WELL_COL, EDGE_DISTANCE)

FIELD_LABELS = {
    WELL_ROW: "Plate location: row",
    WELL_COL: "Plate location: column",
    EDGE_DISTANCE: "Plate location: distance from edge",
}

# Fraction of rows that must parse as wells before these options are offered.
# Well below this and the column is not really a well column; a handful of
# unparseable rows in a real one is normal and becomes blank rather than
# suppressing the whole encoding.
MIN_PARSE_RATE = 0.5


def detect_format(wells: pd.Series) -> int | None:
    """The smallest standard plate geometry that contains every well seen.

    Inferred, never assumed: an export may hold a 96-well screen, a 384-well
    one, or a subset of either, and guessing 96 for a 384-well plate would put
    half the wells out of bounds and silently drop them. Returns None when the
    values do not look like wells at all.

    This reads the LARGEST coordinates present, so a subset infers the
    smallest geometry containing it: four wells in one corner read as a
    6-well plate, and their edge distances would then be relative to a 2x3
    rather than to the plate they came from. Measured against random subsets
    of a real 96-well plate this does not bite -- 5% (4 wells) still detected
    96, because it only takes one well past row F or column 8 -- so it needs
    wells clustered in a corner, which a screen does not produce. Passing the
    format in from the dataset would remove the case entirely, if an export
    ever records it.
    """
    rows: list[int] = []
    cols: list[int] = []
    parsed = 0
    seen = 0
    for raw in wells.dropna().astype(str):
        if not raw.strip():
            continue
        seen += 1
        try:
            # plate_format=None skips the bounds check, so this reads the
            # coordinates first and decides on the geometry afterwards.
            well = parse_well(raw, plate_format=None)
        except WellParseError:
            continue
        parsed += 1
        rows.append(well.row)
        cols.append(well.col)

    if not seen or parsed / seen < MIN_PARSE_RATE:
        return None
    max_row, max_col = max(rows), max(cols)
    for size in sorted(PLATE_GEOMETRIES):
        n_rows, n_cols = PLATE_GEOMETRIES[size]
        if max_row <= n_rows and max_col <= n_cols:
            return size
    return None


def _coordinates(wells: pd.Series, plate_format: int) -> tuple[pd.Series, pd.Series]:
    """1-based (row, col) per entry; NA where the value is not a well."""
    rows: list[int | None] = []
    cols: list[int | None] = []
    for raw in wells.astype(str):
        try:
            well = parse_well(raw, plate_format=plate_format)
        except WellParseError:
            rows.append(None)
            cols.append(None)
            continue
        rows.append(well.row)
        cols.append(well.col)
    return (
        pd.Series(rows, index=wells.index, dtype="Int64"),
        pd.Series(cols, index=wells.index, dtype="Int64"),
    )


def edge_distance(row: int, col: int, n_rows: int, n_cols: int) -> int:
    """Rings in from the plate boundary. 0 is the outer ring.

    The minimum of the four distances to an edge, so a well in a corner and a
    well in the middle of the top row are both ring 0 -- they share the
    exposure that matters, which is having an outside face.
    """
    return min(row - 1, col - 1, n_rows - row, n_cols - col)


def add_columns(frame: pd.DataFrame, well_column: str = "well") -> list[str]:
    """Add the derived location columns to ``frame`` in place.

    Returns the names of the columns actually added -- empty when the frame
    has no usable well column, which is the signal for the UI to not offer
    these options at all rather than offer an encoding with one value.
    """
    if well_column not in frame.columns:
        return []
    wells = frame[well_column]
    plate_format = detect_format(wells)
    if plate_format is None:
        return []

    n_rows, n_cols = PLATE_GEOMETRIES[plate_format]
    rows, cols = _coordinates(wells, plate_format)

    added: list[str] = []

    # Row as its letter, not its number: "B" is what is written on the plate
    # and what a reader recognises. Sorting still works because the letters
    # are in alphabetical order for every geometry here.
    row_labels = rows.map(lambda r: chr(ord("A") + int(r) - 1) if pd.notna(r) else "")
    if row_labels.ne("").any():
        frame[WELL_ROW] = row_labels.astype(str)
        added.append(WELL_ROW)

    # Zero-padded so a plain string sort puts 2 before 10. The explorer sorts
    # category values as strings when assigning colours, so an unpadded
    # column would colour the columns in the order 1, 10, 11, 12, 2, 3...
    # and the palette would read as noise instead of a gradient.
    col_labels = cols.map(lambda c: f"{int(c):02d}" if pd.notna(c) else "")
    if col_labels.ne("").any():
        frame[WELL_COL] = col_labels.astype(str)
        added.append(WELL_COL)

    edge = [
        str(edge_distance(int(r), int(c), n_rows, n_cols))
        if pd.notna(r) and pd.notna(c)
        else ""
        for r, c in zip(rows, cols)
    ]
    edge_series = pd.Series(edge, index=frame.index, dtype=str)
    if edge_series.ne("").any():
        frame[EDGE_DISTANCE] = edge_series
        added.append(EDGE_DISTANCE)

    return added
