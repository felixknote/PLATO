"""Plate map ingestion into a tidy, one-row-per-well frame.

Two layouts are supported, because both are common in practice:

``long``
    One row per well, with a well column and one column per metadata field.
    This is what a LIMS or a picklist exports.

``matrix``
    The plate drawn as a grid — 8 rows x 12 columns for a 96-well plate — with
    one cell per well, exactly as you would lay it out by hand in Excel. Row
    and column labels are optional; without them the position in the grid *is*
    the well, which is why the shape is validated against the plate format
    instead of being assumed.

A matrix cell usually packs several fields into one string (``ftsZ_2`` =
condition plus replicate). ``split_pattern`` unpacks it into real columns, so
filtering by gene collapses replicates instead of treating ``ftsZ_1``,
``ftsZ_2`` and ``ftsZ_3`` as three unrelated conditions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..wells import PLATE_GEOMETRIES, Well, WellParseError, parse_well

RESERVED = {"plate", "well", "well_row", "well_col"}


@dataclass(slots=True)
class PlatemapResult:
    frame: pd.DataFrame  # plate, well, well_row, well_col, + metadata columns
    column_labels: dict[str, str]  # sanitised -> human-readable header
    bad_wells: list[tuple[int, object, str]] = field(default_factory=list)
    duplicate_keys: list[tuple[str, str]] = field(default_factory=list)
    empty_cells: list[str] = field(default_factory=list)
    unsplit_values: list[tuple[str, str]] = field(default_factory=list)  # (well, value)


def sanitise_column(name: object, taken: set[str]) -> str:
    """Make a header safe as a SQL identifier and unique within the table."""
    text = re.sub(r"\W+", "_", str(name).strip()).strip("_")
    if not text:
        text = "column"
    if text[0].isdigit():
        text = f"c_{text}"
    text = text.lower()
    if text in RESERVED:
        text = f"{text}_meta"
    candidate, counter = text, 2
    while candidate in taken:
        candidate = f"{text}_{counter}"
        counter += 1
    taken.add(candidate)
    return candidate


def _read_table(path: Path, sheet: int | str, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"plate map not found: {path}")
    if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        return pd.read_csv(path, sep=sep, dtype=object, **kwargs)
    return pd.read_excel(path, sheet_name=sheet, dtype=object, **kwargs)


def plate_from_filename(path: Path, pattern: str) -> str | None:
    """Pull a plate name out of the plate map file name (e.g. ``..._P1.csv``)."""
    if not pattern:
        return None
    match = re.search(pattern, path.name)
    if match is None:
        return None
    named = match.groupdict().get("plate")
    if named:
        return named
    return match.group(1) if match.groups() else match.group(0)


# -- long layout ----------------------------------------------------------


def read_long_platemap(
    path: Path,
    *,
    sheet: int | str = 0,
    well_column: str = "Well",
    plate_column: str = "",
    default_plate: str = "Plate1",
    plate_format: int | None = 96,
) -> PlatemapResult:
    raw = _read_table(path, sheet)

    if well_column not in raw.columns:
        raise KeyError(
            f"well column {well_column!r} not in plate map. Available: {list(raw.columns)}"
        )
    if plate_column and plate_column not in raw.columns:
        raise KeyError(
            f"plate column {plate_column!r} not in plate map. Available: {list(raw.columns)}"
        )

    metadata_headers = [
        c for c in raw.columns if c != well_column and c != (plate_column or None)
    ]
    taken: set[str] = set()
    column_labels = {sanitise_column(h, taken): str(h) for h in metadata_headers}
    rename = {orig: safe for safe, orig in column_labels.items()}

    rows: list[dict[str, object]] = []
    bad: list[tuple[int, object, str]] = []
    for position, (_, row) in enumerate(raw.iterrows(), start=2):  # header + 1-based
        raw_well = row[well_column]
        if pd.isna(raw_well):
            bad.append((position, raw_well, "empty well cell"))
            continue
        try:
            well = parse_well(raw_well, plate_format)
        except WellParseError as exc:
            bad.append((position, raw_well, str(exc)))
            continue
        plate = str(row[plate_column]).strip() if plate_column else default_plate
        record: dict[str, object] = _well_key(plate, well)
        for header in metadata_headers:
            value = row[header]
            record[rename[header]] = None if pd.isna(value) else value
        rows.append(record)

    return _finish(pd.DataFrame(rows), column_labels, bad_wells=bad)


# -- matrix layout --------------------------------------------------------


def read_matrix_platemap(
    path: Path,
    *,
    sheet: int | str = 0,
    header_row: bool = False,
    index_col: bool = False,
    value_column: str = "condition",
    split_pattern: str = "",
    default_plate: str = "Plate1",
    plate_format: int | None = 96,
) -> PlatemapResult:
    """Read a plate drawn as a grid of cells.

    Args:
        header_row: First row holds column numbers (1..12) rather than data.
        index_col: First column holds row letters (A..H) rather than data.
        value_column: Name given to the raw cell contents.
        split_pattern: Optional named-group regex applied to each cell; every
            group becomes its own column.
    """
    grid = _read_table(
        path,
        sheet,
        header=0 if header_row else None,
        index_col=0 if index_col else None,
    )

    if plate_format is not None:
        expected = PLATE_GEOMETRIES[plate_format]
        if grid.shape != expected:
            raise ValueError(
                f"plate map grid is {grid.shape[0]}x{grid.shape[1]} but a "
                f"{plate_format}-well plate is {expected[0]}x{expected[1]}. "
                "Check header_row / index_col, or plate_format."
            )

    compiled = re.compile(split_pattern) if split_pattern else None
    if compiled is not None and not compiled.groupindex:
        raise ValueError("split_pattern must use named groups, e.g. (?P<gene>...)")

    taken: set[str] = set()
    value_key = sanitise_column(value_column, taken)
    column_labels = {value_key: value_column}
    split_keys: dict[str, str] = {}
    if compiled is not None:
        for group in compiled.groupindex:
            key = sanitise_column(group, taken)
            split_keys[group] = key
            column_labels[key] = group.replace("_", " ").title()

    rows: list[dict[str, object]] = []
    empty: list[str] = []
    unsplit: list[tuple[str, str]] = []

    for row_position, (row_label, series) in enumerate(grid.iterrows(), start=1):
        for col_position, (col_label, cell) in enumerate(series.items(), start=1):
            well = _well_from_labels(
                row_label if index_col else None,
                col_label if header_row else None,
                row_position,
                col_position,
                plate_format,
            )
            if pd.isna(cell) or str(cell).strip() == "":
                empty.append(well.label)
                continue
            value = str(cell).strip()
            record: dict[str, object] = _well_key(default_plate, well)
            record[value_key] = value
            if compiled is not None:
                match = compiled.match(value)
                if match is None:
                    unsplit.append((well.label, value))
                    for key in split_keys.values():
                        record[key] = None
                else:
                    for group, key in split_keys.items():
                        record[key] = match.group(group)
            rows.append(record)

    return _finish(
        pd.DataFrame(rows),
        column_labels,
        empty_cells=empty,
        unsplit_values=unsplit,
    )


def _well_from_labels(
    row_label: object,
    col_label: object,
    row_position: int,
    col_position: int,
    plate_format: int | None,
) -> Well:
    """Prefer explicit labels over grid position; fall back to position."""
    if row_label is not None and col_label is not None:
        try:
            return parse_well(f"{row_label}{col_label}", plate_format)
        except WellParseError:
            pass
    if row_label is not None:
        try:
            return parse_well(f"{row_label}{col_position}", plate_format)
        except WellParseError:
            pass
    return Well(row=row_position, col=col_position)


def _well_key(plate: str, well: Well) -> dict[str, object]:
    return {
        "plate": plate,
        "well": well.label,
        "well_row": well.row,
        "well_col": well.col,
    }


def _finish(
    frame: pd.DataFrame,
    column_labels: dict[str, str],
    **extra,
) -> PlatemapResult:
    duplicates: list[tuple[str, str]] = []
    if not frame.empty:
        counts = frame.groupby(["plate", "well"]).size()
        duplicates = [tuple(k) for k in counts[counts > 1].index]  # type: ignore[misc]
    return PlatemapResult(
        frame=frame,
        column_labels=column_labels,
        duplicate_keys=duplicates,
        **extra,
    )


# -- dispatch -------------------------------------------------------------


def read_platemap(
    path: Path,
    *,
    layout: str = "long",
    sheet: int | str = 0,
    well_column: str = "Well",
    plate_column: str = "",
    header_row: bool = False,
    index_col: bool = False,
    value_column: str = "condition",
    split_pattern: str = "",
    default_plate: str = "Plate1",
    plate_format: int | None = 96,
) -> PlatemapResult:
    """Dispatch to the reader for ``layout``.

    Callers pick the layout at runtime -- from config, or from ``detect``
    trying each in turn -- so the branch lives here rather than being repeated
    at every call site. The signature is spelled out so a caller can pass the
    whole config without knowing which arguments the chosen layout ignores.
    """
    if layout == "matrix":
        return read_matrix_platemap(
            path,
            sheet=sheet,
            header_row=header_row,
            index_col=index_col,
            value_column=value_column,
            split_pattern=split_pattern,
            default_plate=default_plate,
            plate_format=plate_format,
        )
    if layout == "long":
        return read_long_platemap(
            path,
            sheet=sheet,
            well_column=well_column,
            plate_column=plate_column,
            default_plate=default_plate,
            plate_format=plate_format,
        )
    raise ValueError(f"unknown plate map layout: {layout!r} (expected 'long' or 'matrix')")
