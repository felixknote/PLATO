"""Multi-plate session: unions several IndexDB instances behind one API.

Each loaded plate keeps its own config, index database and thumbnail cache —
nothing about ``build_index``/``build_thumbnails`` changes. ``Session`` just
fans queries out to every plate and merges the results, so ``BrowserPanel``
can treat "one plate" and "several plates side by side" identically.

``image_id`` (path relative to a plate's own image root) is only unique
*within* a plate, so a plain image_id is not a safe key once several plates
are loaded — two plates can easily share a relative path like
``WellA01_..._Seq0000.tiff``. Every ``ImageRow`` returned by ``Session.query``
is tagged with ``session_index`` (which plate it came from) so annotations
write back to the plate that actually owns the row.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..config import Config
from ..index.db import ImageRow, IndexDB
from ..wells import parse_well


@dataclass(slots=True)
class PlateSession:
    cfg: Config
    db: IndexDB


class Session:
    """The GUI's view of zero or more loaded plates."""

    def __init__(self) -> None:
        self.plates: list[PlateSession] = []

    def add_plate(self, cfg: Config) -> None:
        self.plates.append(PlateSession(cfg, IndexDB(cfg.db_path)))

    def close(self) -> None:
        for plate in self.plates:
            plate.db.close()
        self.plates.clear()

    @property
    def is_empty(self) -> bool:
        return not self.plates

    # -- introspection ------------------------------------------------------

    @property
    def metadata_columns(self) -> list[str]:
        seen: dict[str, None] = {}
        for plate in self.plates:
            for column in plate.db.metadata_columns:
                seen.setdefault(column, None)
        return list(seen)

    def label(self, column: str) -> str:
        for plate in self.plates:
            if column in plate.db.column_labels:
                return plate.db.label(column)
        return column.replace("_", " ").title()

    def filter_columns(self) -> list[str]:
        seen: dict[str, None] = {}
        for plate in self.plates:
            for column in plate.db.filter_columns():
                seen.setdefault(column, None)
        return list(seen)

    def distinct(self, column: str) -> list[str]:
        values: set[str] = set()
        for plate in self.plates:
            values.update(plate.db.distinct(column))
        return sorted(values)

    def count(self) -> int:
        return sum(plate.db.count() for plate in self.plates)

    def display_limits(self) -> dict[str, tuple[float, float]]:
        merged: dict[str, tuple[float, float]] = {}
        for plate in self.plates:
            for channel, limits in plate.db.display_limits().items():
                merged.setdefault(channel, limits)
        return merged

    # -- querying -------------------------------------------------------------

    def query(
        self,
        filters: dict[str, Sequence[str]] | None = None,
        search: str = "",
        *,
        flagged_only: bool = False,
        order: str = "plate, well_row, well_col, field, channel",
        limit: int | None = None,
    ) -> list[ImageRow]:
        rows: list[ImageRow] = []
        for index, plate in enumerate(self.plates):
            plate_rows = plate.db.query(
                filters, search, flagged_only=flagged_only, order=order, limit=limit
            )
            for row in plate_rows:
                row.session_index = index
            rows.extend(plate_rows)
        if len(self.plates) > 1:
            rows = _resort(rows, order)
            if limit is not None:
                rows = rows[:limit]
        return rows

    # -- annotations ----------------------------------------------------------

    def set_annotation(
        self,
        row: ImageRow,
        *,
        rating: int | None = None,
        flagged: bool | None = None,
        note: str | None = None,
    ) -> None:
        self.plates[row.session_index].db.set_annotation(
            row.image_id, rating=rating, flagged=flagged, note=note
        )

    def export_annotations(self) -> list[dict]:
        """Marked rows from every loaded plate, tagged with their source.

        ``image_id`` is only unique *within* a plate (see the module docstring),
        so a bare image_id is ambiguous the moment two plates are loaded -- two
        plates readily share a relative path like ``WellA01_..._Seq0000.tiff``.
        Every row therefore carries ``plate_source`` (the image root it came
        from) and ``uid`` (session index + image_id), so a batch export can be
        traced back to actual files without guessing which plate owns a row.
        """
        rows: list[dict] = []
        for index, plate in enumerate(self.plates):
            root = str(plate.cfg.images.dir)
            for row in plate.db.export_annotations():
                rows.append({"uid": f"{index}/{row['image_id']}", "plate_source": root, **row})
        return rows


def _well_key(well: str) -> tuple[int, int]:
    """(row, col) for sorting. Unparseable wells sort last rather than raise."""
    try:
        parsed = parse_well(well)
    except Exception:  # noqa: BLE001 - a malformed well must not break sorting
        return (10**6, 10**6)
    return (parsed.row, parsed.col)


def _resort(rows: list[ImageRow], order: str) -> list[ImageRow]:
    if order.strip().upper().startswith("RANDOM"):
        import random

        shuffled = rows[:]
        random.shuffle(shuffled)
        return shuffled
    keys = [part.strip() for part in order.split(",")]
    key_getters = {
        "plate": lambda r: r.plate,
        # Sort on the parsed (row, col), not the label: "A10" sorts before "A2"
        # as a string, which scrambles plate order and makes the two panels of
        # compare mode non-comparable position for position.
        "well_row": lambda r: _well_key(r.well)[0],
        "well_col": lambda r: _well_key(r.well)[1],
        "field": lambda r: r.field or "",
        "channel": lambda r: r.channel or "",
        "well": lambda r: r.well,
        "seq": lambda r: r.metadata.get("seq", "") if r.metadata else "",
    }
    getters = [key_getters[k] for k in keys if k in key_getters]
    if not getters:
        return rows
    return sorted(rows, key=lambda r: tuple(g(r) for g in getters))
