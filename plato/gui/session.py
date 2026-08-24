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

Plate *names* are not unique either: two folders both called ``images`` index
as two plates called "images", and the grid then shows two different wells
labelled "images A01" with no way to tell them apart or to filter to one of
them. :meth:`Session.add_plate` therefore disambiguates the name on the way
in, and refuses a config whose index database is already loaded.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..index.db import ImageRow, IndexDB
from ..ordering import sort_series
from ..wells import parse_well
from .settings import get_show_timepoint

# Filter column name for the plate-name timepoint suffix.
TIMEPOINT = "timepoint"

# Filter column naming the *loaded plate* a row came from.
#
# Distinct from the `plate` column in each index, which comes from the
# filename or the plate map and is not unique across separately loaded plates
# -- two folders indexed with the same config both label every row "Plate1".
# Filtering on that selects both at once, which is precisely the wrong answer
# when the whole reason you loaded two is to tell them apart. This column is
# the session's own disambiguated name, so it always selects exactly one.
SOURCE_PLATE = "source_plate"

# Trailing _T<n> on a plate/folder name: "P13_T1" -> "T1". Anchored to the end
# so it cannot match a _T in the middle of an unrelated name.
_TIMEPOINT_RE = re.compile(r"_(?P<timepoint>T\d+)$", re.IGNORECASE)


def timepoint_of(plate_name: str) -> str:
    """"P13_T1" -> "T1"; "" when the name carries no timepoint suffix."""
    match = _TIMEPOINT_RE.search(plate_name or "")
    return match.group("timepoint").upper() if match else ""


class DuplicatePlateError(ValueError):
    """Raised when a plate whose index is already loaded is added again."""


@dataclass(slots=True)
class PlateSession:
    cfg: Config
    db: IndexDB

    @property
    def name(self) -> str:
        return self.cfg.project.name

    @property
    def source(self) -> Path:
        return self.cfg.images.dir


class Session:
    """The GUI's view of zero or more loaded plates."""

    def __init__(self) -> None:
        self.plates: list[PlateSession] = []

    def add_plate(self, cfg: Config) -> PlateSession:
        """Load one plate, giving it a name unique within the session.

        Two guards, for two different mistakes:

        *Adding the same plate twice* (the same index database) is always an
        error, not a doubling of the grid -- every image would appear twice,
        every count would be wrong, and flagging one copy would leave the
        other unflagged. It raises rather than being silently ignored so the
        caller can say which plate was already loaded.

        *Two distinct plates that happen to share a name* is legitimate -- two
        screens both indexed from a folder called ``images``. But the name is
        what the grid captions, what the SOURCE_PLATE filter offers and what
        an annotation export records, so two identical names leave the plates
        indistinguishable on screen and unselectable apart. The second becomes
        "images (2)".

        The name is display and selection state only; nothing persistent keys
        on it. The index database path is a plate's real identity -- it is
        what the duplicate check compares and what an exported ``uid`` is
        built from -- so renaming a plate never orphans anything.
        """
        resolved = cfg.db_path.resolve()
        for plate in self.plates:
            if plate.cfg.db_path.resolve() == resolved:
                raise DuplicatePlateError(plate.name)
        cfg.project.name = self._unique_name(cfg.project.name)
        plate = PlateSession(cfg, IndexDB(cfg.db_path))
        self.plates.append(plate)
        return plate

    def _unique_name(self, name: str) -> str:
        name = name or "plate"
        taken = {plate.name for plate in self.plates}
        if name not in taken:
            return name
        suffix = 2
        while f"{name} ({suffix})" in taken:
            suffix += 1
        return f"{name} ({suffix})"

    def remove_plate(self, index: int) -> None:
        """Unload one plate, closing its database.

        Rows already handed out carry ``session_index``, which is a position
        in this list, so every one of them is stale afterwards -- the caller
        must re-query rather than reuse them.
        """
        if not 0 <= index < len(self.plates):
            return
        self.plates.pop(index).db.close()

    def close(self) -> None:
        for plate in self.plates:
            plate.db.close()
        self.plates.clear()

    @property
    def is_empty(self) -> bool:
        return not self.plates

    @property
    def plate_names(self) -> list[str]:
        return [plate.name for plate in self.plates]

    # -- introspection ------------------------------------------------------

    @property
    def metadata_columns(self) -> list[str]:
        seen: dict[str, None] = {}
        for plate in self.plates:
            for column in plate.db.metadata_columns:
                seen.setdefault(column, None)
        return list(seen)

    def label(self, column: str) -> str:
        if column == SOURCE_PLATE:
            return "Loaded Plate"
        for plate in self.plates:
            if column in plate.db.column_labels:
                return plate.db.label(column)
        return column.replace("_", " ").title()

    def filter_columns(self) -> list[str]:
        seen: dict[str, None] = {}
        # Offered first, and only when there is a choice to make: with one
        # plate loaded it is a filter with a single value, which is noise.
        if len(self.plates) > 1:
            seen.setdefault(SOURCE_PLATE, None)
        for plate in self.plates:
            for column in plate.db.filter_columns():
                seen.setdefault(column, None)
        if self.timepoints():
            seen.setdefault(TIMEPOINT, None)
        return list(seen)

    def timepoints(self) -> list[str]:
        """Distinct timepoints across loaded plates, empty if none are named.

        Derived from the plate name rather than stored: a plate folder is one
        timepoint (P13_T1), so the suffix is a property of the plate, not of
        any individual image, and needs no schema change to expose. Returns
        empty when the setting is off or no plate name carries a suffix, and
        filter_columns() then never offers the column -- so a screen whose
        folders happen to end in _T-something they do not mean does not
        acquire a meaningless filter.
        """
        if not get_show_timepoint():
            return []
        found: dict[str, None] = {}
        for plate in self.plates:
            value = timepoint_of(plate.cfg.project.name)
            if value:
                found.setdefault(value, None)
        return list(found)

    def distinct(self, column: str) -> list[str]:
        # Sorted by magnitude, not as text: these values become the comparison
        # view's columns left to right, so "1/8x, 1/4x, 1/2x, 1x" has to read
        # as the dose series it is rather than "1/2x, 1/4x, 1/8x, 1x".
        if column == TIMEPOINT:
            return sort_series(self.timepoints())
        if column == SOURCE_PLATE:
            # Load order, not sorted: it is the order they were added in and
            # the order the dialog lists them, so the two agree.
            return self.plate_names
        values: set[str] = set()
        for plate in self.plates:
            values.update(plate.db.distinct(column))
        return sort_series(values)

    def count(self) -> int:
        return sum(plate.db.count() for plate in self.plates)

    def display_limits(self) -> dict[str, tuple[float, float]]:
        """Per-channel (lo, hi) covering every loaded plate.

        Each plate estimates its own limits from its own images, so plates
        imaged on different days arrive with different ones. Taking the first
        plate's and applying them to the rest -- what this used to do -- makes
        a brightness difference between plates a difference in *scaling*, in a
        view whose whole premise is that it is a difference in the sample.

        Pooling instead: lo is the lowest of the plates' los and hi the
        highest of their his, so one window contains every plate's data and
        every plate is stretched identically. A dim plate stays visibly dim
        rather than being levelled up to look like the bright one, which is
        the honest rendering when you are comparing across timepoints.
        """
        pooled: dict[str, tuple[float, float]] = {}
        for plate in self.plates:
            for channel, (lo, hi) in plate.db.display_limits().items():
                current = pooled.get(channel)
                if current is None:
                    pooled[channel] = (lo, hi)
                else:
                    pooled[channel] = (min(current[0], lo), max(current[1], hi))
        return pooled

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
        # Timepoint is a property of the plate, not a column in any plate's
        # index, so it selects whole plates here and must not be forwarded to
        # IndexDB.query -- which would raise on an unknown column.
        wanted: list[str] = []
        if filters and TIMEPOINT in filters:
            filters = dict(filters)
            wanted = [str(v) for v in filters.pop(TIMEPOINT)]

        # Same treatment, for the same reason: the loaded-plate name lives in
        # the session, not in any plate's index, so it selects whole plates
        # here rather than being forwarded as a column IndexDB does not have.
        wanted_plates: list[str] | None = None
        if filters and SOURCE_PLATE in filters:
            filters = dict(filters)
            wanted_plates = [str(v) for v in filters.pop(SOURCE_PLATE)]

        rows: list[ImageRow] = []
        searched = 0
        for index, plate in enumerate(self.plates):
            if wanted and timepoint_of(plate.cfg.project.name) not in wanted:
                continue
            if wanted_plates is not None and plate.name not in wanted_plates:
                continue
            searched += 1
            # A plate whose *loaded* name matches the search term contributes
            # all its rows, searched or not: the name is session state, so no
            # plate's index can match on it, and typing the name of a plate
            # you can see in the sidebar would otherwise return nothing.
            name_matches = bool(search.strip()) and search.strip().lower() in plate.name.lower()
            plate_search = "" if name_matches else search
            # `limit` is NOT forwarded: each plate would return its own first
            # N rows and the merge below would then keep the global first N of
            # those, which is not the global first N -- one plate can rightly
            # own every row of the answer. Fetch all, merge, then truncate.
            plate_rows = plate.db.query(
                filters, plate_search, flagged_only=flagged_only, order=order
            )
            for row in plate_rows:
                row.session_index = index
            rows.extend(plate_rows)
        # Merging is needed whenever more than one plate actually contributed,
        # not whenever more than one is loaded: a timepoint filter can narrow
        # a five-plate session down to one, and that result is already sorted.
        if searched > 1:
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
        Every row therefore carries ``session_plate`` (the plate's name as the
        session disambiguated it, so it matches what the grid showed),
        ``plate_source`` (the image root it came from) and ``uid``, so a batch
        export can be traced back to actual files without guessing which plate
        owns a row.

        ``uid`` is keyed on the *index database* rather than the session index
        or the image root. The session index is a position in a list that
        changes the moment a plate is removed, so it is not stable enough for
        a file that outlives the session; and two plates can legitimately
        share an image folder while having separate indexes, so the image root
        does not separate them either. The index path is what the session
        itself treats as a plate's identity, and it is unique by construction.
        """
        rows: list[dict] = []
        for plate in self.plates:
            index_path = str(plate.cfg.db_path)
            for row in plate.db.export_annotations():
                rows.append(
                    {
                        "uid": f"{index_path}/{row['image_id']}",
                        "session_plate": plate.name,
                        "plate_source": str(plate.source),
                        "plate_index": index_path,
                        **row,
                    }
                )
        return rows


def _well_key(well: str) -> tuple[int, int]:
    """(row, col) for sorting. Unparseable wells sort last rather than raise."""
    try:
        parsed = parse_well(well)
    except Exception:  # noqa: BLE001 - a malformed well must not break sorting
        return (10**6, 10**6)
    return (parsed.row, parsed.col)


def _resort(rows: list[ImageRow], order: str) -> list[ImageRow]:
    """Re-sort rows merged from several plates.

    Only two orders ever reach here: the default plate/well/field/channel and
    RANDOM() for blinded review. Wells sort on the parsed (row, col) rather
    than the label, because "A10" sorts before "A2" as a string, which
    scrambles plate order and makes two panels non-comparable position for
    position.

    ``session_index`` leads the key, ahead of the ``plate`` column. Sorting on
    ``plate`` alone interleaves two separately loaded plates whenever they
    share that label -- the common case, since it defaults to the same value
    for every folder indexed with the same config -- so A01 of one plate and
    A01 of the other land adjacent and indistinguishable. Grouping by the
    plate a row was actually loaded from keeps each one contiguous, which is
    what makes scrolling through a multi-plate grid mean anything.
    """
    if order.strip().upper().startswith("RANDOM"):
        shuffled = rows[:]
        random.shuffle(shuffled)
        return shuffled
    return sorted(
        rows,
        key=lambda r: (
            r.session_index,
            r.plate,
            _well_key(r.well),
            r.field or "",
            r.channel or "",
        ),
    )
