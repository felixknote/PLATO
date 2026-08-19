"""SQLite index.

Two kinds of state live here and must not be confused:

* ``images``, ``platemap``, ``display_limits`` — derived, regenerable at any
  time by re-running ``plato index``.
* ``annotations`` — your ratings, flags and notes. Never dropped on reindex.
  Keyed by ``image_id`` (path relative to the image root), which stays stable
  as long as you do not rename files.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

_BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
    image_id   TEXT PRIMARY KEY,
    path       TEXT NOT NULL,
    plate      TEXT NOT NULL,
    well       TEXT NOT NULL,
    well_row   INTEGER NOT NULL,
    well_col   INTEGER NOT NULL,
    field      TEXT,
    channel    TEXT,
    z          TEXT,
    seq        TEXT,
    point      TEXT,
    mtime      REAL NOT NULL,
    size_bytes INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_images_well ON images (plate, well);
CREATE INDEX IF NOT EXISTS idx_images_channel ON images (channel);

CREATE TABLE IF NOT EXISTS annotations (
    image_id   TEXT PRIMARY KEY,
    rating     INTEGER,
    flagged    INTEGER NOT NULL DEFAULT 0,
    note       TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS display_limits (
    channel TEXT PRIMARY KEY,
    lo      REAL NOT NULL,
    hi      REAL NOT NULL
);
"""


@dataclass(slots=True)
class ImageRow:
    """One row of ``image_view``, as handed to the GUI."""

    image_id: str
    path: str
    plate: str
    well: str
    field: str | None
    channel: str | None
    metadata: dict[str, Any]
    rating: int | None
    flagged: bool
    # Which plate a Session.query() result came from. Unused by IndexDB itself
    # (always 0); Session sets it so annotations route back to the owning db.
    session_index: int = 0


def connect(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open the index. Read-only connections are safe to use from worker threads."""
    if read_only:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path)
        con.execute("PRAGMA journal_mode=WAL")
    con.row_factory = sqlite3.Row
    return con


def initialise(con: sqlite3.Connection) -> None:
    con.executescript(_BASE_SCHEMA)
    con.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    con.commit()


def set_meta(con: sqlite3.Connection, key: str, value: Any) -> None:
    con.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        (key, json.dumps(value)),
    )


def get_meta(con: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except json.JSONDecodeError:
        return row["value"]


def create_image_view(con: sqlite3.Connection, metadata_columns: Sequence[str]) -> None:
    """(Re)create the flat view the GUI queries.

    A LEFT JOIN is used deliberately: images without a plate map row stay
    visible (with NULL metadata) instead of vanishing. Whether that is
    acceptable is a decision for the validation report, not for the join.
    """
    cols = "".join(f", p.{c} AS {c}" for c in metadata_columns)
    con.executescript(
        f"""
        DROP VIEW IF EXISTS image_view;
        CREATE VIEW image_view AS
        SELECT i.image_id, i.path, i.plate, i.well, i.well_row, i.well_col,
               i.field, i.channel, i.z, i.seq, i.point,
               a.rating AS rating, COALESCE(a.flagged, 0) AS flagged, a.note AS note
               {cols}
        FROM images i
        LEFT JOIN platemap p ON p.plate = i.plate AND p.well = i.well
        LEFT JOIN annotations a ON a.image_id = i.image_id;
        """
    )
    con.commit()


class IndexDB:
    """Read/write access to a built index. Thread-safe for reads only."""

    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = path
        self.con = connect(path, read_only=read_only)
        self.metadata_columns: list[str] = get_meta(self.con, "metadata_columns", []) or []
        self.column_labels: dict[str, str] = get_meta(self.con, "column_labels", {}) or {}

    # -- introspection ----------------------------------------------------

    def label(self, column: str) -> str:
        return self.column_labels.get(column, column.replace("_", " ").title())

    def filter_columns(self) -> list[str]:
        """Columns worth offering as filters: plate/well/field/channel + metadata."""
        structural = ["plate", "well", "channel", "seq"]
        present = [c for c in structural if self._has_values(c)]
        return present + list(self.metadata_columns)

    def _has_values(self, column: str) -> bool:
        row = self.con.execute(
            f"SELECT COUNT(*) AS n FROM (SELECT 1 FROM image_view WHERE {column} IS NOT NULL LIMIT 1)"
        ).fetchone()
        return bool(row["n"])

    def distinct(self, column: str) -> list[str]:
        rows = self.con.execute(
            f"SELECT DISTINCT {column} AS v FROM image_view "
            f"WHERE {column} IS NOT NULL ORDER BY {column}"
        ).fetchall()
        return [str(r["v"]) for r in rows]

    def count(self) -> int:
        return int(self.con.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"])

    def display_limits(self) -> dict[str, tuple[float, float]]:
        rows = self.con.execute("SELECT channel, lo, hi FROM display_limits").fetchall()
        return {r["channel"]: (r["lo"], r["hi"]) for r in rows}

    # -- querying ---------------------------------------------------------

    def query(
        self,
        filters: dict[str, Sequence[str]] | None = None,
        search: str = "",
        *,
        flagged_only: bool = False,
        order: str = "plate, well_row, well_col, field, channel",
        limit: int | None = None,
    ) -> list[ImageRow]:
        """Return matching rows.

        Args:
            filters: column -> allowed values (OR within a column, AND across).
            search: case-insensitive substring matched against every metadata
                column plus well and plate.
            flagged_only: restrict to flagged images.
            order: SQL ORDER BY clause. Pass ``RANDOM()`` for blinded review.
        """
        where: list[str] = []
        params: list[Any] = []

        for column, values in (filters or {}).items():
            values = [v for v in values if v != ""]
            if not values:
                continue
            placeholders = ",".join("?" for _ in values)
            where.append(f"CAST({column} AS TEXT) IN ({placeholders})")
            params.extend(str(v) for v in values)

        if search.strip():
            haystack = ["plate", "well", "channel", *self.metadata_columns]
            clause = " OR ".join(
                f"CAST(COALESCE({c}, '') AS TEXT) LIKE ? COLLATE NOCASE" for c in haystack
            )
            where.append(f"({clause})")
            params.extend([f"%{search.strip()}%"] * len(haystack))

        if flagged_only:
            where.append("flagged = 1")

        sql = "SELECT * FROM image_view"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY {order}"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        rows = self.con.execute(sql, params).fetchall()
        return [self._to_row(r) for r in rows]

    def _to_row(self, row: sqlite3.Row) -> ImageRow:
        return ImageRow(
            image_id=row["image_id"],
            path=row["path"],
            plate=row["plate"],
            well=row["well"],
            field=row["field"],
            channel=row["channel"],
            metadata={c: row[c] for c in self.metadata_columns},
            rating=row["rating"],
            flagged=bool(row["flagged"]),
        )

    # -- annotations ------------------------------------------------------

    def set_annotation(
        self,
        image_id: str,
        *,
        rating: int | None = None,
        flagged: bool | None = None,
        note: str | None = None,
    ) -> None:
        # Read-modify-write rather than COALESCE in SQL: None is overloaded
        # here. For `flagged`/`note` it means "leave alone", but for `rating`
        # it is also the value the 0 key writes to *clear* a rating, and
        # COALESCE cannot tell those apart -- it would silently ignore the
        # clear. Only the caller's argument list knows which was meant.
        current = self.con.execute(
            "SELECT rating, flagged, note FROM annotations WHERE image_id = ?", (image_id,)
        ).fetchone()
        new_rating = current["rating"] if current and rating is None else rating
        new_flagged = (
            bool(current["flagged"]) if current and flagged is None else bool(flagged)
        )
        new_note = current["note"] if current and note is None else note
        self.con.execute(
            "INSERT INTO annotations (image_id, rating, flagged, note, updated_at) "
            "VALUES (?, ?, ?, ?, datetime('now')) "
            "ON CONFLICT(image_id) DO UPDATE SET "
            "rating=excluded.rating, flagged=excluded.flagged, note=excluded.note, "
            "updated_at=excluded.updated_at",
            (image_id, new_rating, int(new_flagged), new_note),
        )
        self.con.commit()

    def export_annotations(self) -> list[dict[str, Any]]:
        rows = self.con.execute(
            "SELECT * FROM image_view WHERE rating IS NOT NULL OR flagged = 1"
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.con.close()
