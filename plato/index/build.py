"""Build the index and report on what did not line up.

The report is the point. A left join that silently drops half your treated
wells is the failure mode that survives into a figure, so this module refuses
to be quiet about mismatches.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Config
from ..wells import WellParseError, parse_well
from . import db
from .filenames import ImageRecord, parse_files
from .platemap import plate_from_filename, read_platemap

MAX_EXAMPLES = 20

# The checks that decide whether the join is trustworthy, as
# (attribute, human-readable label). Both `ok` and `summary()` are driven from
# this, so a new check is added in one place rather than three.
PROBLEM_FIELDS: list[tuple[str, str]] = [
    ("unparsed_files", "filenames not parsed"),
    ("images_without_platemap", "images with no plate map row"),
    ("platemap_wells_without_images", "plate map wells with no image"),
    ("duplicate_platemap_keys", "duplicate plate map keys (extra rows dropped)"),
    ("duplicate_image_keys", "duplicate image keys"),
    ("bad_platemap_wells", "unparseable plate map wells"),
    ("unsplit_platemap_values", "plate map cells not split by pattern"),
    ("well_point_mismatches", "filename Well/Point disagreements"),
    ("field_count_outliers", "wells with unusual field count"),
    ("channel_count_outliers", "wells with unusual channel count"),
]


@dataclass(slots=True)
class ValidationReport:
    n_files_seen: int = 0
    n_files_parsed: int = 0
    n_platemap_rows: int = 0

    unparsed_files: list[dict[str, str]] = field(default_factory=list)
    images_without_platemap: list[str] = field(default_factory=list)
    platemap_wells_without_images: list[str] = field(default_factory=list)
    duplicate_platemap_keys: list[list[str]] = field(default_factory=list)
    duplicate_image_keys: list[str] = field(default_factory=list)
    bad_platemap_wells: list[dict[str, Any]] = field(default_factory=list)
    empty_platemap_cells: list[str] = field(default_factory=list)
    unsplit_platemap_values: list[dict[str, str]] = field(default_factory=list)
    well_point_mismatches: list[dict[str, str]] = field(default_factory=list)
    duplicate_rows_dropped: int = 0
    field_count_outliers: list[dict[str, Any]] = field(default_factory=list)
    channel_count_outliers: list[dict[str, Any]] = field(default_factory=list)

    truncated: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(getattr(self, name) for name, _ in PROBLEM_FIELDS)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")

    def summary(self) -> str:
        lines = [
            f"files seen:        {self.n_files_seen}",
            f"files parsed:      {self.n_files_parsed}",
            f"plate map rows:    {self.n_platemap_rows}",
        ]
        for name, label in PROBLEM_FIELDS:
            count = len(getattr(self, name))
            marker = "  " if count == 0 else "! "
            lines.append(f"{marker}{label}: {count}")
        if self.truncated:
            listed = ", ".join(f"{k} ({v} total)" for k, v in self.truncated.items())
            lines.append(f"  (examples truncated for: {listed})")
        return "\n".join(lines)


def _truncate(report: ValidationReport, name: str, items: list) -> list:
    if len(items) > MAX_EXAMPLES:
        report.truncated[name] = len(items)
        return items[:MAX_EXAMPLES]
    return items


def _well_point_mismatches(records: list[ImageRecord]) -> list[dict[str, str]]:
    """Filenames where the Well token and the Point name disagree.

    NIS-Elements repeats the well in the point name. When those two disagree
    the acquisition was renamed or re-pointed mid-run, and the well the image
    is filed under may not be the well it was taken from.
    """
    mismatches: list[dict[str, str]] = []
    for record in records:
        if not record.point:
            continue
        try:
            point_well = parse_well(record.point, None)
        except WellParseError:
            continue  # point name is not a well at all; nothing to compare
        if point_well != record.well:
            mismatches.append(
                {
                    "image_id": record.image_id,
                    "well": record.well.label,
                    "point": record.point,
                }
            )
    return mismatches


def _mode_outliers(
    records: list[ImageRecord], attribute: str
) -> list[dict[str, Any]]:
    """Wells whose number of distinct field/channel values differs from the mode."""
    per_well: dict[tuple[str, str], set[str]] = {}
    for rec in records:
        value = getattr(rec, attribute)
        if value is None:
            return []
        per_well.setdefault((rec.plate, rec.well.label), set()).add(value)
    if not per_well:
        return []
    counts = Counter(len(v) for v in per_well.values())
    expected, _ = counts.most_common(1)[0]
    return [
        {"plate": p, "well": w, "n": len(v), "expected": expected}
        for (p, w), v in sorted(per_well.items())
        if len(v) != expected
    ]


def build_index(cfg: Config, *, verbose: bool = True) -> ValidationReport:
    """Parse filenames + plate map, write the index, return the report."""
    report = ValidationReport()

    # A plate name in the plate map file name (e.g. ..._P1.csv) wins over the
    # config default, so a single config serves a series of per-plate maps.
    plate_name = (
        plate_from_filename(cfg.platemap.path, cfg.platemap.plate_pattern)
        or cfg.platemap.default_plate
    )

    records, unparsed = parse_files(
        cfg.images.dir,
        cfg.images.glob,
        cfg.images.pattern,
        plate_format=cfg.project.plate_format,
        default_plate=plate_name,
        match_on=cfg.images.match_on,
        channel_aliases=cfg.images.channel_aliases,
    )
    report.n_files_seen = len(records) + len(unparsed)
    report.n_files_parsed = len(records)
    report.unparsed_files = _truncate(
        report,
        "unparsed_files",
        [{"path": str(u.path), "reason": u.reason} for u in unparsed],
    )
    if not records:
        raise RuntimeError(
            f"no image filenames matched the pattern under {cfg.images.dir}. "
            "Check [images].pattern and [images].glob."
        )

    pmap = read_platemap(
        cfg.platemap.path,
        layout=cfg.platemap.layout,
        sheet=cfg.platemap.sheet,
        well_column=cfg.platemap.well_column,
        plate_column=cfg.platemap.plate_column,
        header_row=cfg.platemap.header_row,
        index_col=cfg.platemap.index_col,
        value_column=cfg.platemap.value_column,
        split_pattern=cfg.platemap.split_pattern,
        default_plate=plate_name,
        plate_format=cfg.project.plate_format,
    )
    report.n_platemap_rows = len(pmap.frame)
    report.bad_platemap_wells = _truncate(
        report,
        "bad_platemap_wells",
        [{"row": r, "value": str(v), "reason": msg} for r, v, msg in pmap.bad_wells],
    )
    report.empty_platemap_cells = _truncate(
        report, "empty_platemap_cells", list(pmap.empty_cells)
    )
    report.unsplit_platemap_values = _truncate(
        report,
        "unsplit_platemap_values",
        [{"well": w, "value": v} for w, v in pmap.unsplit_values],
    )
    report.well_point_mismatches = _truncate(
        report,
        "well_point_mismatches",
        _well_point_mismatches(records),
    )
    report.duplicate_platemap_keys = _truncate(
        report,
        "duplicate_platemap_keys",
        [list(k) for k in pmap.duplicate_keys],
    )
    if pmap.duplicate_keys:
        # A duplicated (plate, well) would fan out the join and show the same
        # image several times, which reads as extra replicates. Dropping the
        # extra rows is wrong but visible; fanning out is wrong and invisible.
        before = len(pmap.frame)
        pmap.frame = pmap.frame.drop_duplicates(subset=["plate", "well"], keep="first")
        report.duplicate_rows_dropped = before - len(pmap.frame)
        report.n_platemap_rows = len(pmap.frame)

    # -- key coverage ------------------------------------------------------
    image_keys = {(r.plate, r.well.label) for r in records}
    map_keys = set(zip(pmap.frame["plate"], pmap.frame["well"], strict=True))

    report.images_without_platemap = _truncate(
        report,
        "images_without_platemap",
        sorted(f"{p}/{w}" for p, w in image_keys - map_keys),
    )
    report.platemap_wells_without_images = _truncate(
        report,
        "platemap_wells_without_images",
        sorted(f"{p}/{w}" for p, w in map_keys - image_keys),
    )

    id_counts = Counter(r.image_id for r in records)
    report.duplicate_image_keys = _truncate(
        report, "duplicate_image_keys", sorted(k for k, n in id_counts.items() if n > 1)
    )

    report.field_count_outliers = _truncate(
        report, "field_count_outliers", _mode_outliers(records, "field")
    )
    report.channel_count_outliers = _truncate(
        report, "channel_count_outliers", _mode_outliers(records, "channel")
    )

    # -- write -------------------------------------------------------------
    con = db.connect(cfg.db_path)
    db.initialise(con)
    con.execute("DELETE FROM images")
    con.executemany(
        "INSERT OR REPLACE INTO images "
        "(image_id, path, plate, well, well_row, well_col, field, channel, z, "
        "seq, point, mtime, size_bytes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                r.image_id,
                str(r.path),
                r.plate,
                r.well.label,
                r.well.row,
                r.well.col,
                r.field,
                r.channel,
                r.z,
                r.seq,
                r.point,
                r.mtime,
                r.size_bytes,
            )
            for r in records
        ],
    )

    frame = pmap.frame.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            # NaN must become None explicitly: sqlite3 cannot bind float('nan')
            # in a way that "IS NOT NULL" filters treat as missing.
            frame[column] = frame[column].map(
                lambda v: None
                if v is None or (isinstance(v, float) and pd.isna(v))
                else (v if isinstance(v, (int, float, str)) else str(v))
            )
    frame.to_sql("platemap", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS idx_platemap_key ON platemap (plate, well)")

    metadata_columns = list(pmap.column_labels)
    db.set_meta(con, "metadata_columns", metadata_columns)
    db.set_meta(con, "column_labels", pmap.column_labels)
    db.set_meta(con, "image_root", str(cfg.images.dir))
    db.set_meta(con, "project_name", cfg.project.name)
    con.commit()
    db.create_image_view(con, metadata_columns)
    con.close()

    cfg.report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_json(cfg.report_path)
    if verbose:
        print(report.summary())
        print(f"\nindex:  {cfg.db_path}")
        print(f"report: {cfg.report_path}")
    return report


def load_dataframe(cfg: Config) -> pd.DataFrame:
    """The whole joined index as a DataFrame, for notebook use."""
    con = db.connect(cfg.db_path, read_only=True)
    try:
        return pd.read_sql_query("SELECT * FROM image_view", con)
    finally:
        con.close()
