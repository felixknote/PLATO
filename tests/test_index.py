"""Tests for the GUI-free layer. Run: pytest -q"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from plato.cache import build_thumbnails
from plato.config import load_config
from plato.data.index import IndexDB, build_index
from plato.wells import WellParseError, canonical, parse_well

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_demo_data.py"


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("demo")
    subprocess.run(
        [sys.executable, str(SCRIPT), str(out), "--size", "32", "--defects"],
        check=True,
        capture_output=True,
    )
    return out / "plato.toml"


# -- well canonicalisation ------------------------------------------------


@pytest.mark.parametrize("raw", ["A1", "A01", "a1", " a01 ", "A001"])
def test_padding_variants_collapse(raw: str) -> None:
    assert canonical(raw) == "A01"


def test_row_and_column_are_one_based() -> None:
    well = parse_well("H12")
    assert (well.row, well.col) == (8, 12)


@pytest.mark.parametrize("raw", ["H13", "I01", "", "well", "A0", None])
def test_out_of_plate_and_junk_rejected(raw) -> None:
    with pytest.raises(WellParseError):
        parse_well(raw, 96)


def test_384_accepts_what_96_rejects() -> None:
    assert canonical("P24", 384) == "P24"
    with pytest.raises(WellParseError):
        parse_well("P24", 96)


# -- indexing -------------------------------------------------------------


def test_build_index_reports_injected_defects(demo: Path) -> None:
    cfg = load_config(demo)
    report = build_index(cfg, verbose=False)

    assert report.n_files_parsed == 384
    assert len(report.unparsed_files) == 1  # stray_snapshot.tif
    assert len(report.bad_platemap_wells) == 1  # H13
    assert len(report.duplicate_platemap_keys) == 1
    assert report.duplicate_rows_dropped == 1
    assert not report.ok


def test_join_does_not_fan_out(demo: Path) -> None:
    cfg = load_config(demo)
    build_index(cfg, verbose=False)
    db = IndexDB(cfg.db_path)
    try:
        assert len(db.query()) == db.count() == 384
    finally:
        db.close()


def test_mixed_well_padding_still_joins(demo: Path) -> None:
    """The demo plate map deliberately mixes A01 and A1 styles."""
    cfg = load_config(demo)
    build_index(cfg, verbose=False)
    db = IndexDB(cfg.db_path)
    try:
        rows = db.query()
        assert all(r.metadata["gene"] is not None for r in rows)
    finally:
        db.close()


def test_filters_and_search(demo: Path) -> None:
    cfg = load_config(demo)
    build_index(cfg, verbose=False)
    db = IndexDB(cfg.db_path)
    try:
        dapi = db.query({"channel": ["DAPI"]})
        assert len(dapi) == 192
        assert {r.channel for r in dapi} == {"DAPI"}

        combined = db.query({"channel": ["DAPI"], "gene": ["ftsZ"]})
        assert 0 < len(combined) < len(dapi)

        assert db.query(search="Polymyxin")
        assert db.query(search="zzz-nothing") == []
    finally:
        db.close()


def test_annotations_survive_reindex(demo: Path) -> None:
    cfg = load_config(demo)
    build_index(cfg, verbose=False)
    db = IndexDB(cfg.db_path)
    image_id = db.query()[0].image_id
    db.set_annotation(image_id, rating=5, flagged=True)
    db.close()

    build_index(cfg, verbose=False)  # regenerate derived tables
    db = IndexDB(cfg.db_path)
    try:
        flagged = db.query(flagged_only=True)
        assert [r.image_id for r in flagged] == [image_id]
        assert flagged[0].rating == 5
    finally:
        db.close()


# -- thumbnails -----------------------------------------------------------


def test_thumbnails_are_incremental(demo: Path) -> None:
    cfg = load_config(demo)
    build_index(cfg, verbose=False)
    first = build_thumbnails(
        cfg.db_path, cfg.thumb_db_path, size=64, sample_size=20, verbose=False
    )
    assert first == 384
    second = build_thumbnails(
        cfg.db_path, cfg.thumb_db_path, size=64, sample_size=20, verbose=False
    )
    assert second == 0


def test_display_limits_are_shared_across_the_screen(demo: Path) -> None:
    cfg = load_config(demo)
    build_index(cfg, verbose=False)
    build_thumbnails(
        cfg.db_path, cfg.thumb_db_path, size=64, sample_size=20, verbose=False
    )
    db = IndexDB(cfg.db_path)
    try:
        limits = db.display_limits()
        assert set(limits) == {"DAPI", "FM464"}
        assert all(hi > lo for lo, hi in limits.values())
    finally:
        db.close()


def test_timepoint_is_derived_from_plate_name() -> None:
    from plato.data.session import timepoint_of

    assert timepoint_of("P13_T1") == "T1"
    assert timepoint_of("P5_T10") == "T10"
    assert timepoint_of("plate_t7") == "T7"  # case-insensitive
    # No suffix, or a trailing _T that is not a timepoint, yields nothing --
    # filter_columns() then never offers the column.
    assert timepoint_of("P1") == ""
    assert timepoint_of("Screen_TX") == ""
    assert timepoint_of("") == ""
