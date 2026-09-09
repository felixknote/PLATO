"""Multi-plate session behaviour. Run: pytest -q

These are the parts that only go wrong once a second plate is loaded, and that
a single-plate session can never exercise: plate identity, pooled contrast,
and merging two sorted result sets into one.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from plato.config import load_config
from plato.data.session import SOURCE_PLATE, DuplicatePlateError, Session

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_demo_data.py"


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    """A demo screen, indexed and thumbnailed once for the whole module."""
    out = tmp_path_factory.mktemp("session_demo")
    subprocess.run(
        [sys.executable, str(SCRIPT), str(out), "--size", "32"],
        check=True,
        capture_output=True,
    )
    config = out / "plato.toml"
    subprocess.run(
        [sys.executable, "-m", "plato.cli", "index", "-c", str(config)],
        check=True,
        capture_output=True,
    )
    return config


def _second_plate(config: Path, name: str, work_dir_name: str):
    """A config pointing at a *copy* of the built index, so the session sees
    a genuinely different plate without indexing a second dataset."""
    work_dir = config.parent / work_dir_name
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copytree(config.parent / ".plato", work_dir)
    cfg = load_config(config)
    cfg.project.work_dir = work_dir
    cfg.project.name = name
    return cfg


@pytest.fixture
def session(built: Path):
    s = Session()
    s.add_plate(load_config(built))
    yield s
    s.close()


# -- plate identity -------------------------------------------------------


def test_adding_the_same_index_twice_is_refused(session, built: Path) -> None:
    """Loading a plate twice would show every image twice and split its flags
    across two copies, so it raises rather than doubling the grid."""
    before = session.count()
    with pytest.raises(DuplicatePlateError):
        session.add_plate(load_config(built))
    assert session.count() == before
    assert len(session.plates) == 1


def test_colliding_names_are_disambiguated(session, built: Path) -> None:
    first = session.plates[0].name
    added = session.add_plate(_second_plate(built, first, "collide"))
    assert added.name == f"{first} (2)"
    assert len(set(session.plate_names)) == 2


def test_removing_a_plate_leaves_the_rest_queryable(session, built: Path) -> None:
    one = session.count()
    session.add_plate(_second_plate(built, "other", "removable"))
    assert session.count() == one * 2
    session.remove_plate(1)
    assert len(session.plates) == 1
    assert session.count() == one
    assert len(session.query()) == one


# -- the loaded-plate filter ----------------------------------------------


def test_source_plate_is_only_offered_when_there_is_a_choice(
    session, built: Path
) -> None:
    assert SOURCE_PLATE not in session.filter_columns()
    session.add_plate(_second_plate(built, "other", "choice"))
    assert SOURCE_PLATE in session.filter_columns()


def test_source_plate_selects_one_plate_where_the_plate_column_cannot(
    session, built: Path
) -> None:
    """Both plates were indexed from the same config, so both label every row
    with the same `plate` value -- filtering on it selects both. The loaded
    name is what distinguishes them."""
    session.add_plate(_second_plate(built, "other", "select"))
    plate_values = session.distinct("plate")
    assert len(plate_values) == 1, "the premise: the plate column cannot separate them"
    assert len(session.query({"plate": plate_values})) == session.count()

    rows = session.query({SOURCE_PLATE: ["other"]})
    assert rows
    assert {r.session_index for r in rows} == {1}
    assert len(rows) == session.count() // 2


def test_search_matches_the_loaded_plate_name(session, built: Path) -> None:
    """The name is session state, so no plate's index can match on it."""
    session.add_plate(_second_plate(built, "zzunique", "searchable"))
    rows = session.query(search="zzunique")
    assert rows
    assert {r.session_index for r in rows} == {1}


# -- contrast -------------------------------------------------------------


def test_display_limits_are_pooled_not_first_plate_wins(session, built: Path) -> None:
    """Applying the first plate's limits to every plate makes a brightness
    difference between plates a difference in scaling, in a view whose premise
    is that it is a difference in the sample."""
    session.add_plate(_second_plate(built, "bright", "limits"))
    # display_limits is written by `plato thumbs`, which this fixture does not
    # run, so seed the two plates with deliberately different windows.
    for db, (lo, hi) in zip(
        (session.plates[0].db, session.plates[1].db), ((100.0, 900.0), (50.0, 1500.0))
    ):
        db.con.execute(
            "INSERT OR REPLACE INTO display_limits (channel, lo, hi) VALUES ('DAPI', ?, ?)",
            (lo, hi),
        )
        db.con.commit()

    pooled = session.display_limits()
    assert pooled["DAPI"] == (50.0, 1500.0), (
        "the pooled window must contain both plates' data, not just the first's"
    )


# -- merging --------------------------------------------------------------


def test_limit_applies_after_the_merge_not_per_plate(session, built: Path) -> None:
    """Forwarding `limit` to each plate would return each plate's own first N
    and then keep the global first N of those, which is not the global first
    N -- one plate can rightly own every row of the answer."""
    session.add_plate(_second_plate(built, "other", "limited"))
    full = session.query()
    limited = session.query(limit=10)
    assert len(limited) == 10
    assert [r.image_id for r in limited] == [r.image_id for r in full[:10]]


def test_merged_rows_group_by_loaded_plate(session, built: Path) -> None:
    """Sorting on the `plate` column alone interleaves two plates that share
    that label, landing A01 of each adjacent and indistinguishable."""
    session.add_plate(_second_plate(built, "other", "grouped"))
    indices = [r.session_index for r in session.query()]
    assert indices == sorted(indices), "each plate should be contiguous"


def test_annotations_are_tagged_with_the_plate_they_came_from(
    session, built: Path
) -> None:
    session.add_plate(_second_plate(built, "other", "annotated"))
    for index in (0, 1):
        row = session.query({SOURCE_PLATE: [session.plates[index].name]})[0]
        session.set_annotation(row, flagged=True)

    exported = session.export_annotations()
    assert {row["session_plate"] for row in exported} == set(session.plate_names)
    assert len({row["uid"] for row in exported}) == len(exported), "uids must be unique"
