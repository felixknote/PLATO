"""A session holding both experiment arms at once.

The FK_P001 screen is two kinds of plate side by side: CRISPRi plates whose
wells are ``ftsZ_2`` (gene + guide) and antibiotic plates whose wells are
``Ciprofloxacin 1x`` (drug + dose). Their plate maps therefore produce
*different metadata columns*, which is the case a single-plate session and the
existing same-plate-copied-twice tests can never exercise.

What has to hold:

* Both plates load into one session and every image is reachable.
* The filter sidebar offers the union of both plates' columns, so a column
  only one arm has is still selectable.
* Filtering on a column the other plate does not have returns only the plate
  that has it, rather than raising on the plate that does not.
* Annotations route back to the plate that owns the row.
* Exporting a mixed session produces a row per marked image, with blanks
  where a plate has no such column.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from plato.config import (
    Config,
    GuiConfig,
    ImagesConfig,
    PlatemapConfig,
    ProjectConfig,
    ThumbnailsConfig,
)
from plato.cache import build_thumbnails
from plato.data.index import build_index
from plato.data.session import SOURCE_PLATE, Session

NIS_PATTERN = (
    r"^Well(?P<well>[A-P]\d{1,2})_Point(?P<point>[^_]+)_(?P<field>\d+)"
    r"_Channel(?P<channel>.+?)_Seq(?P<seq>\d+)\.tiff?$"
)

GENE_GUIDE_SPLIT = r"^(?P<gene>.+)_(?P<replicate>\d+)$"
DRUG_DOSE_SPLIT = r"^(?P<antibiotic>[^\n]+)\n(?P<concentration>.+)$"

# Two wells per plate is enough to exercise the join; the point of this test
# is heterogeneity, not volume.
WELLS = ["A01", "A02"]


def _make_plate(
    root: Path,
    name: str,
    conditions: dict[str, str],
    split_pattern: str,
) -> Config:
    """Build one plate: images, a matrix plate map, and an index."""
    images = root / name / "images"
    images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for well in WELLS:
        for field in range(2):
            plane = rng.integers(0, 4096, size=(32, 32), dtype=np.uint16)
            tifffile.imwrite(
                images
                / f"Well{well}_Point{well}_{field:04d}"
                f"_ChannelCam-DIA DIC Master Screening_Seq{field:04d}.tiff",
                plane,
            )

    # An 8x12 matrix map, headerless -- position is the well, exactly as the
    # real Plate_Maps CSVs are laid out.
    grid = [["" for _ in range(12)] for _ in range(8)]
    for well, condition in conditions.items():
        row = "ABCDEFGH".index(well[0])
        grid[row][int(well[1:]) - 1] = condition
    platemap = root / name / f"{name}.csv"
    pd.DataFrame(grid).to_csv(platemap, header=False, index=False)

    cfg = Config(
        project=ProjectConfig(name=name, work_dir=root / name / ".plato", plate_format=96),
        images=ImagesConfig(dir=images, glob="**/*.tif*", pattern=NIS_PATTERN),
        platemap=PlatemapConfig(
            path=platemap,
            layout="matrix",
            header_row=False,
            index_col=False,
            value_column="Condition",
            split_pattern=split_pattern,
            default_plate=name,
        ),
        thumbnails=ThumbnailsConfig(size=32),
        gui=GuiConfig(),
        source=None,
    )
    build_index(cfg, verbose=False)
    # Thumbnails too: this is what writes display_limits, which the session
    # pools across plates.
    build_thumbnails(
        cfg.db_path,
        cfg.thumb_db_path,
        size=cfg.thumbnails.size,
        sample_size=4,
        verbose=False,
    )
    return cfg


@pytest.fixture(scope="module")
def mixed(tmp_path_factory):
    """One CRISPRi plate and one antibiotic plate in a single session."""
    root = tmp_path_factory.mktemp("mixed")
    crispri = _make_plate(
        root,
        "CRISPRi_P1",
        {"A01": "ftsZ_1", "A02": "gyrA_2"},
        GENE_GUIDE_SPLIT,
    )
    abx = _make_plate(
        root,
        "ABx_P1",
        {"A01": "Ciprofloxacin\n1x", "A02": "Colistin\n2x"},
        DRUG_DOSE_SPLIT,
    )
    session = Session()
    session.add_plate(crispri)
    session.add_plate(abx)
    yield session
    session.close()


def test_both_arms_load(mixed):
    assert len(mixed.plates) == 2
    # 2 wells x 2 fields x 2 plates
    assert mixed.count() == 8
    assert len(mixed.query()) == 8


def test_filter_columns_are_the_union(mixed):
    columns = mixed.filter_columns()
    # Each arm contributes its own vocabulary, and both survive.
    assert "gene" in columns
    assert "antibiotic" in columns
    assert "concentration" in columns
    # With two plates loaded the session offers its own disambiguated name.
    assert SOURCE_PLATE in columns


def test_filtering_on_one_arms_column_returns_only_that_arm(mixed):
    """The other plate has no such column and must be skipped, not raise."""
    rows = mixed.query({"gene": ["ftsZ"]})
    assert rows
    assert {row.plate for row in rows} == {"CRISPRi_P1"}

    rows = mixed.query({"antibiotic": ["Colistin"]})
    assert rows
    assert {row.plate for row in rows} == {"ABx_P1"}


def test_distinct_spans_both_arms(mixed):
    assert set(mixed.distinct("gene")) == {"ftsZ", "gyrA"}
    assert set(mixed.distinct("antibiotic")) == {"Ciprofloxacin", "Colistin"}


def test_search_matches_across_arms(mixed):
    assert {r.plate for r in mixed.query(search="ftsZ")} == {"CRISPRi_P1"}
    assert {r.plate for r in mixed.query(search="Colistin")} == {"ABx_P1"}


def test_source_plate_filter_selects_one_arm(mixed):
    rows = mixed.query({SOURCE_PLATE: ["ABx_P1"]})
    assert rows
    assert {row.plate for row in rows} == {"ABx_P1"}


def test_annotations_route_to_the_owning_plate(mixed):
    """image_id is only unique within a plate, so a mixed session is where a
    misrouted annotation would show up."""
    abx_row = next(r for r in mixed.query() if r.plate == "ABx_P1")
    mixed.set_annotation(abx_row, flagged=True, rating=4)

    exported = mixed.export_annotations()
    assert len(exported) == 1
    assert exported[0]["session_plate"] == "ABx_P1"
    assert exported[0]["rating"] == 4

    # And the flag is visible when querying that plate back.
    flagged = mixed.query(flagged_only=True)
    assert len(flagged) == 1
    assert flagged[0].plate == "ABx_P1"

    mixed.set_annotation(abx_row, flagged=False, rating=None)


def test_export_unions_columns_across_arms(mixed):
    """A mixed export must carry both vocabularies, blank where absent."""
    rows = mixed.query()
    crispri_row = next(r for r in rows if r.plate == "CRISPRi_P1")
    abx_row = next(r for r in rows if r.plate == "ABx_P1")
    mixed.set_annotation(crispri_row, flagged=True)
    mixed.set_annotation(abx_row, flagged=True)

    exported = mixed.export_annotations()
    assert len(exported) == 2
    fieldnames: list[str] = []
    for row in exported:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    assert "gene" in fieldnames
    assert "antibiotic" in fieldnames

    for row in exported:
        mixed.set_annotation(
            next(r for r in rows if r.image_id == row["image_id"] and r.plate == row["plate"]),
            flagged=False,
        )


def test_display_limits_are_pooled(mixed):
    """One window across both plates, so a difference on screen is a
    difference in the sample rather than in scaling."""
    limits = mixed.display_limits()
    assert limits
    for lo, hi in limits.values():
        assert hi > lo
