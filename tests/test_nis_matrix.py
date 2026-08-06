"""Tests for the layouts and filename grammar of the real screening data.

Fixtures reproduce the two formats exactly:

* NIS-Elements names, e.g.
  ``WellA01_PointA01_0000_ChannelCam-DIA DIC Master Screening_Seq0000.tiff``
* A headerless 8x12 grid plate map whose cells are ``<condition>_<replicate>``,
  with scrambled positions and control names containing spaces and hyphens.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile

from plato.config import load_config
from plato.index import IndexDB, build_index
from plato.index.filenames import compile_pattern, parse_files
from plato.index.platemap import plate_from_filename, read_matrix_platemap

NIS_PATTERN = (
    r"^Well(?P<well>[A-P]\d{1,2})_Point(?P<point>[^_]+)_(?P<field>\d+)"
    r"_Channel(?P<channel>.+?)_Seq(?P<seq>\d+)\.tiff?$"
)
SPLIT = r"^(?P<gene>.+)_(?P<replicate>\d+)$"

GRID = [
    ["rplA_3", "folP_3", "parE_1", "lpxC_2", "ftsZ_3", "ACE-1 NC_6",
     "parE_3", "rpsL_1", "ACE-1 NC_5", "murC_3", "lptA_2", "rpsA_1"],
    ["mrcB_1", "lpxA_2", "MG1655 NC_6", "mrdA_3", "lptC_2", "dnaE_1",
     "ACE-1 NC_4", "ACE-1 NC_3", "ftsZ_2", "rplC_3", "rpoA_1", "folP_2"],
    ["rplA_1", "gyrA_3", "rpoB_1", "rpsL_2", "gyrB_3", "secY_3",
     "mrcA_2", "ftsI_3", "rpoB_2", "rplC_1", "dnaE_2", "ftsI_2"],
    ["folA_2", "mrcB_3", "lpxA_1", "rplC_2", "msbA_1", "murA_1",
     "ftsZ_1", "MG1655 NC_5", "dnaB_2", "rpsA_3", "rplA_2", "rpoA_2"],
    ["parC_2", "dnaE_3", "rpoA_3", "mrcA_1", "dnaB_1", "folA_1",
     "lpxC_3", "mrdA_1", "folA_3", "rpsL_3", "mrdA_2", "gyrA_1"],
    ["msbA_3", "murC_2", "MG1655 NC_4", "murC_1", "dnaB_3", "secY_1",
     "lpxA_3", "gyrB_1", "lptC_3", "rpsA_2", "folP_1", "secY_2"],
    ["lptA_3", "parE_2", "MG1655 NC_1", "gyrA_2", "MG1655 NC_3", "ftsI_1",
     "rpoB_3", "MG1655 NC_2", "secA_3", "mrcB_2", "secA_2", "murA_2"],
    ["mrcA_3", "murA_3", "parC_3", "lptA_1", "ACE-1 NC_2", "lptC_1",
     "secA_1", "ACE-1 NC_1", "parC_1", "msbA_2", "gyrB_2", "lpxC_1"],
]


def write_grid(path, rows=GRID) -> None:
    path.write_text("\r\n".join(",".join(r) for r in rows) + "\r\n", encoding="utf-8")


@pytest.fixture
def screen(tmp_path):
    """A 96-well NIS-style screen: 2 points per well, one DIC channel."""
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.default_rng(0)
    plane = rng.integers(18000, 37000, size=(32, 32), dtype=np.uint16)
    for row_idx, letter in enumerate("ABCDEFGH"):
        for col in range(1, 13):
            well = f"{letter}{col:02d}"
            for point in range(2):
                # A01 keeps the space-separated channel name, the rest use the
                # underscore-sanitised variant. Both must parse.
                channel = (
                    "Cam-DIA DIC Master Screening"
                    if well == "A01"
                    else "Cam-DIA_DIC_Master_Screening"
                )
                name = f"Well{well}_Point{well}_{point:04d}_Channel{channel}_Seq0000.tiff"
                tifffile.imwrite(images / name, plane)

    map_path = tmp_path / "scrambled_plate_map_20260717_101617_P1.csv"
    write_grid(map_path)

    config = tmp_path / "plato.toml"
    config.write_text(
        f"""[project]
name = "nis"
work_dir = "./.plato"
plate_format = 96

[images]
dir = "./images"
glob = "**/*.tif*"
pattern = '{NIS_PATTERN}'
match_on = "name"

[images.channel_aliases]
"Cam-DIA DIC Master Screening" = "DIC"
"Cam-DIA_DIC_Master_Screening" = "DIC"

[platemap]
path = "./{map_path.name}"
layout = "matrix"
header_row = false
index_col = false
value_column = "Condition"
split_pattern = '{SPLIT}'
plate_pattern = '_(?P<plate>P\\d+)\\.'
default_plate = "unknown"

[thumbnails]
size = 64
percentiles = [1.0, 99.5]
sample_size = 20

[masks]
dir = ""
pattern = "{{stem}}_mask.tif"

[gui]
caption_fields = ["gene", "replicate"]
filter_fields = []
thumbnail_size = 180
""",
        encoding="utf-8",
    )
    return config


# -- NIS filename grammar -------------------------------------------------


def test_nis_pattern_captures_all_tokens() -> None:
    compiled = compile_pattern(NIS_PATTERN)
    match = compiled.match(
        "WellA01_PointA01_0000_ChannelCam-DIA DIC Master Screening_Seq0000.tiff"
    )
    assert match is not None
    assert match.groupdict() == {
        "well": "A01",
        "point": "A01",
        "field": "0000",
        "channel": "Cam-DIA DIC Master Screening",
        "seq": "0000",
    }


def test_channel_may_contain_spaces_or_underscores() -> None:
    """Upload/export tools silently swap spaces for underscores; both must work."""
    compiled = compile_pattern(NIS_PATTERN)
    underscored = compiled.match(
        "WellH12_PointH12_0003_ChannelCam-DIA_DIC_Master_Screening_Seq0007.tiff"
    )
    assert underscored is not None
    assert underscored.group("channel") == "Cam-DIA_DIC_Master_Screening"
    assert underscored.group("field") == "0003"
    assert underscored.group("seq") == "0007"


def test_channel_aliases_shorten_the_indexed_label(tmp_path) -> None:
    images = tmp_path / "img"
    images.mkdir()
    (images / "WellA01_PointA01_0000_ChannelCam-DIA DIC Master Screening_Seq0000.tiff").touch()
    records, unparsed = parse_files(
        images,
        "*.tiff",
        NIS_PATTERN,
        channel_aliases={"Cam-DIA DIC Master Screening": "DIC"},
    )
    assert not unparsed
    assert records[0].channel == "DIC"
    assert records[0].seq == "0000"
    assert records[0].point == "A01"


def test_unknown_capture_group_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown groups"):
        compile_pattern(r"^(?P<well>[A-P]\d{2})_(?P<timepoint>\d+)\.tif$")


# -- matrix plate map -----------------------------------------------------


def test_matrix_positions_map_to_wells(tmp_path) -> None:
    path = tmp_path / "map_P1.csv"
    write_grid(path)
    result = read_matrix_platemap(path, value_column="Condition", split_pattern=SPLIT)

    assert len(result.frame) == 96
    by_well = result.frame.set_index("well")
    assert by_well.loc["A01", "condition"] == "rplA_3"   # top-left
    assert by_well.loc["H12", "condition"] == "lpxC_1"   # bottom-right
    assert by_well.loc["D07", "condition"] == "ftsZ_1"


def test_split_pattern_separates_gene_from_replicate(tmp_path) -> None:
    path = tmp_path / "map_P1.csv"
    write_grid(path)
    result = read_matrix_platemap(path, value_column="Condition", split_pattern=SPLIT)
    frame = result.frame

    assert set(frame["replicate"]) == {"1", "2", "3", "4", "5", "6"}
    assert frame["gene"].nunique() == 30
    # Control names contain a space and a hyphen and must survive intact.
    assert "ACE-1 NC" in set(frame["gene"])
    assert (frame["gene"] == "ftsZ").sum() == 3
    assert not result.unsplit_values


def test_wrong_grid_shape_is_an_error_not_a_shift(tmp_path) -> None:
    path = tmp_path / "map_P1.csv"
    write_grid(path, rows=GRID[:7])  # a row went missing
    with pytest.raises(ValueError, match="7x12"):
        read_matrix_platemap(path, split_pattern=SPLIT)


def test_unsplittable_cells_are_reported_not_dropped(tmp_path) -> None:
    rows = [list(r) for r in GRID]
    rows[0][0] = "no_replicate_suffix_here!"
    path = tmp_path / "map_P1.csv"
    write_grid(path, rows=rows)
    result = read_matrix_platemap(path, split_pattern=SPLIT)

    assert result.unsplit_values == [("A01", "no_replicate_suffix_here!")]
    by_well = result.frame.set_index("well")
    assert by_well.loc["A01", "condition"] == "no_replicate_suffix_here!"
    assert pd.isna(by_well.loc["A01", "gene"])


def test_empty_cells_are_reported(tmp_path) -> None:
    rows = [list(r) for r in GRID]
    rows[3][5] = ""
    path = tmp_path / "map_P1.csv"
    write_grid(path, rows=rows)
    result = read_matrix_platemap(path, split_pattern=SPLIT)
    assert result.empty_cells == ["D06"]
    assert len(result.frame) == 95


def test_plate_name_comes_from_the_map_filename(tmp_path) -> None:
    path = tmp_path / "scrambled_plate_map_20260717_101617_P1.csv"
    write_grid(path)
    assert plate_from_filename(path, r"_(?P<plate>P\d+)\.") == "P1"
    assert plate_from_filename(path, r"_(?P<plate>Q\d+)\.") is None


# -- end to end -----------------------------------------------------------


def test_real_format_screen_indexes_cleanly(screen) -> None:
    cfg = load_config(screen)
    report = build_index(cfg, verbose=False)

    assert report.n_files_parsed == 192
    assert report.n_platemap_rows == 96
    assert report.ok, report.summary()


def test_scrambled_layout_is_browsable_by_gene(screen) -> None:
    """The point of the tool: ftsZ replicates sit in unrelated wells."""
    cfg = load_config(screen)
    build_index(cfg, verbose=False)
    db = IndexDB(cfg.db_path)
    try:
        assert db.distinct("plate") == ["P1"]
        assert db.distinct("channel") == ["DIC"]
        assert db.distinct("field") == ["0000", "0001"]

        ftsz = db.query({"gene": ["ftsZ"]})
        assert len(ftsz) == 6  # 3 replicates x 2 points
        wells = {r.well for r in ftsz}
        assert wells == {"A05", "B09", "D07"}
        assert {r.metadata["replicate"] for r in ftsz} == {"1", "2", "3"}

        assert len(db.query({"gene": ["ACE-1 NC"]})) == 12  # 6 replicates x 2
    finally:
        db.close()


def test_well_point_disagreement_is_flagged(screen) -> None:
    """A renamed point is how an image ends up filed under the wrong well."""
    cfg = load_config(screen)
    source = cfg.images.dir / (
        "WellB03_PointB03_0000_ChannelCam-DIA_DIC_Master_Screening_Seq0000.tiff"
    )
    source.rename(
        cfg.images.dir
        / "WellB03_PointG11_0000_ChannelCam-DIA_DIC_Master_Screening_Seq0000.tiff"
    )
    report = build_index(cfg, verbose=False)

    assert len(report.well_point_mismatches) == 1
    mismatch = report.well_point_mismatches[0]
    assert mismatch["well"] == "B03"
    assert mismatch["point"] == "G11"
    assert not report.ok
