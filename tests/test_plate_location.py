"""Colouring by where a point sat on the plate, and the export headline.

Both exist to make an artefact visible that the data alone does not show: the
plate-location encodings reveal geometry-shaped structure (edge effects, a
row-wise dispensing error) that would otherwise be read as biology, and the
headline stops an exported figure from being an unidentifiable field of dots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plato.data import plate_location as pl
from plato.wells import PLATE_GEOMETRIES


def _full_plate(rows: int, cols: int) -> list[str]:
    return [f"{chr(65 + r)}{c:02d}" for r in range(rows) for c in range(1, cols + 1)]


# -- geometry detection --------------------------------------------------------


@pytest.mark.parametrize("size", [96, 384])
def test_detects_the_geometry_it_was_given(size):
    rows, cols = PLATE_GEOMETRIES[size]
    assert pl.detect_format(pd.Series(_full_plate(rows, cols))) == size


def test_a_random_subset_of_a_96_well_plate_still_reads_as_96():
    """Subsampling is the norm here, so detection must survive it.

    detect_format reads the largest coordinates present, so in principle a
    subset could infer a smaller geometry. In practice one well past row F or
    column 8 settles it -- this pins that down to 5% of a plate.
    """
    full = _full_plate(8, 12)
    rng = np.random.default_rng(0)
    for fraction in (0.5, 0.25, 0.1, 0.05):
        n = max(2, int(len(full) * fraction))
        subset = list(rng.choice(full, size=n, replace=False))
        assert pl.detect_format(pd.Series(subset)) == 96


def test_values_that_are_not_wells_detect_nothing():
    assert pl.detect_format(pd.Series(["gyrA", "ftsZ", "murA"])) is None


def test_mostly_unparseable_values_detect_nothing():
    """One stray well among labels is not a well column."""
    assert pl.detect_format(pd.Series(["gyrA", "ftsZ", "murA", "rpoB", "A01"])) is None


def test_an_empty_column_detects_nothing():
    assert pl.detect_format(pd.Series(["", "", ""])) is None


# -- derived columns -----------------------------------------------------------


@pytest.fixture
def plate_frame():
    frame = pd.DataFrame({"well": _full_plate(8, 12)})
    added = pl.add_columns(frame)
    return frame, added


def test_adds_all_three_location_columns(plate_frame):
    _, added = plate_frame
    assert added == [pl.WELL_ROW, pl.WELL_COL, pl.EDGE_DISTANCE]


def test_rows_are_letters_and_columns_are_zero_padded(plate_frame):
    frame, _ = plate_frame
    assert sorted(frame[pl.WELL_ROW].unique()) == list("ABCDEFGH")
    # Zero-padded so a plain string sort orders 02 before 10. Unpadded, the
    # palette assigns colours in the order 1, 10, 11, 12, 2, ... and a
    # column gradient reads as noise.
    assert sorted(frame[pl.WELL_COL].unique()) == [f"{c:02d}" for c in range(1, 13)]


def test_edge_distance_counts_rings_inward(plate_frame):
    frame, _ = plate_frame
    by_well = frame.set_index("well")[pl.EDGE_DISTANCE]
    assert by_well["A01"] == "0"  # corner
    assert by_well["A06"] == "0"  # middle of the top row, still an outer face
    assert by_well["H12"] == "0"  # opposite corner
    assert by_well["B02"] == "1"
    assert by_well["D06"] == "3"  # centre of an 8x12 plate


def test_ring_sizes_partition_the_plate(plate_frame):
    """The arithmetic check: rings must tile the plate exactly.

    An 8x12 outer ring is 2*(8+12)-4 = 36 wells, and every well belongs to
    exactly one ring.
    """
    frame, _ = plate_frame
    counts = frame[pl.EDGE_DISTANCE].value_counts().to_dict()
    assert counts == {"0": 36, "1": 28, "2": 20, "3": 12}
    assert sum(counts.values()) == 96


def test_edge_distance_is_symmetric_under_reflection():
    """A ring is defined by its distance to the nearest edge, whichever one."""
    for row, col in ((1, 1), (8, 12), (1, 12), (8, 1)):
        assert pl.edge_distance(row, col, 8, 12) == 0
    assert pl.edge_distance(4, 6, 8, 12) == pl.edge_distance(5, 7, 8, 12)


def test_unparseable_rows_go_blank_without_losing_the_column():
    """A few bad values must not suppress the encoding for the good ones."""
    frame = pd.DataFrame({"well": ["A01", "B02", "", "junk", "H12"]})
    added = pl.add_columns(frame)
    assert added  # still offered
    assert list(frame[pl.WELL_ROW]) == ["A", "B", "", "", "H"]
    assert list(frame[pl.EDGE_DISTANCE]) == ["0", "1", "", "", "0"]


def test_a_frame_without_a_well_column_gets_nothing():
    frame = pd.DataFrame({"gene": ["gyrA"]})
    assert pl.add_columns(frame) == []
    assert list(frame.columns) == ["gene"]


def test_a_frame_whose_wells_are_labels_gets_nothing():
    """The signal for the UI to not offer these options at all."""
    frame = pd.DataFrame({"well": ["gyrA", "ftsZ"]})
    assert pl.add_columns(frame) == []
    assert pl.WELL_ROW not in frame.columns


# -- reaching the colour dropdown ----------------------------------------------


def test_build_frame_derives_location_for_a_real_plate():
    """Derived in the join layer, so every path that builds a frame has them."""
    from pathlib import Path

    from plato.data.embeddings import EmbeddingDataset
    from plato.data.explorer_model import build_frame, colour_fields

    wells = _full_plate(8, 12)
    raw = pd.DataFrame(
        {
            "well": wells,
            "gene": [["gyrA", "ftsZ"][i % 2] for i in range(len(wells))],
            "image_name": [f"img{i}" for i in range(len(wells))],
        }
    )
    dataset = EmbeddingDataset(
        name="d",
        directory=Path("/x"),
        vectors=np.zeros((len(wells), 4), dtype=np.float32),
        frame=raw,
        run_info={},
    )
    frame, _ = build_frame(dataset)
    offered = colour_fields(frame)
    for column in pl.LOCATION_FIELDS:
        assert column in frame.columns
        assert column in offered


def test_location_options_are_absent_when_there_is_no_plate_geometry():
    from pathlib import Path

    from plato.data.embeddings import EmbeddingDataset
    from plato.data.explorer_model import build_frame, colour_fields

    raw = pd.DataFrame(
        {"gene": ["gyrA", "ftsZ", "murA"], "image_name": ["a", "b", "c"]}
    )
    dataset = EmbeddingDataset(
        name="d",
        directory=Path("/x"),
        vectors=np.zeros((3, 4), dtype=np.float32),
        frame=raw,
        run_info={},
    )
    frame, _ = build_frame(dataset)
    offered = colour_fields(frame)
    for column in pl.LOCATION_FIELDS:
        assert column not in offered


def test_every_location_field_has_a_human_label():
    from plato.data.explorer_model import field_label

    for column in pl.LOCATION_FIELDS:
        label = field_label(column)
        assert label.startswith("Plate location:")
        # Not the raw column name falling through the default.
        assert "_" not in label
