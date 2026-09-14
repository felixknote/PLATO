"""User-defined classes over a column's values, and the well readout.

Faceting splits data the way the metadata already splits it. "Which day was
this plate measured on" is nowhere in the export, but it is where a batch
effect lives -- so a custom grouping maps P1, P2 -> "Day 1" and materialises
as an ordinary derived column, which is what lets faceting, colouring, the
legend and the export headline all work on it unchanged.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from plato.data.custom_groups import (
    CustomGrouping,
    GroupingStore,
    derived_column,
    is_derived,
    source_of,
    suggest_classes,
)
from plato.data.embeddings import EmbeddingDataset
from plato.data.explorer_model import build_frame, distinct_values
from plato.data.workspace import EmbeddingEntry


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


# -- the mapping ---------------------------------------------------------------


def test_several_values_collapse_into_one_class():
    """The stated case: P1 and P2 are one facet because of when they ran."""
    grouping = CustomGrouping(
        source="plate",
        name="measurement day",
        assignments={"P1": "Day 1", "P2": "Day 1", "P3": "Day 2", "P4": "Day 2"},
    )
    out = grouping.apply(pd.Series(["P1", "P2", "P3", "P4"]))
    assert list(out) == ["Day 1", "Day 1", "Day 2", "Day 2"]


def test_unassigned_values_keep_their_own_name():
    """Never an 'other' bucket.

    Grouping four of eight plates and silently merging the rest would hide
    half the data behind one label; keeping them distinct makes a
    half-finished grouping visibly half-finished.
    """
    grouping = CustomGrouping(source="plate", assignments={"P1": "Day 1"})
    assert list(grouping.apply(pd.Series(["P1", "P7", "P8"]))) == ["Day 1", "P7", "P8"]


def test_whitespace_in_a_value_still_matches_its_assignment():
    grouping = CustomGrouping(source="plate", assignments={"P1": "Day 1"})
    assert list(grouping.apply(pd.Series([" P1 "]))) == ["Day 1"]


def test_classes_are_listed_in_first_assigned_order():
    grouping = CustomGrouping(
        source="plate",
        assignments={"P1": "Day 1", "P2": "Day 1", "P3": "Day 2"},
    )
    assert grouping.classes() == ["Day 1", "Day 2"]


def test_the_derived_column_is_named_after_its_source():
    """So two groupings over different columns can coexist."""
    assert derived_column("plate") == "group_plate"
    assert derived_column("drug") == "group_drug"
    assert is_derived("group_plate")
    assert not is_derived("plate")
    assert source_of("group_plate") == "plate"


def test_the_label_names_the_grouping_not_the_column():
    named = CustomGrouping(source="plate", name="measurement day", assignments={"P1": "D"})
    unnamed = CustomGrouping(source="plate", assignments={"P1": "D"})
    assert named.label == "Plate by measurement day"
    assert unnamed.label == "Plate (grouped)"


# -- the suggestion ------------------------------------------------------------


def test_suggestion_pairs_consecutive_values():
    """Two plates a day over four days is the common case by a wide margin."""
    out = suggest_classes([f"P{i}" for i in range(1, 9)], per_class=2, prefix="Day")
    assert out["P1"] == out["P2"] == "Day 1"
    assert out["P3"] == out["P4"] == "Day 2"
    assert out["P7"] == out["P8"] == "Day 4"


def test_suggestion_sorts_naturally_so_p10_follows_p9():
    """A plain string sort would put P10 between P1 and P2."""
    out = suggest_classes(["P1", "P2", "P10"], per_class=1)
    assert list(out) == ["P1", "P2", "P10"]


def test_suggestion_ignores_blanks():
    out = suggest_classes(["P1", "", "  ", "P2"], per_class=1)
    assert list(out) == ["P1", "P2"]


# -- the store -----------------------------------------------------------------


def test_one_grouping_per_source_column():
    """A second grouping over the same column would silently overwrite it."""
    store = GroupingStore()
    store.set(CustomGrouping(source="plate", name="a", assignments={"P1": "X"}))
    store.set(CustomGrouping(source="plate", name="b", assignments={"P1": "Y"}))
    assert len(store) == 1
    assert store.get("plate").name == "b"


def test_an_empty_grouping_removes_rather_than_stores():
    store = GroupingStore()
    store.set(CustomGrouping(source="plate", assignments={"P1": "X"}))
    store.set(CustomGrouping(source="plate", assignments={}))
    assert store.get("plate") is None


def test_apply_all_skips_a_column_this_dataset_does_not_have():
    """Definitions belong to the workspace; datasets differ."""
    store = GroupingStore()
    store.set(CustomGrouping(source="plate", assignments={"P1": "Day 1"}))
    frame = pd.DataFrame({"gene": ["gyrA"]})
    assert store.apply_all(frame) == []
    assert "group_plate" not in frame.columns


def test_round_trips_through_json():
    store = GroupingStore()
    store.set(
        CustomGrouping(
            source="plate", name="day", assignments={"P1": "Day 1", "P2": "Day 1"}
        )
    )
    restored = GroupingStore()
    restored.load_json(store.to_json())
    assert restored.get("plate").assignments == {"P1": "Day 1", "P2": "Day 1"}
    assert restored.get("plate").name == "day"


def test_corrupt_json_leaves_an_empty_store_rather_than_raising():
    """A bad settings value must not stop the explorer opening."""
    store = GroupingStore()
    store.load_json("{not json")
    assert len(store) == 0


def test_saves_and_loads_from_a_file(tmp_path):
    store = GroupingStore()
    store.set(CustomGrouping(source="plate", name="day", assignments={"P1": "Day 1"}))
    path = tmp_path / "groups.json"
    store.save_to(path)
    restored = GroupingStore()
    restored.load_from(path)
    assert restored.get("plate").assignments == {"P1": "Day 1"}


def test_loading_a_missing_file_is_not_an_error(tmp_path):
    store = GroupingStore()
    store.load_from(tmp_path / "nope.json")
    assert len(store) == 0


# -- reaching the explorer -----------------------------------------------------


def _frame(n_plates: int = 8):
    wells = [f"{chr(65 + r)}{c:02d}" for r in range(8) for c in range(1, 13)]
    rows = [(f"P{p}", w) for p in range(1, n_plates + 1) for w in wells]
    raw = pd.DataFrame(
        {
            "plate": [p for p, _ in rows],
            "well": [w for _, w in rows],
            "gene": [["gyrA", "ftsZ"][i % 2] for i in range(len(rows))],
            "image_name": [f"i{i}" for i in range(len(rows))],
        }
    )
    dataset = EmbeddingDataset(
        name="d",
        directory=Path("/x"),
        vectors=np.zeros((len(rows), 8), dtype=np.float32),
        frame=raw,
        run_info={},
    )
    frame, _ = build_frame(dataset)
    return dataset, frame


@pytest.fixture
def explorer(app):
    class _Session:
        pass

    dataset, frame = _frame()
    widget = EmbeddingExplorerFactory(_Session(), dataset, frame)
    return widget


def EmbeddingExplorerFactory(session, dataset, frame):
    from plato.views.explorer import EmbeddingExplorer

    widget = EmbeddingExplorer(session, Path(tempfile.mkdtemp()))
    widget.set_image_mode(False)
    widget._add_entry(EmbeddingEntry(name="d", dataset=dataset, frame=frame))
    return widget


def test_a_grouping_becomes_its_own_display_by_option(explorer):
    from plato.data.custom_groups import CustomGrouping as CG

    grouping = CG(
        source="plate",
        name="measurement day",
        assignments=suggest_classes(
            distinct_values(explorer.frame, "plate"), per_class=2, prefix="Day"
        ),
    )
    explorer.groupings.set(grouping)
    explorer.groupings.apply_all(explorer.frame)
    explorer._rebuild_group_options()

    index = explorer.group_box.findData(grouping.column)
    assert index >= 0
    assert explorer.group_box.itemText(index) == "Plate by measurement day"


def test_a_grouping_is_also_offered_as_a_colour(explorer):
    """'Are the day-1 plates displaced?' is a colour question first."""
    from plato.data.custom_groups import CustomGrouping as CG

    grouping = CG(source="plate", name="day", assignments={"P1": "Day 1", "P2": "Day 1"})
    explorer.groupings.set(grouping)
    explorer.groupings.apply_all(explorer.frame)
    explorer._rebuild_colour_options()
    assert explorer.colour_box.findData(grouping.column) >= 0


def test_the_derived_column_facets_into_the_expected_groups(explorer):
    from plato.data.custom_groups import CustomGrouping as CG
    from plato.views.grid_view import build_groups

    grouping = CG(
        source="plate",
        assignments=suggest_classes(
            distinct_values(explorer.frame, "plate"), per_class=2, prefix="Day"
        ),
    )
    explorer.groupings.set(grouping)
    explorer.groupings.apply_all(explorer.frame)

    values = explorer.frame[grouping.column].astype(str).to_numpy()
    groups, skipped = build_groups(values)
    assert skipped == 0
    # 8 plates, two per class, 96 wells each.
    assert [name for name, _ in groups] == ["Day 1", "Day 2", "Day 3", "Day 4"]
    assert all(len(idx) == 192 for _, idx in groups)


def test_groupings_survive_a_dataset_switch(explorer):
    """Definitions live in the workspace; derived columns live on the frame."""
    from plato.data.custom_groups import CustomGrouping as CG

    grouping = CG(source="plate", name="day", assignments={"P1": "Day 1", "P2": "Day 1"})
    explorer.groupings.set(grouping)
    explorer.groupings.apply_all(explorer.frame)

    dataset, frame = _frame(n_plates=4)
    explorer._add_entry(EmbeddingEntry(name="second", dataset=dataset, frame=frame))
    assert grouping.column in explorer.frame.columns
    assert explorer.group_box.findData(grouping.column) >= 0


# -- the well readout ----------------------------------------------------------


def test_the_description_labels_the_well_and_spells_out_its_position(explorer):
    """'P1 A01' as a bare string reads as one opaque identifier."""
    index = explorer.frame.index[explorer.frame.well == "D06"][0]
    text = explorer._describe(int(index))
    assert "Well:" in text
    assert "D06" in text
    assert "row D" in text
    assert "column 6" in text


def test_an_edge_well_says_so(explorer):
    """The first question asked of a point that looks like an outlier."""
    index = explorer.frame.index[explorer.frame.well == "A01"][0]
    assert "edge well" in explorer._describe(int(index))


def test_an_interior_well_reports_its_distance_instead(explorer):
    index = explorer.frame.index[explorer.frame.well == "D06"][0]
    text = explorer._describe(int(index))
    assert "edge well" not in text
    assert "3 in from edge" in text


def test_the_plate_is_labelled_too(explorer):
    index = explorer.frame.index[explorer.frame.plate == "P3"][0]
    assert "Plate:" in explorer._describe(int(index))
