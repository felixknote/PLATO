"""Marked controls, and their appearance in every encoding.

A control is not another treatment. It is what every other point is judged
against, so the question asked of it -- "where do the controls sit, and does
everything else sit apart from them" -- is the same whatever field is on
screen. Colouring by gene buries that: the controls become a few more entries
in a legend of fifty, findable only if you already know which names they are.

So marking controls does two things. They become their own classes in every
categorical encoding, so a non-targeting row reports "WT (non-targeting gRNA)"
rather than its gene. And they are drawn larger, ringed and last, because at
3 px inside a cluster of 30,000 points a control is invisible whatever colour
it carries.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from plato.data import controls as controls_data
from plato.data.controls import CONTROL_CLASS, CONTROL_FIELD, ControlMarking
from plato.data.embeddings import EmbeddingDataset
from plato.data.explorer_model import CONDITION, GENE, build_frame
from plato.data.workspace import EmbeddingEntry


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


# Two real control kinds plus two treatments. The kinds must stay distinct:
# a solvent control and a non-targeting guide control for entirely different
# things, and one "control" bucket would claim otherwise.
CONDITIONS = ["gyrA_1", "ftsZ_2", "WT NC", "DMSO"]


def _frame(n: int = 96) -> pd.DataFrame:
    wells = [f"{chr(65 + r)}{c:02d}" for r in range(8) for c in range(1, 13)]
    raw = pd.DataFrame(
        {
            "plate": ["P1"] * n,
            "well": [wells[i % len(wells)] for i in range(n)],
            "condition": [CONDITIONS[i % len(CONDITIONS)] for i in range(n)],
            "image_name": [f"i{i}" for i in range(n)],
        }
    )
    dataset = EmbeddingDataset(
        name="d",
        directory=Path("/x"),
        vectors=np.zeros((n, 8), dtype=np.float32),
        frame=raw,
        run_info={},
    )
    frame, _ = build_frame(dataset)
    return frame


# -- MoA/pathway for controls ---------------------------------------------


def test_control_rows_read_control_not_unannotated_in_moa_and_pathway():
    """A control genuinely has no mechanism or gene target -- that is a real,
    known answer, not the same "nobody has annotated this yet" gap
    UNANNOTATED means for a real drug/gene the table just does not cover.
    Before this fix every control fell into UNANNOTATED, so a MoA or pathway
    legend read as mostly gaps instead of naming the controls as controls."""
    from plato.data.explorer_model import CONTROL_LABEL, MOA, PATHWAY

    frame = _frame()
    control_rows = frame[frame["condition"].isin(["WT NC", "DMSO"])]
    treatment_rows = frame[frame["condition"].isin(["gyrA_1", "ftsZ_2"])]

    assert (control_rows[MOA] == CONTROL_LABEL).all()
    assert (control_rows[PATHWAY] == CONTROL_LABEL).all()
    # Treatments are unaffected -- gyrA_1/ftsZ_2 have no MoA table entry in
    # this fixture (no moa_table was passed to build_frame), so they read as
    # UNANNOTATED, not "Control".
    from plato.data.annotations import UNANNOTATED

    assert (treatment_rows[MOA] == UNANNOTATED).all()
    assert (treatment_rows[PATHWAY] == UNANNOTATED).all()


# -- detection -----------------------------------------------------------------


def test_detection_finds_the_labels_the_parser_recognises():
    """The dialog opens pre-filled, so confirming is usually the whole job."""
    detected = controls_data.detect(_frame())
    assert "WT NC" in detected
    assert "DMSO" in detected
    # Treatments are not controls.
    assert "gyrA_1" not in detected


def test_detection_keeps_control_kinds_apart():
    """A vehicle and a non-targeting guide control for different things."""
    detected = controls_data.detect(_frame())
    assert detected["WT NC"] != detected["DMSO"]


def test_detection_on_a_frame_with_no_controls_returns_nothing():
    """An honest empty answer is what makes the confirm step necessary."""
    frame = _frame()
    frame["role"] = "treatment"
    assert controls_data.detect(frame) == {}


def test_detection_does_not_crash_on_a_frame_without_the_columns():
    assert controls_data.detect(pd.DataFrame({"x": [1]})) == {}


# -- the marking ---------------------------------------------------------------


def _marking() -> ControlMarking:
    return ControlMarking(
        source=CONDITION,
        assignments={"WT NC": "Non-targeting", "DMSO": "Vehicle"},
    )


def test_the_mask_selects_exactly_the_marked_rows():
    frame = _frame()
    mask = _marking().mask(frame)
    assert set(frame.loc[mask, CONDITION]) == {"WT NC", "DMSO"}
    assert mask.sum() == (frame[CONDITION].isin(["WT NC", "DMSO"])).sum()


def test_an_empty_marking_marks_nothing():
    frame = _frame()
    assert not ControlMarking(source="").mask(frame).any()


def test_a_marking_over_a_column_this_dataset_lacks_marks_nothing():
    """Markings belong to the workspace; datasets differ."""
    frame = _frame()
    marking = ControlMarking(source="absent", assignments={"x": "y"})
    assert not marking.mask(frame).any()


def test_the_overlay_replaces_a_treatment_field_with_the_control_class():
    """The stated requirement: controls as classes in every display option."""
    frame = _frame()
    overlaid = _marking().overlay(frame[GENE], frame)
    control_rows = _marking().mask(frame)
    assert set(overlaid[control_rows]) == {"Non-targeting", "Vehicle"}


def test_the_overlay_leaves_treatment_rows_alone():
    frame = _frame()
    overlaid = _marking().overlay(frame[GENE], frame)
    treatments = ~_marking().mask(frame)
    assert (overlaid[treatments] == frame[GENE][treatments]).all()


def test_applying_adds_both_derived_columns():
    """Ordinary derived columns, so nothing downstream needs a special case."""
    frame = _frame()
    added = _marking().apply(frame)
    assert set(added) == {CONTROL_CLASS, CONTROL_FIELD}
    assert set(frame[CONTROL_FIELD].unique()) == {
        "Non-targeting",
        "Vehicle",
        controls_data.TREATMENT_LABEL,
    }


def test_the_standalone_field_calls_everything_else_treatment():
    frame = _frame()
    _marking().apply(frame)
    treatments = ~_marking().mask(frame)
    assert (frame.loc[treatments, CONTROL_FIELD] == "Treatment").all()


def test_round_trips_through_json():
    restored = ControlMarking.load_json(_marking().to_json())
    assert restored.source == CONDITION
    assert restored.assignments == _marking().assignments


def test_corrupt_json_leaves_an_unmarked_state_rather_than_raising():
    """A bad settings value must not stop the explorer opening."""
    assert not ControlMarking.load_json("{not json")


def test_saves_and_loads_from_a_file(tmp_path):
    path = tmp_path / "controls.json"
    _marking().save_to(path)
    assert ControlMarking.load_from(path).assignments == _marking().assignments


def test_loading_a_missing_file_is_not_an_error(tmp_path):
    assert not ControlMarking.load_from(tmp_path / "nope.json")


# -- reaching the explorer -----------------------------------------------------


@pytest.fixture
def explorer(app):
    from plato.views.explorer import EmbeddingExplorer

    class _Session:
        pass

    frame = _frame()
    dataset = EmbeddingDataset(
        name="d",
        directory=Path("/x"),
        vectors=np.zeros((len(frame), 8), dtype=np.float32),
        frame=frame,
        run_info={},
    )
    widget = EmbeddingExplorer(_Session(), Path(tempfile.mkdtemp()))
    widget.set_image_mode(False)
    widget._add_entry(EmbeddingEntry(name="d", dataset=dataset, frame=frame))
    return widget


def test_nothing_is_marked_until_asked(explorer):
    """Detection is a suggestion, not an action -- the dialog confirms it."""
    assert not explorer.controls
    assert CONTROL_FIELD not in explorer.frame.columns


def test_marking_puts_the_control_classes_into_an_unrelated_encoding(explorer):
    """Colouring by gene must show the controls as their own legend entries."""
    explorer.controls = _marking()
    explorer._apply_controls()

    universe = explorer._colour_universe(GENE)
    assert "Non-targeting" in universe
    assert "Vehicle" in universe
    # ...and the treatments are still there.
    assert "gyrA" in universe


def test_the_column_the_marking_is_written_against_is_not_overlaid(explorer):
    """Its own values ARE the control names; replacing them is circular."""
    explorer.controls = _marking()
    explorer._apply_controls()
    universe = explorer._colour_universe(CONDITION)
    assert "WT NC" in universe


def test_marking_adds_a_standalone_controls_encoding(explorer):
    explorer.controls = _marking()
    explorer._apply_controls()
    explorer._rebuild_colour_options()
    assert explorer.colour_box.findData(CONTROL_FIELD) >= 0


def test_unmarking_removes_the_derived_columns_and_the_option(explorer):
    explorer.controls = _marking()
    explorer._apply_controls()
    explorer.controls = ControlMarking(source="")
    explorer._apply_controls()
    explorer._rebuild_colour_options()
    assert CONTROL_FIELD not in explorer.frame.columns
    assert explorer.colour_box.findData(CONTROL_FIELD) < 0


def test_controls_are_emphasised_in_the_draw(explorer):
    """At 3 px inside 30,000 points, colour alone cannot find a control."""
    explorer.controls = _marking()
    explorer._apply_controls()
    visible = np.arange(len(explorer.frame))
    mask = explorer._emphasis_for(visible)
    assert mask is not None
    assert mask.sum() == explorer.controls.mask(explorer.frame).sum()


def test_nothing_is_emphasised_when_nothing_is_marked(explorer):
    """The feature must be inert until used."""
    assert explorer._emphasis_for(np.arange(len(explorer.frame))) is None


def test_emphasis_follows_the_visible_subset(explorer):
    """A filtered-out control must not be emphasised into a plot it is not in."""
    explorer.controls = _marking()
    explorer._apply_controls()
    treatments = np.flatnonzero(~explorer.controls.mask(explorer.frame).to_numpy())
    assert explorer._emphasis_for(treatments) is None


def test_the_marking_survives_a_dataset_switch(explorer):
    """Definitions live in the workspace; derived columns live on the frame."""
    explorer.controls = _marking()
    explorer._apply_controls()

    frame = _frame(48)
    dataset = EmbeddingDataset(
        name="second",
        directory=Path("/y"),
        vectors=np.zeros((len(frame), 8), dtype=np.float32),
        frame=frame,
        run_info={},
    )
    explorer._add_entry(EmbeddingEntry(name="second", dataset=dataset, frame=frame))
    assert CONTROL_FIELD in explorer.frame.columns
    assert explorer.controls.mask(explorer.frame).any()
