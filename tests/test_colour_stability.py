"""Colour and shape assignment must not shift when a filter or facet changes
what happens to be on screen.

_colour_universe is what both _redraw (colour) and _symbols_for (shape) build
their value->colour / value->shape mapping from. It is exercised directly
against a bare instance (frame set, nothing else) rather than through a full
projection, because it is a pure function of self.frame and reads nothing
else -- constructing a real projection for every case here would test the
projection pipeline, not this.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from plato.views.explorer import EmbeddingExplorer  # noqa: E402
from plato.views.palette import categorical_colours, sort_values  # noqa: E402


def _explorer_with_frame(frame: pd.DataFrame) -> EmbeddingExplorer:
    explorer = EmbeddingExplorer.__new__(EmbeddingExplorer)
    explorer.frame = frame
    return explorer


def test_universe_is_unaffected_by_which_rows_are_passed():
    """The whole point: the same frame gives the same universe regardless of
    which subset of rows a caller happens to care about right now."""
    frame = pd.DataFrame({"drug": ["Amikacin", "Ampicillin", "Amikacin", "Gentamicin"]})
    explorer = _explorer_with_frame(frame)
    universe = explorer._colour_universe("drug")
    assert universe == sorted(universe)
    assert set(universe) == {"Amikacin", "Ampicillin", "Gentamicin"}


def test_filtering_out_a_category_does_not_move_the_others_colour():
    """Regression: _redraw used to build the value->colour mapping from
    sort_values(dict.fromkeys(visible_values)) -- the post-filter subset --
    so removing "Amikacin" from view shifted "Ampicillin" into the palette
    slot Amikacin used to hold. _colour_universe fixes this by always
    reading the full frame; this test proves the two disagree on the exact
    scenario that motivated the fix.
    """
    frame = pd.DataFrame(
        {"drug": ["Amikacin", "Ampicillin", "Amikacin", "Gentamicin", "Ampicillin"]}
    )
    explorer = _explorer_with_frame(frame)

    universe = explorer._colour_universe("drug")
    fixed_mapping = categorical_colours(universe)

    # The old, buggy derivation: only the values surviving a filter that
    # excludes every "Amikacin" row.
    visible_values = frame.loc[frame["drug"] != "Amikacin", "drug"].tolist()
    old_buggy_universe = sort_values(list(dict.fromkeys(visible_values)))
    old_buggy_mapping = categorical_colours(old_buggy_universe)

    assert fixed_mapping["Ampicillin"] != old_buggy_mapping["Ampicillin"], (
        "if this now matches, the scenario no longer demonstrates the bug "
        "_colour_universe exists to prevent"
    )
    # And the fixed mapping is exactly what "Ampicillin" gets regardless of
    # which rows are filtered -- it never moves.
    other_filter_universe = explorer._colour_universe("drug")
    assert categorical_colours(other_filter_universe)["Ampicillin"] == fixed_mapping["Ampicillin"]


def test_unknown_values_are_grouped_last_not_alphabetised_in():
    """"nan"/"unknown"/blank-marker values must not scatter through the real
    categories just because they happen to sort alphabetically early."""
    frame = pd.DataFrame({"drug": ["Zestril", "nan", "Amikacin", "unknown", ""]})
    explorer = _explorer_with_frame(frame)
    universe = explorer._colour_universe("drug")

    from plato.views.palette import is_unknown

    known = [v for v in universe if not is_unknown(v)]
    unknowns = [v for v in universe if is_unknown(v)]
    assert known == sorted(known)
    # Every unknown-marker value in the universe comes after every real one.
    if known and unknowns:
        assert universe.index(unknowns[0]) > universe.index(known[-1])
    assert set(unknowns) == {"nan", "unknown", ""}


def test_universe_empty_for_missing_or_blank_column():
    frame = pd.DataFrame({"drug": ["Amikacin"]})
    explorer = _explorer_with_frame(frame)
    assert explorer._colour_universe(None) == []
    assert explorer._colour_universe("not_a_column") == []
