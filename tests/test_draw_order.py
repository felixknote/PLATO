"""Overlapping colour groups must not have a fixed winner.

set_points used to add one ScatterPlotItem per (colour, symbol) group, all at
the same z-value, in whatever order the caller's dict happened to iterate --
which for the explorer's own colour mapping is alphabetical by category
label. On a genuinely mixed region that makes whichever category sorts last
("CRISPRi" over "ABx", say) always paint over the other, everywhere the two
overlap: a scientifically meaningless string-sort artefact reads on screen as
"this category dominates". _draw_order (plato/views/scatter.py) fixes this by
splitting each group into chunks and shuffling their paint order.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from plato.views.scatter import (  # noqa: E402
    DRAW_CHUNKS_PER_GROUP,
    EmbeddingScatter,
    _draw_order,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_a_single_group_is_unaffected():
    """One colour, no interleaving question to answer -- must still draw
    every point, just possibly split into more chunks."""
    mask = np.arange(100)
    ordered = _draw_order([("#4a90d9", "o", mask)])
    total = np.concatenate([m for _, _, m in ordered])
    assert sorted(total.tolist()) == list(range(100))


def test_two_groups_interleave_rather_than_fully_separate():
    """The regression case: two evenly-sized groups must not come back as
    one contiguous run of the first colour followed by one contiguous run
    of the second -- that is exactly the "last group always on top" bug.
    """
    a = np.arange(0, 1000)
    b = np.arange(1000, 2000)
    ordered = _draw_order([("colourA", "o", a), ("colourB", "o", b)])

    # Every point must still be drawn, exactly once.
    all_positions = np.concatenate([m for _, _, m in ordered])
    assert sorted(all_positions.tolist()) == list(range(2000))

    # The sequence of colours across draw calls (not points -- draw CALLS,
    # since that is what determines paint order) must alternate rather than
    # being every "colourA" batch followed by every "colourB" batch.
    colour_sequence = [c for c, _, _ in ordered]
    first_b_index = colour_sequence.index("colourB")
    # If interleaved, colourA batches appear both before AND after the first
    # colourB batch. If not interleaved (the bug), every colourA batch comes
    # before every colourB batch.
    assert "colourA" in colour_sequence[first_b_index:], (
        "colourB's batches all came after every colourA batch -- "
        "groups are not interleaved, just re-ordered as whole blocks"
    )


def test_interleaving_is_reproducible():
    """Same input must produce the same draw order every time, or a
    slider tweak that doesn't change colouring would visibly reshuffle
    which points are on top."""
    a = np.arange(0, 500)
    b = np.arange(500, 900)
    first = _draw_order([("A", "o", a), ("B", "o", b)])
    second = _draw_order([("A", "o", a), ("B", "o", b)])
    assert [c for c, _, _ in first] == [c for c, _, _ in second]
    for (_, _, m1), (_, _, m2) in zip(first, second):
        assert np.array_equal(m1, m2)


def test_empty_groups_are_dropped():
    ordered = _draw_order([("A", "o", np.array([], dtype=np.int64)), ("B", "o", np.arange(5))])
    assert all(len(m) for _, _, m in ordered)
    assert sum(len(m) for _, _, m in ordered) == 5


def test_set_points_draws_every_item_at_the_same_z_value(app):
    """Paint order is add order only while every item shares one z-value --
    if that ever drifted apart per group, _draw_order's shuffle would stop
    mattering because z-value would decide paint order instead."""
    widget = EmbeddingScatter()
    n = 200
    coords = np.random.default_rng(0).normal(size=(n, 2)).astype(np.float32)
    rows = np.arange(n, dtype=np.int64)
    colours = ["#4a90d9" if i % 2 == 0 else "#e8a33d" for i in range(n)]
    widget.set_points(coords, rows, colours)
    z_values = {item.zValue() for item in widget._items}
    assert z_values == {0}


def test_set_points_produces_more_than_one_item_per_colour(app):
    """The chunking must actually happen, not collapse back into one item
    per colour (which would silently reintroduce full occlusion)."""
    widget = EmbeddingScatter()
    n = 400
    coords = np.random.default_rng(0).normal(size=(n, 2)).astype(np.float32)
    rows = np.arange(n, dtype=np.int64)
    colours = ["#4a90d9" if i % 2 == 0 else "#e8a33d" for i in range(n)]
    widget.set_points(coords, rows, colours)
    # Two colours, each split into up to DRAW_CHUNKS_PER_GROUP chunks.
    assert len(widget._items) > 2
    assert len(widget._items) <= 2 * DRAW_CHUNKS_PER_GROUP
