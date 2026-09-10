"""The multi-embedding workspace, palettes, shapes and themes.

Stage 1 of the analysis-environment extension. Everything here is headless --
no QApplication needed except for the theme tests, which are marked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plato.data.embeddings import EmbeddingDataset
from plato.data.workspace import (
    DATASET_COLUMN,
    EMBEDDING_COLUMN,
    SOURCE_COMPUTED,
    SOURCE_EXPORT,
    EmbeddingEntry,
    Workspace,
)


def make_entry(name, n=20, dims=8, source=SOURCE_EXPORT, directory="/tmp/ds", **info):
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    vectors = rng.normal(0, 1, (n, dims)).astype(np.float32)
    frame = pd.DataFrame(
        {
            "condition": [f"c{i % 3}" for i in range(n)],
            "gene": ["acrB"] * (n // 2) + ["tolC"] * (n - n // 2),
            "plate": ["P1"] * n,
        }
    )
    dataset = EmbeddingDataset(
        name=name,
        directory=__import__("pathlib").Path(directory),
        vectors=vectors,
        frame=frame,
        run_info={},
    )
    return EmbeddingEntry(
        name=name, dataset=dataset, frame=frame, source=source, info=info
    )


# -- entries -----------------------------------------------------------------


def test_several_embeddings_coexist():
    """The headline requirement: A, B and C open at once."""
    workspace = Workspace()
    for name in ("A", "B", "C"):
        workspace.add(make_entry(name))
    assert len(workspace) == 3
    assert [e.name for e in workspace.entries] == ["A", "B", "C"]


def test_switching_does_not_reload():
    """Switching is a pointer change, not a load."""
    workspace = Workspace()
    first = workspace.add(make_entry("A"))
    second = workspace.add(make_entry("B"))
    assert workspace.current.key == second.key

    assert workspace.set_current(first.key)
    assert workspace.current is first
    # The other entry's vectors are still in memory, not re-read.
    assert workspace.get(second.key).vectors is second.vectors


def test_two_embeddings_can_share_one_dataset():
    """Same points, different vectors: an export and computed descriptors."""
    workspace = Workspace()
    workspace.add(make_entry("screen", source=SOURCE_EXPORT, directory="/data/x"))
    workspace.add(make_entry("screen", source=SOURCE_COMPUTED, directory="/data/x"))

    assert len(workspace) == 2, "loading the same directory twice is legitimate"
    same = workspace.find_by_directory("/data/x")
    assert len(same) == 2
    # Names collide; keys and labels do not.
    assert same[0].key != same[1].key
    assert same[0].label() != same[1].label()


def test_labels_distinguish_models():
    entry = make_entry("run", model="dinov3_vitl16")
    assert "dinov3_vitl16" in entry.label()
    assert "dinov3_vitl16" in entry.describe()


def test_removing_an_entry_moves_current():
    workspace = Workspace()
    first = workspace.add(make_entry("A"))
    second = workspace.add(make_entry("B"))
    workspace.remove(second.key)
    assert workspace.current is first
    workspace.remove(first.key)
    assert workspace.current is None


# -- one selection model -----------------------------------------------------


def test_selection_is_per_entry():
    """Switching embeddings must not carry a selection onto other points."""
    workspace = Workspace()
    first = workspace.add(make_entry("A"))
    second = workspace.add(make_entry("B"))

    workspace.set_current(first.key)
    workspace.set_selection([1, 2, 3])
    workspace.set_current(second.key)
    assert len(workspace.selection()) == 0, "selection must not leak between entries"

    workspace.set_current(first.key)
    assert workspace.selection().tolist() == [1, 2, 3], "and must survive the round trip"


def test_selection_toggling():
    workspace = Workspace()
    workspace.add(make_entry("A"))
    workspace.set_selection([1, 2])
    workspace.toggle_selection(3)
    assert workspace.selection().tolist() == [1, 2, 3]
    workspace.toggle_selection(2)
    assert workspace.selection().tolist() == [1, 3]


def test_selection_rejects_out_of_range_rows():
    """A stale selection must not index off the end of a reloaded frame."""
    workspace = Workspace()
    workspace.add(make_entry("A", n=10))
    workspace.set_selection([5, 999, -2])
    assert workspace.selection().tolist() == [5]


def test_selection_deduplicates():
    workspace = Workspace()
    workspace.add(make_entry("A"))
    workspace.set_selection([4, 4, 4, 1])
    assert workspace.selection().tolist() == [1, 4]


def test_combined_frame_labels_each_source():
    workspace = Workspace()
    workspace.add(make_entry("A", n=5))
    workspace.add(make_entry("B", n=7))
    combined = workspace.combined_frame()
    assert len(combined) == 12
    assert set(combined[EMBEDDING_COLUMN]) == {"A", "B"}
    assert DATASET_COLUMN in combined.columns


# -- palettes ----------------------------------------------------------------


def test_every_palette_is_well_formed():
    from plato.views import palettes

    for palette in palettes.PALETTES:
        assert palette.kind in (
            palettes.CATEGORICAL,
            palettes.SEQUENTIAL,
            palettes.DIVERGING,
        )
        assert len(palette) >= 2
        for colour in palette.colours:
            assert colour.startswith("#") and len(colour) == 7, colour


def test_all_three_palette_kinds_are_offered():
    from plato.views import palettes

    for kind in (palettes.CATEGORICAL, palettes.SEQUENTIAL, palettes.DIVERGING):
        assert palettes.of_kind(kind), f"no {kind} palette"
        assert palettes.default_for(kind).kind == kind


def test_unknown_palette_falls_back():
    from plato.views import palettes

    assert palettes.get("no-such-palette").kind == palettes.CATEGORICAL


def test_okabe_ito_is_available_for_publication():
    from plato.views import palettes

    palette = palettes.get("okabe_ito")
    assert palette.kind == palettes.CATEGORICAL
    assert len(palette) == 8


# -- shapes ------------------------------------------------------------------


def test_shapes_are_assigned_in_order():
    from plato.views import shapes

    mapping, overflow = shapes.assign(["a", "b", "c"])
    assert mapping == {"a": "o", "b": "s", "c": "t"}
    assert overflow == []


def test_too_many_categories_do_not_wrap():
    """Wrapping would say two unrelated groups are the same group."""
    from plato.views import shapes

    values = [f"v{i}" for i in range(shapes.MAX_SHAPES + 5)]
    mapping, overflow = shapes.assign(values)

    assert len(overflow) == 5
    # Everything still has a shape, so rendering never hits a missing key.
    assert set(mapping) == set(values)
    # The distinct ones are genuinely distinct.
    distinct = [mapping[v] for v in values[: shapes.MAX_SHAPES]]
    assert len(set(distinct)) == shapes.MAX_SHAPES
    # And the overflow is honest rather than silently reused.
    assert all(mapping[v] == shapes.DEFAULT_SHAPE for v in overflow)
    assert "only" in shapes.describe_overflow(overflow, len(values))


def test_no_overflow_message_when_it_fits():
    from plato.views import shapes

    assert shapes.describe_overflow([], 3) == ""


# -- themes ------------------------------------------------------------------


def test_both_themes_define_every_colour():
    from plato.gui import themes

    for colours in (themes.DARK_COLOURS, themes.LIGHT_COLOURS):
        for field in colours.__dataclass_fields__:
            value = getattr(colours, field)
            assert value, f"{colours.key}.{field} is empty"
            if field not in ("key", "name"):
                assert value.startswith("#"), f"{colours.key}.{field} = {value}"


def test_light_and_dark_actually_differ():
    from plato.gui import themes

    assert themes.LIGHT_COLOURS.background != themes.DARK_COLOURS.background
    # Text must invert, or the light theme is unreadable.
    assert themes.LIGHT_COLOURS.text != themes.DARK_COLOURS.text


def test_the_stylesheet_template_serves_both_themes():
    """One set of rules, two palettes -- not two copies of the CSS."""
    from plato.gui import theme, themes

    dark = theme.build_stylesheet(themes.DARK_COLOURS)
    light = theme.build_stylesheet(themes.LIGHT_COLOURS)

    assert len(dark) == len(light), "the rules should be identical in structure"
    assert themes.DARK_COLOURS.background in dark
    assert themes.DARK_COLOURS.background not in light
    assert themes.LIGHT_COLOURS.background in light


def test_image_canvas_stays_dark_in_both_themes():
    """A white surround destroys the contrast a micrograph is judged on."""
    from plato.gui import themes

    for colours in (themes.DARK_COLOURS, themes.LIGHT_COLOURS):
        # Crude luminance check: the canvas must be a dark colour.
        value = colours.image_background.lstrip("#")
        red, green, blue = (int(value[i : i + 2], 16) for i in (0, 2, 4))
        assert (red + green + blue) / 3 < 60, f"{colours.key} canvas is not dark"


# -- the explorer actually using the workspace -------------------------------


def test_explorer_keeps_several_embeddings_open(tmp_path):
    """The end-to-end requirement: A, B and C switchable without reloading."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from plato.data.workspace import SOURCE_COMPUTED
    from plato.views.explorer import EmbeddingExplorer

    app = QApplication.instance() or QApplication([])

    class _Session:
        pass

    explorer = EmbeddingExplorer(_Session(), tmp_path)

    def load(name, n, dims, directory, source=SOURCE_EXPORT):
        entry = make_entry(name, n=n, dims=dims, source=source, directory=directory)
        explorer.dataset = entry.dataset
        explorer.frame = entry.frame
        explorer._register_entry(entry.dataset, entry.frame, source=source)
        explorer._active_key = explorer.workspace.current_key
        return explorer.workspace.current_key

    first = load("A", 40, 16, "/data/a")
    second = load("B", 30, 16, "/data/b")
    # Same directory, different vectors: an export and computed descriptors.
    third = load("A", 40, 8, "/data/a", source=SOURCE_COMPUTED)

    assert len(explorer.workspace) == 3
    assert len({first, second, third}) == 3, "keys must be distinct"
    assert len(explorer.workspace.find_by_directory("/data/a")) == 2

    # Selections belong to their embedding.
    explorer.open_box.setCurrentIndex(explorer.open_box.findData(first))
    explorer.scatter.set_selection(np.arange(5))
    explorer.open_box.setCurrentIndex(explorer.open_box.findData(second))
    assert len(explorer.scatter.selected_rows) == 0

    explorer.open_box.setCurrentIndex(explorer.open_box.findData(first))
    assert len(explorer.scatter.selected_rows) == 5

    # Switching is a pointer move, not a reload.
    vectors = explorer.workspace.get(first).vectors
    explorer.open_box.setCurrentIndex(explorer.open_box.findData(third))
    explorer.open_box.setCurrentIndex(explorer.open_box.findData(first))
    assert explorer.workspace.get(first).vectors is vectors
    assert len(explorer.frame) == 40


def test_closing_an_embedding_falls_back_to_another(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from plato.views.explorer import EmbeddingExplorer

    QApplication.instance() or QApplication([])

    class _Session:
        pass

    explorer = EmbeddingExplorer(_Session(), tmp_path)
    for name in ("A", "B"):
        entry = make_entry(name, n=20, directory=f"/data/{name}")
        explorer.dataset = entry.dataset
        explorer.frame = entry.frame
        explorer._register_entry(entry.dataset, entry.frame)
        explorer._active_key = explorer.workspace.current_key

    assert len(explorer.workspace) == 2
    explorer._close_current_embedding()
    assert len(explorer.workspace) == 1
    # The last one cannot be closed; there would be nothing to show.
    explorer._close_current_embedding()
    assert len(explorer.workspace) == 1
