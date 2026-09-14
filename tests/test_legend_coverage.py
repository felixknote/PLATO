"""Every colour-by field either shows a legend or says why it cannot.

A legend is truthful only while each value has its own colour. Past the
active palette's length (8 for Okabe-Ito, 20 for PLATO's own) colours wrap,
so two unrelated values share a swatch -- and a legend then asserts a
distinction the plot cannot make, which is worse than no legend because it
looks authoritative.

The old rule was a flat cap of 24, above every palette. Two things were wrong
with it. "Perturbation (which gene or drug)" lands around 38 values on a real
CRISPRi+ABx export and lost its legend entirely, with nothing on screen
saying a legend existed and was withheld. And between the palette length and
24 the legend was drawn anyway, over colours that had already started
repeating.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from plato.data.embeddings import EmbeddingDataset
from plato.data.explorer_model import build_frame
from plato.data.workspace import EmbeddingEntry
from plato.views import palette, palettes
from plato.views.explorer import MAX_LEGEND_ENTRIES


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


# A realistic two-arm export: CRISPRi genes with replicates, antibiotics at
# three doses, plus controls. This is what makes several fields land in the
# awkward range between "a handful" and "hundreds".
GENES = [f"gene{i}" for i in range(1, 28)]
DRUGS = [
    "Ampicillin",
    "Amikacin",
    "Ciprofloxacin",
    "Colistin",
    "Gentamicin",
    "Meropenem",
    "Rifampicin",
    "Tetracycline",
    "Trimethoprim",
]


def _conditions() -> list[str]:
    out = [f"{g}_{rep}" for g in GENES for rep in (1, 2)]
    out += [f"{d} {dose}" for d in DRUGS for dose in ("0.5x", "1x", "2x")]
    return out + ["WT NC", "DMSO"]


@pytest.fixture
def explorer(app):
    from plato.data.projection import UMAP, ProjectionParams, ProjectionResult
    from plato.views.explorer import EmbeddingExplorer

    class _Session:
        pass

    conditions = _conditions()
    n = len(conditions) * 6
    wells = [f"{chr(65 + r)}{c:02d}" for r in range(8) for c in range(1, 13)]
    raw = pd.DataFrame(
        {
            "plate": [f"P{i % 8 + 1}" for i in range(n)],
            "well": [wells[i % len(wells)] for i in range(n)],
            "condition": [conditions[i % len(conditions)] for i in range(n)],
            "image_name": [f"i{i}" for i in range(n)],
        }
    )
    dataset = EmbeddingDataset(
        name="Aug26",
        directory=Path("/x"),
        vectors=np.zeros((n, 8), dtype=np.float32),
        frame=raw,
        run_info={},
    )
    frame, _ = build_frame(dataset)

    widget = EmbeddingExplorer(_Session(), Path(tempfile.mkdtemp()))
    widget.set_image_mode(False)
    widget._add_entry(EmbeddingEntry(name="Aug26", dataset=dataset, frame=frame))
    rng = np.random.default_rng(0)
    widget.result = ProjectionResult(
        coords=rng.normal(size=(n, 2)).astype(np.float32),
        row_indices=np.arange(n),
        params=ProjectionParams(method=UMAP),
    )
    widget._set_message("")
    return widget


def _fields(explorer):
    return [
        (explorer.colour_box.itemData(i), explorer.colour_box.itemText(i))
        for i in range(explorer.colour_box.count())
    ]


def test_every_field_either_shows_a_legend_or_explains_itself(explorer):
    """The audit. No field may silently drop its legend."""
    unexplained = []
    for index, (column, label) in enumerate(_fields(explorer)):
        explorer.colour_box.setCurrentIndex(index)
        explorer._redraw(reset_view=False)
        drawn = explorer.scatter._legend_entries
        note = explorer.count_label.text()
        if not drawn and "values" not in note:
            unexplained.append(label)
    assert not unexplained, (
        f"these fields drop the legend with no explanation: {unexplained}"
    )


def test_a_field_within_the_palette_gets_a_complete_legend(explorer):
    """Not a truncated one: every value drawn is named."""
    index = explorer.colour_box.findData("plate")
    explorer.colour_box.setCurrentIndex(index)
    explorer._redraw(reset_view=False)
    entries = explorer.scatter._legend_entries
    assert entries is not None
    assert len(entries) == len(explorer._colour_universe("plate"))


def test_perturbation_keeps_its_legend_under_the_old_flat_cap(explorer):
    """The reported case. ~38 values, dropped entirely at a cap of 24.

    It only gets one when the palette is large enough to colour them
    uniquely, which is the real question -- but the FLAT cap must no longer
    be what decides.
    """
    from plato.data.explorer_model import PERTURBATION

    values = explorer._colour_universe(PERTURBATION)
    assert len(values) > 24, "fixture no longer exercises the reported range"
    assert len(values) <= MAX_LEGEND_ENTRIES, "the flat cap must not be the limit"


def test_a_field_past_the_palette_says_the_colours_repeat(explorer):
    """Not just "too many values" -- WHY it matters.

    Two clusters sharing a colour may be unrelated, and that changes how the
    plot is read. A bare count does not say so.
    """
    index = explorer.colour_box.findData("well")
    explorer.colour_box.setCurrentIndex(index)
    explorer._redraw(reset_view=False)
    assert explorer.scatter._legend_entries is None
    assert "colours repeat" in explorer.count_label.text()


def test_the_limit_follows_the_active_palette(explorer):
    """Okabe-Ito has 8 colours; a legend of 12 over it would be a lie."""
    palette.set_active_palette("okabe_ito")
    try:
        assert explorer._legend_limit() == len(palettes.get("okabe_ito").colours)
        index = explorer.colour_box.findData("drug")
        explorer.colour_box.setCurrentIndex(index)
        explorer._redraw(reset_view=False)
        # 12 antibiotics over an 8-colour palette: no legend, and it says why.
        assert explorer.scatter._legend_entries is None
        assert "colours repeat" in explorer.count_label.text()
    finally:
        palette.set_active_palette("plato")


def test_the_same_field_regains_its_legend_on_a_larger_palette(explorer):
    """Switching palette must update the legend, not leave a stale promise."""
    index = explorer.colour_box.findData("drug")
    explorer.colour_box.setCurrentIndex(index)

    palette.set_active_palette("okabe_ito")
    explorer._redraw(reset_view=False)
    assert explorer.scatter._legend_entries is None

    palette.set_active_palette("plato")
    explorer._redraw(reset_view=False)
    assert explorer.scatter._legend_entries is not None


def test_turning_the_legend_off_still_works(explorer):
    """The checkbox must win over any of this."""
    explorer.legend_box.setChecked(False)
    try:
        index = explorer.colour_box.findData("plate")
        explorer.colour_box.setCurrentIndex(index)
        explorer._redraw(reset_view=False)
        assert explorer.scatter._legend_entries is None
    finally:
        explorer.legend_box.setChecked(True)


def test_the_perturbation_label_does_not_read_as_a_two_way_split():
    """"Gene or drug" reads as CRISPRi-vs-ABx, which is Experiment arm.

    This field is the perturbation's NAME, ~38 values. The old label sent
    someone looking for a two-colour plot to a field that cannot give one.
    """
    from plato.data.explorer_model import PERTURBATION, field_label

    assert field_label(PERTURBATION) == "Perturbation (which gene or drug)"


def test_experiment_arm_is_the_actual_two_way_split(explorer):
    """And it does have a legend, because it is two values."""
    index = explorer.colour_box.findData("experiment")
    explorer.colour_box.setCurrentIndex(index)
    explorer._redraw(reset_view=False)
    entries = explorer.scatter._legend_entries
    assert entries is not None
    # Labels carry their point count at this size -- see LEGEND_COUNT_LIMIT.
    names = {label.split("  (")[0] for label, _ in entries}
    assert names == {"CRISPRi", "ABx"}


def test_a_small_legend_carries_point_counts(explorer):
    """Two arms of a joint fit are rarely the same size, and the bigger one
    looks denser for that reason alone. Without counts, "this cluster is
    tighter" is not readable from the picture."""
    index = explorer.colour_box.findData("experiment")
    explorer.colour_box.setCurrentIndex(index)
    explorer._redraw(reset_view=False)
    labels = [label for label, _ in explorer.scatter._legend_entries]
    assert all("(" in label for label in labels)
    total = sum(
        int(label.split("(")[1].rstrip(")").replace(",", "")) for label in labels
    )
    assert total == len(explorer._visible_rows)


def test_a_large_legend_drops_the_counts_to_stay_narrow(explorer):
    """Past LEGEND_COUNT_LIMIT the numbers make the legend too wide."""
    from plato.views.explorer import LEGEND_COUNT_LIMIT

    index = explorer.colour_box.findData("gene")
    explorer.colour_box.setCurrentIndex(index)
    explorer._redraw(reset_view=False)
    entries = explorer.scatter._legend_entries
    if entries and len(entries) > LEGEND_COUNT_LIMIT:
        assert not any("  (" in label for label, _ in entries)
