"""Cluster composition: does it find a real signal and ignore a decoy?

The panel exists to answer "why are these points together", and the failure
mode that matters is a confident wrong answer -- ranking a field that is
uniform across the whole dataset above the one that actually distinguishes
the cluster. Every test here plants a known answer and checks it comes back.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from plato.data.cluster_stats import MAX_CATEGORIES, compose, summarise_field


@pytest.fixture
def planted():
    """1,000 rows. Rows 0-199 are the 'cluster': 70% acrB, 30% tolC.

    ``plate`` is a decoy -- identically distributed inside and out, so a
    ranking that puts it first is broken. ``moa`` is sparse: six rows, all
    inside, so it is maximally distinctive but covers almost nothing.
    """
    rng = np.random.default_rng(0)
    gene = rng.choice(["a", "b", "c", "d", "e"], 1000).tolist()
    for i in range(140):
        gene[i] = "acrB"
    for i in range(140, 200):
        gene[i] = "tolC"
    moa = [""] * 1000
    for i in range(6):
        moa[i] = "efflux"
    return pd.DataFrame(
        {
            "gene": gene,
            "plate": rng.choice(["P1", "P2"], 1000).tolist(),
            "moa": moa,
        }
    )


def test_dominant_value_is_found(planted):
    composition = compose(planted, np.arange(200))
    assert composition.n_selected == 200
    assert composition.n_total == 1000

    gene = next(f for f in composition.fields if f.column == "gene")
    top = gene.dominant
    assert top.value == "acrB"
    assert top.count == 140
    assert top.percent == pytest.approx(70.0)
    # Shares are over annotated points, so they total 100%.
    assert sum(v.percent for v in gene.values) == pytest.approx(100.0)


def test_a_uniformly_distributed_field_does_not_outrank_the_signal(planted):
    """The decoy check: `plate` is identical inside and out."""
    composition = compose(planted, np.arange(200))
    gene = next(f for f in composition.fields if f.column == "gene")
    plate = next(f for f in composition.fields if f.column == "plate")

    assert gene.divergence > plate.divergence
    assert plate.divergence < 0.1
    assert composition.fields[0].column == "gene"


def test_enrichment_is_the_ratio_of_shares():
    """50% inside against 10% outside is 5x, not 5 percentage points."""
    frame = pd.DataFrame(
        {"gene": ["acrB"] * 50 + ["x"] * 50 + ["acrB"] * 90 + ["y"] * 810}
    )
    composition = compose(frame, np.arange(100))
    acrb = next(
        v for v in composition.fields[0].values if v.value == "acrB"
    )
    assert acrb.background_fraction == pytest.approx(0.1)
    assert acrb.enrichment == pytest.approx(5.0)


def test_a_value_absent_from_the_background_is_infinitely_enriched():
    frame = pd.DataFrame({"gene": ["only"] * 10 + ["other"] * 90})
    composition = compose(frame, np.arange(10))
    assert math.isinf(composition.fields[0].values[0].enrichment)


def test_sparse_fields_are_ranked_below_well_covered_ones(planted):
    """Six perfectly distinctive rows must not outrank 200 informative ones.

    Divergence is scaled by coverage precisely so that a column annotating a
    handful of the selected points cannot dominate the ranking on the
    strength of those few.
    """
    composition = compose(planted, np.arange(200))
    gene = next(f for f in composition.fields if f.column == "gene")
    moa = next(f for f in composition.fields if f.column == "moa")

    assert moa.missing == 194
    assert moa.dominant.percent == pytest.approx(100.0)
    # Distinctive, but not more informative than the gene split.
    assert moa.divergence < gene.divergence
    assert moa.divergence > 0.0, "a field found only inside is not featureless"


def test_unknown_columns_are_summarised_too():
    """Metadata this code has never heard of still gets a breakdown."""
    frame = pd.DataFrame({"instrument_serial": ["X"] * 100 + ["Y"] * 900})
    composition = compose(frame, np.arange(100))
    assert composition.fields[0].column == "instrument_serial"
    assert composition.fields[0].dominant.value == "X"


def test_identifier_columns_are_skipped():
    """A column with a distinct value per row is not a category."""
    frame = pd.DataFrame(
        {
            "image_name": [f"img_{i}.tif" for i in range(100)],
            "gene": ["a"] * 50 + ["b"] * 50,
        }
    )
    composition = compose(frame, np.arange(50))
    assert [f.column for f in composition.fields] == ["gene"]


def test_high_cardinality_columns_are_skipped():
    frame = pd.DataFrame({"junk": [str(i) for i in range(MAX_CATEGORIES + 50)]})
    composition = compose(frame, np.arange(MAX_CATEGORIES + 10))
    assert composition.fields == []


def test_stale_indices_are_dropped(planted):
    """A selection can outlive the frame it was made against."""
    composition = compose(planted, [5, 10_000, -3])
    assert composition.n_selected == 1


def test_empty_and_missing_inputs(planted):
    assert compose(planted, []).n_selected == 0
    assert compose(planted, None).n_selected == 0
    assert compose(None, [1]).n_selected == 0
    assert compose(planted, []).headline == "Nothing selected."


def test_headline_names_the_dominant_field(planted):
    composition = compose(planted, np.arange(200))
    headline = composition.headline
    assert "acrB" in headline
    assert "70.0%" in headline


def test_headline_is_honest_when_nothing_dominates():
    """A selection that looks like the background says so."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"gene": rng.choice(list("abcdefgh"), 1000).tolist()})
    composition = compose(frame, rng.choice(1000, 300, replace=False))
    assert "No single field dominates" in composition.headline


def test_missing_values_are_excluded_from_shares():
    """A half-annotated field is not reported as half '(blank)'."""
    selected = pd.Series(["a", "a", "", ""])
    background = pd.Series(["b"] * 10)
    summary = summarise_field(selected, background, "gene")
    assert summary.missing == 2
    assert summary.dominant.count == 2
    assert summary.dominant.percent == pytest.approx(100.0)


def test_divergence_counts_values_absent_from_the_selection():
    """A value common outside and absent inside is a real difference."""
    # Inside: all 'a'. Outside: half 'a', half 'b'. The missing 'b' is what
    # makes the selection distinctive, and only the union catches it.
    frame = pd.DataFrame({"g": ["a"] * 100 + ["a"] * 450 + ["b"] * 450})
    composition = compose(frame, np.arange(100))
    assert composition.fields[0].divergence == pytest.approx(0.5, abs=0.01)
