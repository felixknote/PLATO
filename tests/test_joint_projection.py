"""Projecting several embeddings together as one fit.

Colouring by dataset over separately-fit UMAPs answers nothing: two
independent fits have no shared coordinate system, so a point from one
dataset landing near a point from another is coincidence. The only version
where relative position is actually meaningful is a joint fit over the
concatenated vectors, which is what this module builds and what these tests
verify -- concatenation order, dimensionality compatibility, and that the
combined frame stays aligned with the combined vectors row for row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plato.data.embeddings import EmbeddingDataset
from plato.data.joint_projection import (
    ALIGN_CENTRE,
    ALIGN_NONE,
    ALIGN_ZSCORE,
    IncompatibleEmbeddings,
    align_vectors,
    build_joint_dataset,
    check_compatible,
    make_joint_entry,
)
from plato.data.workspace import DATASET_COLUMN, EMBEDDING_COLUMN, EmbeddingEntry


def _entry(name: str, n: int, dims: int, center: float = 0.0) -> EmbeddingEntry:
    rng = np.random.default_rng(abs(hash(name)) % 2**31)
    frame = pd.DataFrame({"gene": [f"g{i % 3}" for i in range(n)]})
    dataset = EmbeddingDataset(
        name=name,
        directory=Path(f"/data/{name}"),
        vectors=(rng.normal(size=(n, dims)) + center).astype(np.float32),
        frame=frame,
        run_info={},
    )
    return EmbeddingEntry(name=name, dataset=dataset, frame=frame)


def test_needs_at_least_two_entries():
    with pytest.raises(IncompatibleEmbeddings):
        check_compatible([_entry("A", 10, 8)])


def test_mismatched_dimensionality_is_rejected():
    a, b = _entry("A", 10, 64), _entry("B", 10, 32)
    with pytest.raises(IncompatibleEmbeddings) as excinfo:
        check_compatible([a, b])
    message = str(excinfo.value)
    assert "64" in message and "32" in message
    assert "A" in message and "B" in message


def test_matching_dimensionality_is_accepted():
    a, b = _entry("A", 10, 32), _entry("B", 10, 32)
    check_compatible([a, b])  # must not raise


def test_vectors_concatenate_in_entry_order():
    a, b = _entry("A", 5, 8), _entry("B", 7, 8)
    joint = build_joint_dataset([a, b])

    assert joint.vectors.shape == (12, 8)
    np.testing.assert_array_equal(joint.vectors[:5], a.vectors)
    np.testing.assert_array_equal(joint.vectors[5:], b.vectors)


def test_frame_stays_aligned_with_vectors():
    a, b = _entry("A", 5, 8), _entry("B", 7, 8)
    joint = build_joint_dataset([a, b])

    assert len(joint.frame) == 12
    assert (joint.frame[DATASET_COLUMN].iloc[:5] == "A").all()
    assert (joint.frame[DATASET_COLUMN].iloc[5:] == "B").all()
    assert EMBEDDING_COLUMN in joint.frame.columns


def test_sources_map_positions_back_to_their_entry():
    a, b = _entry("A", 5, 8), _entry("B", 7, 8)
    joint = build_joint_dataset([a, b])

    assert joint.sources[0].entry_key == a.key
    assert joint.sources[0].start == 0 and joint.sources[0].stop == 5
    assert joint.sources[1].entry_key == b.key
    assert joint.sources[1].start == 5 and joint.sources[1].stop == 12

    source = joint.source_for(6)
    assert source is not None
    assert source.name == "B"
    assert source.local_index(6) == 1

    source = joint.source_for(2)
    assert source.name == "A"
    assert source.local_index(2) == 2


def test_three_way_join():
    a, b, c = _entry("A", 3, 8), _entry("B", 4, 8), _entry("C", 5, 8)
    joint = build_joint_dataset([a, b, c])

    assert joint.vectors.shape == (12, 8)
    assert len(joint.sources) == 3
    assert joint.sources[2].start == 7 and joint.sources[2].stop == 12


def test_fingerprint_is_stable_for_the_same_order():
    a, b = _entry("A", 5, 8), _entry("B", 7, 8)
    first = build_joint_dataset([a, b])
    second = build_joint_dataset([a, b])
    assert first.fingerprint == second.fingerprint


def test_fingerprint_is_order_sensitive():
    """{A, B} is a different concatenated array from {B, A}."""
    a, b = _entry("A", 5, 8), _entry("B", 7, 8)
    forward = build_joint_dataset([a, b])
    backward = build_joint_dataset([b, a])
    assert forward.fingerprint != backward.fingerprint


def test_fingerprint_changes_if_an_entry_changes():
    a, b = _entry("A", 5, 8), _entry("B", 7, 8)
    baseline = build_joint_dataset([a, b])

    different_b = _entry("B", 7, 8, center=5.0)
    changed = build_joint_dataset([a, different_b])
    assert baseline.fingerprint != changed.fingerprint


def test_a_real_joint_fit_preserves_structure_that_is_genuinely_there():
    """The end-to-end claim: joint position is meaningful, unlike separate fits.

    Two synthetic clusters that really ARE far apart in the vector space, one
    per "dataset". A real UMAP fit over their concatenation must recover that,
    verified by clustering the 2-D output against the true labels.

    Note what this does and does not show. It shows the fit transmits real
    structure faithfully -- it does NOT show that separated arms mean a
    finding, because a batch offset produces this same picture from data with
    no biological difference at all. See the alignment tests below for that
    distinction; here the separation is real by construction.
    """
    pytest.importorskip("umap")
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    from plato.data.projection import UMAP, ProjectionParams, project

    a = _entry("A", 100, 16, center=10.0)
    b = _entry("B", 100, 16, center=-10.0)
    joint = build_joint_dataset([a, b])

    params = ProjectionParams(method=UMAP, n_neighbors=15, pca_components=8)
    result = project(joint.vectors, params, fingerprint=joint.fingerprint)

    assert result.coords.shape[0] == 200
    truth = np.array([0] * 100 + [1] * 100)
    labels = KMeans(2, n_init=10, random_state=0).fit_predict(result.coords)
    assert adjusted_rand_score(truth, labels) > 0.9

    subset = joint.frame.iloc[result.row_indices]
    assert len(subset) == len(result.coords)
    assert set(subset[DATASET_COLUMN].unique()) == {"A", "B"}


# -- alignment -----------------------------------------------------------------
#
# Projecting together gives the arms a shared coordinate system. It does not
# make them comparable: a screen imaged in April and one in August differ in
# illumination and staining, every point of an arm carries that same offset,
# and the fit separates on it cleanly. The picture is indistinguishable from a
# real biological difference, which is why "the datasets separated" is not by
# itself a finding.


def test_alignment_is_off_unless_asked_for():
    """The raw concatenation is the honest default; aligning is a decision."""
    a, b = _entry("A", 40, 8, center=5.0), _entry("B", 40, 8, center=-5.0)
    joint = build_joint_dataset([a, b])
    assert joint.align == ALIGN_NONE
    assert np.allclose(joint.vectors[:40], a.vectors)
    assert np.allclose(joint.vectors[40:], b.vectors)


def test_centring_gives_each_dataset_the_same_origin():
    a, b = _entry("A", 60, 8, center=8.0), _entry("B", 40, 8, center=-8.0)
    joint = build_joint_dataset([a, b], align=ALIGN_CENTRE)
    for source in joint.sources:
        block = joint.vectors[source.start : source.stop]
        assert np.allclose(block.mean(axis=0), 0.0, atol=1e-4)


def test_centring_leaves_within_dataset_distances_untouched():
    """It removes exactly one thing -- the shift -- and no structure.

    This is what makes centring defensible at all: whatever the arm's own
    points said about each other, they still say.
    """
    a, b = _entry("A", 30, 8, center=4.0), _entry("B", 30, 8)
    raw = build_joint_dataset([a, b])
    aligned = build_joint_dataset([a, b], align=ALIGN_CENTRE)

    def within(vectors):
        block = vectors[:30]
        return np.linalg.norm(block[:, None] - block[None], axis=-1)

    assert np.allclose(within(raw.vectors), within(aligned.vectors), atol=1e-3)


def test_zscore_equalises_spread_as_well_as_origin():
    a = _entry("A", 80, 8)
    b = _entry("B", 80, 8)
    b.dataset.vectors = (b.dataset.vectors * 10.0).astype(np.float32)
    joint = build_joint_dataset([a, b], align=ALIGN_ZSCORE)
    spreads = [
        joint.vectors[s.start : s.stop].std(axis=0).mean() for s in joint.sources
    ]
    assert spreads[0] == pytest.approx(spreads[1], rel=0.05)


def test_a_constant_feature_does_not_become_nan():
    """0/0 in the z-score path would poison the whole fit, not one column."""
    a, b = _entry("A", 20, 8), _entry("B", 20, 8)
    a.dataset.vectors[:, 3] = 1.0
    joint = build_joint_dataset([a, b], align=ALIGN_ZSCORE)
    assert np.isfinite(joint.vectors).all()


def test_aligning_removes_a_batch_offset_and_keeps_the_biology():
    """The claim the feature exists for, on data with a known right answer.

    Two arms carrying the SAME two biological classes plus a per-arm offset.
    Raw, the arms separate and the classes are muddled; centred, the classes
    separate and the arms do not.
    """
    pytest.importorskip("sklearn")
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(0)
    dims = 32
    signal = rng.normal(size=(2, dims))

    def arm(name, offset, seed):
        r = np.random.default_rng(seed)
        labels = r.integers(0, 2, 200)
        vectors = signal[labels] + r.normal(scale=0.5, size=(200, dims)) + offset
        frame = pd.DataFrame({"gene": [f"g{v}" for v in labels]})
        dataset = EmbeddingDataset(
            name=name,
            directory=Path(f"/data/{name}"),
            vectors=vectors.astype(np.float32),
            frame=frame,
            run_info={},
        )
        return EmbeddingEntry(name=name, dataset=dataset, frame=frame), labels

    a, la = arm("A", np.zeros(dims), 1)
    b, lb = arm("B", rng.normal(scale=1.5, size=dims), 2)
    by_dataset = np.r_[np.zeros(200), np.ones(200)]
    by_biology = np.r_[la, lb]

    def separation(vectors):
        coords = PCA(2, random_state=0).fit_transform(vectors)
        return (
            silhouette_score(coords, by_dataset),
            silhouette_score(coords, by_biology),
        )

    raw_dataset, raw_biology = separation(build_joint_dataset([a, b]).vectors)
    aligned_dataset, aligned_biology = separation(
        build_joint_dataset([a, b], align=ALIGN_CENTRE).vectors
    )

    # Raw: the offset dominates, and the arms separate at least as well as
    # the biology does -- the trap this feature exists to expose.
    assert raw_dataset > 0.3
    # Centred: the arms stop separating, and the classes start.
    assert aligned_dataset < 0.1
    assert aligned_biology > raw_biology


def test_an_aligned_fit_is_a_different_cache_entry():
    """Otherwise the second combination silently restores the first's layout."""
    a, b = _entry("A", 20, 8), _entry("B", 20, 8)
    plain = build_joint_dataset([a, b]).fingerprint
    centred = build_joint_dataset([a, b], align=ALIGN_CENTRE).fingerprint
    scaled = build_joint_dataset([a, b], align=ALIGN_ZSCORE).fingerprint
    assert len({plain, centred, scaled}) == 3


def test_an_aligned_entry_says_so_in_its_name():
    """The label reaches the export headline; a figure must not misstate this."""
    a, b = _entry("A", 20, 8), _entry("B", 20, 8)
    plain = make_joint_entry([a, b])
    centred = make_joint_entry([a, b], align=ALIGN_CENTRE)
    assert "centred" not in plain.name
    assert "centred" in centred.name
    assert centred.info["align"] == ALIGN_CENTRE
