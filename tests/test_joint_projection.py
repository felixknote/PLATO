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
    IncompatibleEmbeddings,
    build_joint_dataset,
    check_compatible,
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


def test_a_real_joint_fit_separates_the_true_source_datasets():
    """The end-to-end claim: joint position is meaningful, unlike separate fits.

    Two synthetic, well-separated clusters, one per "dataset". A real UMAP fit
    over their concatenation must recover which dataset each point came from,
    verified by clustering the 2-D output and comparing to the true labels.
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
