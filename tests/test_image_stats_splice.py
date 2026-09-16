"""A joint entry's image statistics should reuse what its sources already have.

Computing image statistics on a joint entry used to always measure every row
from scratch, even when one (or both) of the entries it was built from had
already been measured and cached under its own fingerprint. splice_from_sources
fills a joint entry's stats array from its sources' own StatsCache entries
first, so the worker pass -- and the wall-clock cost -- only covers rows no
source's cache already answered.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plato.data.embeddings import EmbeddingDataset
from plato.data.image_stats import STAT_NAMES, StatsCache, empty, splice_from_sources
from plato.data.joint_projection import make_joint_entry
from plato.data.workspace import EmbeddingEntry, Workspace


def _entry(name: str, n: int) -> EmbeddingEntry:
    frame = pd.DataFrame({"image_name": [f"{name}_{i}" for i in range(n)]})
    dataset = EmbeddingDataset(
        name=name,
        directory=Path(f"/x/{name}"),
        vectors=np.zeros((n, 4), dtype=np.float32),
        frame=frame,
        run_info={},
    )
    return EmbeddingEntry(name=name, dataset=dataset, frame=frame)


def _filled(n: int, value: float) -> dict[str, np.ndarray]:
    return {name: np.full(n, value, dtype=np.float32) for name in STAT_NAMES}


def _workspace(*entries: EmbeddingEntry) -> Workspace:
    ws = Workspace()
    for entry in entries:
        ws.add(entry)
    return ws


# -- non-joint entries: unchanged behaviour -----------------------------------


def test_a_plain_entry_has_nothing_to_splice(tmp_path):
    entry = _entry("A", 5)
    ws = _workspace(entry)
    cache = StatsCache(tmp_path)
    values, missing = splice_from_sources(entry, ws, cache)
    assert missing == list(range(5))
    assert all(np.isnan(values[name]).all() for name in STAT_NAMES)


# -- joint entries -------------------------------------------------------------


def test_a_joint_entry_with_no_cached_sources_needs_every_row(tmp_path):
    a, b = _entry("A", 3), _entry("B", 2)
    ws = _workspace(a, b)
    joint = make_joint_entry([a, b])
    ws.add(joint)
    cache = StatsCache(tmp_path)

    values, missing = splice_from_sources(joint, ws, cache)
    assert missing == list(range(5))


def test_a_joint_entry_inherits_a_fully_cached_source(tmp_path):
    a, b = _entry("A", 3), _entry("B", 2)
    ws = _workspace(a, b)
    joint = make_joint_entry([a, b])
    ws.add(joint)
    cache = StatsCache(tmp_path)
    cache.save(a.dataset.fingerprint(), _filled(3, 7.0))

    values, missing = splice_from_sources(joint, ws, cache)
    # A's span (rows 0-2) is filled from its cache; B's span (3-4) is not.
    assert missing == [3, 4]
    assert list(values["brightness"][:3]) == [7.0, 7.0, 7.0]
    assert np.isnan(values["brightness"][3:]).all()


def test_a_joint_entry_needs_nothing_when_every_source_is_cached(tmp_path):
    a, b = _entry("A", 3), _entry("B", 2)
    ws = _workspace(a, b)
    joint = make_joint_entry([a, b])
    ws.add(joint)
    cache = StatsCache(tmp_path)
    cache.save(a.dataset.fingerprint(), _filled(3, 7.0))
    cache.save(b.dataset.fingerprint(), _filled(2, 9.0))

    values, missing = splice_from_sources(joint, ws, cache)
    assert missing == []
    assert list(values["brightness"]) == [7.0, 7.0, 7.0, 9.0, 9.0]


def test_a_stale_source_cache_is_not_used(tmp_path):
    """A cache saved for a different row count must not be spliced in --
    StatsCache.load already refuses this; splice must respect that refusal
    rather than reading the file directly."""
    a, b = _entry("A", 3), _entry("B", 2)
    ws = _workspace(a, b)
    joint = make_joint_entry([a, b])
    ws.add(joint)
    cache = StatsCache(tmp_path)
    # Saved under A's fingerprint but for the wrong row count.
    cache.save(a.dataset.fingerprint(), _filled(99, 7.0))

    values, missing = splice_from_sources(joint, ws, cache)
    assert missing == list(range(5))


def test_a_source_no_longer_open_is_skipped_not_an_error(tmp_path):
    """A joint entry keeps its source_keys even after a source is closed --
    closing an entry the joint was built from must not make splicing crash."""
    a, b = _entry("A", 3), _entry("B", 2)
    ws = _workspace(a, b)
    joint = make_joint_entry([a, b])
    ws.add(joint)
    ws.remove(a.key)
    cache = StatsCache(tmp_path)
    cache.save(b.dataset.fingerprint(), _filled(2, 9.0))

    values, missing = splice_from_sources(joint, ws, cache)
    assert missing == [0, 1, 2]
    assert list(values["brightness"][3:]) == [9.0, 9.0]


def test_splicing_never_mutates_the_sources_own_cached_arrays(tmp_path):
    """empty()/cache.load() results must be copied into the joint array, not
    aliased -- otherwise a later in-place edit of the joint values (the
    worker writes into its own array by row index) could corrupt what a
    source's own cache handed back."""
    a, b = _entry("A", 3), _entry("B", 2)
    ws = _workspace(a, b)
    joint = make_joint_entry([a, b])
    ws.add(joint)
    cache = StatsCache(tmp_path)
    cache.save(a.dataset.fingerprint(), _filled(3, 7.0))

    values, _ = splice_from_sources(joint, ws, cache)
    values["brightness"][0] = 999.0
    reloaded = cache.load(a.dataset.fingerprint(), 3)
    assert reloaded["brightness"][0] == 7.0
