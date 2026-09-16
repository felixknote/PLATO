"""Computing image statistics on a joint entry reuses its sources' own caches.

End-to-end version of test_image_stats_splice.py's unit tests: drives the
real EmbeddingExplorer.compute path (the "Compute image statistics" button)
against a joint entry whose sources already have cached stats, and checks the
worker is only ever asked to measure the rows no source's cache covered.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from plato.data.embeddings import EmbeddingDataset  # noqa: E402
from plato.data.explorer_model import ImageResolver, build_frame  # noqa: E402
from plato.data.image_stats import STAT_NAMES, StatsCache  # noqa: E402
from plato.data.workspace import EmbeddingEntry  # noqa: E402
from plato.views.explorer import EmbeddingExplorer  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class _Session:
    pass


def _arm(name: str, root: Path, n: int, *, seed: int = 0) -> EmbeddingEntry:
    stems = [f"{name}_{i:03d}" for i in range(n)]
    for stem in stems:
        (root / f"{stem}.tiff").write_bytes(b"\0")
    raw = pd.DataFrame({"plate": ["P1"] * n, "well": [f"A{i:02d}" for i in range(n)], "image_name": stems})
    # Distinct, non-degenerate vectors per arm -- EmbeddingDataset.fingerprint
    # hashes vector content, and two all-zero (or otherwise identical) arrays
    # of the same shape collide onto the SAME cache file, which is not a
    # thing that happens with real exports but very much a thing that
    # happens with two lazily-zeroed test fixtures.
    vectors = np.random.default_rng(seed).normal(size=(n, 4)).astype(np.float32)
    dataset = EmbeddingDataset(
        name=name,
        directory=Path("/x") / name,
        vectors=vectors,
        frame=raw,
        run_info={},
    )
    frame, _ = build_frame(dataset)
    entry = EmbeddingEntry(name=name, dataset=dataset, frame=frame)
    entry.resolver = ImageResolver.detect(root, frame)
    assert entry.resolver is not None
    return entry


@pytest.fixture
def explorer(app, tmp_path):
    widget = EmbeddingExplorer(_Session(), tmp_path)
    return widget


def test_only_the_uncached_arm_is_measured(app, explorer, tmp_path, monkeypatch):
    from plato.data.joint_projection import make_joint_entry

    root_a, root_b = tmp_path / "A", tmp_path / "B"
    root_a.mkdir()
    root_b.mkdir()
    entry_a = _arm("armA", root_a, 3, seed=1)
    entry_b = _arm("armB", root_b, 2, seed=2)

    cache = StatsCache(explorer.work_dir / "projections")
    cache.save(entry_a.dataset.fingerprint(), {n: np.full(3, 4.0, dtype=np.float32) for n in STAT_NAMES})

    explorer._add_entry(entry_a)
    explorer._add_entry(entry_b)
    joint = make_joint_entry([entry_a, entry_b])
    explorer._add_entry(joint)

    # Give the joint entry the resolver it would have inherited via Locate
    # all (test_locate_inherits_joint.py covers that mechanism itself) --
    # this test is only about the stats side, so it is set directly.
    combined = entry_a.resolver.combined_with(entry_b.resolver, joint.frame)
    explorer.resolver = combined
    explorer.workspace.current.resolver = combined
    explorer._invalidate_paths()

    captured_rows = {}
    import plato.views.stats_worker as stats_worker_module

    original_start = stats_worker_module.start

    def _spy_start(pool, rows, paths, n_rows, signals, **kwargs):
        captured_rows["rows"] = list(rows)
        captured_rows["seed"] = kwargs.get("seed")
        return original_start(pool, rows, paths, n_rows, signals, **kwargs)

    monkeypatch.setattr(stats_worker_module, "start", _spy_start)

    explorer._compute_image_stats()

    assert captured_rows["rows"] == [3, 4], (
        "only armB's rows (positions 3-4 in the joint frame) should need "
        "measuring; armA's are already in its own cache"
    )
    assert captured_rows["seed"] is not None
    assert list(captured_rows["seed"]["brightness"][:3]) == [4.0, 4.0, 4.0]


def test_nothing_to_measure_when_every_source_is_cached(app, explorer, tmp_path, monkeypatch):
    from plato.data.joint_projection import make_joint_entry

    root_a, root_b = tmp_path / "A2", tmp_path / "B2"
    root_a.mkdir()
    root_b.mkdir()
    entry_a = _arm("armA2", root_a, 2, seed=3)
    entry_b = _arm("armB2", root_b, 2, seed=4)

    cache = StatsCache(explorer.work_dir / "projections")
    cache.save(entry_a.dataset.fingerprint(), {n: np.full(2, 1.0, dtype=np.float32) for n in STAT_NAMES})
    cache.save(entry_b.dataset.fingerprint(), {n: np.full(2, 2.0, dtype=np.float32) for n in STAT_NAMES})

    explorer._add_entry(entry_a)
    explorer._add_entry(entry_b)
    joint = make_joint_entry([entry_a, entry_b])
    explorer._add_entry(joint)
    combined = entry_a.resolver.combined_with(entry_b.resolver, joint.frame)
    explorer.resolver = combined
    explorer.workspace.current.resolver = combined
    explorer._invalidate_paths()

    import plato.views.stats_worker as stats_worker_module

    called = []
    monkeypatch.setattr(
        stats_worker_module, "start", lambda *a, **k: called.append(1) or None
    )

    explorer._compute_image_stats()

    assert called == [], "the worker must never be started when nothing is missing"
    assert explorer._image_stats is not None
    assert list(explorer._image_stats["brightness"]) == [1.0, 1.0, 2.0, 2.0]
