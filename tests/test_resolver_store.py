"""Remembering an image folder across restarts, not just within a session.

Before ResolverStore, EmbeddingExplorer._resolver_cache was a plain in-memory
dict: correct within one run, but every new PLATO session re-hunted the whole
data library (or re-asked via Locate) for every dataset, even ones confirmed
correct a dozen times before. "The connection should not change" means the
answer, once confirmed, is applied automatically next time -- Locate's own
Choose folder... still overrides it whenever it genuinely needs to change.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from plato.data.explorer_model import ImageResolver, build_frame
from plato.data.resolver_store import ResolverStore, rebuild_resolver


# -- ResolverStore: pure persistence, no filesystem scanning ------------------


def test_a_fresh_store_remembers_nothing(tmp_path):
    store = ResolverStore(tmp_path / "image_roots.json")
    assert store.roots_for("/data/A") is None


def test_remember_then_roots_for_round_trips(tmp_path):
    store = ResolverStore(tmp_path / "image_roots.json")
    store.remember("/data/A", [Path("/images/A")])
    assert store.roots_for("/data/A") == [Path("/images/A")]


def test_remember_writes_to_disk_immediately(tmp_path):
    """The whole point is surviving a restart -- nothing in-memory-only."""
    path = tmp_path / "image_roots.json"
    store = ResolverStore(path)
    store.remember("/data/A", [Path("/images/A")])
    assert path.exists()

    reloaded = ResolverStore(path)
    assert reloaded.roots_for("/data/A") == [Path("/images/A")]


def test_different_datasets_are_kept_separate(tmp_path):
    store = ResolverStore(tmp_path / "image_roots.json")
    store.remember("/data/A", [Path("/images/A")])
    store.remember("/data/B", [Path("/images/B")])
    assert store.roots_for("/data/A") == [Path("/images/A")]
    assert store.roots_for("/data/B") == [Path("/images/B")]


def test_remembering_again_overwrites_the_previous_answer(tmp_path):
    """Locate's Choose folder... must be able to change the connection."""
    store = ResolverStore(tmp_path / "image_roots.json")
    store.remember("/data/A", [Path("/images/old")])
    store.remember("/data/A", [Path("/images/new")])
    assert store.roots_for("/data/A") == [Path("/images/new")]


def test_multiple_roots_are_preserved_in_order(tmp_path):
    """A dataset split across several folders (Add another folder...) keeps
    every root, in the order they were added."""
    store = ResolverStore(tmp_path / "image_roots.json")
    roots = [Path("/images/arm1"), Path("/images/arm2")]
    store.remember("/data/joint", roots)
    assert store.roots_for("/data/joint") == roots


def test_forget_removes_a_remembered_answer(tmp_path):
    store = ResolverStore(tmp_path / "image_roots.json")
    store.remember("/data/A", [Path("/images/A")])
    store.forget("/data/A")
    assert store.roots_for("/data/A") is None


def test_forgetting_an_unknown_dataset_is_a_no_op(tmp_path):
    store = ResolverStore(tmp_path / "image_roots.json")
    store.forget("/data/never-seen")  # must not raise


def test_a_missing_file_is_treated_as_empty_not_an_error(tmp_path):
    store = ResolverStore(tmp_path / "does_not_exist.json")
    assert store.roots_for("/data/A") is None


def test_a_corrupt_file_is_treated_as_empty_not_a_crash(tmp_path):
    path = tmp_path / "image_roots.json"
    path.write_text("{not valid json", encoding="utf-8")
    store = ResolverStore(path)
    assert store.roots_for("/data/A") is None
    # And it must still be usable afterwards.
    store.remember("/data/A", [Path("/images/A")])
    assert store.roots_for("/data/A") == [Path("/images/A")]


def test_malformed_entries_in_an_otherwise_valid_file_are_skipped(tmp_path):
    path = tmp_path / "image_roots.json"
    path.write_text('{"/data/good": ["/images/good"], "/data/bad": "not-a-list"}', encoding="utf-8")
    store = ResolverStore(path)
    assert store.roots_for("/data/good") == [Path("/images/good")]
    assert store.roots_for("/data/bad") is None


# -- rebuild_resolver: real filesystem scanning -------------------------------


@pytest.fixture
def screen(tmp_path):
    root = tmp_path / "screen"
    root.mkdir()
    stems = [f"Well{w}_Point{w}_0000_Channel" for w in ("A01", "A02", "A03")]
    for stem in stems:
        (root / f"{stem}.tiff").write_bytes(b"\0")
    return root, stems


def _frame(stems: list[str]) -> pd.DataFrame:
    raw = pd.DataFrame(
        {"plate": ["P1"] * len(stems), "well": [s[4:7] for s in stems], "image_name": stems}
    )
    from plato.data.embeddings import EmbeddingDataset

    dataset = EmbeddingDataset(
        name="x", directory=Path("/x"), vectors=None, frame=raw, run_info={}
    )
    # build_frame only needs .frame's columns here; vectors is unused by it.
    frame, _ = build_frame(dataset)
    return frame


def test_rebuild_resolver_recreates_a_working_resolver(screen):
    root, stems = screen
    frame = _frame(stems)
    resolver = rebuild_resolver([root], frame)
    assert resolver is not None
    resolved = sum(1 for _, row in frame.iterrows() if resolver.path_for(row))
    assert resolved == len(stems)


def test_rebuild_resolver_merges_several_remembered_roots(tmp_path):
    """A dataset remembered as split across two folders (Add another
    folder...) must still resolve fully after a restart."""
    root_a = tmp_path / "armA"
    root_b = tmp_path / "armB"
    root_a.mkdir()
    root_b.mkdir()
    stems_a = ["WellA01_PointA01_0000_Channel"]
    stems_b = ["WellB01_PointB01_0000_Channel"]
    (root_a / f"{stems_a[0]}.tiff").write_bytes(b"\0")
    (root_b / f"{stems_b[0]}.tiff").write_bytes(b"\0")
    frame = _frame(stems_a + stems_b)

    resolver = rebuild_resolver([root_a, root_b], frame)
    assert resolver is not None
    resolved = sum(1 for _, row in frame.iterrows() if resolver.path_for(row))
    assert resolved == 2


def test_rebuild_resolver_returns_none_for_a_root_that_no_longer_resolves(tmp_path):
    """A remembered folder that has gone stale (moved, deleted, unmounted)
    must read as "not located", not crash or silently show nothing."""
    stems = ["WellA01_PointA01_0000_Channel"]
    frame = _frame(stems)
    missing_root = tmp_path / "never_existed"
    assert rebuild_resolver([missing_root], frame) is None


def test_rebuild_resolver_skips_a_dead_root_among_working_ones(tmp_path):
    """One stale root (of several remembered) must not sink the whole
    dataset -- the roots that still exist keep resolving their share."""
    root_a = tmp_path / "stillThere"
    root_a.mkdir()
    stems = ["WellA01_PointA01_0000_Channel"]
    (root_a / f"{stems[0]}.tiff").write_bytes(b"\0")
    frame = _frame(stems)
    dead_root = tmp_path / "goneNow"

    resolver = rebuild_resolver([dead_root, root_a], frame)
    assert resolver is not None
    assert resolver.path_for(frame.iloc[0]) is not None
