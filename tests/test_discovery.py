"""Recursive discovery of embedding exports and plate image folders.

The scan runs against network shares and mechanical disks, so what matters
here is not just "does it find things" but "does it stop looking once it has
found them, and does it stay bounded when asked to."
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plato.data import discovery as D


def _make_embedding(directory: Path, n_rows: int = 10) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    header = "a,b\n"
    rows = "\n".join(f"{i},{i}" for i in range(n_rows))
    (directory / "features_metadata.csv").write_text(header + rows + "\n")
    (directory / "features_all.npz").write_bytes(b"")


def _make_plate(directory: Path, n_images: int = 20) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n_images):
        (directory / f"img_{i}.tif").write_bytes(b"")


def test_finds_embedding_and_plate(tmp_path):
    _make_embedding(tmp_path / "screens" / "exportA", n_rows=500)
    _make_plate(tmp_path / "screens" / "plateB", n_images=50)

    result = D.scan([tmp_path])

    kinds = {f.path.name: f.kind for f in result.found}
    assert kinds["exportA"] == D.EMBEDDING
    assert kinds["plateB"] == D.PLATE


def test_row_count_hint_is_correct(tmp_path):
    _make_embedding(tmp_path / "exportA", n_rows=500)
    result = D.scan([tmp_path])
    found = next(f for f in result.found if f.path.name == "exportA")
    assert found.detail == "500 rows"


def test_does_not_descend_into_a_found_embedding(tmp_path):
    """An export's own directory holds only its own files, never a nested one."""
    export = tmp_path / "exportA"
    _make_embedding(export)
    nested = export / "junk_subdir"
    _make_embedding(nested)

    result = D.scan([tmp_path])
    names = [f.path.name for f in result.found]
    assert names == ["exportA"]


def test_does_not_descend_into_a_found_plate(tmp_path):
    """A plate's own image folder must not be walked looking for more datasets."""
    plate = tmp_path / "plateB"
    _make_plate(plate)
    nested = plate / "thumbs"
    _make_embedding(nested)

    result = D.scan([tmp_path])
    names = [f.path.name for f in result.found]
    assert names == ["plateB"]


def test_entry_budget_truncates(tmp_path):
    wide = tmp_path / "wide"
    wide.mkdir()
    for i in range(200):
        (wide / f"d{i}").mkdir()

    result = D.scan([tmp_path], max_entries=50, max_seconds=30)
    assert result.truncated
    assert result.directories_visited < 210


def test_depth_limit_stops_the_walk(tmp_path):
    deep = tmp_path
    for i in range(10):
        deep = deep / f"lvl{i}"
        deep.mkdir()
    _make_embedding(deep)

    shallow = D.scan([tmp_path], max_depth=3)
    assert shallow.found == []

    full = D.scan([tmp_path], max_depth=12)
    assert len(full.found) == 1


def test_missing_root_does_not_raise(tmp_path):
    result = D.scan([tmp_path / "does" / "not" / "exist", tmp_path])
    assert result.found == []
    assert not result.truncated


def test_duplicate_paths_across_overlapping_roots_are_collapsed(tmp_path):
    _make_embedding(tmp_path / "a" / "exportA")
    result = D.scan([tmp_path, tmp_path / "a"])
    paths = [str(f.path) for f in result.found]
    assert len(paths) == len(set(paths))
    assert len(result.found) == 1


def test_looks_like_image_folder_requires_a_minimum(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not D.looks_like_image_folder(empty)

    almost = tmp_path / "almost"
    almost.mkdir()
    for i in range(D.IMAGE_SAMPLE_MIN_HITS - 1):
        (almost / f"i{i}.tif").write_bytes(b"")
    assert not D.looks_like_image_folder(almost)

    enough = tmp_path / "enough"
    enough.mkdir()
    for i in range(D.IMAGE_SAMPLE_MIN_HITS):
        (enough / f"i{i}.tif").write_bytes(b"")
    assert D.looks_like_image_folder(enough)


def test_nonimage_files_do_not_count(tmp_path):
    directory = tmp_path / "docs"
    directory.mkdir()
    for i in range(10):
        (directory / f"note_{i}.txt").write_text("x")
    assert not D.looks_like_image_folder(directory)


@pytest.mark.parametrize("ext", [".tif", ".tiff", ".png", ".jpg", ".jpeg"])
def test_every_declared_image_extension_is_recognised(tmp_path, ext):
    directory = tmp_path / "plate"
    directory.mkdir()
    for i in range(D.IMAGE_SAMPLE_MIN_HITS):
        (directory / f"i{i}{ext}").write_bytes(b"")
    assert D.looks_like_image_folder(directory)


def test_scan_of_empty_tree_finds_nothing(tmp_path):
    (tmp_path / "empty1").mkdir()
    (tmp_path / "empty2" / "empty3").mkdir(parents=True)
    result = D.scan([tmp_path])
    assert result.found == []
    assert not result.truncated
