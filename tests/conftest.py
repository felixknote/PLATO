"""Locating the optional real datasets these tests can run against.

Everything here is discovery, never a hard-coded path. Most of the suite runs
on synthetic fixtures and needs none of this; the few tests that check
behaviour against the real screen skip themselves when it is not reachable, so
the suite passes on a machine that has never seen the data.

Point the tests at real data with:

    PLATO_TEST_EMBEDDING_ROOT   a folder of embedding exports
    PLATO_TEST_IMAGE_ROOT       a folder of screens (one subfolder per plate)

Both are optional.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _root(name: str) -> Path | None:
    raw = os.environ.get(name, "")
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


def embedding_root() -> Path | None:
    """A directory containing embedding exports, if one is configured."""
    return _root("PLATO_TEST_EMBEDDING_ROOT")


def image_root() -> Path | None:
    """A directory of screens, if one is configured."""
    return _root("PLATO_TEST_IMAGE_ROOT")


def find_export(predicate) -> Path | None:
    """The first embedding export whose metadata satisfies ``predicate``.

    Found by content rather than by folder name: these directories get
    renamed, and a test that hard-codes a name fails for a reason unrelated to
    the code under test.
    """
    root = embedding_root()
    if root is None:
        return None
    import pandas as pd

    for directory in sorted(root.iterdir()):
        metadata = directory / "features_metadata.csv"
        if not metadata.exists():
            continue
        try:
            # The whole column, not a head: exports are grouped by arm, so the
            # first few thousand rows of a two-arm export are all one arm and
            # a capped read would reject the very file being looked for.
            frame = pd.read_csv(metadata, dtype=str).fillna("")
        except (OSError, ValueError):
            continue
        if predicate(frame):
            return directory
    return None


@pytest.fixture(scope="session")
def real_embedding_root() -> Path:
    root = embedding_root()
    if root is None:
        pytest.skip("set PLATO_TEST_EMBEDDING_ROOT to test against real exports")
    return root


@pytest.fixture(scope="session")
def real_image_root() -> Path:
    root = image_root()
    if root is None:
        pytest.skip("set PLATO_TEST_IMAGE_ROOT to test against real images")
    return root
