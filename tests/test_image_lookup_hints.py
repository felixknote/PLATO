"""hint_columns and disambiguation for a frame whose plate is constant.

Found while wiring up Learned_Embeddings folds: every fold is single-plate by
construction, so its `plate` column has exactly one distinct value. The old
`1 < distinct` rule excluded any such column from `hint_columns`, which meant
a fold's rows could never be told apart from a same-named file on every OTHER
plate in a multi-plate image root -- every row silently failed to resolve,
even though the root held the right images and the frame carried the right
plate value. The fix offers a constant column too; resolution still needs the
value to match a candidate path (a wrong root, or one with no plate folders,
still fails, just as it should).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from plato.data.image_lookup import ImageIndex, hint_columns


def test_hint_columns_offers_a_constant_column():
    frame = pd.DataFrame({"plate": ["P1", "P1", "P1"], "image_name": ["a", "b", "c"]})

    hints = hint_columns(frame, exclude=("image_name",))

    assert "plate" in hints


def test_hint_columns_still_excludes_a_per_row_identifier():
    frame = pd.DataFrame(
        {"plate": ["P1"] * 4, "row_id": ["r0", "r1", "r2", "r3"], "image_name": ["a", "b", "c", "d"]}
    )

    hints = hint_columns(frame, exclude=("image_name",))

    assert "row_id" not in hints
    assert "plate" in hints


def test_constant_plate_disambiguates_a_name_shared_across_plates(tmp_path):
    """The scenario that motivated the fix: the same file name exists under
    several plate folders, and the only thing that tells them apart is a
    plate value that never varies WITHIN this dataset's own frame."""
    root = tmp_path
    for plate in ("P1", "P2", "P3"):
        plate_dir = root / plate
        plate_dir.mkdir()
        (plate_dir / "WellA01_PointA01_0000_Seq0000.tiff").write_bytes(b"")

    index = ImageIndex.build(root)
    frame = pd.DataFrame({"plate": ["P2"], "image_name": ["WellA01_PointA01_0000_Seq0000"]})
    hints = hint_columns(frame, exclude=("image_name",))

    resolved = index.resolve(
        "WellA01_PointA01_0000_Seq0000", [str(frame.iloc[0][c]) for c in hints]
    )

    assert resolved is not None
    assert resolved.parent.name == "P2"


def test_no_hints_available_falls_back_to_first_candidate(tmp_path):
    """With nothing at all to disambiguate by, resolve() still returns
    something rather than silently dropping every ambiguous row -- unchanged
    behaviour, guarded so the constant-column change doesn't quietly alter
    the truly-no-hints case."""
    root = tmp_path
    for plate in ("P1", "P2"):
        plate_dir = root / plate
        plate_dir.mkdir()
        (plate_dir / "shared.tiff").write_bytes(b"")

    index = ImageIndex.build(root)
    resolved = index.resolve("shared", [])

    assert resolved is not None
