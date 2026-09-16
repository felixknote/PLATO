"""A joint entry's images should never need locating a second time.

Combining two already-resolved entries used to leave the joint entry showing
"Not located yet." in the Locate dialog, even though the answer is just "both
of the above" -- a joint entry's rows are exactly its sources' rows
concatenated, and ImageResolver.combined_with already knows how to fold two
resolvers together (it is what handles a dataset split across two folders).
This wires that fold into the dialog so a joint row inherits automatically:
on open if its sources are already resolved, or the moment they are resolved
in the same dialog session.
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
from plato.data.joint_projection import make_joint_entry  # noqa: E402
from plato.data.workspace import EmbeddingEntry  # noqa: E402
from plato.views.locate_all_dialog import LocateAllDialog  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


NAMES = {
    "armA": [f"20260825_Well{w}_Point{w}_{i:04d}_Channel" for w in ("A01", "A02") for i in range(3)],
    "armB": [f"20260901_Well{w}_Point{w}_{i:04d}_Channel" for w in ("B01", "B02") for i in range(3)],
}


@pytest.fixture
def screens(tmp_path):
    """Two screen folders, each holding only its own arm's images."""
    roots = {}
    for arm, stems in NAMES.items():
        root = tmp_path / arm
        root.mkdir()
        for stem in stems:
            (root / f"{stem}.tiff").write_bytes(b"\0")
        roots[arm] = root
    return roots


def _entry(arm: str) -> EmbeddingEntry:
    stems = NAMES[arm]
    raw = pd.DataFrame(
        {
            "experiment": [arm] * len(stems),
            "plate": ["P1"] * len(stems),
            "well": [s.split("Well")[1][:3] for s in stems],
            "image_name": stems,
        }
    )
    dataset = EmbeddingDataset(
        name=arm,
        directory=Path("/x") / arm,
        vectors=np.zeros((len(stems), 8), dtype=np.float32),
        frame=raw,
        run_info={},
    )
    frame, _ = build_frame(dataset)
    return EmbeddingEntry(name=arm, dataset=dataset, frame=frame)


def _resolved(resolver, frame) -> int:
    return sum(1 for _, row in frame.iterrows() if resolver.path_for(row))


def test_a_joint_row_inherits_when_both_sources_are_already_resolved(app, screens):
    entry_a = _entry("armA")
    entry_b = _entry("armB")
    entry_a.resolver = ImageResolver.detect(screens["armA"], entry_a.frame)
    entry_b.resolver = ImageResolver.detect(screens["armB"], entry_b.frame)
    assert entry_a.resolver is not None
    assert entry_b.resolver is not None

    joint = make_joint_entry([entry_a, entry_b])
    dialog = LocateAllDialog([entry_a, entry_b, joint])

    row = dialog._rows[joint.key]
    assert row.pending_resolver is not None
    assert _resolved(row.pending_resolver, joint.frame) == len(joint.frame)
    assert "Inherited" in row.status_label.text()
    assert "armA" in row.status_label.text()
    assert "armB" in row.status_label.text()


def test_a_joint_row_stays_unresolved_when_no_source_is_resolved(app, screens):
    entry_a = _entry("armA")
    entry_b = _entry("armB")
    joint = make_joint_entry([entry_a, entry_b])
    dialog = LocateAllDialog([entry_a, entry_b, joint])

    row = dialog._rows[joint.key]
    assert row.pending_resolver is None
    assert row.status_label.text() == "Not located yet."


def test_a_joint_row_partially_inherits_from_one_resolved_source(app, screens):
    """Only one arm located: the joint row still gets that arm's images, and
    its own report (over the JOINT frame) says the coverage honestly rather
    than claiming the source arm's own 100%."""
    entry_a = _entry("armA")
    entry_b = _entry("armB")
    entry_a.resolver = ImageResolver.detect(screens["armA"], entry_a.frame)
    assert entry_a.resolver is not None

    joint = make_joint_entry([entry_a, entry_b])
    dialog = LocateAllDialog([entry_a, entry_b, joint])

    row = dialog._rows[joint.key]
    assert row.pending_resolver is not None
    resolved = _resolved(row.pending_resolver, joint.frame)
    assert 0 < resolved < len(joint.frame)
    report = row.pending_resolver.report
    assert report is not None
    assert report.fraction < 1.0


def test_resolving_a_source_mid_dialog_fills_in_the_joint_row(app, screens):
    """The cascade: a source resolved AFTER the dialog is already open must
    still reach any joint row waiting on it, not only ones resolved before
    the dialog was constructed."""
    entry_a = _entry("armA")
    entry_b = _entry("armB")
    joint = make_joint_entry([entry_a, entry_b])
    dialog = LocateAllDialog([entry_a, entry_b, joint])

    joint_row = dialog._rows[joint.key]
    assert joint_row.pending_resolver is None

    resolver_a = ImageResolver.detect(screens["armA"], entry_a.frame)
    resolver_b = ImageResolver.detect(screens["armB"], entry_b.frame)
    dialog._on_scanned(entry_a.key, resolver_a, False)
    dialog._on_scanned(entry_b.key, resolver_b, False)

    assert joint_row.pending_resolver is not None
    assert _resolved(joint_row.pending_resolver, joint.frame) == len(joint.frame)


def test_a_user_chosen_joint_resolver_is_never_overwritten_by_a_source_change(app, screens):
    """Once the USER has picked a folder for a joint row -- not an inherit,
    a real choice -- a source resolving afterwards must not silently
    replace it. Re-running Locate all is how a stale choice gets revisited,
    not an automatic overwrite behind the user's back."""
    entry_a = _entry("armA")
    entry_b = _entry("armB")
    joint = make_joint_entry([entry_a, entry_b])
    dialog = LocateAllDialog([entry_a, entry_b, joint])
    joint_row = dialog._rows[joint.key]

    # The user picks a folder for the joint row directly, before either
    # source is resolved -- e.g. everything actually lives under one shared
    # root and they know it without going arm by arm.
    chosen = ImageResolver.detect(screens["armA"], joint.frame)
    joint_row.set_result(chosen)
    assert joint_row.pending_resolver is chosen

    entry_a.resolver = ImageResolver.detect(screens["armA"], entry_a.frame)
    dialog._on_scanned(entry_a.key, entry_a.resolver, False)
    assert joint_row.pending_resolver is chosen


def test_an_auto_inherited_row_refreshes_as_more_sources_resolve(app, screens):
    """The partial-inherit case must upgrade in place: resolving the second
    arm while the dialog is still open should reach the joint row
    immediately, not require reopening Locate all to pick up the improvement."""
    entry_a = _entry("armA")
    entry_b = _entry("armB")
    entry_a.resolver = ImageResolver.detect(screens["armA"], entry_a.frame)

    joint = make_joint_entry([entry_a, entry_b])
    dialog = LocateAllDialog([entry_a, entry_b, joint])
    joint_row = dialog._rows[joint.key]
    assert 0 < _resolved(joint_row.pending_resolver, joint.frame) < len(joint.frame)

    entry_b.resolver = ImageResolver.detect(screens["armB"], entry_b.frame)
    dialog._on_scanned(entry_b.key, entry_b.resolver, False)
    assert _resolved(joint_row.pending_resolver, joint.frame) == len(joint.frame)


def test_results_includes_the_inherited_joint_resolver(app, screens):
    """The dialog's own results() must carry the inherited resolver through,
    exactly as it does for a manually chosen one -- explorer.py applies
    whatever results() returns onto the workspace without distinguishing how
    a row got its resolver."""
    entry_a = _entry("armA")
    entry_b = _entry("armB")
    entry_a.resolver = ImageResolver.detect(screens["armA"], entry_a.frame)
    entry_b.resolver = ImageResolver.detect(screens["armB"], entry_b.frame)

    joint = make_joint_entry([entry_a, entry_b])
    dialog = LocateAllDialog([entry_a, entry_b, joint])

    results = dialog.results()
    assert joint.key in results
    assert _resolved(results[joint.key], joint.frame) == len(joint.frame)


def test_a_non_joint_entry_is_never_treated_as_inheriting(app, screens):
    """_apply_inherited must be a no-op for an ordinary entry -- it has no
    source_keys to fold, and must fall back to "Not located yet." exactly as
    before this feature existed."""
    entry_a = _entry("armA")
    entry_b = _entry("armB")
    entry_b.resolver = ImageResolver.detect(screens["armB"], entry_b.frame)

    dialog = LocateAllDialog([entry_a, entry_b])
    row_a = dialog._rows[entry_a.key]
    assert row_a.pending_resolver is None
    assert row_a.status_label.text() == "Not located yet."
