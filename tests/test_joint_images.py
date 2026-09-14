"""Images for an entry whose rows came from several screens at once.

A joint entry (plato.data.joint_projection) concatenates several datasets, so
its rows point at images under several roots -- the CRISPRi folder holds half
of them and the ABx folder the other half. The resolver hunt was written when
a dataset always came from one screen: it collected matching roots, took the
first, and reported the rest as "ambiguous".

For a joint entry that is wrong in a way nothing reports. The chosen root
resolves its own half perfectly, so there is no error and no warning -- the
other arm's points simply never show an image. Measured on a real two-arm
entry before the fix: 30 of 60 rows resolved.

Two roots matching means two different things, and they want opposite
handling: rival answers for the whole dataset (pick one, say it was a guess)
versus complementary pieces of it (merge them). Coverage tells them apart.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plato.data.embeddings import EmbeddingDataset
from plato.data.explorer_model import ImageResolver, build_frame
from plato.data.joint_projection import make_joint_entry
from plato.data.workspace import EmbeddingEntry
from plato.views.explorer import _choose_resolver

# One filename shape, two folders -- the real case. NIS writes the same
# convention for every plate of every screen, which is exactly why a root
# matching some rows proves nothing about the rest.
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


def test_one_arms_root_alone_covers_only_that_arm(screens):
    """The premise: this is why taking the first match loses half the data."""
    joint = make_joint_entry([_entry("armA"), _entry("armB")])
    resolver = ImageResolver.detect(screens["armA"], joint.frame)
    assert resolver is not None
    assert _resolved(resolver, joint.frame) == len(NAMES["armA"])
    assert _resolved(resolver, joint.frame) < len(joint.frame)


def test_complementary_roots_are_merged_not_ranked(screens):
    """The fix: every row of a joint entry gets its image."""
    joint = make_joint_entry([_entry("armA"), _entry("armB")])
    matches = [
        r
        for r in (ImageResolver.detect(root, joint.frame) for root in screens.values())
        if r is not None
    ]
    assert len(matches) == 2, "both arms' roots should match some rows"

    chosen, ambiguous = _choose_resolver(matches, joint.frame)
    assert _resolved(chosen, joint.frame) == len(joint.frame)
    # Merged roots are all in use, so none of them is a guess.
    assert ambiguous == []
    assert len(chosen.roots) == 2


def test_a_single_source_entry_still_picks_one_root(screens):
    """The fix must not turn ordinary ambiguity into a merge.

    When a root covers the dataset on its own, any other match is a rival
    answer to the same question -- one of them is the wrong screen, and
    merging would quietly show images from it.
    """
    entry = _entry("armA")
    matches = [
        r
        for r in (ImageResolver.detect(root, entry.frame) for root in screens.values())
        if r is not None
    ]
    chosen, _ = _choose_resolver(matches, entry.frame)
    assert len(chosen.roots) == 1
    assert _resolved(chosen, entry.frame) == len(entry.frame)


def test_genuine_ambiguity_is_still_reported_as_a_guess(screens, tmp_path):
    """Two roots holding the SAME images: one must win and say so."""
    twin = tmp_path / "twin"
    twin.mkdir()
    for stem in NAMES["armA"]:
        (twin / f"{stem}.tiff").write_bytes(b"\0")

    entry = _entry("armA")
    matches = [
        ImageResolver.detect(screens["armA"], entry.frame),
        ImageResolver.detect(twin, entry.frame),
    ]
    chosen, ambiguous = _choose_resolver(matches, entry.frame)
    assert len(chosen.roots) == 1
    assert len(ambiguous) == 1, "the losing root must be reported, not merged"


def test_no_match_at_all_is_not_an_error(screens):
    assert _choose_resolver([], _entry("armA").frame) == (None, [])


def test_a_three_way_joint_entry_reaches_every_arm(screens, tmp_path):
    """Two matching roots used to be the cap; three sources needs three."""
    third = tmp_path / "armC"
    third.mkdir()
    stems = [f"20261001_WellC0{i}_PointC0{i}_0000_Channel" for i in range(1, 4)]
    for stem in stems:
        (third / f"{stem}.tiff").write_bytes(b"\0")
    NAMES["armC"] = stems
    try:
        joint = make_joint_entry([_entry("armA"), _entry("armB"), _entry("armC")])
        roots = [screens["armA"], screens["armB"], third]
        matches = [
            r
            for r in (ImageResolver.detect(root, joint.frame) for root in roots)
            if r is not None
        ]
        chosen, _ = _choose_resolver(matches, joint.frame)
        assert _resolved(chosen, joint.frame) == len(joint.frame)
    finally:
        NAMES.pop("armC", None)
