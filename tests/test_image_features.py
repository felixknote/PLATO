"""Descriptors computed from images, and the dataset built from loaded plates.

This is what makes the explorer usable before anyone has run an embedding
export, so the properties that matter are: the descriptor separates visibly
different images, it does not separate on brightness alone, and a plate with
no thumbnail cache fails with a message that says what to do.
"""

from __future__ import annotations

import numpy as np
import pytest

from plato.data.image_features import FEATURE_NAMES, describe, standardise


def _rods(rng, size=96):
    """An image with strong local edges, like cells."""
    plane = rng.normal(1000, 40, size=(size, size)).astype(np.float32)
    for _ in range(30):
        y, x = rng.integers(4, size - 12, size=2)
        plane[y : y + 3, x : x + 10] += 900
    return plane


def _empty(rng, size=96):
    """Flat noise, like an empty well."""
    return rng.normal(1000, 40, size=(size, size)).astype(np.float32)


def test_descriptor_has_a_fixed_length():
    rng = np.random.default_rng(0)
    assert len(describe(_rods(rng))) == len(FEATURE_NAMES)
    assert len(describe(_empty(rng))) == len(FEATURE_NAMES)


def test_descriptor_separates_texture_from_flat():
    """Objects on a background must be distinguishable from bare noise.

    Not via mean edge strength: every image is normalised to its own 1-99
    percentile range, so pure noise is stretched to full contrast and scores a
    HIGHER mean gradient than a mostly-empty field containing a few bright
    objects. The features that actually carry "there are objects here" are the
    strong-edge tail and the skew of the intensity distribution.
    """
    rng = np.random.default_rng(0)
    textured = describe(_rods(rng))
    flat = describe(_empty(rng))

    edge_p90 = FEATURE_NAMES.index("edge_p90")
    skewness = FEATURE_NAMES.index("skewness")
    assert textured[edge_p90] > flat[edge_p90]
    # A few bright objects on a dim background skew hard positive; noise does
    # not skew at all.
    assert textured[skewness] > 1.0
    assert abs(flat[skewness]) < 0.5

    # And the two descriptors must not be near-identical overall.
    assert np.linalg.norm(textured - flat) > 1.0


def test_descriptor_ignores_a_pure_brightness_shift():
    """A plate imaged brighter must not separate on brightness alone.

    Otherwise the projection reports which day a plate was acquired rather
    than anything biological.
    """
    rng = np.random.default_rng(1)
    plane = _rods(rng)
    brighter = plane * 2.5 + 500
    np.testing.assert_allclose(describe(plane), describe(brighter), atol=1e-3)


def test_descriptor_survives_degenerate_input():
    flat = np.full((32, 32), 7.0, dtype=np.float32)
    values = describe(flat)
    assert len(values) == len(FEATURE_NAMES)
    assert np.all(np.isfinite(values))

    empty = np.full((8, 8), np.nan, dtype=np.float32)
    assert np.all(np.isfinite(describe(empty)))


def test_standardise_centres_and_scales():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(50, 6)).astype(np.float32) * 100 + 30
    out = standardise(data)
    np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-4)
    np.testing.assert_allclose(out.std(axis=0), 1, atol=1e-4)


def test_standardise_handles_a_constant_feature():
    data = np.column_stack(
        [np.arange(20, dtype=np.float32), np.full(20, 5.0, dtype=np.float32)]
    )
    out = standardise(data)
    assert np.all(np.isfinite(out))
    # A feature that never varies carries no information and maps to zero.
    np.testing.assert_allclose(out[:, 1], 0)


def test_build_without_thumbnails_says_what_to_do():
    from plato.data.plate_features import build

    class _Plate:
        class cfg:  # noqa: N801
            class project:  # noqa: N801
                work_dir = "."

            thumb_db_path = "does-not-exist.sqlite"

        class db:  # noqa: N801
            @staticmethod
            def count():
                return 0

    class _Session:
        plates = [_Plate()]

    with pytest.raises(ValueError, match="thumbs"):
        build(_Session())
