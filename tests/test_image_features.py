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


# -- computing features for an export whose vectors are missing --------------


def _export_with_metadata_only(tmp_path, n=40):
    """An export directory that describes images but has no feature array."""
    import pandas as pd
    import tifffile

    directory = tmp_path / "export"
    directory.mkdir()
    images = tmp_path / "images" / "P1"
    images.mkdir(parents=True)

    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        name = f"img_{i:04d}"
        tifffile.imwrite(
            images / f"{name}.tiff",
            rng.integers(0, 4096, size=(48, 48), dtype=np.uint16),
        )
        rows.append({"plate": "P1", "well": f"A{i % 12 + 1:02d}",
                     "label": f"gene{i % 4}_1", "image_name": name})
    pd.DataFrame(rows).to_csv(directory / "features_metadata.csv", index=False)
    return directory, tmp_path / "images"


def test_computes_features_for_an_export_missing_its_vectors(tmp_path):
    """Metadata without vectors must not be a dead end."""
    from plato.data.explorer_model import ImageResolver, build_frame
    from plato.data.export_features import COMPUTED_FILENAME, build

    directory, image_root = _export_with_metadata_only(tmp_path)
    import pandas as pd

    metadata = pd.read_csv(directory / "features_metadata.csv", dtype=str)
    resolver = ImageResolver.detect(image_root, metadata)
    assert resolver is not None

    dataset = build(directory, resolver)
    assert dataset.n_points == 40
    assert dataset.run_info.get("computed") is True
    # The export's own annotation survives, which is the whole point.
    frame, _ = build_frame(dataset)
    assert set(frame["gene"]) == {"gene0", "gene1", "gene2", "gene3"}

    # A complete run is cached and reused.
    assert (directory / COMPUTED_FILENAME).exists()
    again = build(directory, resolver=None)
    assert again.n_points == 40


def test_partial_runs_are_not_cached(tmp_path):
    """A sampled run has fewer rows than the metadata and must not be stored.

    The row-count guard would reject it on load, so writing it would leave a
    file that looks like an answer and never is.
    """
    from plato.data.explorer_model import ImageResolver
    from plato.data.export_features import COMPUTED_FILENAME, build

    directory, image_root = _export_with_metadata_only(tmp_path)
    import pandas as pd

    metadata = pd.read_csv(directory / "features_metadata.csv", dtype=str)
    resolver = ImageResolver.detect(image_root, metadata)

    dataset = build(directory, resolver, max_images=10)
    assert dataset.n_points <= 10
    assert not (directory / COMPUTED_FILENAME).exists()


def test_computing_without_a_resolver_says_what_to_do(tmp_path):
    from plato.data.export_features import build

    directory, _ = _export_with_metadata_only(tmp_path)
    with pytest.raises(ValueError, match="Locate"):
        build(directory, resolver=None)
