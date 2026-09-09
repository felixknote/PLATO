"""Feature vectors computed from images, for when there is no embedding export.

The explorer's normal input is a precomputed DINO export. That is the better
input -- a self-supervised model sees far more than any hand-written
descriptor -- but it means the whole tab is unusable until somebody has run
the extractor, which is exactly when you most want to look at a new plate.

So: descriptors computed from the thumbnails PLATO has already cached. They
are cheap (the cache is local SQLite, already downscaled, no network read),
they need no GPU or model weights, and they are enough to answer the questions
a projection is actually asked at this stage -- does this plate separate from
that one, are these replicates consistent, is that well an outlier.

They are NOT a substitute for learned embeddings for a phenotype claim, and
the UI says so.

Each vector is intensity statistics plus a small texture and gradient
summary: enough to separate "dense rods", "filaments", "lysed", "empty"
without pretending to be a morphology pipeline.
"""

from __future__ import annotations

import numpy as np

# Histogram bins over the normalised intensity range. Coarse on purpose: this
# is a shape-of-the-distribution summary, not a fingerprint.
HISTOGRAM_BINS = 16

FEATURE_NAMES = (
    "mean",
    "std",
    "p05",
    "p25",
    "median",
    "p75",
    "p95",
    "skewness",
    "edge_mean",
    "edge_std",
    "edge_p90",
    "gradient_energy",
    "focus",
    "row_variation",
    "column_variation",
    *(f"hist_{i:02d}" for i in range(HISTOGRAM_BINS)),
)


def describe(plane: np.ndarray) -> np.ndarray:
    """A fixed-length descriptor for one image plane.

    Every feature is computed on a per-image normalised copy, so a plate that
    was imaged brighter than another does not separate on brightness alone --
    which would otherwise dominate the projection and tell you only which day
    the plate was acquired.
    """
    data = np.asarray(plane, dtype=np.float32)
    if data.ndim > 2:
        data = data.reshape(-1, *data.shape[-2:])[0]

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((data - lo) / (hi - lo), 0.0, 1.0)

    flat = norm.ravel()
    mean = float(flat.mean())
    std = float(flat.std())
    p05, p25, median, p75, p95 = (
        float(v) for v in np.percentile(flat, [5, 25, 50, 75, 95])
    )
    # Third moment, guarded: a flat image has zero spread and no skew.
    skewness = float(((flat - mean) ** 3).mean() / (std**3)) if std > 1e-6 else 0.0

    # Gradients carry the texture: cells give strong local edges, an empty
    # well does not.
    dy, dx = np.gradient(norm)
    edges = np.hypot(dx, dy)
    edge_flat = edges.ravel()
    edge_mean = float(edge_flat.mean())
    edge_std = float(edge_flat.std())
    edge_p90 = float(np.percentile(edge_flat, 90))
    gradient_energy = float((edge_flat**2).mean())

    # Variance of the Laplacian: the standard focus measure, and a good
    # separator of a defocused acquisition from a real phenotype.
    laplacian = (
        -4 * norm[1:-1, 1:-1]
        + norm[:-2, 1:-1]
        + norm[2:, 1:-1]
        + norm[1:-1, :-2]
        + norm[1:-1, 2:]
    )
    focus = float(laplacian.var()) if laplacian.size else 0.0

    # Illumination structure: a vignetted or gradient-lit field varies
    # smoothly across rows/columns in a way a biological difference does not.
    row_variation = float(norm.mean(axis=1).std())
    column_variation = float(norm.mean(axis=0).std())

    histogram, _ = np.histogram(flat, bins=HISTOGRAM_BINS, range=(0.0, 1.0))
    histogram = histogram.astype(np.float32) / max(1, flat.size)

    return np.asarray(
        [
            mean,
            std,
            p05,
            p25,
            median,
            p75,
            p95,
            skewness,
            edge_mean,
            edge_std,
            edge_p90,
            gradient_energy,
            focus,
            row_variation,
            column_variation,
            *histogram.tolist(),
        ],
        dtype=np.float32,
    )


def standardise(vectors: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance per feature.

    These descriptors live on wildly different scales -- a histogram bin is a
    fraction, the Laplacian variance is not -- so without this the projection
    would be driven by whichever feature happens to have the largest numbers.
    """
    data = np.asarray(vectors, dtype=np.float32)
    if data.size == 0:
        return data
    centre = data.mean(axis=0, keepdims=True)
    spread = data.std(axis=0, keepdims=True)
    # A constant feature carries no information; leaving its spread at 1
    # maps it to zero rather than producing NaNs.
    np.maximum(spread, 1e-6, out=spread)
    return (data - centre) / spread
