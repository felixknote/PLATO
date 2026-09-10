"""Raw per-image statistics, for colouring an embedding by what the pixels do.

Colouring by metadata answers "where are the acrB points?". Colouring by
brightness or focus answers a different and equally important question: is
this cluster real, or is it an acquisition artefact? A region that separates
cleanly on nothing but mean intensity is usually telling you about the
microscope, not the biology -- and that is worth seeing before it becomes a
figure.

Deliberately NOT ``image_features.describe``. That normalises every image to
its own 1-99 percentile range before measuring, so a plate imaged brighter
does not separate on brightness alone -- correct when the descriptors ARE the
embedding, and exactly wrong here, where raw brightness is the signal being
looked for. These are the raw numbers, in the image's own units.

Cheap by construction: a coarse stride, a handful of statistics, and one pass
over the thread pool. Measured on real 3200x3200 plates: ~5 ms per image at
stride 8, so a 24k-point dataset takes a few seconds across all cores. Results
are cached next to the projections, keyed by the dataset fingerprint, so the
pass happens once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Every statistic offered, in display order. Keep the keys stable: they are
# written into the cache and used as colour-by column names.
STAT_NAMES: tuple[str, ...] = (
    "brightness",
    "contrast",
    "focus",
    "saturation",
    "background",
    "foreground",
)

STAT_LABELS = {
    "brightness": "Brightness (mean)",
    "contrast": "Contrast (std)",
    "focus": "Focus (edge energy)",
    "saturation": "Saturated pixels (%)",
    "background": "Background (5th pct)",
    "foreground": "Foreground (95th pct)",
}

STAT_HELP = {
    "brightness": "Mean pixel value. Separates plates imaged at different "
    "exposures, and flags a cluster that is only a brightness difference.",
    "contrast": "Standard deviation of pixel values. Low contrast usually "
    "means an empty or out-of-focus field.",
    "focus": "Mean absolute gradient. The standard cheap focus measure: high "
    "for sharp edges, low for a blurred or empty field.",
    "saturation": "Percentage of pixels at the very top of the dtype range. "
    "Anything above a few percent is clipped and its intensities are lost.",
    "background": "5th percentile. Where the empty parts of the field sit, "
    "which is what an offset or a stray light change moves.",
    "foreground": "95th percentile. Where the bright objects sit, largely "
    "independent of how much of the field they cover.",
}

# Read every Nth pixel. These statistics are aggregates over millions of
# pixels, so an eighth of them estimates each one far more precisely than the
# difference between images ever needs -- and costs an eighth of the read.
STATS_STRIDE = 8


def describe_raw(plane: np.ndarray) -> dict[str, float]:
    """The statistics above for one plane, in the image's own units."""
    data = np.asarray(plane)
    if data.ndim > 2:
        data = data.reshape(-1, *data.shape[-2:])[0]
    values = data[np.isfinite(data)] if data.dtype.kind == "f" else data.ravel()
    if values.size == 0:
        return {name: float("nan") for name in STAT_NAMES}

    as_float = values.astype(np.float32, copy=False)
    p05, p95 = (float(v) for v in np.percentile(as_float, [5, 95]))

    # Ceiling for the saturation test: the dtype's maximum for integers, the
    # observed maximum for floats, which have no inherent full-scale value.
    if data.dtype.kind in "ui":
        ceiling = float(np.iinfo(data.dtype).max)
    else:
        ceiling = float(as_float.max())
    saturated = (
        float((as_float >= ceiling * 0.999).mean() * 100.0) if ceiling > 0 else 0.0
    )

    # Gradient energy on the strided plane. Mean absolute difference between
    # neighbouring pixels in both directions -- the cheapest focus measure
    # that behaves monotonically with blur.
    if data.ndim == 2 and min(data.shape) > 1:
        small = data[::4, ::4].astype(np.float32)
        gy = np.abs(np.diff(small, axis=0)).mean() if small.shape[0] > 1 else 0.0
        gx = np.abs(np.diff(small, axis=1)).mean() if small.shape[1] > 1 else 0.0
        focus = float((gx + gy) / 2.0)
    else:
        focus = 0.0

    return {
        "brightness": float(as_float.mean()),
        "contrast": float(as_float.std()),
        "focus": focus,
        "saturation": saturated,
        "background": p05,
        "foreground": p95,
    }


class StatsCache:
    """Per-dataset statistics on disk, keyed by fingerprint."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def _path(self, fingerprint: str) -> Path:
        return self.directory / f"imagestats_{fingerprint}.npz"

    def load(self, fingerprint: str, n_rows: int) -> dict[str, np.ndarray] | None:
        path = self._path(fingerprint)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as handle:
                out = {name: handle[name] for name in handle.files}
        except (OSError, ValueError, KeyError):
            return None
        # A cache written for a different row count cannot be aligned to this
        # frame; recomputing is cheap and being wrong is not.
        if any(len(v) != n_rows for v in out.values()):
            return None
        return out

    def save(self, fingerprint: str, stats: dict[str, np.ndarray]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self._path(fingerprint), **stats)


def empty(n_rows: int) -> dict[str, np.ndarray]:
    """All-NaN arrays, the state before anything has been measured."""
    return {
        name: np.full(n_rows, np.nan, dtype=np.float32) for name in STAT_NAMES
    }
