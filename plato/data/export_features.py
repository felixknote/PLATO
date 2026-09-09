"""Computing feature vectors for an export whose metadata exists but whose
vectors do not.

A DINO export is metadata plus a feature array. The metadata is small and gets
written first; the array is large and can be missing -- the run was
interrupted, the file was never copied, the save step failed. That leaves a
directory that describes 32,256 images perfectly and cannot be plotted at all.

Rather than a dead end, the images themselves can be described. The metadata is
still the export's own, so every gene, guide, drug and dose stays intact and
the projection is colourable exactly as it would have been; only the vectors
differ, and they are simple descriptors rather than learned embeddings.

Vectors are cached beside the metadata as ``features_computed.npz``, so this
is paid once. The name is deliberately NOT ``features_all.npz``: that name
belongs to the real export, and a computed stand-in must never be mistaken for
one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .embeddings import METADATA_FILENAME, EmbeddingDataset
from .image_features import FEATURE_NAMES, describe, standardise

COMPUTED_FILENAME = "features_computed.npz"

# Read every Nth pixel. These descriptors are summary statistics over a
# 2720 px plane; a quarter-resolution read changes them negligibly and cuts
# the network cost fourfold.
READ_STRIDE = 4


def cached_vectors(directory: Path, expected_rows: int) -> np.ndarray | None:
    """Previously computed vectors for this export, if they still fit."""
    path = Path(directory) / COMPUTED_FILENAME
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as handle:
            vectors = handle["embeddings"]
    except (OSError, ValueError, KeyError):
        return None
    return vectors if vectors.shape[0] == expected_rows else None


def build(
    directory: Path,
    resolver,
    *,
    progress=None,
    max_images: int | None = None,
) -> EmbeddingDataset:
    """Describe the images an export refers to, keeping its own metadata.

    ``resolver`` maps a metadata row to a file on disk; without one there is
    nothing to read and this raises.

    Rows whose image cannot be found are dropped, and the returned dataset's
    frame is trimmed to match -- vectors and metadata must stay row-aligned,
    which is the one invariant the whole explorer depends on.
    """
    directory = Path(directory)
    frame = pd.read_csv(directory / METADATA_FILENAME, dtype=str).fillna("")

    cached = cached_vectors(directory, len(frame))
    if cached is not None:
        if progress:
            progress(len(frame), len(frame), "loaded previously computed features")
        return _dataset(directory, cached, frame)

    if resolver is None:
        raise ValueError(
            "The original images have not been located yet.\n\n"
            "Use Locate… to point PLATO at them, then compute again."
        )

    full_rows = len(frame)
    if max_images is not None and len(frame) > max_images:
        # Sampled evenly rather than from the head: exports are grouped by
        # arm, so the first N rows would be one arm only.
        keep = np.linspace(0, len(frame) - 1, max_images).astype(int)
        frame = frame.iloc[np.unique(keep)].reset_index(drop=True)

    from ..cache import read_plane

    total = len(frame)
    vectors: list[np.ndarray] = []
    kept: list[int] = []
    for position, (_, row) in enumerate(frame.iterrows()):
        if progress and position % 50 == 0:
            progress(position, total, f"describing images… {position:,}/{total:,}")
        path = resolver.path_for(row)
        if path is None:
            continue
        try:
            plane = read_plane(Path(path), stride=READ_STRIDE)
        except Exception:  # noqa: BLE001 - one unreadable file must not stop the run
            continue
        vectors.append(describe(plane))
        kept.append(position)

    if not vectors:
        raise ValueError(
            "None of this export's images could be read.\n\n"
            "Check that Locate… points at the right folder."
        )

    frame = frame.iloc[kept].reset_index(drop=True)
    data = standardise(np.vstack(vectors))

    # Cache only a COMPLETE run. A sampled or partially-readable run has
    # fewer rows than the metadata, so the row-count guard would reject it on
    # load anyway -- writing it would leave a file that looks like an answer
    # and is never used.
    if len(frame) == full_rows:
        try:
            np.savez_compressed(directory / COMPUTED_FILENAME, embeddings=data)
        except OSError:
            # A read-only share is not a reason to fail; it just means this is
            # recomputed next time.
            pass
    return _dataset(directory, data, frame)


def _dataset(directory: Path, vectors: np.ndarray, frame: pd.DataFrame) -> EmbeddingDataset:
    return EmbeddingDataset(
        name=f"{directory.name} (computed)",
        directory=directory,
        vectors=np.ascontiguousarray(vectors, dtype=np.float32),
        frame=frame,
        run_info={
            "model": f"image descriptors ({len(FEATURE_NAMES)} features)",
            "computed": True,
        },
    )
