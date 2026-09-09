"""Precomputed feature vectors and their 2-D projections.

A DINO export is three files that must be read together:

``features_all.npz``
    ``embeddings``: float32 (N, D). One row per image. The other keys are
    integer label indices, which we ignore -- the metadata CSV carries the
    same information as strings, and strings survive a change of label map.
``features_metadata.csv``
    One row per embedding, in the SAME ORDER. This is the only thing tying a
    vector to an image, so a length mismatch is fatal rather than something to
    paper over: silently truncating would mislabel every point after the first
    missing row.
``features_label_map.json``
    label -> index. Read only to report what the export believed it contained.

The embedding index is deliberately NOT treated as a filename. The join back
to an image goes through the metadata columns (plate + image_name), because
the row order is an artefact of whatever order the extractor walked the disk
in, and that is not a contract anyone maintains.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# The array key written by the DINO export. The same file also carries a
# `*_indices` array whose name varies by dataset (gene_indices, drug_indices,
# label_indices); none of them are read here.
EMBEDDING_KEY = "embeddings"

# Candidate feature files, in the order they are tried. The .npz is what the
# exporter writes today; the others let a dataset be dropped in without a
# re-export if it was saved some other way.
FEATURE_FILENAMES = ("features_all.npz", "features.npy", "features.parquet")

METADATA_FILENAME = "features_metadata.csv"
LABEL_MAP_FILENAME = "features_label_map.json"
RUN_METADATA_FILENAME = "metadata.json"


class EmbeddingError(RuntimeError):
    """Raised when a dataset directory cannot be loaded as embeddings."""


@dataclass(slots=True)
class EmbeddingDataset:
    """One directory of embeddings plus the metadata describing its rows."""

    name: str
    directory: Path
    vectors: np.ndarray  # (N, D) float32
    frame: pd.DataFrame  # N rows, metadata columns
    run_info: dict  # contents of metadata.json, or {}

    @property
    def n_points(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def n_dimensions(self) -> int:
        return int(self.vectors.shape[1])

    def columns(self) -> list[str]:
        return [str(c) for c in self.frame.columns]

    def fingerprint(self) -> str:
        """Stable id for this dataset's vectors, used to key cached projections.

        Hashes the shape and a strided sample rather than the whole array: a
        full hash of 140 MB costs more than most projections save, and a
        change to the features that leaves shape, dtype, endpoints and every
        4096th value identical is not a thing that happens in practice.
        """
        digest = hashlib.sha256()
        digest.update(str(self.vectors.shape).encode())
        digest.update(str(self.vectors.dtype).encode())
        flat = self.vectors.ravel()
        digest.update(np.ascontiguousarray(flat[::4096]).tobytes())
        digest.update(np.ascontiguousarray(flat[:256]).tobytes())
        digest.update(np.ascontiguousarray(flat[-256:]).tobytes())
        return digest.hexdigest()[:16]


def find_feature_file(directory: Path) -> Path | None:
    """The first recognised feature array in ``directory``, or None."""
    for name in FEATURE_FILENAMES:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def is_dataset_dir(directory: Path) -> bool:
    """True if the directory looks like an embedding export.

    Keyed on the metadata CSV, not on the feature array: a directory whose
    features have not been written yet is still a dataset, and one we want to
    list and report on rather than hide.
    """
    return (directory / METADATA_FILENAME).exists()


def discover_datasets(root: Path) -> list[Path]:
    """Every embedding dataset directory under ``root``, root itself included."""
    if not root.is_dir():
        return []
    found = [root] if is_dataset_dir(root) else []
    found.extend(sorted(p for p in root.iterdir() if p.is_dir() and is_dataset_dir(p)))
    return found


def _read_vectors(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as handle:
            if EMBEDDING_KEY not in handle:
                raise EmbeddingError(
                    f"{path.name} has no '{EMBEDDING_KEY}' array "
                    f"(found: {', '.join(handle.files) or 'nothing'})"
                )
            vectors = handle[EMBEDDING_KEY]
    elif path.suffix == ".npy":
        vectors = np.load(path, allow_pickle=False)
    else:
        vectors = pd.read_parquet(path).to_numpy()

    vectors = np.asarray(vectors)
    if vectors.ndim != 2:
        raise EmbeddingError(f"expected a 2-D feature array, got shape {vectors.shape}")
    return np.ascontiguousarray(vectors, dtype=np.float32)


def load_dataset(directory: Path, name: str | None = None) -> EmbeddingDataset:
    """Load one embedding directory.

    Raises ``EmbeddingError`` -- naming the exact missing path -- rather than
    returning something half-loaded. An explorer with no vectors is not a
    degraded view, it is a blank one, and the reason has to reach the user.
    """
    directory = Path(directory)
    metadata_path = directory / METADATA_FILENAME
    if not metadata_path.exists():
        raise EmbeddingError(f"no {METADATA_FILENAME} in {directory}")

    feature_path = find_feature_file(directory)
    if feature_path is None:
        raise EmbeddingError(
            f"no feature array in {directory}\n\n"
            f"Expected one of: {', '.join(FEATURE_FILENAMES)}.\n"
            f"{metadata_path.name} is present, so the export ran but its "
            f"vectors were not saved here. Re-run the export, or drop the "
            f".npz into this folder."
        )

    frame = pd.read_csv(metadata_path, dtype=str).fillna("")
    vectors = _read_vectors(feature_path)

    if len(frame) != vectors.shape[0]:
        raise EmbeddingError(
            f"{feature_path.name} has {vectors.shape[0]} rows but "
            f"{metadata_path.name} has {len(frame)}. They are matched by "
            f"position, so a mismatch means every point could be mislabelled."
        )

    run_info: dict = {}
    run_path = directory / RUN_METADATA_FILENAME
    if run_path.exists():
        try:
            run_info = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            run_info = {}

    return EmbeddingDataset(
        name=name or directory.name,
        directory=directory,
        vectors=vectors,
        frame=frame,
        run_info=run_info,
    )
