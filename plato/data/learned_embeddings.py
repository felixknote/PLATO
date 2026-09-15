"""A trained classifier's own embedding space, as an ``EmbeddingDataset``.

The DINO exports in :mod:`plato.data.embeddings` are frozen features: one
fixed vector per image, computed once and never touched by a downstream
training run. A leave-one-plate-out fold under
``Learned_Embeddings\\<run>\\fold_Plate_N\\`` is a different thing entirely --
it is what the classifier trained on the OTHER plates actually saw when it
looked at this one, plus what it predicted. Loading it alongside the DINO
arms lets confounder-hunting ask a second question: not just "does the raw
feature space separate these conditions", but "does the trained model's
space separate them, and where does it get the label wrong".

A fold directory holds five files, read together:

``crop_projections_fold_Plate_N.npy``
    float16 (n_images, n_positions, n_crops, dim) -- e.g. (4032, 100, 9, 256).
    NOT a flat (N, D) array: the trainer projected every crop of every grid
    position separately, so what an image "is" in this space is the mean of
    900 vectors, not a single one that was ever computed directly.
``crop_projections_index_fold_Plate_N.json``
    A list of N dicts (``image_path``, ``modality``, ``true_label``,
    ``true_idx``), in the SAME ROW ORDER as the array's first axis. This is
    the join back to metadata; the array carries no identifying information
    of its own.
``test_positions_fold_Plate_N.csv``
    Per-POSITION predictions, N * n_positions rows (403,200 for a 4032-image,
    100-position fold): ``image_path``, ``predicted_label``, ``prob_true``,
    ``correct``, among others. Grouped by ``image_path`` and aggregated down
    to one row per image -- see ``_aggregate_predictions``.
``test_positions_summary_fold_Plate_N.json``
    The run's config (checkpoint, class counts, grid size) and fold-level
    accuracy. Kept verbatim in ``EmbeddingDataset.run_info`` for display; nothing
    here is parsed back out of it.

Six folds (Plate_1..Plate_6) sit side by side in one run directory, each the
output of a DIFFERENT trained model (the one that held that plate out). They
are loaded as six separate datasets, never concatenated -- mixing them would
plot points from six unrelated embedding spaces as if position meant the same
thing in all of them, which it does not. A user who wants to compare folds
does so the same way they compare CRISPRi and ABx today: the existing
joint-projection UI, applied by hand.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from .embeddings import EmbeddingDataset, EmbeddingError

FOLD_DIR_RE = re.compile(r"^fold_(?P<plate>.+)$")

INDEX_SUFFIX = "crop_projections_index_{plate}.json"
ARRAY_SUFFIX = "crop_projections_{plate}.npy"
POSITIONS_SUFFIX = "test_positions_{plate}.csv"
SUMMARY_SUFFIX = "test_positions_summary_{plate}.json"

# Columns the frame guarantees, whatever the fold's own naming. The index
# JSON's own `image_path` is a Linux training-machine path
# ("/home/.../WellA01_....tiff"); PLATO's ImageResolver matches by filename
# STEM (explorer_model.IMAGE_NAME), not by path, so the stem is split out of
# it here rather than assumed to already exist as its own column.
IMAGE_PATH = "image_path"
IMAGE_NAME = "image_name"
PLATE = "plate"
MODALITY = "modality"
LABEL = "label"
PREDICTED_LABEL = "predicted_label"
PROB_TRUE = "prob_true"
CORRECT = "correct"

# modality -> the two-arm split the rest of PLATO already colours by. Not
# CRISPRi/ABx directly, because a third modality showing up in a future run
# should read as itself, not silently vanish.
_MODALITY_ARM = {"drug": "ABx", "mutant": "CRISPRi"}

# A fold directory's plate spelling ("Plate_1") vs the folder name every real
# screen on disk actually uses ("P1") -- e.g. Z:\Analysis\DINO exports write
# plate="P1" because that is the plate's own image folder name. Without this,
# the frame's `plate` column reads "Plate_1", which shares no token with the
# "P1" folder ImageResolver is trying to disambiguate against, and EVERY row
# in a fold with more than one same-named plate folder (all of them, since
# P1..P6 share every filename) fails to resolve to an image.
_PLATE_NUMBER_RE = re.compile(r"(\d+)$")


def _disk_plate_name(plate: str) -> str:
    match = _PLATE_NUMBER_RE.search(plate)
    return f"P{match.group(1)}" if match else plate


def fold_name(directory: Path) -> str | None:
    """The plate name a fold directory is for (``"Plate_1"``), or None."""
    match = FOLD_DIR_RE.match(directory.name)
    return match.group("plate") if match else None


def is_fold_dir(directory: Path) -> bool:
    """True if ``directory`` looks like one ``fold_Plate_N`` export.

    Keyed on the index JSON, the one file every fold must have for the
    embeddings to mean anything -- the same "one defining file" convention
    ``embeddings.is_dataset_dir`` uses for a DINO export.
    """
    plate = fold_name(directory)
    if plate is None:
        return False
    return (directory / INDEX_SUFFIX.format(plate=directory.name)).exists()


def discover_folds(root: Path) -> list[Path]:
    """Every ``fold_Plate_N`` directory directly under ``root``.

    ``root`` is typically the run directory (``Dec25&Apr26 CRISPRi & ABx``)
    that holds the folds as siblings; ``root`` itself is included too, so
    pointing this at one fold directly also works.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    found = [root] if is_fold_dir(root) else []
    found.extend(sorted(p for p in root.iterdir() if p.is_dir() and is_fold_dir(p)))
    return found


@dataclass(slots=True)
class _PredictionAgg:
    predicted_label: str
    prob_true: float
    correct: str  # "1"/"0" as text, like every other frame column


def _aggregate_predictions(positions: pd.DataFrame) -> dict[str, _PredictionAgg]:
    """One prediction summary per image, from per-position rows.

    ``predicted_label`` is the majority vote across the image's positions
    (ties broken by whichever label sorts first, which is at least
    deterministic); ``prob_true`` is the mean confidence; ``correct`` is
    whether the majority vote matches the true label reported by the SAME
    row set, not re-derived from elsewhere. This mirrors
    ``image_majority_vote_acc`` in the run's own summary JSON rather than
    inventing a different aggregation the summary doesn't agree with.
    """
    out: dict[str, _PredictionAgg] = {}
    for image_path, rows in positions.groupby(IMAGE_PATH, sort=False):
        votes = rows[PREDICTED_LABEL].value_counts()
        top = votes.max()
        winner = sorted(votes[votes == top].index)[0]
        true_label = rows["true_label"].iloc[0] if "true_label" in rows.columns else None
        out[image_path] = _PredictionAgg(
            predicted_label=str(winner),
            prob_true=float(rows[PROB_TRUE].mean()),
            correct="1" if true_label is not None and winner == true_label else "0",
        )
    return out


def is_dataset_dir(directory: Path) -> bool:
    """True if ``directory`` is a fold, checked the way ``embeddings.is_dataset_dir``
    checks a DINO export -- a name callers can use interchangeably with that
    function when they want "either kind of loadable directory"."""
    return is_fold_dir(directory)


def load_fold(directory: Path, name: str | None = None) -> EmbeddingDataset:
    """Load one ``fold_Plate_N`` directory as an ``EmbeddingDataset``.

    Each image's 900 (position x crop) vectors are mean-pooled into one
    256-dim vector, so the result drops into the explorer exactly like a
    DINO export: one row = one image = one point. Position-level detail
    (which grid square, which crop) is discarded here by design -- viewing
    at that granularity was explicitly out of scope for this loader.
    """
    directory = Path(directory)
    plate = fold_name(directory)
    if plate is None:
        raise EmbeddingError(f"{directory} is not a fold_Plate_N directory")

    index_path = directory / INDEX_SUFFIX.format(plate=directory.name)
    array_path = directory / ARRAY_SUFFIX.format(plate=directory.name)
    positions_path = directory / POSITIONS_SUFFIX.format(plate=directory.name)
    summary_path = directory / SUMMARY_SUFFIX.format(plate=directory.name)

    if not index_path.exists():
        raise EmbeddingError(f"no {index_path.name} in {directory}")
    if not array_path.exists():
        raise EmbeddingError(f"no {array_path.name} in {directory}")

    try:
        records = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EmbeddingError(f"could not read {index_path.name}: {exc}") from exc
    if not isinstance(records, list):
        raise EmbeddingError(f"{index_path.name}: expected a JSON list")

    crops = np.load(array_path)
    if crops.ndim != 4:
        raise EmbeddingError(
            f"{array_path.name}: expected a 4-D (images, positions, crops, dim) "
            f"array, got shape {crops.shape}"
        )
    if crops.shape[0] != len(records):
        raise EmbeddingError(
            f"{array_path.name} has {crops.shape[0]} images but "
            f"{index_path.name} has {len(records)} rows. They are matched by "
            f"position, so a mismatch means every point could be mislabelled."
        )

    # Pool position and crop axes together: every one of the 900 vectors
    # describes the same image, so a plain mean over both is the same as
    # pooling positions then crops.
    vectors = crops.reshape(crops.shape[0], -1, crops.shape[-1]).mean(axis=1)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    frame = pd.DataFrame(records)
    for column in (IMAGE_PATH, MODALITY):
        if column not in frame.columns:
            frame[column] = ""
    frame[LABEL] = frame.get("true_label", "")
    frame = frame[[IMAGE_PATH, MODALITY, LABEL]].astype(str).fillna("")
    # PurePosixPath, not Path: these are Linux training-machine paths, and
    # parsing them with Windows' Path would treat the whole string as one
    # filename (no "/" separator recognised) instead of splitting it.
    frame[IMAGE_NAME] = frame[IMAGE_PATH].map(
        lambda p: PurePosixPath(p).stem if p else ""
    )
    frame["experiment"] = frame[MODALITY].map(_MODALITY_ARM).fillna(frame[MODALITY])
    frame[PLATE] = _disk_plate_name(plate)

    if positions_path.exists():
        try:
            positions = pd.read_csv(positions_path)
        except (OSError, pd.errors.ParserError) as exc:
            raise EmbeddingError(f"could not read {positions_path.name}: {exc}") from exc
        missing = {IMAGE_PATH, PREDICTED_LABEL, PROB_TRUE} - set(positions.columns)
        if missing:
            raise EmbeddingError(
                f"{positions_path.name} is missing column(s): {', '.join(sorted(missing))}"
            )
        agg = _aggregate_predictions(positions)
        frame[PREDICTED_LABEL] = [agg[p].predicted_label if p in agg else "" for p in frame[IMAGE_PATH]]
        frame[PROB_TRUE] = [f"{agg[p].prob_true:.4f}" if p in agg else "" for p in frame[IMAGE_PATH]]
        frame[CORRECT] = [agg[p].correct if p in agg else "" for p in frame[IMAGE_PATH]]

    run_info: dict = {}
    if summary_path.exists():
        try:
            run_info = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            run_info = {}
    run_info.setdefault("fold", plate)
    run_info.setdefault("model", f"trained classifier embedding ({vectors.shape[1]}-dim, {plate})")

    return EmbeddingDataset(
        name=name or directory.name,
        directory=directory,
        vectors=vectors,
        frame=frame,
        run_info=run_info,
    )
