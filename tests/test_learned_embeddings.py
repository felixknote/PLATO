"""Loading a trained classifier's Learned_Embeddings fold as an EmbeddingDataset.

Verified against a real fold_Plate_1 directory before these fixtures were
written (see plato-learned-embeddings-roadmap memory): the majority-vote
accuracy this module computes from raw per-position rows matched the run's
own reported image_majority_vote_acc to within 0.03 percentage points, and
the plate-token fix in image_lookup.hint_columns was found because the real
resolver returned zero matches against a real, correct image root until it
was applied.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plato.data import discovery
from plato.data.embeddings import EmbeddingError, discover_datasets, load_dataset
from plato.data.explorer_model import build_frame
from plato.data.learned_embeddings import (
    discover_folds,
    fold_name,
    is_fold_dir,
    load_fold,
)


def _make_fold(
    directory: Path,
    *,
    n_images: int = 6,
    n_positions: int = 4,
    n_crops: int = 2,
    dim: int = 8,
    seed: int = 0,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    plate = directory.name  # "fold_Plate_1"

    labels = ["Meropenem_2x", "Levofloxacin_1x", "rpsL_3", "lpxA_1"]
    modalities = ["drug", "drug", "mutant", "mutant"]
    records = []
    for i in range(n_images):
        label = labels[i % len(labels)]
        modality = modalities[i % len(modalities)]
        records.append(
            {
                "image_path": f"/home/s529821/Data/P1/WellA{i:02d}_PointA{i:02d}_0000_Seq{i:04d}.tiff",
                "modality": modality,
                "true_label": label,
                "true_idx": i % len(labels),
                "num_positions": n_positions,
                "num_crops_per_position": n_crops,
                "proj_dim": dim,
            }
        )
    (directory / f"crop_projections_index_{plate}.json").write_text(json.dumps(records))

    crops = rng.normal(size=(n_images, n_positions, n_crops, dim)).astype(np.float16)
    np.save(directory / f"crop_projections_{plate}.npy", crops)

    # Per-position predictions: mostly correct, with image index 0 made
    # deliberately WRONG on every position so aggregation is exercised for
    # both outcomes rather than only the "always correct" case.
    rows = []
    for i, record in enumerate(records):
        true_label = record["true_label"]
        wrong_label = labels[(labels.index(true_label) + 1) % len(labels)]
        for p in range(n_positions):
            predicted = wrong_label if i == 0 else true_label
            rows.append(
                {
                    "image_path": record["image_path"],
                    "modality": record["modality"],
                    "timepoint": "",
                    "position_idx": p,
                    "pos_x": p % 2,
                    "pos_y": p // 2,
                    "true_label": true_label,
                    "true_idx": record["true_idx"],
                    "predicted_label": predicted,
                    "predicted_idx": labels.index(predicted),
                    "prob_true": 0.9 if predicted == true_label else 0.2,
                    "correct": int(predicted == true_label),
                }
            )
    pd.DataFrame(rows).to_csv(directory / f"test_positions_{plate}.csv", index=False)

    summary = {
        "config": {"num_classes": len(labels), "checkpoint": "checkpoint_epoch.pth"},
        "results": {"image_majority_vote_acc": 100.0 * (n_images - 1) / n_images},
    }
    (directory / f"test_positions_summary_{plate}.json").write_text(json.dumps(summary))


def test_is_fold_dir(tmp_path):
    fold = tmp_path / "fold_Plate_1"
    _make_fold(fold)
    assert is_fold_dir(fold)
    assert not is_fold_dir(tmp_path)  # parent has no index file of its own


def test_fold_name():
    assert fold_name(Path("fold_Plate_1")) == "Plate_1"
    assert fold_name(Path("not_a_fold")) is None


def test_discover_folds_finds_all_siblings(tmp_path):
    run = tmp_path / "Dec25&Apr26 CRISPRi & ABx"
    for n in (1, 2, 3):
        _make_fold(run / f"fold_Plate_{n}")
    (run / "not_a_fold").mkdir(parents=True)

    found = discover_folds(run)

    assert [p.name for p in found] == ["fold_Plate_1", "fold_Plate_2", "fold_Plate_3"]


def test_discover_datasets_also_finds_folds(tmp_path):
    """The generic embeddings.discover_datasets sees folds too, so pointing
    the dataset dropdown at a Learned_Embeddings run directory lists its
    folds the same way it lists DINO exports."""
    run = tmp_path / "run"
    _make_fold(run / "fold_Plate_1")

    found = discover_datasets(run)

    assert [p.name for p in found] == ["fold_Plate_1"]


def test_scan_recognises_folds_as_embeddings(tmp_path):
    run = tmp_path / "screens" / "fold_Plate_1"
    _make_fold(run)

    result = discovery.scan([tmp_path])

    kinds = {f.path.name: f.kind for f in result.found}
    assert kinds["fold_Plate_1"] == discovery.EMBEDDING


def test_load_fold_pools_positions_and_crops(tmp_path):
    fold = tmp_path / "fold_Plate_1"
    _make_fold(fold, n_images=6, n_positions=4, n_crops=2, dim=8)

    dataset = load_fold(fold)

    assert dataset.n_points == 6
    assert dataset.n_dimensions == 8
    assert dataset.name == "fold_Plate_1"


def test_load_fold_pooled_vector_is_the_mean(tmp_path):
    fold = tmp_path / "fold_Plate_1"
    _make_fold(fold, n_images=3, n_positions=4, n_crops=2, dim=5)

    crops = np.load(fold / "crop_projections_fold_Plate_1.npy")
    dataset = load_fold(fold)

    expected = crops.reshape(crops.shape[0], -1, crops.shape[-1]).mean(axis=1)
    np.testing.assert_allclose(dataset.vectors, expected, rtol=1e-2)


def test_load_fold_via_generic_load_dataset(tmp_path):
    """load_dataset dispatches to load_fold without the caller knowing --
    this is the exact path plato.views.explorer._load_embedding_directory
    uses for whatever directory the user picked."""
    fold = tmp_path / "fold_Plate_1"
    _make_fold(fold)

    dataset = load_dataset(fold)

    assert dataset.n_points == 6


def test_load_fold_missing_array_raises(tmp_path):
    fold = tmp_path / "fold_Plate_1"
    _make_fold(fold)
    (fold / "crop_projections_fold_Plate_1.npy").unlink()

    with pytest.raises(EmbeddingError):
        load_fold(fold)


def test_load_fold_row_mismatch_raises(tmp_path):
    fold = tmp_path / "fold_Plate_1"
    _make_fold(fold, n_images=6)
    # Truncate the index to create a length mismatch against the array.
    index_path = fold / "crop_projections_index_fold_Plate_1.json"
    records = json.loads(index_path.read_text())
    index_path.write_text(json.dumps(records[:3]))

    with pytest.raises(EmbeddingError):
        load_fold(fold)


def test_load_fold_aggregates_predictions_per_image(tmp_path):
    fold = tmp_path / "fold_Plate_1"
    _make_fold(fold, n_images=6, n_positions=4)

    dataset = load_fold(fold)

    assert "predicted_label" in dataset.frame.columns
    assert "prob_true" in dataset.frame.columns
    assert "correct" in dataset.frame.columns
    # Image 0 was made wrong on every position; every other image was made
    # correct on every position -- so majority vote must agree.
    assert dataset.frame.loc[0, "correct"] == "0"
    assert (dataset.frame.loc[1:, "correct"] == "1").all()


def test_load_fold_sets_plate_and_experiment_from_modality(tmp_path):
    fold = tmp_path / "fold_Plate_1"
    _make_fold(fold, n_images=4)

    dataset = load_fold(fold)

    assert (dataset.frame["plate"] == "P1").all()
    assert set(dataset.frame["experiment"]) == {"ABx", "CRISPRi"}


def test_build_frame_carries_prediction_columns_through(tmp_path):
    fold = tmp_path / "fold_Plate_1"
    _make_fold(fold, n_images=4)
    dataset = load_fold(fold)

    frame, _ = build_frame(dataset)

    assert "predicted_label" in frame.columns
    assert "prob_true" in frame.columns
    assert "correct" in frame.columns
    # Conditions parse: drug rows get a drug+concentration, mutant rows a
    # gene+guide, exactly like any other export's labels.
    assert (frame["drug"] != "").any()
    assert (frame["gene"] != "").any()


def test_build_frame_omits_prediction_columns_for_ordinary_datasets(tmp_path):
    """A normal DINO export never had predictions made against it -- the
    columns must not appear at all, which is what keeps them off the
    colour-by dropdown for every dataset that isn't a fold."""
    from plato.data.embeddings import EmbeddingDataset

    frame_in = pd.DataFrame({"label": ["ftsZ_1", "ftsZ_2"], "plate": ["P1", "P1"]})
    dataset = EmbeddingDataset(
        name="ordinary",
        directory=tmp_path,
        vectors=np.zeros((2, 4), dtype=np.float32),
        frame=frame_in,
        run_info={},
    )

    frame, _ = build_frame(dataset)

    assert "predicted_label" not in frame.columns
    assert "prob_true" not in frame.columns
    assert "correct" not in frame.columns


def test_run_info_carries_summary_and_fold_name(tmp_path):
    fold = tmp_path / "fold_Plate_1"
    _make_fold(fold, n_images=4)

    dataset = load_fold(fold)

    assert dataset.run_info["fold"] == "Plate_1"
    assert "results" in dataset.run_info
    assert dataset.run_info["results"]["image_majority_vote_acc"] == pytest.approx(75.0)
