"""Embedding Explorer: data layer and offscreen widget tests.

The data-layer tests run everywhere on synthetic exports. The tests that need
the real DINO features skip themselves when Z: is not mounted, so the suite
still passes on a machine without the share.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plato.data.annotations import (
    UNANNOTATED,
    AnnotationTable,
    parse_condition,
)
from plato.data.embeddings import (
    EmbeddingError,
    discover_datasets,
    is_dataset_dir,
    load_dataset,
)
from plato.data.explorer_model import (
    ImageResolver,
    build_frame,
    colour_fields,
    distinct_values,
    filter_fields,
)
from plato.data.projection import (
    TSNE,
    UMAP,
    ProjectionCache,
    ProjectionParams,
    project,
)

DINO_ROOT = Path(r"Z:\Analysis\DINO")
IMAGE_ROOT = Path(r"Z:\Data\FK_P001_EX0039_2026_08_28_CRISPRI & ABx Experiment")

needs_share = pytest.mark.skipif(
    not DINO_ROOT.is_dir(), reason="Z: embedding share not mounted"
)


# -- fixtures ---------------------------------------------------------------


def _write_dataset(directory: Path, *, n: int = 120, dim: int = 16, with_features=True):
    """A minimal but structurally faithful embedding export."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    plates = ["ABx_P1", "CRISPRi_P1"]
    labels = ["Ciprofloxacin 1x", "Ciprofloxacin 2x", "ftsZ_1", "gyrA_2", "WT NC"]
    rows = []
    for i in range(n):
        plate = plates[i % 2]
        label = labels[i % len(labels)]
        rows.append(
            {
                "experiment": plate.split("_")[0],
                "plate": plate,
                "well": f"A{i % 12 + 1:02d}",
                "label": label,
                "image_name": f"img_{i:04d}",
            }
        )
    pd.DataFrame(rows).to_csv(directory / "features_metadata.csv", index=False)
    if with_features:
        np.savez_compressed(
            directory / "features_all.npz",
            embeddings=rng.normal(size=(n, dim)).astype(np.float32),
            label_indices=np.arange(n, dtype=np.int32),
        )
    (directory / "metadata.json").write_text(json.dumps({"model": "test"}))
    return directory


@pytest.fixture()
def dataset_dir(tmp_path):
    return _write_dataset(tmp_path / "demo")


# -- condition parsing ------------------------------------------------------


@pytest.mark.parametrize(
    "label,gene,guide,drug,dose,role",
    [
        ("ftsZ_2", "ftsZ", "2", "", "", "treatment"),
        ("ACE-1 NC_6", "ACE-1 NC", "6", "", "", "control"),
        ("MG1655 NC_1", "MG1655 NC", "1", "", "", "control"),
        ("Ciprofloxacin 1x", "", "", "Ciprofloxacin", "1x", "treatment"),
        ("Penicillin G 0.25x", "", "", "Penicillin G", "0.25x", "treatment"),
        ("Polymyxin B 2x", "", "", "Polymyxin B", "2x", "treatment"),
        ("WT", "", "", "WT", "", "control"),
        ("WT NC", "", "", "WT NC", "", "control"),
    ],
)
def test_parse_condition(label, gene, guide, drug, dose, role):
    parsed = parse_condition(label)
    assert (parsed.gene, parsed.guide) == (gene, guide)
    assert (parsed.drug, parsed.concentration) == (drug, dose)
    assert parsed.role == role


def test_parse_condition_handles_blank_and_unknown():
    assert parse_condition("").label == ""
    # An unrecognised label is still a condition and must survive.
    odd = parse_condition("some new thing")
    assert odd.label == "some new thing"
    assert odd.perturbation == "some new thing"


# -- annotation tables ------------------------------------------------------


def test_annotation_table_from_inverted_json(tmp_path):
    path = tmp_path / "moa.json"
    path.write_text(json.dumps({"Gyrase": ["Ciprofloxacin", "Norfloxacin"]}))
    table = AnnotationTable.load(path)
    assert table.get("Ciprofloxacin") == "Gyrase"
    assert table.get("Nonesuch") == UNANNOTATED


def test_annotation_table_from_dose_keyed_json(tmp_path):
    path = tmp_path / "moa.json"
    path.write_text(json.dumps({"Ciprofloxacin_1xIC50": "Gyrase"}))
    table = AnnotationTable.load(path)
    # The dose suffix must be stripped, so every dose shares one annotation.
    assert table.get("Ciprofloxacin") == "Gyrase"


def test_annotation_lookup_ignores_spacing_and_case(tmp_path):
    path = tmp_path / "moa.json"
    path.write_text(json.dumps({"Cell wall (PBP 1)": ["PenicillinG"]}))
    table = AnnotationTable.load(path)
    # Plate maps write "Penicillin G"; the MoA table writes "PenicillinG".
    assert table.get("Penicillin G") == "Cell wall (PBP 1)"


def test_annotation_csv_ignores_comments(tmp_path):
    path = tmp_path / "genes.csv"
    path.write_text("gene,pathway\n# a comment\nftsZ,Cell division\nsecA,\n")
    table = AnnotationTable.load(path)
    assert table.get("ftsZ") == "Cell division"
    # A blank annotation is not an annotation.
    assert table.get("secA") == UNANNOTATED


def test_shipped_pathway_template_is_blank():
    """The shipped template must stay empty: PLATO must never invent biology."""
    template = Path(__file__).resolve().parents[1] / "annotations" / "gene_pathway.csv"
    if not template.exists():
        pytest.skip("template not present")
    assert len(AnnotationTable.load(template)) == 0


# -- embedding loading ------------------------------------------------------


def test_load_dataset(dataset_dir):
    dataset = load_dataset(dataset_dir)
    assert dataset.n_points == 120
    assert dataset.n_dimensions == 16
    assert dataset.fingerprint()


def test_missing_features_names_the_path(tmp_path):
    directory = _write_dataset(tmp_path / "nofeat", with_features=False)
    assert is_dataset_dir(directory)
    with pytest.raises(EmbeddingError) as excinfo:
        load_dataset(directory)
    # The message has to say where to put the file, not just that it failed.
    assert "features_all.npz" in str(excinfo.value)


def test_row_count_mismatch_is_fatal(tmp_path):
    directory = _write_dataset(tmp_path / "bad")
    np.savez_compressed(
        directory / "features_all.npz",
        embeddings=np.zeros((7, 16), dtype=np.float32),
    )
    with pytest.raises(EmbeddingError, match="rows"):
        load_dataset(directory)


def test_discover_datasets(tmp_path):
    _write_dataset(tmp_path / "a")
    _write_dataset(tmp_path / "b")
    (tmp_path / "not_a_dataset").mkdir()
    assert len(discover_datasets(tmp_path)) == 2


# -- the joined frame -------------------------------------------------------


def test_build_frame_resolves_fields(dataset_dir):
    dataset = load_dataset(dataset_dir)
    moa = AnnotationTable({"Ciprofloxacin": "Gyrase"})
    frame, resolver = build_frame(dataset, moa_table=moa)
    assert len(frame) == dataset.n_points
    assert set(frame["role"]) == {"treatment", "control"}
    assert frame.loc[frame["drug"] == "Ciprofloxacin", "moa"].eq("Gyrase").all()
    # Genes get no MoA -- that column annotates drugs.
    assert frame.loc[frame["gene"] == "ftsZ", "moa"].eq(UNANNOTATED).all()
    assert resolver is None  # no image root supplied

    assert "gene" in colour_fields(frame)
    assert "drug" in filter_fields(frame)
    assert distinct_values(frame, "concentration") == ["1x", "2x"]


def test_image_resolver_detects_layout(tmp_path):
    root = tmp_path / "images"
    (root / "ABx_P1").mkdir(parents=True)
    for i in range(0, 120, 2):
        (root / "ABx_P1" / f"img_{i:04d}.tiff").write_bytes(b"x")
    directory = _write_dataset(tmp_path / "ds")
    dataset = load_dataset(directory)
    frame, _ = build_frame(dataset)
    resolver = ImageResolver.detect(root, frame[frame["plate"] == "ABx_P1"])
    assert resolver is not None
    assert resolver.template == "{plate}/{name}"
    row = frame[frame["plate"] == "ABx_P1"].iloc[0]
    assert resolver.path_for(row) is not None
    # A row whose file is absent resolves to None rather than a dead path.
    missing = frame[frame["plate"] == "CRISPRi_P1"].iloc[0]
    assert resolver.path_for(missing) is None


# -- projection -------------------------------------------------------------


def test_projection_caches(tmp_path):
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(150, 12)).astype(np.float32)
    cache = ProjectionCache(tmp_path / "proj")
    params = ProjectionParams(method=UMAP, n_neighbors=10, pca_components=6)

    first = project(vectors, params, fingerprint="abc", cache=cache)
    assert first.coords.shape == (150, 2)
    assert not first.from_cache

    second = project(vectors, params, fingerprint="abc", cache=cache)
    assert second.from_cache
    np.testing.assert_allclose(first.coords, second.coords)


def test_projection_subsamples(tmp_path):
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(300, 12)).astype(np.float32)
    params = ProjectionParams(method=UMAP, n_neighbors=10, max_points=100, pca_components=6)
    result = project(vectors, params)
    assert result.coords.shape == (100, 2)
    assert len(result.row_indices) == 100
    # row_indices must index the ORIGINAL rows, or hover shows the wrong image.
    assert result.row_indices.max() < 300


def test_tsne_runs(tmp_path):
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(120, 12)).astype(np.float32)
    result = project(vectors, ProjectionParams(method=TSNE, perplexity=10, pca_components=6))
    assert result.coords.shape == (120, 2)


def test_params_key_separates_methods():
    umap_key = ProjectionParams(method=UMAP).key()
    tsne_key = ProjectionParams(method=TSNE).key()
    assert umap_key != tsne_key
    # Changing a parameter must change the cache key, or a stale projection
    # would be served for new settings.
    assert ProjectionParams(n_neighbors=15).key() != ProjectionParams(n_neighbors=50).key()


# -- palette ----------------------------------------------------------------


def test_palette_is_stable_and_marks_unknown():
    from plato.views.palette import UNKNOWN_COLOUR, colours_for

    mapping, continuous = colours_for(["ftsZ", "gyrA", UNANNOTATED])
    assert not continuous
    assert mapping[UNANNOTATED] == UNKNOWN_COLOUR
    assert mapping["ftsZ"] != mapping["gyrA"]
    # Same input, same colours, every time.
    assert colours_for(["ftsZ", "gyrA", UNANNOTATED])[0] == mapping


def test_palette_detects_dose_series():
    from plato.views.palette import colours_for, sort_values

    values = ["0.25x", "0.5x", "1x", "2x"]
    mapping, continuous = colours_for(values)
    assert continuous
    assert len(set(mapping.values())) == 4
    assert sort_values(["1x", "0.25x", "2x"]) == ["0.25x", "1x", "2x"]


# -- real data --------------------------------------------------------------


@needs_share
def test_real_metadata_parses_completely():
    """Every condition in the real screen must parse into gene or drug."""
    frame = pd.read_csv(DINO_ROOT / "new 8 plates" / "features_metadata.csv", dtype=str)
    unparsed = [
        label
        for label in frame["label"].unique()
        if not (parse_condition(label).gene or parse_condition(label).drug)
    ]
    assert unparsed == []


@needs_share
@pytest.mark.skipif(not IMAGE_ROOT.is_dir(), reason="image share not mounted")
def test_real_images_resolve():
    frame = pd.read_csv(DINO_ROOT / "new 8 plates" / "features_metadata.csv", dtype=str)
    resolver = ImageResolver.detect(IMAGE_ROOT, frame)
    assert resolver is not None
    sample = frame.sample(n=8, random_state=0)
    assert all(resolver.path_for(row) is not None for _, row in sample.iterrows())


# -- image resolution regressions -------------------------------------------


def test_resolver_prefers_a_root_that_actually_resolves(tmp_path):
    """Resolution must be proven against the frame, not guessed from paths.

    The first version probed before the frame existed and only looked at
    loaded plates' directories, so the explorer silently had no previews.
    """
    # A decoy root that exists but holds nothing this dataset refers to.
    decoy = tmp_path / "decoy"
    (decoy / "ABx_P1").mkdir(parents=True)
    (decoy / "ABx_P1" / "unrelated.tiff").write_bytes(b"x")

    real = tmp_path / "real"
    (real / "ABx_P1").mkdir(parents=True)
    for i in range(0, 120, 2):
        (real / "ABx_P1" / f"img_{i:04d}.tiff").write_bytes(b"x")

    dataset = load_dataset(_write_dataset(tmp_path / "ds"))
    frame, _ = build_frame(dataset)
    abx = frame[frame["plate"] == "ABx_P1"]

    assert ImageResolver.detect(decoy, abx) is None
    resolver = ImageResolver.detect(real, abx)
    assert resolver is not None
    assert resolver.path_for(abx.iloc[0]) is not None


def test_missing_backend_says_what_to_install(monkeypatch):
    """The explorer's libraries are an optional extra, so a missing one must
    name the install command rather than surfacing "No module named 'umap'"."""
    import builtins

    from plato.data.projection import MissingDependency

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in {"umap", "openTSNE", "sklearn"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(MissingDependency) as excinfo:
        project(
            np.random.default_rng(0).normal(size=(40, 8)).astype(np.float32),
            ProjectionParams(n_neighbors=5, pca_components=4),
        )
    assert "pip install" in str(excinfo.value)
