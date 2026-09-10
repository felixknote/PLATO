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

from conftest import embedding_root, find_export, image_root  # noqa: E402

needs_share = pytest.mark.skipif(
    embedding_root() is None,
    reason="set PLATO_TEST_EMBEDDING_ROOT to test against real exports",
)


def _crispri_abx_metadata() -> pd.DataFrame:
    """The real CRISPRi+ABx export's metadata, found by content.

    Deliberately not by folder name: these directories get renamed, and a test
    that hard-codes a name fails for a reason that has nothing to do with the
    code under test. The dataset is identified by what it contains -- both
    experiment arms in one export.
    """
    directory = find_export(
        lambda frame: "experiment" in frame.columns
        and {"CRISPRi", "ABx"} <= set(frame["experiment"])
    )
    if directory is None:
        pytest.skip("no CRISPRi+ABx export found")
    return pd.read_csv(directory / "features_metadata.csv", dtype=str).fillna("")


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


# Dec25Apr26 CRISPRi & ABx's own naming convention: underscores throughout,
# with an explicit ATC induction state instead of a bare replicate number.
# Before this was recognised, every one of these 28 real labels fell through
# to parse_condition's last resort and came back as a DRUG -- so a CRISPRi
# gene knockdown was coloured, filtered and grouped as if it were an
# antibiotic.
@pytest.mark.parametrize(
    "label,gene,guide,role,has_control_kind",
    [
        ("ACE-1_NC_1_minusATC", "ACE-1", "NC_1_minusATC", "control", True),
        ("ACE-1_NC_6_plusATC", "ACE-1", "NC_6_plusATC", "control", True),
        ("ACE-1_minusATC", "ACE-1", "minusATC", "treatment", False),
        ("ACE-1_plusATC", "ACE-1", "plusATC", "treatment", False),
        ("MG1655_NC_1_minusATC", "MG1655", "NC_1_minusATC", "control", True),
        ("MG1655_plusATC", "MG1655", "plusATC", "treatment", False),
    ],
)
def test_parse_condition_atc_induction_suffix(label, gene, guide, role, has_control_kind):
    parsed = parse_condition(label)
    assert parsed.gene == gene
    assert parsed.guide == guide
    assert parsed.drug == "", "an ATC-suffixed CRISPRi label must never become a drug"
    assert parsed.role == role
    assert bool(parsed.control_kind) == has_control_kind


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


# Dec25Apr26 CRISPRi & ABx has no dedicated experiment column: only a
# "source" column with THREE values (control/drug/mutant) that do not line
# up with the CRISPRi/ABx arm split -- "control" holds BOTH CRISPRi's
# non-targeting rows AND its induced-gene treatment rows. Falling back to
# that column directly, as the arm, mixed a CRISPRi sub-role into what
# looked like a clean two-way comparison.
def _write_mixed_arm_dataset(directory, *, n=120, dim=16):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    # Half CRISPRi-shaped labels (mix of NC control and treatment guides),
    # half ABx-shaped labels -- no "experiment"/"arm"/"screen" column, only
    # "source", exactly like the real export.
    rows = []
    for i in range(n):
        if i % 2 == 0:
            is_control = i % 4 == 0
            label = "ACE-1_NC_1_minusATC" if is_control else "ftsZ_1"
            source = "control" if is_control else "mutant"
        else:
            label = "Ciprofloxacin 1x"
            source = "drug"
        rows.append(
            {
                "source": source,
                "plate": "P1",
                "well": f"A{i % 12 + 1:02d}",
                "label": label,
                "image_name": f"img_{i:04d}",
            }
        )
    pd.DataFrame(rows).to_csv(directory / "features_metadata.csv", index=False)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    np.savez(
        directory / "features_all.npz",
        embeddings=vectors,
        label_indices=np.arange(n, dtype=np.int32),
    )
    return directory


def test_experiment_arm_is_derived_from_gene_vs_drug_not_a_raw_source_column(tmp_path):
    directory = _write_mixed_arm_dataset(tmp_path / "mixed")
    dataset = load_dataset(directory)
    frame, _ = build_frame(dataset)

    # The real bug: "source" has three values that do not match the arm.
    assert set(dataset.frame["source"]) == {"control", "drug", "mutant"}

    # The derived experiment column must be a clean two-way CRISPRi/ABx
    # split, matching which rows actually parsed to a gene vs a drug.
    assert set(frame["experiment"]) == {"CRISPRi", "ABx"}
    assert (frame.loc[frame["gene"] != "", "experiment"] == "CRISPRi").all()
    assert (frame.loc[frame["drug"] != "", "experiment"] == "ABx").all()

    # And the CRISPRi control (NC) rows must NOT have been folded into a
    # separate arm just because they shared "source=control" with nothing --
    # they are CRISPRi like the rest of that gene's rows.
    nc_rows = frame[frame["control_kind"] != ""]
    assert not nc_rows.empty
    assert (nc_rows["experiment"] == "CRISPRi").all()


def test_explicit_experiment_column_is_never_overridden_by_derivation(dataset_dir):
    """Aug26 CRISPRi & ABx has "ABx"/"CRISPRi" directly; a real column must
    win over anything derived from parsing."""
    dataset = load_dataset(dataset_dir)
    frame, _ = build_frame(dataset)
    # dataset_dir's fixture writes a real "experiment" column (ABx_P1/
    # CRISPRi_P1 plate names split on "_"); confirm it was kept verbatim
    # rather than replaced by a gene/drug-derived guess.
    assert set(frame["experiment"]) == {"ABx", "CRISPRi"}


def test_image_resolver_finds_files_in_a_plate_layout(tmp_path):
    root = tmp_path / "images"
    (root / "ABx_P1").mkdir(parents=True)
    for i in range(0, 120, 2):
        (root / "ABx_P1" / f"img_{i:04d}.tiff").write_bytes(b"x")
    directory = _write_dataset(tmp_path / "ds")
    dataset = load_dataset(directory)
    frame, _ = build_frame(dataset)
    abx = frame[frame["plate"] == "ABx_P1"]

    resolver = ImageResolver.detect(root, abx)
    assert resolver is not None
    assert resolver.path_for(abx.iloc[0]) is not None
    # A row whose file is absent resolves to None rather than a dead path.
    missing = frame[frame["plate"] == "CRISPRi_P1"].iloc[0]
    assert resolver.path_for(missing) is None


@pytest.mark.parametrize(
    "layout",
    [
        "{plate}",                      # CRISPRi_P1/file.tiff
        "{arm}/{tail}",                 # CRISPRi/P1/file.tiff
        "{plate}/images",               # CRISPRi_P1/images/file.tiff
        "",                             # everything in one folder
    ],
)
def test_image_resolver_is_layout_agnostic(tmp_path, layout):
    """Folder arrangement must not matter.

    The same screen is stored differently on every machine -- one folder per
    plate, arms split into subfolders, a flat dump. Matching on file name
    rather than on a guessed path template makes all of them work.
    """
    dataset = load_dataset(_write_dataset(tmp_path / "ds"))
    frame, _ = build_frame(dataset)

    root = tmp_path / "images"
    for _, row in frame.iterrows():
        arm, _, tail = row["plate"].rpartition("_")
        relative = layout.format(plate=row["plate"], arm=arm, tail=tail)
        directory = root / relative if relative else root
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{row['image_name']}.tiff").write_bytes(b"x")

    resolver = ImageResolver.detect(root, frame)
    assert resolver is not None, f"layout {layout!r} not resolved"
    assert all(
        resolver.path_for(row) is not None for _, row in frame.head(6).iterrows()
    )


def test_resolver_disambiguates_repeated_file_names(tmp_path):
    """The same file name under many plate folders must still resolve.

    Every plate of a timepoint screen contains WellA01_...Seq0000.tiff, so the
    name alone is ambiguous and the row's other columns have to decide.
    """
    dataset = load_dataset(_write_dataset(tmp_path / "ds"))
    frame, _ = build_frame(dataset)
    # Give every row the SAME file name, so only the folder distinguishes them.
    frame = frame.copy()
    frame["image_name"] = "WellA01_PointA01_0000"

    root = tmp_path / "images"
    for plate in sorted(set(frame["plate"])):
        (root / plate).mkdir(parents=True)
        (root / plate / "WellA01_PointA01_0000.tiff").write_bytes(b"x")

    resolver = ImageResolver.detect(root, frame)
    assert resolver is not None
    for _, row in frame.head(8).iterrows():
        path = resolver.path_for(row)
        assert path is not None
        # It must pick the folder matching THIS row's plate.
        assert row["plate"] in path.parts


def test_resolver_refuses_a_coincidental_name_match(tmp_path):
    """A name that matches but whose metadata matches nothing is not a hit."""
    dataset = load_dataset(_write_dataset(tmp_path / "ds"))
    frame, _ = build_frame(dataset)
    frame = frame.copy()
    frame["image_name"] = "WellA01_PointA01_0000"

    root = tmp_path / "images"
    for unrelated in ("SomeOtherScreen_P9", "AndAnother_P8"):
        (root / unrelated).mkdir(parents=True)
        (root / unrelated / "WellA01_PointA01_0000.tiff").write_bytes(b"x")

    resolver = ImageResolver.for_root(root, frame)
    # Files are there and the names match, but nothing about the rows does.
    assert resolver.index.n_files == 2
    assert resolver.report is not None and not resolver.report.ok


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


def test_a_cancelled_projection_is_not_cached(tmp_path):
    """Cancel must stop the result from silently reappearing on the next run.

    Neither UMAP nor openTSNE can be interrupted mid-fit, so the computation
    still runs to completion -- but it must not look like it succeeded.
    Caching it anyway would mean the very next request for the same
    parameters instantly returns the "cancelled" result, which is what
    actually broke here before this was fixed.
    """
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(150, 12)).astype(np.float32)
    cache = ProjectionCache(tmp_path / "proj")
    params = ProjectionParams(method=UMAP, n_neighbors=10, pca_components=6)

    result = project(
        vectors, params, fingerprint="abc", cache=cache, is_cancelled=lambda: True
    )
    assert result.coords.shape == (150, 2), "the fit still runs to completion"

    # Nothing was written: a fresh request recomputes rather than finding a
    # cache entry the cancelled run should not have been able to leave.
    again = project(vectors, params, fingerprint="abc", cache=cache)
    assert not again.from_cache


def test_an_uncancelled_projection_still_caches(tmp_path):
    """The cancellation hook must not break the ordinary case."""
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(150, 12)).astype(np.float32)
    cache = ProjectionCache(tmp_path / "proj")
    params = ProjectionParams(method=UMAP, n_neighbors=10, pca_components=6)

    project(vectors, params, fingerprint="abc", cache=cache, is_cancelled=lambda: False)
    again = project(vectors, params, fingerprint="abc", cache=cache)
    assert again.from_cache


def test_clear_removes_every_cached_projection(tmp_path):
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(150, 12)).astype(np.float32)
    cache = ProjectionCache(tmp_path / "proj")
    params_a = ProjectionParams(method=UMAP, n_neighbors=10, pca_components=6)
    params_b = ProjectionParams(method=UMAP, n_neighbors=20, pca_components=6)

    project(vectors, params_a, fingerprint="abc", cache=cache)
    project(vectors, params_b, fingerprint="abc", cache=cache)
    assert len(cache.entries()) == 2
    assert any(cache.directory.glob("*.npz"))

    removed = cache.clear()

    assert removed == 2
    assert cache.entries() == []
    assert not any(cache.directory.glob("*.npz"))
    assert not any(cache.directory.glob("*.json"))

    # And it must genuinely force a recompute, not just look empty.
    again = project(vectors, params_a, fingerprint="abc", cache=cache)
    assert not again.from_cache


def test_clear_on_an_empty_or_missing_cache_is_a_harmless_noop(tmp_path):
    cache = ProjectionCache(tmp_path / "never_created")
    assert cache.clear() == 0

    cache.directory.mkdir(parents=True)
    assert cache.clear() == 0


def test_clear_counts_an_orphaned_npz_without_its_json(tmp_path):
    """An interrupted save (array written, metadata write never landed)
    must still be swept up -- it is disk space this exists to reclaim, and
    would otherwise never be counted or removed by anything else."""
    cache = ProjectionCache(tmp_path / "proj")
    cache.directory.mkdir(parents=True)
    (cache.directory / "orphan.npz").write_bytes(b"not a real npz, just bytes")

    removed = cache.clear()

    assert removed == 1
    assert not any(cache.directory.glob("*.npz"))


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
    frame = _crispri_abx_metadata()
    unparsed = [
        label
        for label in frame["label"].unique()
        if not (parse_condition(label).gene or parse_condition(label).drug)
    ]
    assert unparsed == []


@needs_share
def test_real_images_resolve():
    """The real export's rows resolve to real files under the real screen."""
    root = image_root()
    if root is None:
        pytest.skip("set PLATO_TEST_IMAGE_ROOT to test against real images")
    frame = _crispri_abx_metadata()

    # The screen may be the configured root itself or one of its subfolders,
    # depending on how this machine stores it.
    candidates = [root] + [d for d in sorted(root.iterdir()) if d.is_dir()]
    resolver = next(
        (r for r in (ImageResolver.detect(c, frame) for c in candidates) if r), None
    )
    if resolver is None:
        pytest.skip("the configured image root does not hold this export's screen")
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


# -- progress reporting -----------------------------------------------------


def test_umap_progress_parser_is_monotonic():
    """Phases and epochs both advance the bar, and it never goes backwards."""
    from plato.data.progress_taps import UmapProgressParser

    seen = []
    parser = UmapProgressParser(lambda f, m: seen.append((f, m)))
    parser.feed("Wed Sep 9 2026 Finding Nearest Neighbors\n")
    parser.feed("Epochs completed:  50%| ##  250/500 [00:01]\n")
    # tqdm rewrites its line; an older value must not rewind the bar.
    parser.feed("Epochs completed:  10%| #    50/500 [00:00]\n")
    parser.feed("Wed Sep 9 2026 Finished embedding\n")

    fractions = [f for f, _ in seen]
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0
    assert all(0.0 <= f <= 1.0 for f in fractions)


def test_umap_progress_parser_ignores_noise():
    """An unrecognised line advances nothing rather than raising."""
    from plato.data.progress_taps import UmapProgressParser

    seen = []
    parser = UmapProgressParser(lambda f, m: seen.append(f))
    parser.feed("something entirely unexpected\n\n")
    assert seen == []


def test_tsne_callback_survives_the_second_pass():
    """openTSNE restarts its counter for the main pass; the bar must not."""
    from plato.data.progress_taps import tsne_callback

    seen = []
    callback = tsne_callback(lambda f, m: seen.append(f), 250, offset=0.35, span=0.65)
    for iteration in (50, 150, 250):  # early exaggeration
        callback(iteration, 1.0, None)
    for iteration in (50, 150, 250):  # main pass, counter restarts
        callback(iteration, 1.0, None)

    assert seen == sorted(seen), "progress went backwards between passes"
    assert seen[-1] <= 1.0


def test_projection_reports_progress():
    """A real fit drives the fraction callback from start to finish."""
    seen = []
    project(
        np.random.default_rng(0).normal(size=(200, 12)).astype(np.float32),
        ProjectionParams(method=UMAP, n_neighbors=10, pca_components=6),
        on_progress=lambda f, m: seen.append(f),
    )
    assert seen, "no progress was reported"
    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(1.0)


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


# -- progress reporting -----------------------------------------------------


def test_umap_progress_parser_is_monotonic():
    """Phases and epochs both advance the bar, and it never goes backwards."""
    from plato.data.progress_taps import UmapProgressParser

    seen = []
    parser = UmapProgressParser(lambda f, m: seen.append((f, m)))
    parser.feed("Wed Sep 9 2026 Finding Nearest Neighbors\n")
    parser.feed("Epochs completed:  50%| ##  250/500 [00:01]\n")
    # tqdm rewrites its line; an older value must not rewind the bar.
    parser.feed("Epochs completed:  10%| #    50/500 [00:00]\n")
    parser.feed("Wed Sep 9 2026 Finished embedding\n")

    fractions = [f for f, _ in seen]
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0
    assert all(0.0 <= f <= 1.0 for f in fractions)


def test_umap_progress_parser_ignores_noise():
    """An unrecognised line advances nothing rather than raising."""
    from plato.data.progress_taps import UmapProgressParser

    seen = []
    parser = UmapProgressParser(lambda f, m: seen.append(f))
    parser.feed("something entirely unexpected\n\n")
    assert seen == []


def test_tsne_callback_survives_the_second_pass():
    """openTSNE restarts its counter for the main pass; the bar must not."""
    from plato.data.progress_taps import tsne_callback

    seen = []
    callback = tsne_callback(lambda f, m: seen.append(f), 250, offset=0.35, span=0.65)
    for iteration in (50, 150, 250):  # early exaggeration
        callback(iteration, 1.0, None)
    for iteration in (50, 150, 250):  # main pass, counter restarts
        callback(iteration, 1.0, None)

    assert seen == sorted(seen), "progress went backwards between passes"
    assert seen[-1] <= 1.0


def test_projection_reports_progress():
    """A real fit drives the fraction callback from start to finish."""
    seen = []
    project(
        np.random.default_rng(0).normal(size=(200, 12)).astype(np.float32),
        ProjectionParams(method=UMAP, n_neighbors=10, pca_components=6),
        on_progress=lambda f, m: seen.append(f),
    )
    assert seen, "no progress was reported"
    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(1.0)


# -- suggested parameters ---------------------------------------------------


def test_suggested_neighbours_scale_with_the_dataset():
    """Sublinear (0.4*sqrt(n)), clamped to [15, 100] -- not the old linear
    1.5%-of-n-clamped-to-500. With ~50-100 true conditions in this project's
    30k-50k point exports, a condition is roughly a few hundred points; a
    neighbourhood of 500 already spans several conditions and starts
    averaging distinct phenotypes together before the projection sees them.

    0.4*sqrt(n) is calibrated to land near 30 at 5,000 points, which a
    silhouette-score sweep against a real 32,256-point DINO export (54
    ground-truth perturbations) found to be a genuine interior peak, not
    merely smaller-is-better down to a floor. See suggest()'s docstring.
    """
    from plato.data.projection import suggest

    assert suggest(100_000).n_neighbors == 100  # only the ceiling saturates now
    assert suggest(36_288).n_neighbors < 100
    assert suggest(6_000).n_neighbors < 100
    # Two exports of comparable size must not collapse to one identical,
    # coincidentally-clamped value the way the old linear-to-500 scaling did.
    assert suggest(6_000).n_neighbors != suggest(20_000).n_neighbors
    small = suggest(300, learned=False)
    assert small.n_neighbors == 15
    # Never at or above the sample: every point being everyone's neighbour
    # erases the local structure a projection exists to show.
    for n in (20, 50, 300, 5_000):
        assert suggest(n).n_neighbors < n


def test_suggested_geometry_follows_the_vector_kind():
    """Learned embeddings want cosine after an L2 normalise; hand-computed
    descriptors are already standardised and live in a Euclidean space."""
    from plato.data.projection import suggest

    learned = suggest(20_000, learned=True)
    assert learned.metric == "cosine"
    assert learned.normalize
    assert learned.pca_components > 0

    computed = suggest(2_000, learned=False)
    assert computed.metric == "euclidean"
    assert not computed.normalize
    # Reducing 31 descriptors to 50 components does nothing but cost a fit.
    assert computed.pca_components == 0


def test_suggested_perplexity_stays_usable():
    """Flat 4, not scaled with n. Contrary to general t-SNE literature
    guidance (which would suggest perplexity in the hundreds for tens of
    thousands of points), a direct silhouette-score sweep against a real
    32,256-point DINO export (54 ground-truth perturbations) found
    perplexity 3-5 clearly best at BOTH a 5,000-point subsample and the
    full 32k-point scale -- the low-perplexity result was not a subsampling
    artefact. See suggest()'s docstring for the full measurement."""
    from plato.data.projection import suggest

    for n in (20, 50, 500, 5_000, 50_000):
        perplexity = suggest(n).perplexity
        assert 2 <= perplexity <= 4
        # openTSNE needs perplexity < n/3, which the runner also enforces --
        # but a SUGGESTED value should not already need that clamp itself.
        assert perplexity < max(5, n / 3)

    assert suggest(50_000).perplexity == 4
    # Only a genuinely tiny dataset pulls it below 4.
    assert suggest(10).perplexity < 4
    assert suggest(500).perplexity < 200


def test_subsampling_only_kicks_in_when_it_has_to():
    from plato.data.projection import suggest

    assert suggest(30_000).max_points is None
    assert suggest(200_000).max_points == 20_000


def test_deterministic_flag_changes_the_cache_key():
    """A seeded run and a threaded run are different results, not one."""
    from plato.data.projection import ProjectionParams

    assert (
        ProjectionParams(deterministic=True).key()
        != ProjectionParams(deterministic=False).key()
    )


def test_suggested_params_favour_speed():
    """UMAP is single-threaded once seeded, so exploration defaults to fast."""
    from plato.data.projection import suggest

    assert suggest(20_000).deterministic is False


def test_deterministic_runs_repeat_exactly():
    vectors = np.random.default_rng(0).normal(size=(150, 12)).astype(np.float32)
    params = ProjectionParams(n_neighbors=10, pca_components=6, deterministic=True)
    first = project(vectors, params)
    second = project(vectors, params)
    np.testing.assert_allclose(first.coords, second.coords)


def test_parallel_runs_preserve_cluster_structure():
    """Threading costs the coordinate frame, not the clusters.

    An unseeded UMAP may rotate or mirror the layout between runs, so the
    coordinates differ; what must not differ is which points group together,
    since that is what any conclusion rests on.
    """
    pytest.importorskip("sklearn")
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(0)
    centres = rng.normal(size=(4, 20)) * 6
    labels = rng.integers(0, 4, size=600)
    vectors = (centres[labels] + rng.normal(size=(600, 20))).astype(np.float32)

    params = ProjectionParams(n_neighbors=25, pca_components=10, deterministic=False)
    first = project(vectors, params).coords
    second = project(vectors, params).coords

    grouping = lambda coords: KMeans(4, n_init=10, random_state=0).fit_predict(coords)
    assert adjusted_rand_score(grouping(first), grouping(second)) > 0.9


# -- subsampling as a share -------------------------------------------------


def test_subsample_percentages_scale_with_the_dataset():
    """A percentage means the same thing to a 300-point set and a 36k one,
    which an absolute count does not."""
    from plato.views.explorer import _cap_for

    assert _cap_for("100% (all)", 24_192) is None
    assert _cap_for("50%", 24_192) == 12_096
    assert _cap_for("10%", 24_192) == 2_419
    assert _cap_for("50%", 300) == 150


def test_subsample_never_returns_a_useless_handful():
    """1% of a small set must still leave enough points to lay out."""
    from plato.views.explorer import _cap_for

    assert _cap_for("1%", 300) >= 10
    assert _cap_for("1%", 50) >= 10
    # And never more points than exist.
    assert _cap_for("50%", 4) <= 4


def test_subsample_choice_round_trips_a_suggested_cap():
    from plato.views.explorer import FULL_SAMPLE, _choice_for

    assert _choice_for(None, 36_288) == FULL_SAMPLE
    # A cap at or above the dataset size is not a subsample.
    assert _choice_for(50_000, 36_288) == FULL_SAMPLE
    assert _choice_for(20_000, 200_000) == "10%"


def test_subsample_ignores_unparseable_text():
    """The combo is editable elsewhere; nonsense must not crash a run."""
    from plato.views.explorer import _cap_for, _percent_of

    assert _percent_of("nonsense") is None
    assert _cap_for("nonsense", 1_000) is None


def test_normalised_data_searches_with_euclidean():
    """Cosine on unit-norm rows is redundant and 2.5x slower.

    L2-normalising puts every row on the unit sphere, where angle and
    Euclidean distance are monotonically related, so the neighbours are the
    same either way -- but UMAP has a fast path for Euclidean and none for
    cosine on near-identical norms. Measured 20.3s vs 8.2s at 3k points, with
    the resulting layouts agreeing at ARI 1.000.
    """
    from plato.data.projection import ProjectionParams, effective_metric

    assert effective_metric(ProjectionParams(metric="cosine", normalize=True)) == "euclidean"
    # Without normalisation, cosine still means something and is kept.
    assert effective_metric(ProjectionParams(metric="cosine", normalize=False)) == "cosine"
    # Anything else passes through untouched.
    assert effective_metric(ProjectionParams(metric="euclidean", normalize=True)) == "euclidean"
    assert effective_metric(ProjectionParams(metric="manhattan", normalize=True)) == "manhattan"


# -- the worker path the Compute button actually takes -----------------------
#
# Every projection test above calls project() directly, which is why all 158
# of them passed while the explorer was hard-broken: a refactor overwrote
# _ProjectionTask and _WorkerSignals, so pressing Compute raised NameError on
# the GUI thread and the bar sat at "starting…" forever. These cover the wiring
# between the button and the function, not the function.


def test_explorer_defines_every_name_it_uses():
    """No reference in the module resolves to nothing.

    A NameError inside a Qt slot does not reach a test that never invokes the
    slot -- it surfaces as a UI that quietly does nothing. Checking the module
    is closed under its own references catches that class of breakage for the
    whole file at once, however it is introduced.
    """
    import ast
    import builtins
    from pathlib import Path

    source = Path("plato/views/explorer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    defined = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            defined.update(node.names)

    # Module-level names only; attributes (self.foo) are not resolvable here.
    used = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    missing = sorted(name for name in used - defined if not name.startswith("__"))
    assert not missing, f"explorer.py references undefined names: {missing}"


def test_projection_task_reports_progress_and_finishes(tmp_path):
    """The worker the Compute button starts must actually drive the bar.

    Asserts the two things the stuck-at-"starting" bug broke: that the task
    runs at all, and that it emits rising fractions rather than only a final
    result.
    """
    pytest.importorskip("umap")
    from plato.data.projection import ProjectionCache, ProjectionParams
    from plato.views.explorer import _ProjectionTask, _WorkerSignals

    rng = np.random.default_rng(0)
    vectors = np.vstack(
        [rng.normal(6, 1, (60, 8)), rng.normal(-6, 1, (60, 8))]
    ).astype(np.float32)

    # Run the QRunnable's body directly: no event loop, so this stays a unit
    # test, but it is the same code path the thread pool executes.
    seen: list[tuple[float, str]] = []
    finished: list[object] = []
    failed: list[str] = []

    class _Recorder:
        """Stands in for the Qt signals, recording instead of emitting."""

        class _Slot:
            def __init__(self, sink):
                self.emit = sink

        def __init__(self):
            self.progress = self._Slot(lambda _text: None)
            self.advanced = self._Slot(lambda f, m: seen.append((f, m)))
            self.finished = self._Slot(finished.append)
            self.failed = self._Slot(failed.append)

    params = ProjectionParams(n_neighbors=5, pca_components=0)
    task = _ProjectionTask(
        vectors, params, "wiring", ProjectionCache(tmp_path / "proj"), _Recorder()
    )
    task.run()

    assert not failed, failed
    assert len(finished) == 1
    assert finished[0].coords.shape == (120, 2)

    fractions = [fraction for fraction, _ in seen]
    assert fractions, "the progress bar would never leave 'starting…'"
    assert fractions == sorted(fractions), "progress must never go backwards"
    assert max(fractions) > 0.5

    # The signal names the task uses must exist on the real signals object,
    # which the recorder above cannot verify.
    for name in ("progress", "advanced", "finished", "failed"):
        assert hasattr(_WorkerSignals(), name)


# -- export filenames --------------------------------------------------------


def test_export_name_identifies_dataset_method_encoding_and_time():
    from plato.views.explorer import _suggested_export_name

    name = _suggested_export_name(
        dataset="Aug26 CRISPRi & ABx",
        method="tsne",
        column="plate",
        group_column=None,
        fmt="png",
        stamp="20260911_093000",
    )
    assert name == "Aug26_CRISPRi_ABx_tsne_by_plate_20260911_093000.png"


def test_export_name_for_a_facet_grid_names_the_grouping_not_the_colour():
    from plato.views.explorer import _suggested_export_name

    name = _suggested_export_name(
        dataset="Aug26 CRISPRi & ABx",
        method="umap",
        column="plate",
        group_column="experiment_arm",
        fmt="svg",
        stamp="20260911_093000",
    )
    assert name == (
        "Aug26_CRISPRi_ABx_umap_grid_by_experiment_arm"
        "_colour_plate_20260911_093000.svg"
    )


def test_export_name_omits_colour_suffix_when_it_matches_the_grouping():
    from plato.views.explorer import _suggested_export_name

    name = _suggested_export_name(
        dataset="Aug26 CRISPRi & ABx",
        method="umap",
        column="experiment_arm",
        group_column="experiment_arm",
        fmt="png",
        stamp="20260911_093000",
    )
    assert name == "Aug26_CRISPRi_ABx_umap_grid_by_experiment_arm_20260911_093000.png"


def test_export_name_survives_a_missing_dataset_label():
    from plato.views.explorer import _suggested_export_name

    name = _suggested_export_name(
        dataset="",
        method="tsne",
        column=None,
        group_column=None,
        fmt="png",
        stamp="20260911_093000",
    )
    assert name == "tsne_plot_20260911_093000.png"
