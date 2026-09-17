"""Layering several MoA sources, so a local correction never hides the rest.

A single "first file found wins" lookup (the original find_default_moa) was
right while there was one table to consult. It stopped being right the
moment a PLATO-local drug_moa.csv was added for drugs missing from the lab's
own AI4AB table: if drug_moa.csv existed at all, EVERY drug not in it would
have read as unannotated, because the lookup never got as far as looking at
AI4AB's table. load_default_moa exists to combine them instead -- a drug
named in more than one source keeps the higher-priority source's value, and
every other drug still resolves from whichever lower-priority source has it.
"""

from __future__ import annotations

import json

import pytest

from plato.data.annotations import (
    UNANNOTATED,
    AnnotationTable,
    load_default_moa,
    load_or_empty,
)


# -- merged_with -----------------------------------------------------------------


def test_merging_keeps_the_calling_tables_own_entries():
    local = AnnotationTable({"Amikacin": "Ribosome"})
    lab = AnnotationTable({"Ciprofloxacin": "Gyrase"})
    merged = local.merged_with(lab)
    assert merged.get("Amikacin") == "Ribosome"


def test_merging_fills_gaps_from_the_other_table():
    local = AnnotationTable({"Amikacin": "Ribosome"})
    lab = AnnotationTable({"Ciprofloxacin": "Gyrase"})
    merged = local.merged_with(lab)
    assert merged.get("Ciprofloxacin") == "Gyrase"


def test_the_calling_table_wins_a_genuine_conflict():
    """Priority is about the CALLER, not about which table is bigger.

    A local correction is meant to override the lab table for the one drug
    it names -- that is the entire reason to add it locally rather than
    editing the lab's own file.
    """
    local = AnnotationTable({"Ciprofloxacin": "Corrected class"})
    lab = AnnotationTable({"Ciprofloxacin": "Gyrase"})
    merged = local.merged_with(lab)
    assert merged.get("Ciprofloxacin") == "Corrected class"


def test_merging_with_an_empty_table_changes_nothing():
    local = AnnotationTable({"Amikacin": "Ribosome"})
    merged = local.merged_with(AnnotationTable.empty())
    assert merged.get("Amikacin") == "Ribosome"
    assert len(merged) == 1


def test_an_empty_table_merged_with_a_full_one_gets_everything():
    lab = AnnotationTable({"Ciprofloxacin": "Gyrase", "Rifampicin": "RNA polymerase"})
    merged = AnnotationTable.empty().merged_with(lab)
    assert len(merged) == 2
    assert merged.get("Rifampicin") == "RNA polymerase"


def test_a_three_way_merge_still_respects_priority_order():
    """load_default_moa's own use: several sources, ordered strictly."""
    highest = AnnotationTable({"X": "from highest"})
    middle = AnnotationTable({"X": "from middle", "Y": "from middle"})
    lowest = AnnotationTable({"X": "from lowest", "Y": "from lowest", "Z": "from lowest"})
    merged = highest.merged_with(middle).merged_with(lowest)
    assert merged.get("X") == "from highest"
    assert merged.get("Y") == "from middle"
    assert merged.get("Z") == "from lowest"


# -- load_default_moa -------------------------------------------------------------


@pytest.fixture
def two_source_root(tmp_path):
    """A PLATO-local drug_moa.csv plus a lab-style JSON table, at the exact
    relative paths DEFAULT_MOA_CANDIDATES looks for.

    The lab-table candidate is "../AI4AB/..." relative to the search root --
    a SIBLING of it -- so the search root has to be a subdirectory of
    tmp_path, not tmp_path itself. Writing AI4AB into tmp_path.parent would
    leak it across every test whose tmp_path shares that same numbered
    pytest parent.
    """
    plato_root = tmp_path / "PLATO"
    annotations_dir = plato_root / "annotations"
    annotations_dir.mkdir(parents=True)
    (annotations_dir / "drug_moa.csv").write_text(
        "drug,moa\nAmikacin,Ribosome\nAmpicillin,Cell wall (PBP 1)\n"
    )
    ai4ab_dir = tmp_path / "AI4AB" / "analysis" / "E_coli_params"
    ai4ab_dir.mkdir(parents=True, exist_ok=True)
    (ai4ab_dir / "moa_dict_inv.json").write_text(
        json.dumps(
            {
                "Gyrase": ["Ciprofloxacin"],
                "Cell wall (PBP 1)": ["PenicillinG"],
            }
        )
    )
    return plato_root


def test_layering_covers_drugs_from_both_sources(two_source_root):
    table = load_default_moa([two_source_root])
    assert table.get("Amikacin") == "Ribosome"
    assert table.get("Ciprofloxacin") == "Gyrase"


def test_a_drug_named_in_the_local_file_is_not_shadowed_by_the_lab_table(
    two_source_root,
):
    """The bug a first-match lookup would have: drug_moa.csv existing at all
    would previously have meant the lab's moa_dict_inv.json was never even
    opened, so Ciprofloxacin -- not in the local file -- would have read as
    unannotated instead of falling through to the lab's own entry."""
    table = load_default_moa([two_source_root])
    assert table.get("Ciprofloxacin") != UNANNOTATED
    assert table.get("Ciprofloxacin") == "Gyrase"


def test_the_local_file_wins_when_a_drug_is_in_both(tmp_path):
    plato_root = tmp_path / "PLATO"
    annotations_dir = plato_root / "annotations"
    annotations_dir.mkdir(parents=True)
    (annotations_dir / "drug_moa.csv").write_text(
        "drug,moa\nCiprofloxacin,Corrected class\n"
    )
    ai4ab_dir = tmp_path / "AI4AB" / "analysis" / "E_coli_params"
    ai4ab_dir.mkdir(parents=True, exist_ok=True)
    (ai4ab_dir / "moa_dict_inv.json").write_text(
        json.dumps({"Gyrase": ["Ciprofloxacin"]})
    )
    table = load_default_moa([plato_root])
    assert table.get("Ciprofloxacin") == "Corrected class"


def test_no_sources_found_returns_an_empty_table_not_an_error(tmp_path):
    table = load_default_moa([tmp_path])
    assert len(table) == 0
    assert table.get("Anything") == UNANNOTATED


def test_a_malformed_local_file_does_not_lose_the_lab_table(tmp_path):
    """A broken local override must not take the good source down with it."""
    plato_root = tmp_path / "PLATO"
    annotations_dir = plato_root / "annotations"
    annotations_dir.mkdir(parents=True)
    (annotations_dir / "drug_moa.csv").write_text("not,even\na,real,csv,shape\n\x00")
    ai4ab_dir = tmp_path / "AI4AB" / "analysis" / "E_coli_params"
    ai4ab_dir.mkdir(parents=True, exist_ok=True)
    (ai4ab_dir / "moa_dict_inv.json").write_text(
        json.dumps({"Gyrase": ["Ciprofloxacin"]})
    )
    table = load_default_moa([plato_root])
    assert table.get("Ciprofloxacin") == "Gyrase"


def test_only_the_local_file_present_still_works(tmp_path):
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()
    (annotations_dir / "drug_moa.csv").write_text("drug,moa\nAmikacin,Ribosome\n")
    table = load_default_moa([tmp_path])
    assert table.get("Amikacin") == "Ribosome"


def test_only_the_lab_table_present_still_works(tmp_path):
    plato_root = tmp_path / "PLATO"
    plato_root.mkdir()
    ai4ab_dir = tmp_path / "AI4AB" / "analysis" / "E_coli_params"
    ai4ab_dir.mkdir(parents=True, exist_ok=True)
    (ai4ab_dir / "moa_dict_inv.json").write_text(
        json.dumps({"Gyrase": ["Ciprofloxacin"]})
    )
    table = load_default_moa([plato_root])
    assert table.get("Ciprofloxacin") == "Gyrase"


# -- the shipped drug_moa.csv, if present ------------------------------------------


def test_shipped_drug_moa_additions_do_not_collide_with_ai4ab():
    """The whole point of a layer file: it should ADD drugs, not repeat ones
    AI4AB already has correctly -- a name in both is either a deliberate,
    documented correction or an accidental duplicate worth catching."""
    from pathlib import Path

    from plato.data.annotations import DEFAULT_MOA_CANDIDATES

    root = Path(__file__).resolve().parents[1]
    local_path = root / DEFAULT_MOA_CANDIDATES[0]
    if not local_path.exists():
        pytest.skip("no local drug_moa.csv shipped")

    ai4ab_path = None
    for candidate in DEFAULT_MOA_CANDIDATES[1:]:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            ai4ab_path = resolved
            break
    if ai4ab_path is None:
        pytest.skip("AI4AB table not present on this machine")

    local = load_or_empty(local_path)
    lab = load_or_empty(ai4ab_path)
    overlap = set(local._by_key) & set(lab._by_key)
    assert not overlap, (
        f"drug_moa.csv repeats {overlap}, already in AI4AB's table -- "
        "either drop the duplicate or document why it corrects AI4AB"
    )


def test_shipped_drug_moa_covers_iclaprim_and_rifabutin():
    """The two drugs genuinely absent from AI4AB's table under any spelling
    (unlike Cefepim/Doxicyclin/Clavulanic Acid/Penicillin below, which are
    spelling variants of drugs AI4AB already has). Classified by their real
    mechanism, cross-referenced against Krentzel/Kho et al. 2025's own MoA
    taxonomy where it applied (it does not cover either of these two) and a
    web search otherwise -- see drug_moa.csv's own comment for sources."""
    from pathlib import Path

    from plato.data.annotations import DEFAULT_MOA_CANDIDATES, load_default_moa

    root = Path(__file__).resolve().parents[1]
    if not (root / DEFAULT_MOA_CANDIDATES[0]).exists():
        pytest.skip("no local drug_moa.csv shipped")

    table = load_default_moa([root])
    # Iclaprim: a DHFR inhibitor, same target as AI4AB's own Trimethoprim.
    assert table.get("Iclaprim") == table.get("Trimethoprim") == "DNA synthesis"
    # Rifabutin: a rifamycin, same target as AI4AB's own Rifampicin.
    assert table.get("Rifabutin") == table.get("Rifampicin") == "RNA polymerase"


def test_shipped_drug_moa_bridges_real_spelling_variants():
    """Cefepim/Doxicyclin/Clavulanic Acid/Penicillin are how different real
    exports spell drugs AI4AB's table already covers under a different
    spelling -- AnnotationTable's own whitespace/case folding does not
    bridge a genuine spelling difference, so each needs its own alias
    entry, mapped to the SAME class as its AI4AB-spelled counterpart."""
    from pathlib import Path

    from plato.data.annotations import DEFAULT_MOA_CANDIDATES, load_default_moa

    root = Path(__file__).resolve().parents[1]
    if not (root / DEFAULT_MOA_CANDIDATES[0]).exists():
        pytest.skip("no local drug_moa.csv shipped")
    if not any((root / c).resolve().exists() for c in DEFAULT_MOA_CANDIDATES[1:]):
        pytest.skip("AI4AB table not present on this machine")

    table = load_default_moa([root])
    assert table.get("Cefepim") == table.get("Cefepime")
    assert table.get("Doxicyclin") == table.get("Doxycycline")
    assert table.get("Clavulanic Acid") == table.get("Clavulanate")
    assert table.get("Penicillin") == table.get("PenicillinG")
    assert UNANNOTATED not in (
        table.get("Cefepim"),
        table.get("Doxicyclin"),
        table.get("Clavulanic Acid"),
        table.get("Penicillin"),
    )
