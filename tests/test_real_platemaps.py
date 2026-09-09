"""Detection against plate maps shaped like the real ones.

Both of these were live bugs found by loading the actual FK_P001 maps:

* the real maps carry column numbers AND row letters, and detection only ever
  tried a bare grid, so Load Data refused every one of them;
* the exported antibiotic maps write "Ciprofloxacin 1x" on one line, while the
  only drug/dose pattern expected the Excel original's newline, so the whole
  ABx arm indexed as one opaque condition column with no antibiotic or
  concentration to filter by.

The fixtures here are written in the same shapes as the real files, so these
run anywhere; the tests at the bottom use the real files when Z: is mounted.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from plato.data.index.platemap import read_platemap
from plato.gui.detect import detect_platemap_layout

MAPS = Path(
    r"Z:\Data\FK_P001_EX0039_2026_08_28_CRISPRI & ABx Experiment\Plate_Maps"
)
needs_share = pytest.mark.skipif(not MAPS.is_dir(), reason="plate map share not mounted")

ROWS = list("ABCDEFGH")


def _write_grid(path: Path, cells: list[list[str]], *, header: bool, index: bool) -> Path:
    """Write an 8x12 grid, optionally with 1..12 headers and A..H row labels."""
    frame = pd.DataFrame(cells)
    if index:
        frame.index = ROWS
    if header:
        frame.columns = [str(c) for c in range(1, 13)]
    frame.to_csv(path, header=header, index=index)
    return path


def _gene_cells() -> list[list[str]]:
    genes = ["ftsZ", "gyrA", "lptA", "murA", "ACE-1 NC", "MG1655 NC"]
    return [
        [f"{genes[(r + c) % len(genes)]}_{(c % 3) + 1}" for c in range(12)]
        for r in range(8)
    ]


def _drug_cells(separator: str) -> list[list[str]]:
    drugs = ["Ciprofloxacin", "Polymyxin B", "Penicillin G", "Colistin"]
    doses = ["0.25x", "0.5x", "1x", "2x"]
    cells = [
        [f"{drugs[(r + c) % len(drugs)]}{separator}{doses[c % 4]}" for c in range(12)]
        for r in range(8)
    ]
    # Controls carry no dose and must survive unsplit.
    cells[0][0] = "WT"
    cells[0][1] = "WT NC"
    return cells


@pytest.mark.parametrize("header,index", [(True, True), (False, False), (True, False), (False, True)])
def test_detects_every_grid_shape(tmp_path, header, index):
    """A map may or may not carry its row/column labels; all four load."""
    path = _write_grid(tmp_path / "p.csv", _gene_cells(), header=header, index=index)
    guess = detect_platemap_layout(path)
    assert guess is not None, f"header={header} index={index} not detected"
    assert guess.layout == "matrix"
    result = read_platemap(
        path,
        layout="matrix",
        header_row=guess.header_row,
        index_col=guess.index_col,
        split_pattern=guess.split_pattern,
        plate_format=96,
    )
    assert len(result.frame) == 96


def test_labelled_grid_splits_gene_and_replicate(tmp_path):
    path = _write_grid(tmp_path / "crispri.csv", _gene_cells(), header=True, index=True)
    guess = detect_platemap_layout(path)
    assert "gene" in guess.split_pattern
    result = read_platemap(
        path,
        layout="matrix",
        header_row=guess.header_row,
        index_col=guess.index_col,
        split_pattern=guess.split_pattern,
        plate_format=96,
    )
    assert {"gene", "replicate"} <= set(result.frame.columns)
    assert "ftsZ" in set(result.frame["gene"])
    # A control name containing a space and a hyphen must stay intact.
    assert "ACE-1 NC" in set(result.frame["gene"])


@pytest.mark.parametrize("separator", [" ", "\n"])
def test_drug_dose_splits_on_either_separator(tmp_path, separator):
    """The Excel original uses a newline; the exported CSV uses a space."""
    path = _write_grid(
        tmp_path / "abx.csv", _drug_cells(separator), header=True, index=True
    )
    guess = detect_platemap_layout(path)
    assert guess is not None
    assert "antibiotic" in guess.split_pattern, f"no drug split for {separator!r}"
    result = read_platemap(
        path,
        layout="matrix",
        header_row=guess.header_row,
        index_col=guess.index_col,
        split_pattern=guess.split_pattern,
        plate_format=96,
    )
    drugs = set(result.frame["antibiotic"].dropna())
    # Two-word names must not be cut at the first space.
    assert "Polymyxin B" in drugs
    assert "Penicillin G" in drugs
    assert set(result.frame["concentration"].dropna()) == {"0.25x", "0.5x", "1x", "2x"}
    # Controls have no dose and are left whole rather than misparsed.
    assert {"WT", "WT NC"} <= {value for _, value in result.unsplit_values}


# -- the real files ---------------------------------------------------------


@needs_share
@pytest.mark.parametrize("arm,expected", [("CRISPRi", "gene"), ("ABx", "antibiotic")])
def test_real_plate_maps_detect(arm, expected):
    guess = detect_platemap_layout(MAPS / arm / "P1.csv")
    assert guess is not None, f"{arm} map not detected"
    assert guess.header_row and guess.index_col
    assert expected in guess.split_pattern


@needs_share
def test_real_abx_map_splits_all_drugs():
    guess = detect_platemap_layout(MAPS / "ABx" / "P1.csv")
    result = read_platemap(
        MAPS / "ABx" / "P1.csv",
        layout="matrix",
        header_row=guess.header_row,
        index_col=guess.index_col,
        split_pattern=guess.split_pattern,
        plate_format=96,
    )
    drugs = set(result.frame["antibiotic"].dropna())
    assert "Polymyxin B" in drugs
    assert "Penicillin G" in drugs
    assert {"WT", "WT NC"} <= {value for _, value in result.unsplit_values}
