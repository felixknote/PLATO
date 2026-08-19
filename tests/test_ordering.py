"""Filter-value ordering.

These values become the comparison view's columns left to right, so their
order is what a reader interprets as the x-axis of the comparison.
"""

from __future__ import annotations

import pytest

from plato.ordering import numeric_part, sort_series


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1x", 1.0),
        ("1/2x", 0.5),
        ("1/8x", 0.125),
        ("10 ug/mL", 10.0),
        ("T12", 12.0),
        ("0.5", 0.5),
        ("1e2", 100.0),
        ("WT", None),
        ("Ciprofloxacin", None),
        ("", None),
    ],
)
def test_numeric_part_reads_the_magnitude(value, expected) -> None:
    parsed = numeric_part(value)
    assert (None if parsed is None else parsed[1]) == expected


def test_division_by_zero_is_not_a_number() -> None:
    assert numeric_part("1/0x") is None


def test_dose_series_ascends_rather_than_sorting_as_text() -> None:
    # The failure this guards: lexicographic order gives 1/2x, 1/4x, 1/8x, 1x,
    # which reads left to right as a dose series that it is not.
    assert sort_series(["1x", "1/2x", "1/8x", "1/4x", "2x"]) == [
        "1/8x",
        "1/4x",
        "1/2x",
        "1x",
        "2x",
    ]


def test_numeric_concentrations_sort_by_value() -> None:
    assert sort_series(["0", "0.5", "10", "2"]) == ["0", "0.5", "2", "10"]


def test_timepoints_sort_by_index_not_by_text() -> None:
    assert sort_series(["T10", "T2", "T1"]) == ["T1", "T2", "T10"]


def test_controls_sort_after_the_doses_they_accompany() -> None:
    assert sort_series(["1/2x", "WT", "1x", "1/8x", "ACE-1 NC"]) == [
        "1/8x",
        "1/2x",
        "1x",
        "ACE-1 NC",
        "WT",
    ]


def test_labels_containing_digits_stay_alphabetical() -> None:
    # "FM464" carries a number but a channel list is not a series; ordering it
    # by magnitude would put FM464 before DAPI for no discernible reason.
    assert sort_series(["FM464", "DAPI"]) == ["DAPI", "FM464"]


def test_names_sort_alphabetically_case_insensitively() -> None:
    assert sort_series(["ciprofloxacin", "Ampicillin", "DMSO"]) == [
        "Ampicillin",
        "ciprofloxacin",
        "DMSO",
    ]


def test_empty_input_is_handled() -> None:
    assert sort_series([]) == []
