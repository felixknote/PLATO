"""Ordering for filter values that read as a series.

Filter values arrive as strings, and a plain string sort puts a dose series in
an order that is actively misleading: "1/2x", "1/4x", "1/8x", "1x" reads
left-to-right as increasing dose when it is nothing of the sort, and the
weakest dose lands in the middle. Numeric columns are no better -- "10" sorts
before "2".

The comparison view lays these values out as columns left to right, so their
order *is* the x-axis of the figure someone reads off the screen. Sorting them
by magnitude is what makes that axis mean what it looks like it means.

Values that carry no number (gene names, antibiotic names, "WT") keep plain
alphabetical order, and sort after the numeric ones rather than interleaving
with them -- a control column belongs at one end of a dose series, not in the
middle of it.
"""

from __future__ import annotations

import re

# The first number anywhere in the value, optionally a fraction ("1/2"),
# optionally with a decimal point or exponent. Not anchored to the start: a
# timepoint is "T1"/"T10" and a dose is "1/8x", so the number that orders the
# series sits behind a prefix as often as it leads.
_NUMBER = re.compile(
    r"(?P<numerator>\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"\s*(?:/\s*(?P<denominator>\d*\.?\d+))?"
)


def numeric_part(value: str) -> tuple[str, float] | None:
    """The value's leading text and its first number, or None when it has none.

    Handles the fractional-MIC form a compound screen uses ("1/8x" -> 0.125),
    plain numbers ("10 ug/mL" -> 10.0), and a prefixed index ("T10" -> 10.0).
    The prefix is returned alongside so that values from different series
    ("T1" vs "P1") group by prefix before ordering by number.
    """
    match = _NUMBER.search(value)
    if match is None:
        return None
    try:
        number = float(match.group("numerator"))
        denominator = match.group("denominator")
        if denominator is not None:
            divisor = float(denominator)
            if divisor == 0:
                return None
            number /= divisor
    except ValueError:  # pragma: no cover - regex already constrains the shape
        return None
    return value[: match.start()].strip().casefold(), number


def series_key(value: object) -> tuple[int, str, float, str]:
    """Sort key ordering a dose/timepoint series by magnitude.

    Numeric values sort by their prefix then their magnitude; everything else
    sorts alphabetically after them, so controls sit at one end rather than in
    the middle of the series.
    """
    text = str(value)
    parsed = numeric_part(text)
    if parsed is None:
        return (1, "", 0.0, text.casefold())
    prefix, number = parsed
    return (0, prefix, number, text.casefold())


def sort_series(values: object) -> list[str]:
    """Sort filter values into the order they should be read left to right.

    Numeric ordering is applied only when the values actually look like a
    series -- most of them carry a number. A dose series with a couple of
    controls in it ("1/8x", "1/4x", "1x", "WT") still counts, and the controls
    sort to the end; a set of labels that merely contains digits ("FM464"
    beside "DAPI") does not, and stays alphabetical. Ranking the latter by
    magnitude would put "FM464" before "DAPI" for no reason a reader of the
    column order could guess.
    """
    text = [str(v) for v in values]
    non_empty = [t for t in text if t.strip()]
    if not non_empty:
        return sorted(text, key=lambda t: t.casefold())
    numeric = sum(numeric_part(t) is not None for t in non_empty)
    if numeric * 2 > len(non_empty):
        return sorted(text, key=series_key)
    return sorted(text, key=lambda t: t.casefold())
