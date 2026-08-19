"""Canonical well identifiers.

`A1`, `A01`, `a1`, ` A1 ` must all collapse to the same key before any join
between plate map and filenames happens. Everything downstream uses
``(row, col)`` integers internally and the zero-padded label for display.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WELL_RE = re.compile(r"^\s*([A-Za-z]{1,2})\s*0*(\d{1,2})\s*$")

# Plate geometries we accept. Anything outside these bounds is a parse error,
# not a silently accepted well.
PLATE_GEOMETRIES: dict[int, tuple[int, int]] = {
    6: (2, 3),
    12: (3, 4),
    24: (4, 6),
    48: (6, 8),
    96: (8, 12),
    384: (16, 24),
    1536: (32, 48),
}


class WellParseError(ValueError):
    """Raised when a string cannot be interpreted as a well identifier."""


@dataclass(frozen=True, slots=True)
class Well:
    """A well position. ``row`` and ``col`` are 1-based."""

    row: int
    col: int

    @property
    def label(self) -> str:
        """Zero-padded canonical label, e.g. ``A01``."""
        return f"{self.row_letter}{self.col:02d}"

    @property
    def row_letter(self) -> str:
        # The largest plate here is 1536-well = 32 rows, so a row is always a
        # single letter; two-letter rows would need a geometry that does not
        # exist in PLATE_GEOMETRIES.
        return chr(ord("A") + self.row - 1)


def _letters_to_row(letters: str) -> int:
    value = 0
    for ch in letters.upper():
        value = value * 26 + (ord(ch) - ord("A") + 1)
    return value


def parse_well(raw: object, plate_format: int | None = 96) -> Well:
    """Parse an arbitrary well string into a :class:`Well`.

    Args:
        raw: The value from a plate map cell or a filename capture group.
        plate_format: Expected number of wells. If given, out-of-range positions
            raise instead of being accepted. Pass ``None`` to skip the check.

    Raises:
        WellParseError: If the value is not a well or lies outside the plate.
    """
    if raw is None:
        raise WellParseError("well is None")
    match = _WELL_RE.match(str(raw))
    if match is None:
        raise WellParseError(f"cannot parse well identifier: {raw!r}")
    row = _letters_to_row(match.group(1))
    col = int(match.group(2))
    if col == 0:
        raise WellParseError(f"column 0 is not a valid well: {raw!r}")
    if plate_format is not None:
        if plate_format not in PLATE_GEOMETRIES:
            raise WellParseError(f"unsupported plate format: {plate_format}")
        n_rows, n_cols = PLATE_GEOMETRIES[plate_format]
        if row > n_rows or col > n_cols:
            raise WellParseError(
                f"well {raw!r} lies outside a {plate_format}-well plate "
                f"({n_rows}x{n_cols})"
            )
    return Well(row=row, col=col)


def canonical(raw: object, plate_format: int | None = 96) -> str:
    """Convenience wrapper returning the canonical label directly."""
    return parse_well(raw, plate_format).label
