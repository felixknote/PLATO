"""User-defined classes over an existing column's values.

Faceting and colouring both work by reading the distinct values of one column,
so they can only ever split data the way the metadata already splits it. That
is not always the split that matters. Plates P1..P8 were acquired two per day
over four days, and "which day was this measured on" is not written anywhere
in the export -- but it is exactly the variable a batch effect lives in, and
without it the only way to look for one is to read a plate facet grid and hold
the pairing in your head.

A custom group is a named class over some of a column's values::

    plate:  P1, P2 -> "Day 1"
            P3, P4 -> "Day 2"

The result materialises as an ordinary derived column, so everything
downstream -- faceting, colouring, the legend, filters, the export headline --
works on it unchanged rather than growing a special case. That is the whole
design: this module decides *what the classes are*, and nothing else has to
know they were user-defined.

**Values left unassigned keep their own name** rather than collapsing into an
"other" bucket. Grouping four of eight plates by day and silently merging the
rest would quietly hide half the data behind one label; keeping them distinct
means a half-finished grouping is visibly half-finished.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Derived columns are named after their source so several groupings can
# coexist -- a day grouping over plate and a class grouping over drug are
# different columns, not a single overwritten one.
COLUMN_PREFIX = "group_"


def derived_column(source: str) -> str:
    """The column name a grouping over ``source`` materialises into."""
    return f"{COLUMN_PREFIX}{source}"


def is_derived(column: str) -> bool:
    return column.startswith(COLUMN_PREFIX)


def source_of(column: str) -> str:
    """The column a derived grouping was built from."""
    return column[len(COLUMN_PREFIX) :] if is_derived(column) else column


@dataclass
class CustomGrouping:
    """One named grouping over one column.

    ``assignments`` maps a source value to a class name. Values absent from it
    are their own class, so a partial grouping stays honest about what it has
    not covered.
    """

    source: str
    name: str = ""
    assignments: dict[str, str] = field(default_factory=dict)

    @property
    def column(self) -> str:
        return derived_column(self.source)

    @property
    def label(self) -> str:
        """What the dropdown shows."""
        from .explorer_model import field_label

        base = field_label(self.source)
        return f"{base} by {self.name}" if self.name else f"{base} (grouped)"

    def classes(self) -> list[str]:
        """Distinct class names, in first-assigned order."""
        return list(dict.fromkeys(self.assignments.values()))

    def apply(self, values: pd.Series) -> pd.Series:
        """Map a column's values to their class names.

        An unassigned value maps to itself -- see the module docstring for why
        that is deliberate rather than an oversight.
        """
        text = values.astype(str).str.strip()
        return text.map(lambda v: self.assignments.get(v, v))

    def covers(self, values) -> int:
        """How many distinct values of ``values`` this grouping assigns."""
        distinct = {str(v).strip() for v in values}
        return len(distinct & set(self.assignments))

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "name": self.name,
            "assignments": dict(self.assignments),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> CustomGrouping:
        assignments = raw.get("assignments") or {}
        return cls(
            source=str(raw.get("source") or ""),
            name=str(raw.get("name") or ""),
            # Both sides forced to str: JSON round-trips dict keys as strings
            # anyway, and a frame column is compared as str everywhere else.
            assignments={str(k): str(v) for k, v in assignments.items() if str(v)},
        )


class GroupingStore:
    """Every grouping defined for one workspace, keyed by source column.

    One grouping per source column: two different day-groupings over `plate`
    would produce the same derived column name and silently overwrite each
    other, and "several ways to group the same field" is not a need that has
    come up. Editing an existing one replaces it.
    """

    def __init__(self) -> None:
        self._by_source: dict[str, CustomGrouping] = {}

    def __len__(self) -> int:
        return len(self._by_source)

    def __iter__(self):
        return iter(self._by_source.values())

    def get(self, source: str) -> CustomGrouping | None:
        return self._by_source.get(source)

    def set(self, grouping: CustomGrouping) -> None:
        """Add or replace. An empty grouping removes instead of storing."""
        if not grouping.source:
            return
        if not grouping.assignments:
            self._by_source.pop(grouping.source, None)
            return
        self._by_source[grouping.source] = grouping

    def remove(self, source: str) -> None:
        self._by_source.pop(source, None)

    def apply_all(self, frame: pd.DataFrame) -> list[str]:
        """Materialise every grouping whose source column exists.

        Returns the derived column names added. Called after the frame is
        built and again whenever a grouping changes, so the derived columns
        always reflect the current definitions.
        """
        added: list[str] = []
        for grouping in self._by_source.values():
            if grouping.source not in frame.columns:
                continue
            frame[grouping.column] = grouping.apply(frame[grouping.source])
            added.append(grouping.column)
        return added

    def labels(self) -> dict[str, str]:
        """Derived column -> the label a dropdown should show for it."""
        return {g.column: g.label for g in self._by_source.values()}

    # -- persistence ------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(
            {"groupings": [g.to_dict() for g in self._by_source.values()]}, indent=2
        )

    def load_json(self, text: str) -> None:
        """Replace the store's contents from ``to_json`` output.

        Malformed input leaves the store empty rather than raising: a corrupt
        settings value must not stop the explorer opening.
        """
        self._by_source.clear()
        try:
            raw = json.loads(text or "{}")
        except (ValueError, TypeError):
            return
        for entry in raw.get("groupings") or []:
            if isinstance(entry, dict):
                self.set(CustomGrouping.from_dict(entry))

    def save_to(self, path: Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    def load_from(self, path: Path) -> None:
        try:
            self.load_json(Path(path).read_text(encoding="utf-8"))
        except OSError:
            self._by_source.clear()


def suggest_classes(values, per_class: int = 2, prefix: str = "Group") -> dict[str, str]:
    """A starting assignment putting consecutive values into classes of ``n``.

    The common case by a wide margin -- plates acquired two per day, in
    order -- so the dialog opens on something close to the answer instead of
    an empty grid. It is a *suggestion*: the ordering it assumes (that the
    column sorts the way the runs happened) is true often enough to save
    typing and obvious enough to correct when it is not.
    """
    ordered = sorted({str(v).strip() for v in values if str(v).strip()}, key=_natural)
    out: dict[str, str] = {}
    if per_class < 1:
        return out
    for index, value in enumerate(ordered):
        out[value] = f"{prefix} {index // per_class + 1}"
    return out


def _natural(text: str):
    """Sort key that orders P2 before P10."""
    import re

    parts = re.split(r"(\d+)", text)
    return [int(p) if p.isdigit() else p.lower() for p in parts]
