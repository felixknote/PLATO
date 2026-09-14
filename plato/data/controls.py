"""Which wells are controls, and making them visible in every encoding.

A control is not another treatment. It is the thing every other point is
judged against, so the question asked of it is always the same -- "where do
the controls sit, and does everything else sit apart from them" -- whatever
field happens to be on screen. Colouring by gene hides that: the controls are
scattered through a legend of fifty genes as just more entries, and finding
them means knowing which names are the non-targeting ones.

So marking controls does two things:

**They become their own classes.** In any categorical encoding, a control row
reports its control class instead of its ordinary value -- "WT (non-targeting
gRNA)" rather than whichever gene it nominally carries. One legend entry per
control kind, in every field, so the comparison is available without changing
what you are looking at. Treatment rows are untouched.

**They are drawn to be found.** Marking is also what lets the view emphasise
them (see ``views/explorer.py``): a control inside a dense cluster of 30,000
points is invisible at 3 px whatever colour it is.

``parse_condition`` already recognises the lab's own control labels, so
detection starts from that rather than from an empty dialog -- but it is a
dictionary of known names (``annotations.CONTROL_LABELS``), and a new screen
naming its controls differently would get nothing. Hence the confirm step:
the guess is shown, and correcting it is picking values from a list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .annotations import CONTROL
from .explorer_model import CONTROL_KIND, ROLE

# The derived column marking each row's control class, empty for treatments.
CONTROL_CLASS = "control_class"
# The standalone encoding: control class, or "Treatment".
CONTROL_FIELD = "control_group"
TREATMENT_LABEL = "Treatment"

FIELD_LABELS = {
    CONTROL_CLASS: "Control class",
    CONTROL_FIELD: "Controls vs treatment",
}


def detect(frame: pd.DataFrame) -> dict[str, str]:
    """Guess the control assignment from what the condition parser found.

    Returns ``{condition label: control class}``. Only labels the parser
    already flagged as controls are included, so this is a starting point to
    confirm rather than an answer -- a screen whose controls are named in a
    way ``annotations`` has not seen returns nothing, which is the honest
    result and exactly why the dialog exists.
    """
    if frame is None or ROLE not in frame.columns:
        return {}
    from .explorer_model import CONDITION

    if CONDITION not in frame.columns:
        return {}

    is_control = frame[ROLE].astype(str) == CONTROL
    if not is_control.any():
        return {}

    out: dict[str, str] = {}
    subset = frame.loc[is_control, [CONDITION, CONTROL_KIND]].astype(str)
    for label, kind in zip(subset[CONDITION], subset[CONTROL_KIND]):
        label = label.strip()
        if not label:
            continue
        # control_kind is the parser's own name for what this controls for
        # ("WT (non-targeting gRNA)", "Vehicle (DMSO)"). Falling back to the
        # raw label keeps a recognised-but-unnamed control visible rather
        # than collapsing it into a generic bucket.
        out.setdefault(label, kind.strip() or label)
    return out


@dataclass
class ControlMarking:
    """Which values of one column are controls, and what each one controls for.

    ``source`` is the column the assignment is written against -- usually the
    condition label, but a screen that carries its controls in a dedicated
    column can mark that one instead.
    """

    source: str
    assignments: dict[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.source and self.assignments)

    def classes(self) -> list[str]:
        """Distinct control classes, in first-assigned order."""
        return list(dict.fromkeys(self.assignments.values()))

    def mask(self, frame: pd.DataFrame) -> pd.Series:
        """True for rows that are controls."""
        if not self or self.source not in frame.columns:
            return pd.Series(False, index=frame.index)
        text = frame[self.source].astype(str).str.strip()
        return text.isin(self.assignments)

    def class_for(self, frame: pd.DataFrame) -> pd.Series:
        """Each row's control class, empty string for treatments."""
        if not self or self.source not in frame.columns:
            return pd.Series("", index=frame.index, dtype=object)
        text = frame[self.source].astype(str).str.strip()
        return text.map(lambda v: self.assignments.get(v, ""))

    def apply(self, frame: pd.DataFrame) -> list[str]:
        """Materialise the control columns onto ``frame``.

        Returns the column names added. Like custom groupings, these are
        ordinary derived columns, so faceting, filtering, the legend and the
        export headline need no special case for them.
        """
        classes = self.class_for(frame)
        frame[CONTROL_CLASS] = classes
        frame[CONTROL_FIELD] = classes.where(classes.astype(bool), TREATMENT_LABEL)
        return [CONTROL_CLASS, CONTROL_FIELD]

    def overlay(self, values: pd.Series, frame: pd.DataFrame) -> pd.Series:
        """Replace a column's values with the control class, where there is one.

        This is what "controls show up as additional classes in all display
        options" means concretely: colouring by gene, a non-targeting row
        reports "WT (non-targeting gRNA)" instead of its gene, so the control
        is one legend entry rather than scattered among the treatments.
        """
        classes = self.class_for(frame)
        text = values.astype(str)
        return text.where(~classes.astype(bool), classes)

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        return {"source": self.source, "assignments": dict(self.assignments)}

    @classmethod
    def from_dict(cls, raw: dict) -> ControlMarking:
        assignments = raw.get("assignments") or {}
        return cls(
            source=str(raw.get("source") or ""),
            assignments={str(k): str(v) for k, v in assignments.items() if str(v)},
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def load_json(cls, text: str) -> ControlMarking:
        """Never raises: a corrupt setting must not stop the explorer opening."""
        try:
            raw = json.loads(text or "{}")
        except (ValueError, TypeError):
            return cls(source="")
        return cls.from_dict(raw) if isinstance(raw, dict) else cls(source="")

    def save_to(self, path: Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load_from(cls, path: Path) -> ControlMarking:
        try:
            return cls.load_json(Path(path).read_text(encoding="utf-8"))
        except OSError:
            return cls(source="")
