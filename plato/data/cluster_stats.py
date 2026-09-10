"""What is a lasso'd region actually made of?

A projection poses one question -- "these points sit together, why?" -- and
answering it by eye means colouring by one field at a time until something
lines up. This does that sweep numerically: for every categorical column the
dataset actually has, count the selected points, rank the values, and say
which fields separate the cluster from the background.

Two numbers per value, because prevalence alone is misleading. A cluster that
is 48% `acrB` is only interesting if the dataset as a whole is not also 48%
`acrB`; conversely a value at 3% of the selection can be the strongest signal
in the plot if it is 0.05% of everything else. So each entry carries both its
share of the selection and its *enrichment* over the background, and fields
are ranked by how far their composition departs from the background rather
than by the size of their largest bucket.

Nothing here knows what a gene or an antibiotic is. Columns come from the
frame, labels from FIELD_LABELS, and a dataset with entirely different
metadata gets a summary of whatever it does have.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .explorer_model import FIELD_LABELS, IMAGE_NAME, IMAGE_PATH, field_label

# Columns that are identifiers rather than categories. Summarising them is
# noise: every point has its own value, so the "composition" is a list of
# distinct filenames as long as the selection.
_NEVER_SUMMARISE = frozenset({IMAGE_NAME, IMAGE_PATH})

# A column with more distinct values than this in the selection is treated as
# an identifier, not a category. Deliberately generous -- a 384-well plate is
# a legitimate thing to break a cluster down by.
MAX_CATEGORIES = 400


@dataclass(slots=True)
class ValueShare:
    """One value of one field, inside the selection and outside it."""

    value: str
    count: int
    fraction: float  # share of the selection, 0..1
    background_fraction: float  # share of everything else, 0..1

    @property
    def percent(self) -> float:
        return self.fraction * 100.0

    @property
    def enrichment(self) -> float:
        """How many times over-represented this value is in the selection.

        ``inf`` when the value appears only inside the selection, which is the
        strongest signal there is and should not be flattened to a number that
        sorts alongside ordinary ratios.
        """
        if self.background_fraction <= 0.0:
            return math.inf if self.fraction > 0 else 1.0
        return self.fraction / self.background_fraction


@dataclass(slots=True)
class FieldSummary:
    """Composition of the selection along one metadata column."""

    column: str
    label: str
    values: list[ValueShare] = field(default_factory=list)
    # Selected points with no value for this column. Excluded from the shares
    # above, so a half-annotated field is not reported as half "(blank)".
    missing: int = 0
    # How far this field's composition departs from the background, 0..1.
    divergence: float = 0.0

    @property
    def dominant(self) -> ValueShare | None:
        return self.values[0] if self.values else None

    @property
    def is_uniform(self) -> bool:
        """One value accounts for the whole selection."""
        return len(self.values) == 1 and self.missing == 0


@dataclass(slots=True)
class ClusterComposition:
    """The full breakdown of one selection."""

    n_selected: int
    n_total: int
    fields: list[FieldSummary] = field(default_factory=list)

    @property
    def headline(self) -> str:
        """One sentence naming the most distinctive thing about the cluster."""
        if not self.n_selected:
            return "Nothing selected."
        for summary in self.fields:
            top = summary.dominant
            if top is None:
                continue
            # Needs to be both common inside and unusual outside: a field that
            # is 90% one value everywhere says nothing about this region.
            if top.fraction >= 0.25 and top.enrichment >= 1.5:
                ratio = (
                    "only here"
                    if math.isinf(top.enrichment)
                    else f"{top.enrichment:.1f}x the background"
                )
                return (
                    f"Mostly {summary.label}: <b>{top.value}</b> "
                    f"({top.percent:.1f}% of the selection, {ratio})"
                )
        return "No single field dominates this cluster."


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    """One column as plain strings, with nulls flattened to empty."""
    return frame[column].astype(str).replace({"nan": "", "None": ""}).fillna("")


def summarise_field(
    selected: pd.Series, background: pd.Series, column: str
) -> FieldSummary | None:
    """Compose one column, or None if it is not a usable category here."""
    present = selected[selected.ne("")]
    missing = int(len(selected) - len(present))
    if present.empty:
        return None

    counts = present.value_counts()
    if len(counts) > MAX_CATEGORIES:
        return None

    total = int(counts.sum())
    background = background[background.ne("")]
    background_counts = background.value_counts()
    background_total = int(background_counts.sum())

    values = [
        ValueShare(
            value=str(value),
            count=int(count),
            fraction=count / total,
            background_fraction=(
                float(background_counts.get(value, 0)) / background_total
                if background_total
                else 0.0
            ),
        )
        for value, count in counts.items()
    ]
    # Prevalence first: the table answers "what is in here", and enrichment is
    # a column of the table rather than the sort key. Ties break on the name,
    # not on enrichment -- an unbounded value would otherwise let a single
    # background-absent row outrank an equally common, well-evidenced one, and
    # make the order of two 50/50 values depend on which was rarer elsewhere.
    values.sort(key=lambda v: (-v.count, v.value))

    # Total variation distance between the selection and the background: half
    # the summed absolute difference in shares, so 0 means identical
    # composition and 1 means no overlap at all. This is what ranks fields --
    # a field is interesting when the cluster does NOT look like the rest.
    #
    # Summed over the union of both vocabularies, not just the values present
    # in the selection: a value that is common in the background and absent
    # here is exactly as much of a difference as the reverse, and counting
    # only one direction understates it.
    if background_total:
        seen = {v.value: v.fraction for v in values}
        vocabulary = set(seen) | {str(k) for k in background_counts.index}
        divergence = 0.5 * sum(
            abs(
                seen.get(name, 0.0)
                - float(background_counts.get(name, 0)) / background_total
            )
            for name in vocabulary
        )
    else:
        # Nothing outside to compare against (the selection is everything, or
        # the field is blank everywhere else). A field annotated only inside
        # the selection is maximally distinctive, not featureless.
        divergence = 1.0 if values else 0.0

    # Weight by how much of the selection the field actually annotates. A
    # column with a value for six of two hundred selected points may be
    # perfectly divergent over those six and still says far less about the
    # cluster than one covering all of them; without this, sparse columns
    # dominate the ranking on the strength of a handful of rows.
    coverage = total / (total + missing) if (total + missing) else 0.0
    divergence *= coverage

    return FieldSummary(
        column=column,
        label=field_label(column),
        values=values,
        missing=missing,
        divergence=float(min(1.0, divergence)),
    )


def compose(
    frame: pd.DataFrame,
    rows,
    *,
    columns: list[str] | None = None,
) -> ClusterComposition:
    """Break a selection down along every categorical column available.

    Args:
        frame: the explorer's metadata frame.
        rows: row indices of the selected points.
        columns: restrict to these columns; by default every column of the
            frame that behaves like a category is used, so a dataset carrying
            metadata this code has never heard of is still summarised.
    """
    n_total = 0 if frame is None else int(len(frame))
    indices = np.asarray(rows, dtype=np.int64).ravel() if rows is not None else np.empty(0, np.int64)
    if frame is None or len(indices) == 0 or n_total == 0:
        return ClusterComposition(n_selected=0, n_total=n_total)

    # Guard against stale indices: a selection can outlive a reprojection.
    indices = indices[(indices >= 0) & (indices < n_total)]
    if len(indices) == 0:
        return ClusterComposition(n_selected=0, n_total=n_total)

    if columns is None:
        # Prefer the known fields, in their established display order, then
        # anything else the frame happens to carry. Discovery, not assumption.
        known = [c for c in FIELD_LABELS if c in frame.columns]
        extra = [
            c
            for c in frame.columns
            if c not in FIELD_LABELS and c not in _NEVER_SUMMARISE
        ]
        columns = known + extra

    mask = np.zeros(n_total, dtype=bool)
    mask[indices] = True
    inside = frame.iloc[indices]
    outside = frame.loc[~mask]

    summaries: list[FieldSummary] = []
    for column in columns:
        if column in _NEVER_SUMMARISE or column not in frame.columns:
            continue
        try:
            summary = summarise_field(
                _series(inside, column), _series(outside, column), column
            )
        except (TypeError, ValueError):
            # A column of an exotic dtype is not worth crashing the panel for.
            continue
        if summary is None:
            continue
        # A field with one value across the WHOLE dataset describes the
        # dataset, not this cluster.
        if summary.is_uniform and summary.divergence <= 0.0:
            continue
        summaries.append(summary)

    # Most distinctive first, so the answer to "why are these together" is at
    # the top rather than wherever the column happens to sit in the frame.
    summaries.sort(key=lambda s: (-s.divergence, s.label))
    return ClusterComposition(
        n_selected=int(len(indices)), n_total=n_total, fields=summaries
    )
