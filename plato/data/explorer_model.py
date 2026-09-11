"""The table behind the embedding plot: one row per point, all colourable
fields resolved, plus the path back to the original image.

This is the join layer. It takes an ``EmbeddingDataset`` (vectors + whatever
columns that export happened to write) and produces a frame with a stable set
of columns the view can rely on, whichever dataset was loaded. Datasets differ
-- one has ``gene``, another ``drug``, another a single ``label`` -- so the
view must not have to know which.

Image resolution is by metadata, never by row position. The embedding row
order is an artefact of the extractor's directory walk; the columns are the
contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .annotations import UNANNOTATED, AnnotationTable, parse_condition
from .embeddings import EmbeddingDataset
from .image_lookup import (
    ImageIndex,
    LookupReport,
    hint_columns,
    probe,
)

# Columns the explorer guarantees, whatever the source dataset looked like.
CONDITION = "condition"
EXPERIMENT = "experiment"
PLATE = "plate"
WELL = "well"
GENE = "gene"
GUIDE = "guide"
DRUG = "drug"
CONCENTRATION = "concentration"
MOA = "moa"
PATHWAY = "pathway"
ROLE = "role"
CONTROL_KIND = "control_kind"
PERTURBATION = "perturbation"
IMAGE_NAME = "image_name"
IMAGE_PATH = "image_path"

# Source columns that may carry the condition string, best first. Datasets
# name it differently depending on which export wrote them.
_CONDITION_SOURCES = ("label", "condition", "gene", "drug", "compound", "treatment")

# Human labels for the colour-by / filter dropdowns.
FIELD_LABELS = {
    CONDITION: "Condition",
    EXPERIMENT: "Experiment arm",
    PLATE: "Plate",
    WELL: "Well",
    GENE: "Gene",
    GUIDE: "Guide / replicate",
    DRUG: "Antibiotic",
    CONCENTRATION: "Concentration",
    MOA: "Mechanism of action",
    PATHWAY: "Pathway (gene target)",
    ROLE: "Control vs treatment",
    CONTROL_KIND: "Control type",
    PERTURBATION: "Perturbation (gene or drug)",
}

# Offered as colour-by options in this order, when the column has values.
COLOUR_FIELDS = (
    CONDITION,
    GENE,
    GUIDE,
    DRUG,
    CONCENTRATION,
    MOA,
    PATHWAY,
    ROLE,
    CONTROL_KIND,
    PERTURBATION,
    EXPERIMENT,
    PLATE,
    WELL,
)

# Offered as filters. Deliberately fewer than the colour fields: filtering by
# well or by individual condition is noise at 30k points.
FILTER_FIELDS = (
    EXPERIMENT,
    ROLE,
    GENE,
    DRUG,
    MOA,
    PATHWAY,
    CONCENTRATION,
    PLATE,
    GUIDE,
)


def field_label(column: str) -> str:
    return FIELD_LABELS.get(column, column.replace("_", " ").title())


@dataclass
class ImageResolver:
    """Turns a metadata row into a path on disk.

    Backed by :mod:`plato.data.image_lookup`, which indexes the file names
    actually present under a root and matches rows to them by name, using the
    row's other columns only to break ties. That replaced a list of guessed
    path templates keyed on fixed column names, which broke on every new
    export: the four exports on this machine carry four different schemas
    (``plate``, ``plate_timepoint``, ``source``, ``experiment``) and their
    screens are arranged four different ways on disk. Nothing here assumes
    either.
    """

    root: Path
    index: ImageIndex
    name_column: str = IMAGE_NAME
    hints: list[str] = field(default_factory=list)
    report: LookupReport | None = None

    @property
    def roots(self) -> list[Path]:
        """Every folder this resolver actually scanned, primary first."""
        return self.index.roots

    @classmethod
    def detect(
        cls, root: Path, frame: pd.DataFrame, *, samples: int = 24
    ) -> ImageResolver | None:
        """Index ``root`` and keep it if it resolves any of ``frame``'s rows."""
        resolver = cls.for_root(root, frame, samples=samples)
        return resolver if resolver.report and resolver.report.ok else None

    @classmethod
    def for_root(
        cls, root: Path, frame: pd.DataFrame, *, samples: int = 24
    ) -> ImageResolver:
        """Index ``root`` and report how well it matches, match or not.

        Unlike :meth:`detect` this always returns a resolver, so a caller can
        explain *why* a folder did not work instead of only that it did not.
        """
        root = Path(root)
        if frame is None or frame.empty or IMAGE_NAME not in frame.columns:
            return cls(root=root, index=ImageIndex(root=root))
        index = ImageIndex.build(root)
        hints = hint_columns(frame, exclude=(IMAGE_NAME,))
        report = probe(
            index, frame, name_column=IMAGE_NAME, hints=hints, samples=samples
        )
        return cls(root=root, index=index, hints=hints, report=report)

    def combined_with(self, other: ImageResolver, frame: pd.DataFrame, *, samples: int = 24) -> ImageResolver:
        """A resolver covering this one's roots plus ``other``'s roots.

        Used when one folder only holds part of a dataset's images (an arm,
        a plate, an incremental delivery) -- adding a second folder should
        resolve more rows, not replace the first folder's matches.
        """
        index = self.index.merged_with(other.index)
        hints = hint_columns(frame, exclude=(IMAGE_NAME,)) if frame is not None else self.hints
        report = (
            probe(index, frame, name_column=self.name_column, hints=hints, samples=samples)
            if frame is not None and not frame.empty
            else self.report
        )
        return ImageResolver(
            root=self.root, index=index, name_column=self.name_column, hints=hints, report=report
        )

    def path_for(self, row) -> Path | None:
        """The image for one row, or None if it is not under this root."""
        stem = str(row.get(self.name_column) or "").strip()
        if not stem:
            return None
        return self.index.resolve(stem, [str(row.get(c, "")) for c in self.hints])


def _first_present(frame: pd.DataFrame, names) -> str | None:
    for name in names:
        if name in frame.columns:
            values = frame[name].astype(str).str.strip()
            if values.ne("").any():
                return name
    return None


def build_frame(
    dataset: EmbeddingDataset,
    *,
    moa_table: AnnotationTable | None = None,
    pathway_table: AnnotationTable | None = None,
    image_root: Path | None = None,
) -> tuple[pd.DataFrame, ImageResolver | None]:
    """One row per embedding, with every colourable field resolved.

    Row order matches ``dataset.vectors`` exactly; the projection's
    ``row_indices`` index into this frame.
    """
    source = dataset.frame
    frame = pd.DataFrame(index=range(len(source)))

    # Carry through the source columns we understand, normalised to our names.
    for column in (EXPERIMENT, PLATE, WELL, IMAGE_NAME):
        frame[column] = (
            source[column].astype(str).str.strip() if column in source.columns else ""
        )
    # Some exports name the arm differently, or not at all. Tried as a
    # fallback BEFORE parsing, because it may be the only signal available on
    # a file with neither a real experiment column nor parseable conditions.
    had_explicit_experiment = frame[EXPERIMENT].ne("").any()
    if not had_explicit_experiment:
        for candidate in ("arm", "screen", "source"):
            if candidate in source.columns:
                frame[EXPERIMENT] = source[candidate].astype(str).str.strip()
                break

    condition_column = _first_present(source, _CONDITION_SOURCES)
    frame[CONDITION] = (
        source[condition_column].astype(str).str.strip() if condition_column else ""
    )

    parsed = [parse_condition(value) for value in frame[CONDITION]]
    frame[GENE] = [c.gene for c in parsed]
    frame[GUIDE] = [c.guide for c in parsed]
    frame[DRUG] = [c.drug for c in parsed]
    frame[CONCENTRATION] = [c.concentration for c in parsed]
    frame[ROLE] = [c.role for c in parsed]
    frame[CONTROL_KIND] = [c.control_kind for c in parsed]
    frame[PERTURBATION] = [c.perturbation for c in parsed]

    # Whether each row is a genetic (CRISPRi) or chemical (antibiotic)
    # perturbation is exactly what "which arm" means, and it is now known
    # precisely from parsing -- more reliably than guessing from whichever
    # column happens to be named "source"/"arm"/"screen", which on
    # Dec25Apr26 CRISPRi & ABx is a three-way split (control/drug/mutant)
    # that does NOT line up with the two-way CRISPRi/ABx arm split: its
    # "control" bucket holds BOTH CRISPRi roles (non-targeting AND induced
    # gene), so colouring by that "arm" mixed a CRISPRi sub-role into what
    # looked like a clean two-way comparison. Only used when there was no
    # explicit experiment column to begin with, so a real one (Aug26 CRISPRi
    # & ABx already has "ABx"/"CRISPRi" directly) is never overridden by a
    # guess.
    if not had_explicit_experiment:
        has_gene = pd.Series([bool(c.gene) for c in parsed])
        has_drug = pd.Series([bool(c.drug) for c in parsed])
        if (has_gene | has_drug).any():
            derived = pd.Series([""] * len(parsed), dtype=object)
            derived[has_gene] = "CRISPRi"
            derived[~has_gene & has_drug] = "ABx"
            # Keep the raw fallback wherever parsing found neither -- an
            # unparsed label is still worth grouping by its own column value
            # rather than being blanked out.
            still_unknown = ~has_gene & ~has_drug
            if still_unknown.any():
                derived[still_unknown.to_numpy()] = frame[EXPERIMENT][
                    still_unknown.to_numpy()
                ]
            frame[EXPERIMENT] = derived.to_numpy()

    # MoA annotates drugs; pathway annotates gene targets. Both are external
    # tables -- absent one, the column is present but entirely unannotated,
    # which is what keeps the dropdown honest rather than empty.
    moa = moa_table or AnnotationTable.empty()
    pathway = pathway_table or AnnotationTable.empty()
    frame[MOA] = [moa.get(c.drug) if c.drug else UNANNOTATED for c in parsed]
    frame[PATHWAY] = [pathway.get(c.gene) if c.gene else UNANNOTATED for c in parsed]

    resolver = ImageResolver.detect(image_root, frame) if image_root else None
    return frame, resolver


def colour_fields(frame: pd.DataFrame) -> list[str]:
    """Colour-by options that actually have more than one value here."""
    out = []
    for column in COLOUR_FIELDS:
        if column not in frame.columns:
            continue
        values = frame[column].astype(str)
        distinct = values[values.ne("")].nunique()
        if distinct > 1 or (distinct == 1 and column in (ROLE, EXPERIMENT)):
            out.append(column)
    return out


def filter_fields(frame: pd.DataFrame, *, max_values: int = 120) -> list[str]:
    """Filter options: present, multi-valued, and not absurdly high-cardinality."""
    out = []
    for column in FILTER_FIELDS:
        if column not in frame.columns:
            continue
        values = frame[column].astype(str)
        distinct = values[values.ne("")].nunique()
        if 1 < distinct <= max_values:
            out.append(column)
    return out


def categorical_fields(
    frame: pd.DataFrame, *, max_values: int = 60, min_values: int = 2
) -> list[str]:
    """Every column that behaves like a category, known or not.

    ``filter_fields`` deliberately offers a curated shortlist, because a
    filter panel with one box per column is unusable. Grouping and shape are
    different: the question "split by whatever this dataset happens to carry"
    is a legitimate one, and a dataset with its own columns -- a timepoint, a
    donor, an instrument -- should be groupable by them without this module
    having heard of them.

    So the known fields come first, in their established display order, then
    anything else the frame carries that has a usable number of distinct
    values. Identifier-like columns (a distinct value per row) and constants
    are excluded, since neither makes a group.
    """
    ordered: list[str] = []
    seen: set[str] = set()

    def offer(column: str) -> None:
        if column in seen or column not in frame.columns:
            return
        if column in (IMAGE_NAME, IMAGE_PATH):
            return
        values = frame[column].astype(str)
        distinct = values[values.ne("")].nunique()
        if min_values <= distinct <= max_values:
            ordered.append(column)
        seen.add(column)

    for column in FILTER_FIELDS:
        offer(column)
    for column in FIELD_LABELS:
        offer(column)
    for column in frame.columns:
        offer(str(column))
    return ordered


def distinct_values(frame: pd.DataFrame, column: str) -> list[str]:
    """Sorted distinct values, with concentrations ordered by magnitude."""
    if column not in frame.columns:
        return []
    values = sorted({v for v in frame[column].astype(str) if v})
    if column == CONCENTRATION:
        from ..ordering import sort_series

        return sort_series(values)
    return values
