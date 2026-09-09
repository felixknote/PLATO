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

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .annotations import UNANNOTATED, AnnotationTable, parse_condition
from .embeddings import EmbeddingDataset

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


@dataclass(slots=True)
class ImageResolver:
    """Turns a metadata row into a path on disk.

    The layout that holds for this screen is ``<root>/<plate>/<image_name>.tiff``
    (verified against the real export). Rather than hard-coding that, the
    resolver tries a short list of layouts once, on a sample of real rows, and
    keeps whichever actually resolves -- so a differently organised dataset
    works without a code change, and one that resolves nothing says so instead
    of silently producing dead paths.
    """

    root: Path
    template: str = ""
    suffix: str = ".tiff"

    # Layouts tried, in order. Placeholders come from the metadata row:
    #   {plate}       the plate column verbatim, e.g. "CRISPRi_P1"
    #   {arm}         everything before the last "_", e.g. "CRISPRi"
    #   {plate_tail}  everything after it, e.g. "P1"
    #
    # The arm/tail split matters because the same screen is organised two ways
    # in practice: one folder per plate on the share ("CRISPRi_P1/"), and a
    # folder per arm containing plates on a local copy ("CRISPRi/P1/"). A
    # resolver that only knew {plate} found nothing in the second and told the
    # user their images were missing when they were not.
    TEMPLATES = (
        "{plate}/{name}",
        "{arm}/{plate_tail}/{name}",
        "{plate_tail}/{name}",
        "{name}",
        "{plate}/images/{name}",
        "images/{plate}/{name}",
        "{arm}/{plate_tail}/images/{name}",
    )
    SUFFIXES = (".tiff", ".tif", "")

    @classmethod
    def detect(cls, root: Path, frame: pd.DataFrame, *, samples: int = 12) -> ImageResolver | None:
        """Find the layout that resolves the most sampled rows, or None."""
        root = Path(root)
        if not root.is_dir() or frame.empty or IMAGE_NAME not in frame.columns:
            return None
        probe = frame.head(samples * 40).sample(
            n=min(samples, len(frame)), random_state=0
        )
        best: tuple[int, str, str] | None = None
        for template in cls.TEMPLATES:
            for suffix in cls.SUFFIXES:
                resolver = cls(root=root, template=template, suffix=suffix)
                hits = sum(
                    1
                    for _, row in probe.iterrows()
                    if resolver.path_for(row) is not None
                )
                if hits and (best is None or hits > best[0]):
                    best = (hits, template, suffix)
                if best is not None and best[0] == len(probe):
                    break
        if best is None:
            return None
        return cls(root=root, template=best[1], suffix=best[2])

    def path_for(self, row) -> Path | None:
        """The image for one row, or None if it is not on disk."""
        name = str(row.get(IMAGE_NAME) or "").strip()
        if not name:
            return None
        plate = str(row.get(PLATE) or "").strip()
        arm, _, tail = plate.rpartition("_")
        relative = self.template.format(
            plate=plate,
            name=name,
            # With no "_" in the plate name, rpartition puts everything in the
            # tail; falling back to the whole plate keeps {arm} meaningful
            # rather than empty.
            arm=arm or plate,
            plate_tail=tail or plate,
        )
        candidate = self.root / f"{relative}{self.suffix}"
        return candidate if candidate.exists() else None


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
    # Some exports name the arm differently, or not at all.
    if frame[EXPERIMENT].eq("").all():
        for candidate in ("source", "arm", "screen"):
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


def distinct_values(frame: pd.DataFrame, column: str) -> list[str]:
    """Sorted distinct values, with concentrations ordered by magnitude."""
    if column not in frame.columns:
        return []
    values = sorted({v for v in frame[column].astype(str) if v})
    if column == CONCENTRATION:
        from ..ordering import sort_series

        return sort_series(values)
    return values
