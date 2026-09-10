"""Condition strings -> the biology they stand for.

Two separate jobs, kept apart because one is derivable and the other is not:

*Parsing* a condition label into its parts is derivable from the data. The
plate maps pack everything into one string -- ``ftsZ_2`` is gene + guide,
``Ciprofloxacin 1x`` is drug + dose -- so gene, guide, drug, concentration and
control-vs-treatment all come out of the label itself, with no external table.

*Annotating* a gene with a pathway or a drug with a mechanism of action is
NOT derivable. Nothing in the dataset records it: the plate maps carry only
the condition string, and there is no MoA column anywhere under Z:\\Data.
That mapping is therefore loaded from a JSON/CSV file supplied by the user,
and anything the file does not cover reads as UNANNOTATED rather than being
guessed. Colouring a figure by an invented MoA is how a plotting convenience
becomes a wrong result in a thesis.

The default annotation source is the lab's own validated table from
AI4AB/analysis/E_coli_params (``moa_dict.json`` / ``moa_dict_inv.json``),
which is why its format is one of the ones understood here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

UNANNOTATED = "unannotated"

# CRISPRi condition: "<gene>_<guide>", e.g. "ftsZ_2", "ACE-1 NC_6".
# The gene group is lazy-anchored to the LAST underscore so control names
# containing spaces and hyphens ("ACE-1 NC") survive intact -- the same
# convention plato.toml already uses for this screen.
GENE_GUIDE_RE = re.compile(r"^(?P<gene>.+)_(?P<guide>\d+)$")

# ABx condition: "<drug> <dose>x", e.g. "Ciprofloxacin 1x", "Penicillin G 0.25x".
# The drug group is greedy up to the final dose token, so two-word drug names
# ("Penicillin G", "Polymyxin B") stay whole.
DRUG_DOSE_RE = re.compile(r"^(?P<drug>.+?)\s+(?P<dose>[\d.]+x)$")

# Condition labels that are controls rather than perturbations. Matched
# case-insensitively against the whole label after collapsing whitespace.
CONTROL_LABELS = {
    "wt": "WT (untreated)",
    # NC is a non-targeting guide, not a solvent: the CRISPRi machinery is
    # present and directed at nothing, which is the control for the effect of
    # knockdown itself. Naming it a vehicle control would put it in the same
    # class as DMSO below, which controls for something else entirely.
    "wt nc": "WT (non-targeting gRNA)",
    "dmso": "Vehicle (DMSO)",
    "water": "Vehicle (water)",
}
# CRISPRi non-targeting control strains, matched against the parsed gene.
CONTROL_GENES = {
    "ace-1 nc": "ACE-1 (non-targeting)",
    "mg1655 nc": "MG1655 (parental)",
}

TREATMENT = "treatment"
CONTROL = "control"


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _fold(value: str) -> str:
    """Comparison key: lowercase, no spaces, no punctuation.

    Lets "Penicillin G", "PenicillinG" and "penicillin_g" all match the same
    annotation entry -- the lab's MoA table writes them the second way and the
    plate maps the first.
    """
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


@dataclass(slots=True)
class Condition:
    """One condition label, split into its parts."""

    label: str
    gene: str = ""
    guide: str = ""
    drug: str = ""
    concentration: str = ""
    role: str = TREATMENT
    control_kind: str = ""

    @property
    def perturbation(self) -> str:
        """Gene or drug, whichever this condition is. The natural grouping
        key when CRISPRi and ABx points share one plot."""
        return self.gene or self.drug or self.label


def parse_condition(label: object) -> Condition:
    """Split a plate-map condition string into gene/guide or drug/dose.

    Unrecognised labels come back with just ``label`` set and no parts, which
    is the honest result: an unparsed condition is still a real condition and
    must stay visible in the plot rather than being dropped.
    """
    text = _clean(label)
    if not text:
        return Condition(label="")

    folded = text.lower()
    if folded in CONTROL_LABELS:
        return Condition(
            label=text,
            drug=text,
            role=CONTROL,
            control_kind=CONTROL_LABELS[folded],
        )

    match = GENE_GUIDE_RE.match(text)
    if match:
        gene = _clean(match.group("gene"))
        kind = CONTROL_GENES.get(gene.lower(), "")
        return Condition(
            label=text,
            gene=gene,
            guide=match.group("guide"),
            role=CONTROL if kind else TREATMENT,
            control_kind=kind,
        )

    match = DRUG_DOSE_RE.match(text)
    if match:
        return Condition(
            label=text,
            drug=_clean(match.group("drug")),
            concentration=match.group("dose"),
        )

    # A bare name with no guide or dose (a gene without replicates, a drug at
    # a single concentration). Treated as a perturbation, not a control --
    # only the explicit control lists above make something a control.
    return Condition(label=text, drug=text)


class AnnotationTable:
    """gene -> pathway and drug -> MoA, loaded from a user-supplied file.

    Understands three shapes, so the lab's existing tables work unchanged:

    * ``{"Drug_1xIC50": "Gyrase", ...}``  (moa_dict.json)
    * ``{"Gyrase": ["Ciprofloxacin", ...], ...}``  (moa_dict_inv.json)
    * a CSV with a name column and a moa/pathway column

    Lookup is whitespace- and case-insensitive, because the same drug is
    written "Penicillin G" in the plate map and "PenicillinG" in the MoA table.
    """

    def __init__(self, mapping: dict[str, str] | None = None, source: Path | None = None) -> None:
        self._by_key: dict[str, str] = {}
        self.source = source
        for name, value in (mapping or {}).items():
            self.add(name, value)

    def add(self, name: object, value: object) -> None:
        key = _fold(_clean(name))
        text = _clean(value)
        if key and text:
            self._by_key.setdefault(key, text)

    def get(self, name: object) -> str:
        """The annotation for ``name``, or UNANNOTATED."""
        return self._by_key.get(_fold(_clean(name)), UNANNOTATED)

    def covers(self, name: object) -> bool:
        return _fold(_clean(name)) in self._by_key

    @property
    def values(self) -> list[str]:
        return sorted(set(self._by_key.values()))

    def __len__(self) -> int:
        return len(self._by_key)

    # -- loading ----------------------------------------------------------

    @classmethod
    def from_json(cls, path: Path) -> AnnotationTable:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        table = cls(source=Path(path))
        if not isinstance(data, dict):
            raise ValueError(f"{path.name}: expected a JSON object")
        for key, value in data.items():
            if isinstance(value, str):
                # moa_dict.json: "Drug_1xIC50" -> "MoA". The dose suffix is
                # stripped so every dose of a drug shares one annotation.
                table.add(key.split("_")[0], value)
            elif isinstance(value, (list, tuple)):
                # moa_dict_inv.json: "MoA" -> [drug, drug, ...]
                for item in value:
                    table.add(item, key)
        return table

    @classmethod
    def from_csv(cls, path: Path) -> AnnotationTable:
        # `comment="#"` so a shipped template can explain itself in the file
        # the user is meant to edit, rather than in documentation they would
        # have to go and find.
        frame = pd.read_csv(path, dtype=str, comment="#", skip_blank_lines=True).fillna("")
        if frame.shape[1] < 2:
            raise ValueError(f"{path.name}: need at least two columns (name, annotation)")
        columns = {str(c).strip().lower(): c for c in frame.columns}
        name_col = next(
            (columns[c] for c in ("gene", "drug", "name", "condition", "compound") if c in columns),
            frame.columns[0],
        )
        value_col = next(
            (columns[c] for c in ("moa", "pathway", "mechanism", "class", "annotation") if c in columns),
            frame.columns[1],
        )
        table = cls(source=Path(path))
        for _, row in frame.iterrows():
            table.add(row[name_col], row[value_col])
        return table

    @classmethod
    def load(cls, path: Path) -> AnnotationTable:
        path = Path(path)
        if path.suffix.lower() == ".json":
            return cls.from_json(path)
        return cls.from_csv(path)

    @classmethod
    def empty(cls) -> AnnotationTable:
        return cls()


# Where to look for annotation tables when the user has not chosen a file.
#
# The MoA candidates are the lab's own validated tables from the AI4AB
# project, which sits beside PLATO on this machine. Reusing them rather than
# shipping a copy means one table stays authoritative: correcting a drug's MoA
# there fixes it here too.
#
# The pathway candidate is a template shipped with PLATO, deliberately blank
# (see annotations/gene_pathway.csv). Nothing is invented if neither is found;
# the column simply reads as unannotated.
DEFAULT_MOA_CANDIDATES = (
    Path("annotations/drug_moa.csv"),
    Path("../AI4AB/analysis/E_coli_params/moa_dict_inv.json"),
    Path("../AI4AB/analysis/E_coli_params/moa_dict.json"),
)

DEFAULT_PATHWAY_CANDIDATES = (
    Path("annotations/gene_pathway.csv"),
)


def _find(search_roots: list[Path], candidates) -> Path | None:
    for root in search_roots:
        for relative in candidates:
            candidate = (root / relative).resolve()
            if candidate.exists():
                return candidate
    return None


def find_default_moa(search_roots: list[Path]) -> Path | None:
    return _find(search_roots, DEFAULT_MOA_CANDIDATES)


def find_default_pathway(search_roots: list[Path]) -> Path | None:
    return _find(search_roots, DEFAULT_PATHWAY_CANDIDATES)


def load_or_empty(path: Path | None) -> AnnotationTable:
    """Load an annotation file, degrading to an empty table on any problem.

    A malformed annotation file must not stop the explorer opening: the plot
    is still meaningful coloured by gene or drug, and an empty table shows
    every point as unannotated, which is visibly wrong rather than subtly so.
    """
    if path is None:
        return AnnotationTable.empty()
    try:
        return AnnotationTable.load(path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return AnnotationTable.empty()
