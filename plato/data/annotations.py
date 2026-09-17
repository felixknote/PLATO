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

# A second CRISPRi naming convention, seen on Dec25Apr26 CRISPRi & ABx:
# "<gene>[_NC[_<replicate>]]_{plus,minus}ATC", e.g. "ACE-1_NC_1_minusATC",
# "MG1655_plusATC", "rpsA_NC_3_plusATC" -- underscores throughout rather than
# the space before "NC" that GENE_GUIDE_RE's exports use, and an explicit
# induction state (whether the CRISPRi system was induced with anhydrotetra-
# cycline) rather than a bare replicate number.
#
# Tried BEFORE GENE_GUIDE_RE's fallthrough matters here: without this,
# "ACE-1_NC_1_minusATC" matched neither existing pattern and fell all the way
# to parse_condition's last resort, which treats an unmatched label as a
# DRUG -- so every one of these 28 real labels was being coloured, filtered
# and grouped as if it were an antibiotic, not a CRISPRi condition.
ATC_INDUCTION_RE = re.compile(
    r"^(?P<gene>[A-Za-z0-9-]+?)(?:_(?P<nc>NC)(?:_(?P<replicate>\d+))?)?"
    r"_(?P<state>plus|minus)ATC$"
)

# ABx condition: "<drug> <dose>x" or "<drug>_<dose>x", e.g. "Ciprofloxacin 1x"
# (space, most exports) and "Avibactam_0.25x" (underscore, 2026_04_ABx). The
# drug group is non-greedy up to the LAST space-or-underscore before the dose
# token, so two-word drug names keep their own internal space either way --
# "Polymyxin B_0.25x" -> drug "Polymyxin B", not "Polymyxin" with "B" folded
# into a garbled dose. Before this only the space form matched, so every
# underscore-separated label fell through unparsed: the whole string became
# the "drug" (89 near-duplicate values instead of ~22 real ones), dose was
# empty, and MoA lookup failed since it is keyed on the bare drug name.
DRUG_DOSE_RE = re.compile(r"^(?P<drug>.+?)[\s_]+(?P<dose>[\d.]+x)$")

# A third ABx naming convention, seen on 2026_07_ABx: "<drug>/<dose>x", dose
# itself sometimes a fraction -- "Avibactam/1/2x", "Avibactam/1/8x",
# "Avibactam/1x". Not folded into DRUG_DOSE_RE above: that one already
# treats "/" as an ordinary drug-name character (none of the real drug names
# contain one, so this is unambiguous), and a fractional dose needs its own
# optional second "/<denominator>" group DRUG_DOSE_RE has no equivalent of.
# ordering.numeric_part already parses "1/8x" as a fraction (0.125) for
# sorting, so the extracted dose string needs no further conversion here.
DRUG_SLASH_DOSE_RE = re.compile(r"^(?P<drug>[^/]+)/(?P<dose>[\d.]+(?:/[\d.]+)?x)$")

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
    # 2026_04_ABx's own name for its DMSO vehicle control (confirmed
    # directly, not inferred from the label alone -- "drug_control" reads
    # generically and other exports could plausibly mean something else by
    # it). Before this it fell through to the last-resort branch, was
    # treated as a real drug named "drug_control", and its 1,008 rows
    # (8.3% of the dataset) showed up as their own fake antibiotic in the
    # legend, unannotated in MoA, rather than as the DMSO control they are.
    "drug_control": "Vehicle (DMSO)",
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

    match = ATC_INDUCTION_RE.match(text)
    if match:
        gene = _clean(match.group("gene"))
        is_nc = match.group("nc") is not None
        replicate = match.group("replicate")
        state = match.group("state")
        # CONTROL_GENES is keyed on "<gene> nc" (a space, from the export
        # that dict was built against); this convention never writes that
        # combined form, so a non-targeting guide here is recognised by the
        # NC marker THIS regex captured, not by a dictionary lookup.
        kind = f"{gene} (non-targeting)" if is_nc else ""
        # Guide records everything that distinguishes this row from another
        # row of the same gene: whether it is the non-targeting control,
        # which replicate, and the induction state -- e.g. "NC_1_plusATC" or
        # just "plusATC" for a bare gene with no replicate.
        guide_bits = [part for part in ("NC" if is_nc else None, replicate) if part]
        guide_bits.append(f"{state}ATC")
        return Condition(
            label=text,
            gene=gene,
            guide="_".join(guide_bits),
            role=CONTROL if is_nc else TREATMENT,
            control_kind=kind,
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

    match = DRUG_SLASH_DOSE_RE.match(text)
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

    def merged_with(self, other: AnnotationTable) -> AnnotationTable:
        """This table's entries, plus anything ``other`` has that this does not.

        ``add`` is first-writer-wins (``setdefault``), so calling this on the
        table you want to take PRIORITY is what makes it a layer rather than
        a replacement: a PLATO-local drug_moa.csv correcting or extending a
        few drugs still falls back to AI4AB's validated table for everything
        it does not mention, instead of silently losing the rest the moment
        the local file exists at all.
        """
        merged = AnnotationTable(source=self.source)
        merged._by_key.update(self._by_key)
        for key, value in other._by_key.items():
            merged._by_key.setdefault(key, value)
        return merged


# Where to look for annotation tables when the user has not chosen a file.
#
# Listed in PRIORITY order, highest first, and every one that exists is
# layered together (see load_default_moa) rather than only the first found --
# annotations/drug_moa.csv exists so a drug missing from, or wrong in, the
# lab's AI4AB table can be corrected or added locally without touching that
# table or losing everything else it already has right. The AI4AB tables
# themselves are the lab's own validated data, from the sibling project that
# sits beside PLATO on this machine; reusing them rather than shipping a copy
# means correcting a drug's MoA there fixes it here too.
#
# The pathway candidate is a template shipped with PLATO. It ships filled in
# for the CRISPRi genes this project actually screens (see the file's own
# header for provenance); a gene not in it simply reads as unannotated
# rather than something being invented for it.
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


def _find_all(search_roots: list[Path], candidates) -> list[Path]:
    """Every candidate that exists, in priority order, first root first.

    Distinct from ``_find``: that stops at the first match, which is right
    for the pathway table (one template, nothing to layer) but wrong for MoA,
    where a PLATO-local correction and the lab's validated table are meant to
    combine rather than have one hide the other.
    """
    found = []
    for root in search_roots:
        for relative in candidates:
            candidate = (root / relative).resolve()
            if candidate.exists():
                found.append(candidate)
    return found


def find_default_moa(search_roots: list[Path]) -> Path | None:
    """The single highest-priority MoA file, if any exists.

    Kept for callers that only want one file (or a path to show the user);
    ``load_default_moa`` is what the explorer actually loads from, since a
    single "first found" path cannot express layering several sources.
    """
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


def load_default_moa(search_roots: list[Path]) -> AnnotationTable:
    """Every MoA source that exists, layered highest-priority first.

    A drug named in more than one source keeps the value from whichever
    source is earlier in DEFAULT_MOA_CANDIDATES -- a PLATO-local
    drug_moa.csv wins over AI4AB's table for a drug it names, and AI4AB's
    table is still consulted for every drug the local file does not
    mention. One malformed file degrades to being skipped, same as
    load_or_empty, rather than losing every other source along with it.
    """
    table = AnnotationTable.empty()
    for path in reversed(_find_all(search_roots, DEFAULT_MOA_CANDIDATES)):
        # Lowest priority first, so each merged_with layers a HIGHER-priority
        # source on top -- merged_with keeps its own caller's entries and
        # only fills gaps from the argument, so the last merge (the
        # highest-priority file) has to be the one calling it.
        loaded = load_or_empty(path)
        table = loaded.merged_with(table)
    return table
