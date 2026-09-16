"""Project several embeddings together, as one UMAP/t-SNE run over their union.

Colouring by dataset only tells you where each screen's points landed WHEN
EACH WAS PROJECTED SEPARATELY -- two independently-fit UMAPs have no shared
coordinate system, so a CRISPRi point and an antibiotic point sitting close
together on screen is coincidence, not similarity. The only way position
means the same thing across datasets is to project them together: concatenate
the vectors, run one fit, and every point -- whichever dataset it came from --
is now embedded relative to every other point in the same fit.

This module does that concatenation, and optionally removes the offset
between the arms first -- see ``align_vectors``. The fit itself is
``plato.data.projection.project``, unchanged: it already takes a plain
``(n, d)`` array and does not need to know or care that its rows came from
more than one source.

**Seeing the arms separate is usually the point, not a problem.** Datasets
imaged months apart differ in illumination, staining and focus, so every
point of an arm carries the same offset and the fit separates them cleanly.
At this stage -- looking for confounders before anything is trained -- that
separation is the measurement: it is what a batch effect LOOKS like, and a
joint fit is how you see its size relative to everything else in the data.
Frozen DINO features are used precisely because they report it undisguised.

So the raw concatenation is the default and the normal way to work. What the
picture cannot do on its own is tell a batch offset from a genuine global
difference between the screens, since both put the arms in different places.
``align_vectors`` is the second look for that one question: centring removes
the constant shift, so structure that survives it was not the shift. It is
a diagnostic to reach for deliberately, not a correction to apply first --
run on data whose arms genuinely differ, it erases the evidence.

**The one real constraint**: every entry combined must share the same vector
space -- same dimensionality, same model, same preprocessing. Concatenating a
1024-d DINO export with a 31-d hand-computed descriptor set would not fail
loudly; it would just produce a meaningless fit where 31 padded-or-truncated
dimensions dominate or starve the comparison. So this module checks
dimensionality up front and refuses the combination instead of guessing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .embeddings import EmbeddingDataset
from .workspace import (
    DATASET_COLUMN,
    EMBEDDING_COLUMN,
    SOURCE_JOINT,
    EmbeddingEntry,
)


# Whether to remove the offset between datasets before the fit.
#
# Raw concatenation (ALIGN_NONE) is the default and the normal way to work.
# A screen run in April and one run in August differ in illumination,
# staining and focus; a deep encoder reports that faithfully as an additive
# shift shared by every point of an arm, and the fit lays the arms out apart
# because they ARE apart. Seeing that, and how large it is next to the
# biological structure, is the job at this stage.
#
# Centring is the follow-up question, not the starting point: "does anything
# survive removing the shift". Measured on synthetic arms carrying the same
# two biological classes plus a per-arm offset (silhouette over the first two
# PCs, higher = more separated by that variable):
#
#     raw concatenation    dataset 0.56   biology 0.55
#     L2 normalised        dataset 0.51   biology 0.59
#     per-arm centred      dataset 0.00   biology 0.91
#
# L2 normalisation -- which the cosine path already applies -- barely touches
# it, because the shift survives projection onto the sphere.
ALIGN_NONE = "none"
ALIGN_CENTRE = "centre"
ALIGN_ZSCORE = "zscore"

ALIGN_LABELS = {
    ALIGN_NONE: "None — keep the offset (default)",
    ALIGN_CENTRE: "Centre each dataset",
    ALIGN_ZSCORE: "Centre and scale each dataset",
}


class IncompatibleEmbeddings(ValueError):
    """Raised when the chosen entries cannot be projected together.

    Carries enough detail to explain WHY, since "cannot combine" alone would
    leave the user guessing which pair is the problem.
    """


@dataclass(slots=True)
class JointSource:
    """One entry's contribution to a joint projection: its span of rows.

    ``start``/``stop`` are positions in the concatenated vector array (and
    therefore in the combined frame, which is built in the same order), so a
    joint result's row can always be traced back to which entry and which of
    that entry's own rows it came from.
    """

    entry_key: str
    name: str
    start: int
    stop: int

    def contains(self, position: int) -> bool:
        return self.start <= position < self.stop

    def local_index(self, position: int) -> int:
        """This entry's own row index for a position in the joint array."""
        return position - self.start


def align_vectors(vectors: np.ndarray, sources: list[JointSource], mode: str) -> np.ndarray:
    """Remove the between-dataset offset from a concatenated array.

    Centring subtracts each arm's own mean, so the arms share an origin and
    only within-arm structure is left for the fit to find. It removes exactly
    one thing -- the constant shift -- and preserves every relative distance
    inside an arm.

    ``ALIGN_ZSCORE`` additionally divides by each arm's per-feature standard
    deviation, for the case where one dataset is not merely shifted but more
    variable overall (a noisier imaging session). It is the stronger claim of
    the two: it asserts the arms *should* have equal spread, which is wrong
    if a real treatment effect is what widens one of them.

    **This is a deliberate distortion, not a correction.** Centring cannot
    tell a batch offset from a genuine global difference between the two
    populations, and will erase the second as readily as the first. If the
    claim IS "these two screens differ", aligning removes the evidence for
    it -- and when the screens were imaged apart, that difference is exactly
    what you came to look at.

    So ``ALIGN_NONE`` is the default and stays the normal view. Use this to
    ask one follow-up question -- "what is left once the shift is gone" --
    and read the answer as a second picture alongside the first, never as a
    cleaned-up replacement for it.
    """
    if mode == ALIGN_NONE or not sources:
        return vectors

    out = np.array(vectors, dtype=np.float32, copy=True)
    for source in sources:
        block = out[source.start : source.stop]
        if block.size == 0:
            continue
        block -= block.mean(axis=0, keepdims=True)
        if mode == ALIGN_ZSCORE:
            spread = block.std(axis=0, keepdims=True)
            # A constant feature has no spread to normalise; dividing would
            # turn 0/0 into NaN and poison the whole fit.
            np.divide(block, spread, out=block, where=spread > 1e-6)
    return out


@dataclass(slots=True)
class JointDataset:
    """The concatenation of several entries' vectors and metadata.

    Built once, then handed to ``plato.data.projection.project`` exactly as
    a single dataset's vectors would be -- the fit has no notion of "joint",
    it is simply given a bigger array.
    """

    vectors: np.ndarray  # (sum of n_i, d) float32
    frame: pd.DataFrame  # combined metadata, EMBEDDING_COLUMN/DATASET_COLUMN added
    sources: list[JointSource]
    fingerprint: str
    align: str = ALIGN_NONE

    @property
    def n_points(self) -> int:
        return int(self.vectors.shape[0])

    def source_for(self, position: int) -> JointSource | None:
        """Which entry a row in the combined array/frame came from."""
        for source in self.sources:
            if source.contains(position):
                return source
        return None


def check_compatible(entries: list[EmbeddingEntry]) -> None:
    """Raise :class:`IncompatibleEmbeddings` if these cannot be combined.

    Dimensionality is the load-bearing check: a joint UMAP/t-SNE fit needs
    every row in the same vector space, and mismatched dimensionality is the
    one incompatibility that cannot be worked around (unlike, say, differing
    row counts, which concatenation handles trivially).
    """
    if len(entries) < 2:
        raise IncompatibleEmbeddings("Choose at least two embeddings to combine.")

    dims = {e.n_dimensions for e in entries}
    if len(dims) > 1:
        detail = ", ".join(f"{e.name} ({e.n_dimensions}-d)" for e in entries)
        raise IncompatibleEmbeddings(
            "These embeddings do not share a vector space, so combining them "
            "would not be a meaningful comparison -- their dimensionality "
            f"differs: {detail}.\n\n"
            "Only embeddings produced by the same model and preprocessing "
            "(matching dimensionality) can be projected together."
        )


def _disambiguate_shared_plate_names(frame: pd.DataFrame) -> None:
    """Prefix ``plate`` with its dataset wherever the SAME plate name is
    written by more than one source, in place.

    Real screens on this machine are not consistently namespaced: four of
    six real DINO exports write a bare "P1".."P6"-ish plate column, and two
    write "ABx_P1"/"CRISPRi_P1" -- so "P1" alone names FOUR unrelated
    physical plates from four different runs once they are concatenated.
    Faceting or grouping by plate on a joint entry with no fix would (and
    did) silently pool points from unrelated plates into one facet, and any
    custom grouping built over it inherited the same collision.

    Deliberately conditional: a joint entry whose sources never share a
    plate name (the common case, when every export already namespaces its
    own plate column, or the plates genuinely do not collide) is left with
    its plain "P1"/"P2" values -- there is nothing here to disambiguate, and
    rewriting it anyway would make every existing joint entry's plate labels
    longer for no reason.
    """
    from .explorer_model import PLATE

    if PLATE not in frame.columns or DATASET_COLUMN not in frame.columns:
        return
    plates = frame[PLATE].astype(str)
    non_empty = plates.ne("")
    # A plate name is ambiguous if it is written by more than one dataset --
    # checked on the (dataset, plate) pair count vs the plate's own count,
    # not on raw row count, since one dataset legitimately has many rows per
    # plate.
    pairs = frame.loc[non_empty, [DATASET_COLUMN, PLATE]].drop_duplicates()
    ambiguous = set(pairs[PLATE][pairs.duplicated(subset=PLATE, keep=False)])
    if not ambiguous:
        return
    mask = non_empty & plates.isin(ambiguous)
    frame.loc[mask, PLATE] = (
        frame.loc[mask, DATASET_COLUMN].astype(str) + " · " + plates[mask]
    )


def build_joint_dataset(
    entries: list[EmbeddingEntry], *, align: str = ALIGN_NONE
) -> JointDataset:
    """Concatenate ``entries`` into one vector array and one metadata frame.

    Raises :class:`IncompatibleEmbeddings` via :func:`check_compatible` first.
    Row order is entry order, then each entry's own row order -- the same
    order the vectors and the frame are built in, so a position in one always
    matches the same position in the other.
    """
    check_compatible(entries)

    vector_parts: list[np.ndarray] = []
    frame_parts: list[pd.DataFrame] = []
    sources: list[JointSource] = []
    cursor = 0
    for entry in entries:
        vector_parts.append(np.ascontiguousarray(entry.vectors, dtype=np.float32))
        part = entry.frame.copy()
        part[EMBEDDING_COLUMN] = entry.label()
        part[DATASET_COLUMN] = entry.dataset.name
        frame_parts.append(part)

        n = entry.n_points
        sources.append(JointSource(entry.key, entry.name, cursor, cursor + n))
        cursor += n

    vectors = np.concatenate(vector_parts, axis=0)
    frame = pd.concat(frame_parts, ignore_index=True)
    _disambiguate_shared_plate_names(frame)
    # After concatenation, before the fit: alignment needs the arms in one
    # array to know where each starts, and the fit must never see the
    # unaligned version.
    vectors = align_vectors(vectors, sources, align)

    return JointDataset(
        vectors=vectors,
        frame=frame,
        sources=sources,
        align=align,
        # Part of the fingerprint, so an aligned and an unaligned combination
        # of the same entries are different cache entries. Without it the
        # second would silently restore the first's cached layout.
        fingerprint=_joint_fingerprint(entries, align),
    )


def make_joint_entry(
    entries: list[EmbeddingEntry], *, name: str = "", align: str = ALIGN_NONE
) -> EmbeddingEntry:
    """Build a real ``EmbeddingEntry`` from a joint fit of ``entries``.

    Wraps the concatenated vectors/frame in an ``EmbeddingDataset`` so this
    drops straight into the ``Workspace`` and every downstream mechanism --
    switching, per-parameter projection caching, selection, colouring,
    filtering -- works completely unchanged: they all operate on
    ``EmbeddingEntry`` and know nothing about where its vectors came from.

    ``directory`` is a synthetic, non-existent path built from the source
    entries' own SOURCE DATASET directories (never from ``entry.key``, which
    is a random id assigned fresh every time an entry is loaded -- keying on
    it would make the same combination of the same datasets get a different
    synthetic directory every session). A joint entry is never re-loaded from
    disk, so nothing reads this as a real path, but ResolverStore does use it
    as a persistence key: combining the same datasets a second time, even in
    a later session, must resolve to the same remembered answer, or "locate
    the images once" silently never applies to any joint entry at all.
    """
    joint = build_joint_dataset(entries, align=align)
    label = name or " + ".join(e.name for e in entries)
    if align != ALIGN_NONE:
        # In the name itself, not only in info: the label is what reaches the
        # open-embeddings list, the window and the export headline, and an
        # aligned fit that presents as a plain one is a figure that misstates
        # what was done to the data.
        label = f"{label} [{'centred' if align == ALIGN_CENTRE else 'centred+scaled'}]"
    directory = Path("<joint>") / "+".join(sorted(str(e.dataset.directory) for e in entries))

    dataset = EmbeddingDataset(
        name=label,
        directory=directory,
        vectors=joint.vectors,
        frame=joint.frame,
        run_info={},
    )
    entry = EmbeddingEntry(
        name=label,
        dataset=dataset,
        frame=joint.frame,
        source=SOURCE_JOINT,
        info={
            "source_names": [e.name for e in entries],
            "source_keys": [e.key for e in entries],
            "source_counts": [e.n_points for e in entries],
            "align": align,
        },
    )
    return entry


def _joint_fingerprint(entries: list[EmbeddingEntry], align: str = ALIGN_NONE) -> str:
    """A fingerprint over the ordered set of entries.

    Order matters -- {A, B} concatenated is not the same array as {B, A} --
    so the fingerprint is order-sensitive, unlike a set hash. Built from each
    entry's own fingerprint rather than rehashing the vectors, since that
    fingerprint already exists and is itself cheap (a strided sample, not the
    whole array; see EmbeddingDataset.fingerprint).
    """
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry.key.encode())
        digest.update(entry.fingerprint().encode())
    digest.update(align.encode())
    return digest.hexdigest()[:16]
