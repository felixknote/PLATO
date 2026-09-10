"""Project several embeddings together, as one UMAP/t-SNE run over their union.

Colouring by dataset only tells you where each screen's points landed WHEN
EACH WAS PROJECTED SEPARATELY -- two independently-fit UMAPs have no shared
coordinate system, so a CRISPRi point and an antibiotic point sitting close
together on screen is coincidence, not similarity. The only way position
means the same thing across datasets is to project them together: concatenate
the vectors, run one fit, and every point -- whichever dataset it came from --
is now embedded relative to every other point in the same fit.

This module does exactly that concatenation and nothing else. The fit itself
is ``plato.data.projection.project``, unchanged: it already takes a plain
``(n, d)`` array and does not need to know or care that its rows came from
more than one source.

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


def build_joint_dataset(entries: list[EmbeddingEntry]) -> JointDataset:
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

    return JointDataset(
        vectors=vectors,
        frame=frame,
        sources=sources,
        fingerprint=_joint_fingerprint(entries),
    )


def make_joint_entry(entries: list[EmbeddingEntry], *, name: str = "") -> EmbeddingEntry:
    """Build a real ``EmbeddingEntry`` from a joint fit of ``entries``.

    Wraps the concatenated vectors/frame in an ``EmbeddingDataset`` so this
    drops straight into the ``Workspace`` and every downstream mechanism --
    switching, per-parameter projection caching, selection, colouring,
    filtering -- works completely unchanged: they all operate on
    ``EmbeddingEntry`` and know nothing about where its vectors came from.

    ``directory`` is a synthetic, non-existent path built from the source
    entries' own directories: it has to be something, since EmbeddingDataset
    requires one, but a joint entry is never re-loaded from disk so nothing
    reads it as real.
    """
    joint = build_joint_dataset(entries)
    label = name or " + ".join(e.name for e in entries)
    directory = Path("<joint>") / "+".join(sorted(e.key for e in entries))

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
        },
    )
    return entry


def _joint_fingerprint(entries: list[EmbeddingEntry]) -> str:
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
    return digest.hexdigest()[:16]
