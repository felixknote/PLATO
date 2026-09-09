"""Data layer: everything that knows about files, indexes and embeddings, and
nothing that knows about Qt.

``session`` unions the image indexes of several loaded plates; ``embeddings``
does the same job for precomputed feature vectors and their projections;
``annotations`` maps conditions to the biology (MoA, pathway) they belong to.
The views layer composes these; none of them import from it.
"""

from .session import DuplicatePlateError, PlateSession, Session

__all__ = ["DuplicatePlateError", "PlateSession", "Session"]
