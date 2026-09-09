"""An embedding dataset built from the plates already loaded in the browser.

Lets the explorer work with no DINO export at all: point PLATO at images and a
plate map, and the projection tab becomes usable immediately, coloured by the
same metadata the grid filters on.

Vectors come from the thumbnail cache rather than the originals. That is the
whole reason this is fast enough to be worth offering -- the cache is local
SQLite holding already-downscaled PNGs, so a plate is a few thousand small
decodes instead of a few thousand 14 MB reads across a share.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .embeddings import EmbeddingDataset
from .image_features import FEATURE_NAMES, describe, standardise

# Name shown in the dataset dropdown for this synthetic dataset.
LOADED_PLATES = "Loaded plates (computed)"


def _decode(png: bytes) -> np.ndarray | None:
    import io

    from PIL import Image

    try:
        with Image.open(io.BytesIO(png)) as image:
            return np.asarray(image.convert("L"), dtype=np.float32)
    except Exception:  # noqa: BLE001 - a corrupt tile must not stop the build
        return None


def build(session, *, progress=None) -> EmbeddingDataset:
    """Describe every cached thumbnail in ``session`` as a feature vector.

    ``progress`` is an optional ``callable(done, total, message)``.

    Raises ``ValueError`` when nothing could be described, naming the likely
    cause -- almost always that thumbnails have not been built yet.
    """
    from ..cache import ThumbnailCache

    rows: list[dict] = []
    vectors: list[np.ndarray] = []
    missing_cache = 0

    plates = list(getattr(session, "plates", []))
    total = sum(plate.db.count() for plate in plates)
    done = 0

    for plate in plates:
        cache_path = Path(plate.cfg.thumb_db_path)
        if not cache_path.exists():
            missing_cache += 1
            continue
        cache = ThumbnailCache(cache_path)
        try:
            for row in plate.db.query():
                done += 1
                if progress and done % 200 == 0:
                    progress(done, total, f"describing images… {done:,}/{total:,}")
                png = cache.get(row.image_id)
                if png is None:
                    continue
                plane = _decode(png)
                if plane is None:
                    continue
                vectors.append(describe(plane))
                rows.append(
                    {
                        "plate": plate.name,
                        "well": row.well,
                        "field": row.field or "",
                        "channel": row.channel or "",
                        "image_name": row.image_id,
                        **{
                            key: ("" if value is None else str(value))
                            for key, value in row.metadata.items()
                        },
                    }
                )
        finally:
            cache.close()

    if not vectors:
        raise ValueError(
            "No cached thumbnails to describe.\n\n"
            + (
                "Build them first with 'plato thumbs', or reload the plate."
                if missing_cache
                else "The loaded plates have an empty thumbnail cache."
            )
        )

    frame = pd.DataFrame(rows).fillna("")
    # A condition column is what the explorer colours by; the plate map's own
    # columns are already in `frame`, so the first metadata column that varies
    # stands in for it when nothing is called "label".
    if "label" not in frame.columns:
        for column in frame.columns:
            if column in {"plate", "well", "field", "channel", "image_name"}:
                continue
            if frame[column].nunique() > 1:
                frame["label"] = frame[column]
                break

    return EmbeddingDataset(
        name=LOADED_PLATES,
        directory=Path(plates[0].cfg.project.work_dir) if plates else Path("."),
        vectors=standardise(np.vstack(vectors)),
        frame=frame,
        run_info={
            "model": f"image descriptors ({len(FEATURE_NAMES)} features)",
            "computed": True,
        },
    )
