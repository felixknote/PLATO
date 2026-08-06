"""Pre-computed thumbnail cache.

Design decisions that matter:

* Thumbnails are built **offline**, never on scroll. Decoding a 16-bit TIFF
  takes tens of milliseconds; doing that during a scroll event is what makes
  naive browsers unusable.
* They live in one SQLite file, not 50k PNGs on disk. On a network share the
  per-file metadata round-trips dominate everything else.
* Intensity scaling is **fixed per channel across the whole screen**, estimated
  once from a random sample. Per-image autoscaling makes a dead well and a
  healthy well look identical, which is exactly the comparison you are trying
  to make by eye.
"""

from __future__ import annotations

import io
import sqlite3
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageFilter

_SCHEMA = """
CREATE TABLE IF NOT EXISTS thumbnails (
    image_id   TEXT PRIMARY KEY,
    mtime      REAL NOT NULL,
    size_bytes INTEGER NOT NULL,
    width      INTEGER NOT NULL,
    height     INTEGER NOT NULL,
    png        BLOB NOT NULL
);
"""


@dataclass(slots=True)
class ThumbnailJob:
    image_id: str
    path: Path
    channel: str | None
    mtime: float
    size_bytes: int


def read_plane(path: Path) -> np.ndarray:
    """Read a TIFF and reduce it to a single 2-D plane.

    Multi-page / multi-channel files are reduced by taking the first plane of
    every leading axis. This is deliberate and lossy: the thumbnail is a
    navigation aid, and the full-resolution viewer shows all planes.
    """
    array = tifffile.imread(path)
    array = np.asarray(array)
    while array.ndim > 2:
        # Collapse leading axes, but prefer a channel axis of small extent last.
        array = array[0]
    return array


def _percentile_limits(
    values: np.ndarray, percentiles: tuple[float, float]
) -> tuple[float, float]:
    lo, hi = np.percentile(values, percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(values)), float(np.max(values))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def estimate_display_limits(
    jobs: Iterable[ThumbnailJob],
    *,
    percentiles: tuple[float, float] = (1.0, 99.5),
    sample_size: int = 200,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Estimate one (lo, hi) pair per channel from a random sample of images."""
    rng = np.random.default_rng(seed)
    by_channel: dict[str, list[ThumbnailJob]] = {}
    for job in jobs:
        by_channel.setdefault(job.channel or "_", []).append(job)

    limits: dict[str, tuple[float, float]] = {}
    for channel, channel_jobs in by_channel.items():
        n = min(sample_size, len(channel_jobs))
        picks = rng.choice(len(channel_jobs), size=n, replace=False)
        samples: list[np.ndarray] = []
        for idx in picks:
            try:
                plane = read_plane(channel_jobs[int(idx)].path)
            except Exception:  # noqa: BLE001 - a broken file must not abort the pass
                continue
            flat = plane.ravel()
            if flat.size > 20_000:
                flat = flat[:: flat.size // 20_000]
            samples.append(flat.astype(np.float32))
        if not samples:
            continue
        limits[channel] = _percentile_limits(np.concatenate(samples), percentiles)
    return limits


def _downscale(plane: np.ndarray, target: int) -> np.ndarray:
    """Integer-factor block mean, then let PIL do the final resize."""
    factor = max(1, min(plane.shape[0] // target, plane.shape[1] // target))
    if factor > 1:
        h = (plane.shape[0] // factor) * factor
        w = (plane.shape[1] // factor) * factor
        plane = (
            plane[:h, :w]
            .reshape(h // factor, factor, w // factor, factor)
            .mean(axis=(1, 3))
        )
    return plane


def scale_to_uint8(plane: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    """Apply fixed (lo, hi) display limits and quantise to 8-bit."""
    lo, hi = limits
    scaled = np.clip((plane.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def render_thumbnail(
    path: Path, limits: tuple[float, float], size: int
) -> tuple[bytes, int, int]:
    """Render one file to PNG bytes using fixed display limits.

    A subtle unsharp mask is applied after downscaling: block-mean downscaling
    softens detail that per-image autoscaling would otherwise recover, and the
    thumbnail grid is the primary navigation view, so a little sharpening back
    keeps well-to-well differences visible at a glance.
    """
    plane = read_plane(path).astype(np.float32)
    plane = _downscale(plane, size)
    image = Image.fromarray(scale_to_uint8(plane, limits), mode="L")
    image.thumbnail((size, size), Image.Resampling.BILINEAR)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=60, threshold=2))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue(), image.width, image.height


def build_thumbnails(
    db_path: Path,
    thumb_db_path: Path,
    *,
    size: int = 256,
    percentiles: tuple[float, float] = (1.0, 99.5),
    sample_size: int = 200,
    workers: int = 4,
    force: bool = False,
    verbose: bool = True,
) -> int:
    """Build (or refresh) the thumbnail cache. Returns the number rendered.

    Incremental: an image is re-rendered only if its mtime or size changed.
    """
    from ..index import db as index_db

    source = index_db.connect(db_path, read_only=True)
    rows = source.execute(
        "SELECT image_id, path, channel, mtime, size_bytes FROM images"
    ).fetchall()
    jobs = [
        ThumbnailJob(r["image_id"], Path(r["path"]), r["channel"], r["mtime"], r["size_bytes"])
        for r in rows
    ]
    source.close()

    thumb_db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(thumb_db_path)
    con.executescript(_SCHEMA)
    con.execute("PRAGMA journal_mode=WAL")

    existing = {
        r[0]: (r[1], r[2])
        for r in con.execute("SELECT image_id, mtime, size_bytes FROM thumbnails")
    }
    if force:
        todo = jobs
    else:
        todo = [
            j
            for j in jobs
            if existing.get(j.image_id) != (j.mtime, j.size_bytes)
        ]

    if not todo:
        if verbose:
            print(f"thumbnail cache up to date ({len(jobs)} images)")
        con.close()
        return 0

    limits = estimate_display_limits(
        jobs, percentiles=percentiles, sample_size=sample_size
    )
    write_con = index_db.connect(db_path)
    write_con.executemany(
        "INSERT OR REPLACE INTO display_limits (channel, lo, hi) VALUES (?, ?, ?)",
        [(ch, lo, hi) for ch, (lo, hi) in limits.items()],
    )
    write_con.commit()
    write_con.close()
    if verbose:
        for channel, (lo, hi) in sorted(limits.items()):
            print(f"display limits  {channel}: [{lo:.1f}, {hi:.1f}]")

    def render(job: ThumbnailJob):
        try:
            png, w, h = render_thumbnail(
                job.path, limits.get(job.channel or "_", (0.0, 1.0)), size
            )
        except Exception as exc:  # noqa: BLE001
            return job, None, str(exc)
        return job, (png, w, h), None

    failures: list[tuple[str, str]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for job, result, error in pool.map(render, todo):
            if result is None:
                failures.append((job.image_id, error or "unknown error"))
                continue
            png, w, h = result
            con.execute(
                "INSERT OR REPLACE INTO thumbnails "
                "(image_id, mtime, size_bytes, width, height, png) VALUES (?, ?, ?, ?, ?, ?)",
                (job.image_id, job.mtime, job.size_bytes, w, h, png),
            )
            done += 1
            if verbose and done % 250 == 0:
                con.commit()
                print(f"  {done}/{len(todo)}")
    con.commit()
    con.close()

    if verbose:
        print(f"rendered {done} thumbnails -> {thumb_db_path}")
        if failures:
            print(f"! {len(failures)} files failed:")
            for image_id, error in failures[:10]:
                print(f"    {image_id}: {error}")
    return done


class ThumbnailCache:
    """Read-only accessor. One instance per thread."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)

    def get(self, image_id: str) -> bytes | None:
        row = self.con.execute(
            "SELECT png FROM thumbnails WHERE image_id = ?", (image_id,)
        ).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self.con.close()
