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
import os
import sqlite3
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageFilter


def default_worker_count() -> int:
    """Thumbnail rendering is I/O + numpy/PIL bound, both of which release
    the GIL, so threads keep scaling well past the CPU count on this
    workload (measured: near-linear to ~16, still improving to ~32 on a
    36-core machine). Scale with the machine instead of a flat default so a
    2-core laptop and a 36-core workstation each get a sensible number."""
    return min(32, max(4, (os.cpu_count() or 4) * 2))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS thumbnails (
    image_id      TEXT PRIMARY KEY,
    mtime         REAL NOT NULL,
    size_bytes    INTEGER NOT NULL,
    width         INTEGER NOT NULL,
    height        INTEGER NOT NULL,
    png           BLOB NOT NULL,
    autoscale_png BLOB
);
"""


@dataclass(slots=True)
class ThumbnailJob:
    image_id: str
    path: Path
    channel: str | None
    mtime: float
    size_bytes: int


# Read every Nth pixel when building thumbnails. The thumbnail is a 256px
# navigation tile rendered from a 2720px plane -- a ~10x reduction -- so
# reading every 2nd pixel still supplies ~5x more data than the output needs.
# Measured against a full read over a random sample of a real plate, the final
# PNGs differ by at most 11/255 (mean 0.66), i.e. not visible in a thumbnail
# grid, for ~1.8x less read time.
#
# 2 is deliberately conservative. Larger strides fall off a cliff: stride 3
# measured 66/255 worst-case and stride 5 measured 48/255, because the stride
# stops lining up with the block-mean factor below and starts aliasing real
# structure. Do not raise this without re-measuring the PNG difference.
#
# This applies to thumbnails ONLY. The full-resolution viewer reads its own
# planes and is unaffected -- previews are for navigation, the viewer is what
# you inspect an image in.
THUMBNAIL_READ_STRIDE = 2


def read_plane(path: Path, *, stride: int = 1) -> np.ndarray:
    """Read a TIFF and reduce it to a single 2-D plane.

    Multi-page / multi-channel files are reduced by taking the first plane of
    every leading axis. This is deliberate and lossy: the thumbnail is a
    navigation aid, and the full-resolution viewer shows all planes.

    ``stride`` subsamples the plane on read; it defaults to 1 (every pixel) so
    that any caller other than the thumbnail renderer keeps full fidelity.
    """
    array = _read_array(path)
    while array.ndim > 2:
        # Collapse leading axes, but prefer a channel axis of small extent last.
        array = array[0]
    if stride > 1:
        # Slice FIRST, then copy. When _read_array returned a memmap the array
        # is still lazy at this point, so taking every Nth row means the other
        # N-1 are never paged in at all; materialising the full plane and then
        # discarding most of it reads the whole thing for nothing.
        #
        # These files are large (measured: 266 MB, 13 planes of 3200x3200), so
        # the difference is the difference between a preview that appears and
        # one you wait for. Cold, over the network, at stride 4: 302 ms before,
        # 106 ms after, with bit-identical output.
        array = np.ascontiguousarray(array[::stride, ::stride])
    return array


def _read_array(path: Path) -> np.ndarray:
    """Read the raw array, memory-mapping it when the file layout allows.

    NIS-Elements writes these planes uncompressed, and for an uncompressed
    strip TIFF the pixels are already laid out exactly as numpy wants them,
    so the OS can map the bytes straight into the array instead of tifffile
    reassembling 2720 one-row strips in Python. Measured on a real plate this
    is the single biggest win in the pipeline -- reading was ~70% of the
    per-image cost -- and it is bit-identical to imread(), because it is the
    same bytes by a cheaper route (verified over a random sample: the decoded
    arrays compare equal and the final PNGs differ by 0).

    memmap() raises for anything it cannot map (compressed, tiled, or an
    unusual layout), so fall back to imread() rather than assuming; correctness
    does not depend on which branch is taken.
    """
    try:
        return np.asarray(tifffile.memmap(path))
    except Exception:  # noqa: BLE001 - any unmappable layout falls back
        return np.asarray(tifffile.imread(path))


def _percentile_limits(
    values: np.ndarray, percentiles: tuple[float, float]
) -> tuple[float, float]:
    lo, hi = np.percentile(values, percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(values)), float(np.max(values))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _sample_flat(path: Path) -> np.ndarray | None:
    try:
        # Same stride as the renderer: these samples set the display limits the
        # thumbnails are scaled with, so they should see the same pixels.
        plane = read_plane(path, stride=THUMBNAIL_READ_STRIDE)
    except Exception:  # noqa: BLE001 - a broken file must not abort the pass
        return None
    flat = plane.ravel()
    if flat.size > 20_000:
        flat = flat[:: flat.size // 20_000]
    return flat.astype(np.float32)


def estimate_display_limits(
    jobs: Iterable[ThumbnailJob],
    *,
    percentiles: tuple[float, float] = (1.0, 99.5),
    sample_size: int = 200,
    workers: int | None = None,
) -> dict[str, tuple[float, float]]:
    """Estimate one (lo, hi) pair per channel from a random sample of images.

    Reads are I/O bound, so sampled files are decoded in a thread pool rather
    than one at a time — with a couple hundred samples per channel this
    otherwise dominates the time before rendering even starts.
    """
    # Fixed seed: the sample decides the screen-wide display limits, so the
    # same screen should scale identically every time it is rebuilt.
    rng = np.random.default_rng(0)
    by_channel: dict[str, list[ThumbnailJob]] = {}
    for job in jobs:
        by_channel.setdefault(job.channel or "_", []).append(job)

    picks_by_channel: dict[str, list[Path]] = {}
    for channel, channel_jobs in by_channel.items():
        n = min(sample_size, len(channel_jobs))
        picks = rng.choice(len(channel_jobs), size=n, replace=False)
        picks_by_channel[channel] = [channel_jobs[int(idx)].path for idx in picks]

    limits: dict[str, tuple[float, float]] = {}
    with ThreadPoolExecutor(max_workers=workers or default_worker_count()) as pool:
        for channel, paths in picks_by_channel.items():
            samples = [s for s in pool.map(_sample_flat, paths) if s is not None]
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


def _finish_png(plane_u8: np.ndarray, size: int) -> bytes:
    image = Image.fromarray(plane_u8, mode="L")
    image.thumbnail((size, size), Image.Resampling.BILINEAR)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=60, threshold=2))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def render_thumbnail(
    path: Path, limits: tuple[float, float], size: int, *, autoscale: bool = False
) -> tuple[bytes, bytes | None, int, int]:
    """Render one file to PNG bytes using fixed display limits.

    A subtle unsharp mask is applied after downscaling: block-mean downscaling
    softens detail that per-image autoscaling would otherwise recover, and the
    thumbnail grid is the primary navigation view, so a little sharpening back
    keeps well-to-well differences visible at a glance.

    If ``autoscale`` is set, a second PNG is also rendered using that image's
    own percentile limits — pre-computed so the GUI's fixed/autoscale toggle
    is an instant cache lookup rather than a live re-render.
    """
    raw_plane = read_plane(path, stride=THUMBNAIL_READ_STRIDE).astype(np.float32)
    plane = _downscale(raw_plane, size)
    png = _finish_png(scale_to_uint8(plane, limits), size)

    autoscale_png = None
    if autoscale:
        # Percentiles from a full-resolution 2720x2720 plane cost ~4x more
        # than from a ~20k-value subsample and land on the same limits to
        # well within display precision, same trick estimate_display_limits
        # already uses for the screen-wide estimate.
        flat = raw_plane.ravel()
        if flat.size > 20_000:
            flat = flat[:: flat.size // 20_000]
        own_limits = _percentile_limits(flat, (1.0, 99.5))
        autoscale_png = _finish_png(scale_to_uint8(plane, own_limits), size)

    size_after = Image.open(io.BytesIO(png))
    return png, autoscale_png, size_after.width, size_after.height


def build_thumbnails(
    db_path: Path,
    thumb_db_path: Path,
    *,
    size: int = 256,
    percentiles: tuple[float, float] = (1.0, 99.5),
    sample_size: int = 200,
    workers: int | None = None,
    force: bool = False,
    verbose: bool = True,
    autoscale: bool = True,
    progress=None,
) -> int:
    """Build (or refresh) the thumbnail cache. Returns the number rendered.

    Incremental: an image is re-rendered only if its mtime or size changed.

    ``progress`` is an optional ``callable(done, total, message)`` invoked from
    the calling thread as work completes. It exists so a GUI can show a real
    progress bar: this is minutes of work on a full plate, and a window that
    simply stops responding for that long is indistinguishable from a hang.
    """
    from ..data.index import db as index_db

    if workers is None:
        workers = default_worker_count()

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

    if progress:
        progress(0, len(todo), "estimating display limits…")
    limits = estimate_display_limits(
        jobs, percentiles=percentiles, sample_size=sample_size, workers=workers
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
            png, autoscale_png, w, h = render_thumbnail(
                job.path,
                limits.get(job.channel or "_", (0.0, 1.0)),
                size,
                autoscale=autoscale,
            )
        except Exception as exc:  # noqa: BLE001
            return job, None, str(exc)
        return job, (png, autoscale_png, w, h), None

    failures: list[tuple[str, str]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for job, result, error in pool.map(render, todo):
            if result is None:
                failures.append((job.image_id, error or "unknown error"))
                continue
            png, autoscale_png, w, h = result
            con.execute(
                "INSERT OR REPLACE INTO thumbnails "
                "(image_id, mtime, size_bytes, width, height, png, autoscale_png) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (job.image_id, job.mtime, job.size_bytes, w, h, png, autoscale_png),
            )
            done += 1
            if progress and (done % 25 == 0 or done == len(todo)):
                progress(done, len(todo), f"rendering thumbnails… {done}/{len(todo)}")
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

    def get(self, image_id: str, *, autoscale: bool = False) -> bytes | None:
        column = "autoscale_png" if autoscale else "png"
        row = self.con.execute(
            f"SELECT {column}, png FROM thumbnails WHERE image_id = ?", (image_id,)
        ).fetchone()
        if row is None:
            return None
        # Fall back to the fixed-levels PNG if no autoscaled variant was
        # rendered for this image (e.g. cache built with autoscale=False).
        return row[0] if row[0] is not None else row[1]

    def close(self) -> None:
        self.con.close()
