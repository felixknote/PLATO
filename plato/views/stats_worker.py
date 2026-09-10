"""Measuring a whole dataset's images, in parallel, without freezing the GUI.

One task per chunk of rows rather than one per image: at ~6 ms an image the
per-task overhead of a QRunnable would be a noticeable fraction of the work,
and a chunk amortises it away while still letting every core participate.

Progress is reported per chunk, cancellation is checked between images, and a
file that cannot be read leaves NaN for that row rather than failing the pass.
NaN is the honest value -- the statistic is genuinely unknown -- and the
colouring layer already knows how to grey out points with no value.
"""

from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal

from ..data.image_stats import STAT_NAMES, STATS_STRIDE, describe_raw

# Rows per task, at most. Large enough that per-task overhead vanishes next to
# the ~6 ms an image costs.
MAX_CHUNK = 64

# Fewest chunks to split a pass into, so the bar has something to report. A
# fixed chunk size would put a 30-image dataset in one task and move the bar
# 0 -> 100% with nothing in between, which reads as a hang rather than as
# speed. Also keeps every core busy on a small dataset instead of one.
MIN_CHUNKS = 24


def chunk_size(n_rows: int) -> int:
    """Rows per task for a pass over ``n_rows`` images."""
    if n_rows <= 0:
        return MAX_CHUNK
    # Floor, not ceiling: rounding up overshoots and yields FEWER chunks than
    # asked for (30 rows / 24 rounds to 2, giving 15 tasks). Rounding down
    # errs towards more chunks, which is the safe direction for a progress bar.
    return max(1, min(MAX_CHUNK, n_rows // MIN_CHUNKS))


class StatsSignals(QObject):
    # (rows done, rows total)
    progress = Signal(int, int)
    # {stat name: array over all rows}
    finished = Signal(object)
    failed = Signal(str)


class StatsRun:
    """Shared state for one measuring pass across many chunk tasks."""

    def __init__(self, n_rows: int, signals: StatsSignals) -> None:
        self.signals = signals
        self.n_rows = n_rows
        self.cancelled = False
        self._lock = threading.Lock()
        self._done = 0
        self._remaining = 0
        self.values = {
            name: np.full(n_rows, np.nan, dtype=np.float32) for name in STAT_NAMES
        }

    def cancel(self) -> None:
        self.cancelled = True

    def chunk_finished(self, measured: int) -> None:
        """Record a finished chunk and emit the final result after the last."""
        with self._lock:
            self._done += measured
            self._remaining -= 1
            done, remaining = self._done, self._remaining
        try:
            self.signals.progress.emit(done, self.n_rows)
            if remaining <= 0 and not self.cancelled:
                self.signals.finished.emit(self.values)
        except RuntimeError:
            # The panel can be closed while tasks are still finishing.
            pass

    def expect(self, tasks: int) -> None:
        with self._lock:
            self._remaining = tasks


class StatsTask(QRunnable):
    """Measures one chunk of rows."""

    def __init__(self, run: StatsRun, rows: list[int], paths: list) -> None:
        super().__init__()
        self._run = run
        self._rows = rows
        self._paths = paths

    def run(self) -> None:  # pragma: no cover - worker thread
        from ..cache import read_plane

        measured = 0
        for row, path in zip(self._rows, self._paths):
            if self._run.cancelled:
                break
            measured += 1
            if path is None:
                continue
            try:
                plane = read_plane(path, stride=STATS_STRIDE)
                stats = describe_raw(plane)
            except Exception:  # noqa: BLE001 - an unreadable file stays NaN
                continue
            for name, value in stats.items():
                self._run.values[name][row] = value
        self._run.chunk_finished(measured)


def start(pool, rows: list[int], paths: list, n_rows: int, signals: StatsSignals) -> StatsRun:
    """Queue the whole pass and return its handle.

    ``rows`` and ``paths`` are parallel: paths[i] is where rows[i] lives, or
    None when the image was not found.
    """
    run = StatsRun(n_rows, signals)
    size = chunk_size(len(rows))
    chunks = [
        (rows[i : i + size], paths[i : i + size]) for i in range(0, len(rows), size)
    ]
    if not chunks:
        run.expect(0)
        try:
            signals.finished.emit(run.values)
        except RuntimeError:
            pass
        return run
    # Set the expected count BEFORE starting any task, or a chunk that
    # finishes immediately could see zero remaining and emit early.
    run.expect(len(chunks))
    for chunk_rows, chunk_paths in chunks:
        pool.start(StatsTask(run, chunk_rows, chunk_paths))
    return run
