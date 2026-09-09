"""Turning UMAP's and t-SNE's chatter into a real progress bar.

Neither library offers a progress callback, but both will narrate what they
are doing, and the narration is structured enough to drive a determinate bar:

* **UMAP** with ``verbose=True`` prints named phases ("Finding Nearest
  Neighbors", "Construct embedding") and then its optimiser's epoch counter
  ("completed 150 / 500 epochs"). Phases are mapped to spans of the overall
  bar and the epoch counter fills the last, longest span. Output goes through
  ``print``/``tqdm`` to stdout/stderr, so it is captured by redirecting those
  on the worker thread -- which is safe here because the projection already
  runs on its own thread and nothing else writes there.

* **openTSNE** takes real ``callbacks``, called every ``callbacks_every_iters``
  iterations with the current error, so its progress is read directly rather
  than parsed.

The parsing is deliberately forgiving: an unrecognised line advances nothing
rather than raising, so a version that renames a phase degrades to a slower-
moving bar instead of a crash. The fractions are approximate by nature -- they
describe how far through the *phases* the fit is, not how much wall time is
left -- which is what a progress bar is for.
"""

from __future__ import annotations

import contextlib
import io
import re
import threading
from typing import Callable

# UMAP's phase banners, in the order they occur, each mapped to the fraction
# of the whole fit that is COMPLETE when that phase begins. The gaps between
# consecutive values are how much of the bar each phase owns.
#
# Measured on a 24k x 1024 fit: the neighbour search dominates, and embedding
# optimisation is the only phase that reports its own sub-progress, so it gets
# the largest span and is filled smoothly by the epoch counter.
UMAP_PHASES: tuple[tuple[str, float], ...] = (
    ("Construct fuzzy simplicial set", 0.05),
    ("Finding Nearest Neighbors", 0.10),
    ("Finished Nearest Neighbor Search", 0.45),
    ("Construct embedding", 0.50),
    ("Finished embedding", 1.00),
)

# "completed  150  /  500 epochs" -- spacing varies between versions. This is
# the form UMAP prints through `print`, captured by the stdout redirect.
_EPOCH_RE = re.compile(r"completed\s+(\d+)\s*/\s*(\d+)\s+epochs", re.IGNORECASE)

# "Epochs completed:  30%| ###  152/500 [00:01]" -- the tqdm bar itself, which
# only arrives through the stream passed in tqdm_kwds. It updates more often
# than the printed form, so it is what actually makes the bar move smoothly.
_TQDM_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")

# The span of the bar the epoch counter fills: from "Construct embedding" to done.
_EPOCH_SPAN = (0.50, 1.00)


class UmapProgressParser:
    """Feeds UMAP's verbose output to a ``(fraction, message)`` callback."""

    def __init__(self, report: Callable[[float, str], None]) -> None:
        self._report = report
        self._fraction = 0.0
        self._phase = "preparing"

    def feed(self, text: str) -> None:
        for line in text.replace("\r", "\n").splitlines():
            self._feed_line(line.strip())

    def _feed_line(self, line: str) -> None:
        if not line:
            return

        match = _EPOCH_RE.search(line) or _TQDM_RE.search(line)
        if match:
            done, total = int(match.group(1)), int(match.group(2))
            if total > 0:
                low, high = _EPOCH_SPAN
                self._set(low + (high - low) * (done / total), f"optimising layout — epoch {done}/{total}")
            return

        for phase, fraction in UMAP_PHASES:
            if phase.lower() in line.lower():
                self._phase = _readable(phase)
                self._set(fraction, self._phase)
                return

    def _set(self, fraction: float, message: str) -> None:
        # Never go backwards: tqdm rewrites its line, and a redraw that
        # re-reports an earlier epoch would make the bar stutter.
        fraction = max(self._fraction, min(1.0, fraction))
        self._fraction = fraction
        self._report(fraction, message)


def _readable(phase: str) -> str:
    return {
        "Construct fuzzy simplicial set": "building the neighbour graph",
        "Finding Nearest Neighbors": "finding nearest neighbours",
        "Finished Nearest Neighbor Search": "neighbours found",
        "Construct embedding": "optimising layout",
        "Finished embedding": "finishing",
    }.get(phase, phase.lower())


class _TeeStream(io.TextIOBase):
    """A write-only stream that forwards everything to a parser."""

    def __init__(self, parser: UmapProgressParser) -> None:
        self._parser = parser
        self._lock = threading.Lock()

    def write(self, text: str) -> int:  # noqa: D102
        if text:
            with self._lock:
                self._parser.feed(text)
        return len(text)

    def flush(self) -> None:  # noqa: D102
        return None


class UmapProgressTap:
    """Both channels UMAP reports through, wired to one parser.

    ``tqdm_kwds`` must be passed to ``umap.UMAP``: tqdm captured the real
    stderr at import time, so redirecting stderr alone never sees the epoch
    bar -- and the epoch bar is the only part of a UMAP fit that reports
    sub-progress.
    """

    def __init__(self, report: Callable[[float, str], None]) -> None:
        self._parser = UmapProgressParser(report)
        self.stream = _TeeStream(self._parser)

    @property
    def tqdm_kwds(self) -> dict:
        return {"file": self.stream, "disable": False, "mininterval": 0.3}

    @contextlib.contextmanager
    def capture(self):
        """Redirect print-based output for the duration of the fit."""
        with contextlib.redirect_stdout(self.stream), contextlib.redirect_stderr(self.stream):
            yield


@contextlib.contextmanager
def capture_umap_progress(report: Callable[[float, str], None] | None):
    """Route UMAP's verbose output into ``report`` for the duration of a fit.

    Redirects stdout and stderr, so only call this around the fit itself, on a
    thread that has no other reason to print. Yields nothing; if ``report`` is
    None it is a no-op and the library stays quiet.
    """
    if report is None:
        yield
        return
    stream = _TeeStream(UmapProgressParser(report))
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        yield


def tsne_callback(
    report: Callable[[float, str], None] | None, n_iter: int, *, offset: float, span: float
):
    """An openTSNE callback that reports optimisation progress.

    openTSNE optimises in two passes -- early exaggeration, then the main run
    -- and the iteration counter RESTARTS for the second. Reported naively the
    bar would run to the end, snap back, and run again. The callback therefore
    tracks how far it has already come and never reports a lower fraction, so
    the two passes read as one continuous run.
    """
    if report is None:
        return None

    state = {"fraction": offset, "pass_offset": offset, "previous": 0}

    def callback(iteration: int, error: float, embedding) -> None:  # noqa: ANN001
        # A counter that went backwards means a new pass began; anchor the
        # remaining span to wherever the bar had reached.
        if iteration < state["previous"]:
            state["pass_offset"] = state["fraction"]
        state["previous"] = iteration

        remaining = max(0.0, offset + span - state["pass_offset"])
        fraction = state["pass_offset"] + remaining * min(1.0, iteration / max(1, n_iter))
        state["fraction"] = max(state["fraction"], min(offset + span, fraction))
        report(state["fraction"], f"optimising layout — iteration {iteration}/{n_iter}")

    return callback
