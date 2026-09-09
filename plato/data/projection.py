"""UMAP / t-SNE projections of an embedding matrix, computed once and cached.

Parameter defaults are taken from the recipe already used for these DINO
features in AI4AB/analysis/utils.py (``plot_umap``): cosine metric,
n_neighbors=500, min_dist=1.0, fixed seed. That recipe is tuned for 1024-d
deep features, where distances are angular and neighbourhoods are large.

It is deliberately NOT the morphology recipe from the multiplate UMAP notebook
(RobustScaler -> QuantileTransformer -> PCA(10), n_neighbors=15, min_dist=0.1),
which is tuned for ~30 hand-crafted shape descriptors. Applying that here would
quantile-flatten 1024 already-comparable dimensions and then throw away all but
10 of them.

A projection of 30k x 1024 takes minutes, so every result is cached on disk
under the work directory, keyed by the dataset fingerprint plus every
parameter that affects the output. Change a parameter and you get a different
cache entry rather than a stale one.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


def _enable_numba_cache() -> None:
    """Let numba keep its compiled kernels between runs.

    UMAP is numba code, and numba compiles on first use. Without a cache
    directory that compilation happens again on every application launch --
    measured here at ~12 s before the first projection even starts, which
    reads as "UMAP is broken" rather than "the compiler is running".

    Set before numba is imported, since it reads this at import time. A
    user-set NUMBA_CACHE_DIR always wins.
    """
    if os.environ.get("NUMBA_CACHE_DIR"):
        return
    try:
        from platformdirs import user_cache_dir

        base = Path(user_cache_dir("plato", "plato"))
    except ImportError:
        base = Path.home() / ".cache" / "plato"
    cache = base / "numba"
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    os.environ["NUMBA_CACHE_DIR"] = str(cache)


_enable_numba_cache()

# Shared by both methods: PCA down to this many dimensions before the
# neighbour search. 1024-d cosine neighbour search is the slow part, and for
# DINO features the first ~50 components carry the structure the projection
# then lays out. Set to 0 to skip PCA and project the raw vectors.
DEFAULT_PCA_COMPONENTS = 50

RANDOM_STATE = 1

UMAP = "UMAP"
TSNE = "t-SNE"
METHODS = (UMAP, TSNE)


@dataclass(slots=True)
class ProjectionParams:
    """Everything that changes a projection's output.

    Every field is part of the cache key, so two projections that differ in
    any of them are stored separately rather than one overwriting the other.
    """

    method: str = UMAP
    # L2-normalise rows before projecting. On for cosine geometry, which is
    # what the DINO recipe assumes.
    normalize: bool = True
    pca_components: int = DEFAULT_PCA_COMPONENTS
    metric: str = "cosine"
    # UMAP
    n_neighbors: int = 500
    min_dist: float = 1.0
    # t-SNE
    perplexity: float = 30.0
    # Subsample cap. None projects every point.
    max_points: int | None = None
    # Reproducible layout, or a fast one.
    #
    # UMAP refuses to use more than one thread once a random_state is set --
    # its parallel optimiser is not order-deterministic, so a seed and threads
    # cannot both be honoured. Measured on this machine (36 cores): 8k points
    # take 46.6 s seeded and 4.5 s parallel, a 10x difference, and at 24k
    # points the parallel run takes 34 s.
    #
    # What is lost is the exact coordinates, not the structure: across
    # independent unseeded runs of a 24k set, k-means labels of the resulting
    # layouts agreed at ARI 1.000. So the clusters, and any conclusion drawn
    # from them, are the same; the plot may be rotated or mirrored.
    #
    # Default False -- exploration wants speed, and a projection is re-run
    # whenever a parameter changes. Turn it on for a figure that has to be
    # regenerated exactly.
    deterministic: bool = False

    def key(self) -> str:
        relevant = asdict(self)
        if self.method == UMAP:
            relevant.pop("perplexity")
        else:
            relevant.pop("n_neighbors")
            relevant.pop("min_dist")
        parts = [f"{k}={relevant[k]}" for k in sorted(relevant)]
        return "_".join(parts).replace(" ", "").replace("/", "-")


def suggest(n_points: int, *, learned: bool = True) -> ProjectionParams:
    """Sensible starting parameters for a dataset of this size and kind.

    There is no single right default. ``n_neighbors=500`` is the tuned value
    for tens of thousands of DINO vectors, where neighbourhoods are large and
    the question is global structure; applied to a few hundred points it
    exceeds the sample and every point becomes a neighbour of every other,
    which erases exactly the local structure a projection exists to show.

    So the defaults scale:

    * **n_neighbors** ~ 1.5% of the points, clamped to [15, 500]. That lands
      on the tuned 500 for the 30k-row exports and on something sane for a
      few hundred computed descriptors.
    * **perplexity** ~ n/100, clamped to [10, 50]; openTSNE additionally
      requires it below n/3, which the runner enforces.
    * **subsample** kicks in only above ~40k points, where a full UMAP starts
      costing more minutes than the first look is worth.
    * **PCA** is skipped for low-dimensional inputs -- reducing 31 image
      descriptors to 50 components does nothing but cost a fit.

    ``learned`` says whether the vectors are learned embeddings (cosine
    geometry, worth normalising) or hand-computed descriptors, which are
    already standardised per feature and live in a Euclidean space.
    """
    n_points = max(1, int(n_points))

    neighbours = int(round(n_points * 0.015))
    neighbours = max(15, min(500, neighbours))
    # Never at or above the sample size: UMAP clamps it anyway, but a value
    # that has to be clamped is not a sensible default to show the user.
    neighbours = min(neighbours, max(2, n_points - 1))

    perplexity = float(max(10, min(50, n_points // 100)))

    max_points = None if n_points <= 40_000 else 20_000

    return ProjectionParams(
        method=UMAP,
        normalize=learned,
        pca_components=DEFAULT_PCA_COMPONENTS if learned else 0,
        metric="cosine" if learned else "euclidean",
        n_neighbors=neighbours,
        min_dist=1.0 if learned else 0.1,
        perplexity=perplexity,
        max_points=max_points,
        deterministic=False,
    )


@dataclass(slots=True)
class ProjectionResult:
    """2-D coordinates for a (possibly subsampled) set of embedding rows."""

    coords: np.ndarray  # (M, 2) float32
    # Index into the dataset's rows for each coordinate. Identity unless the
    # projection was subsampled; carried explicitly so hover/click can always
    # get back to the right metadata row.
    row_indices: np.ndarray
    params: ProjectionParams
    seconds: float = 0.0
    from_cache: bool = False
    notes: list[str] = field(default_factory=list)


class MissingDependency(RuntimeError):
    """A projection backend is not installed.

    These live behind an optional extra, so the message has to say what to
    install rather than surfacing a bare "No module named 'umap'" in a dialog.
    """

    def __init__(self, package: str) -> None:
        super().__init__(
            f"{package} is not installed.\n\n"
            f"The Embedding Explorer needs it. Install the optional extra:\n"
            f'    pip install -e ".[gui,embed]"'
        )


def _prepare(vectors: np.ndarray, params: ProjectionParams) -> np.ndarray:
    """Normalise and optionally PCA-reduce before the neighbour search."""
    data = np.ascontiguousarray(vectors, dtype=np.float32)

    if params.normalize:
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        # A zero vector has no direction to preserve; leaving the norm at 1
        # maps it to the origin instead of producing NaNs that would poison
        # every distance computed against it.
        np.maximum(norms, 1e-12, out=norms)
        data = data / norms

    n_components = min(params.pca_components, data.shape[0], data.shape[1])
    if params.pca_components and n_components >= 2:
        # NOTE: PCA also decides the metric the caller should use -- see
        # effective_metric(). Once the rows are L2-normalised and projected,
        # every vector has norm ~1 and cosine distance is degenerate, which
        # makes UMAP's approximate neighbour search do 2.5x the work for an
        # identical layout (measured: 20.3s vs 8.2s at 3k points, ARI 1.000).
        try:
            from sklearn.decomposition import PCA
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise MissingDependency("scikit-learn") from exc

        data = PCA(n_components=n_components, random_state=RANDOM_STATE).fit_transform(data)
        data = np.ascontiguousarray(data, dtype=np.float32)
    return data


def _subsample(n_rows: int, params: ProjectionParams) -> np.ndarray:
    if params.max_points is None or n_rows <= params.max_points:
        return np.arange(n_rows)
    rng = np.random.default_rng(RANDOM_STATE)
    return np.sort(rng.choice(n_rows, size=params.max_points, replace=False))


def effective_metric(params: ProjectionParams) -> str:
    """The metric to actually search with, given how the data was prepared.

    Cosine distance measures angle. L2-normalising already puts every row on
    the unit sphere, where angle and Euclidean distance are monotonically
    related -- so after normalisation (and the PCA that follows it) Euclidean
    gives the same neighbours far more cheaply, because UMAP's neighbour
    search has a fast path for it and cosine on near-identical norms does not.

    Measured on 3k x 50: 20.3 s asking for cosine, 8.2 s asking for Euclidean,
    with k-means labels of the two layouts agreeing at ARI 1.000.
    """
    if params.metric == "cosine" and params.normalize:
        return "euclidean"
    return params.metric


def _run_umap(data: np.ndarray, params: ProjectionParams, progress=None) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise MissingDependency("umap-learn") from exc

    from .progress_taps import UmapProgressTap

    # n_neighbors must stay below the sample count; the tuned default of 500
    # is larger than some datasets are.
    n_neighbors = int(max(2, min(params.n_neighbors, data.shape[0] - 1)))

    # UMAP has no progress callback, but it will narrate its phases and its
    # optimiser's epochs; the tap turns that into fractions. Without a
    # reporter it stays silent, exactly as before.
    tap = UmapProgressTap(progress) if progress is not None else None
    # random_state and n_jobs are mutually exclusive in UMAP; passing both
    # prints a warning and silently drops the threads.
    threading = (
        {"random_state": RANDOM_STATE}
        if params.deterministic
        else {"n_jobs": -1}
    )
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=params.min_dist,
        metric=effective_metric(params),
        verbose=tap is not None,
        tqdm_kwds=tap.tqdm_kwds if tap is not None else None,
        **threading,
    )
    if tap is None:
        return np.asarray(reducer.fit_transform(data), dtype=np.float32)
    with tap.capture():
        return np.asarray(reducer.fit_transform(data), dtype=np.float32)


def _run_tsne(data: np.ndarray, params: ProjectionParams, progress=None) -> np.ndarray:
    try:
        from openTSNE import TSNE
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise MissingDependency("openTSNE") from exc

    from .progress_taps import tsne_callback

    # openTSNE requires perplexity < n_samples / 3.
    perplexity = float(max(5.0, min(params.perplexity, (data.shape[0] - 1) / 3.0)))
    n_iter = 500

    # Unlike UMAP, openTSNE takes a real callback. The neighbour search before
    # optimisation reports nothing, so it owns the first third of the bar and
    # the callback fills the rest.
    if progress is not None:
        progress(0.05, "finding nearest neighbours")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric=effective_metric(params),
        # Unlike UMAP, openTSNE parallelises with a seed set, so this stays
        # reproducible AND threaded.
        random_state=RANDOM_STATE,
        n_jobs=-1,
        n_iter=n_iter,
        callbacks=tsne_callback(progress, n_iter, offset=0.35, span=0.65),
        callbacks_every_iters=25,
    )
    return np.asarray(tsne.fit(data), dtype=np.float32)


def warm_up() -> None:
    """Compile UMAP's kernels on a tiny input, so a real run does not.

    Cheap when the numba cache is warm (a few hundred ms), and worth the one
    slow call otherwise: the compilation happens once either way, and doing it
    here means it happens while the user is still choosing a dataset rather
    than in the middle of their first projection.

    Never raises -- a warm-up that fails costs nothing but the warmth.
    """
    try:
        import umap

        vectors = np.linspace(0, 1, 64 * 4, dtype=np.float32).reshape(64, 4)
        umap.UMAP(
            n_components=2, n_neighbors=5, metric="cosine", n_jobs=-1
        ).fit_transform(vectors)
    except Exception:  # noqa: BLE001 - best effort only
        pass


class ProjectionCache:
    """Disk cache of computed projections, keyed by dataset + parameters."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def _paths(self, fingerprint: str, params: ProjectionParams) -> tuple[Path, Path]:
        stem = f"{fingerprint}__{params.method}__{params.key()}"
        # Parameter keys are long and contain characters a filesystem would
        # rather not see; hash the tail and keep a readable prefix.
        import hashlib

        digest = hashlib.sha256(stem.encode()).hexdigest()[:16]
        base = self.directory / f"{params.method.replace('-', '')}_{fingerprint}_{digest}"
        return base.with_suffix(".npz"), base.with_suffix(".json")

    def load(self, fingerprint: str, params: ProjectionParams) -> ProjectionResult | None:
        array_path, _ = self._paths(fingerprint, params)
        if not array_path.exists():
            return None
        try:
            with np.load(array_path, allow_pickle=False) as handle:
                coords = handle["coords"]
                row_indices = handle["row_indices"]
        except (OSError, ValueError, KeyError):
            # A truncated cache file (interrupted write, full disk) should
            # cost a recompute, not an unreadable explorer.
            return None
        return ProjectionResult(
            coords=coords, row_indices=row_indices, params=params, from_cache=True
        )

    def save(self, fingerprint: str, result: ProjectionResult) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        array_path, meta_path = self._paths(fingerprint, result.params)
        np.savez_compressed(
            array_path, coords=result.coords, row_indices=result.row_indices
        )
        meta_path.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "seconds": round(result.seconds, 2),
                    "n_points": int(result.coords.shape[0]),
                    **asdict(result.params),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def entries(self) -> list[dict]:
        """Every cached projection's metadata, for display."""
        if not self.directory.is_dir():
            return []
        out = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                out.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return out


def project(
    vectors: np.ndarray,
    params: ProjectionParams,
    *,
    fingerprint: str = "",
    cache: ProjectionCache | None = None,
    progress=None,
    on_progress=None,
) -> ProjectionResult:
    """Project ``vectors`` to 2-D, reusing a cached result when one exists.

    ``progress`` is an optional callable taking a status string.
    ``on_progress`` is an optional ``callable(fraction, message)`` driven by
    the backend's own reporting, for a determinate progress bar.
    """
    import time

    if cache is not None and fingerprint:
        cached = cache.load(fingerprint, params)
        if cached is not None:
            if progress:
                progress("loaded cached projection")
            return cached

    notes: list[str] = []
    row_indices = _subsample(vectors.shape[0], params)
    if len(row_indices) < vectors.shape[0]:
        notes.append(f"subsampled to {len(row_indices):,} of {vectors.shape[0]:,} points")

    started = time.perf_counter()
    if progress:
        progress("preparing vectors…")
    data = _prepare(vectors[row_indices], params)

    if progress:
        progress(f"running {params.method} on {data.shape[0]:,} x {data.shape[1]}…")

    # The libraries report a fraction and a phase; `on_progress` receives both,
    # while `progress` stays the plain status-text callback it always was.
    reporter = None
    if on_progress is not None:
        def reporter(fraction: float, message: str) -> None:  # noqa: ANN202
            on_progress(fraction, message)

    if params.method == UMAP:
        coords = _run_umap(data, params, reporter)
    elif params.method == TSNE:
        coords = _run_tsne(data, params, reporter)
    else:
        raise ValueError(f"unknown projection method: {params.method}")

    result = ProjectionResult(
        coords=coords,
        row_indices=row_indices,
        params=params,
        seconds=time.perf_counter() - started,
        notes=notes,
    )
    if cache is not None and fingerprint:
        cache.save(fingerprint, result)
    return result
