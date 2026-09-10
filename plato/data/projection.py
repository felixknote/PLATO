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
    # -- t-SNE
    #
    # Exposed because t-SNE is easy to misread and the defaults hide the
    # ways it misleads (see distill.pub/2016/misread-tsne). Cluster SIZES
    # mean nothing, distances BETWEEN clusters mean little, and both change
    # with perplexity -- so reading one t-SNE at one perplexity is exactly the
    # mistake these controls exist to prevent.
    perplexity: float = 30.0
    # Neighbourhood size. Low values fragment real clusters into sub-blobs;
    # high values merge distinct ones. openTSNE requires < n/3, enforced in
    # the runner. Looking at two or three values is the standard advice.
    n_iter: int = 500
    # Iterations AFTER early exaggeration. A run that has not converged shows
    # structure that is an artefact of where it stopped -- the classic
    # "pinched" or filament-shaped clusters. 500 is a floor, not a target;
    # 1000-2000 is normal for a converged layout.
    early_exaggeration_iter: int = 250
    early_exaggeration: float = 12.0
    # Early exaggeration inflates attraction so clusters can separate before
    # fine structure is fitted. Too little and clusters stay entangled; too
    # much and everything collapses to points.
    late_exaggeration: float = 0.0
    # Applied during the main phase; 0 means "off" (openTSNE's None). Values
    # above 1 spread clusters apart and are sometimes used to make structure
    # legible in very large datasets. It changes the picture, so it is off by
    # default and stated when on.
    learning_rate: float = 0.0
    # 0 means openTSNE's "auto" (n/early_exaggeration), which is almost always
    # right. Too low and the layout does not move; too high and it explodes.
    initialization: str = "pca"
    # "pca" is reproducible and preserves global structure, so distances
    # between well-separated clusters carry some meaning. "random" is the
    # classic t-SNE default and shows only local structure. "spectral" uses
    # the neighbour graph.
    seed: int = 1
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
        tsne_only = (
            "perplexity",
            "n_iter",
            "early_exaggeration_iter",
            "early_exaggeration",
            "late_exaggeration",
            "learning_rate",
            "initialization",
        )
        if self.method == UMAP:
            for name in tsne_only:
                relevant.pop(name, None)
        else:
            relevant.pop("n_neighbors")
            relevant.pop("min_dist")
        parts = [f"{k}={relevant[k]}" for k in sorted(relevant)]
        return "_".join(parts).replace(" ", "").replace("/", "-")


def suggest(n_points: int, *, learned: bool = True) -> ProjectionParams:
    """Sensible starting parameters for a dataset of this size and kind.

    There is no single right default. These formulas are tuned for THIS
    project's actual data: DINO features (1024-d, cosine geometry) over
    E. coli perturbation screens where the true structure is on the order of
    50-100 conditions (tens of CRISPRi genes x guides, tens of antibiotics x
    doses) in datasets of roughly 5k-50k images. That "many conditions in a
    large N" regime is exactly where a linear-in-N neighbourhood size goes
    wrong: it was previously scaled at 1.5% of the points, uncapped in
    practice below its 500 ceiling, which by ~30k points puts each point's
    neighbourhood at several times the size of a single condition (a
    condition here is roughly N / 100 points) -- UMAP and t-SNE alike then
    average distinct phenotypes together before the projection ever sees
    them, which reads on screen as "these conditions collapsed" when the
    actual cause is a parameter, not the biology.

    So the defaults instead:

    * **UMAP n_neighbors** ~ 0.4*sqrt(n), clamped to [15, 100]. Sublinear
      instead of linear, and capped well below N/100-ish points-per-condition
      rather than at 500 -- umap-learn's own documented range is 2-200, and
      100-200 is already described as favouring broad topology over local
      structure. Confirmed empirically: a silhouette-score sweep of
      n_neighbors in {5, 8, 10, 12, 15, 30, 50, 75, 100, 150, 200} against
      the real 32,256-point "Aug26 CRISPRi & ABx" DINO export (54
      ground-truth perturbations, dose/guide collapsed) found n_neighbors=30
      a genuine interior peak -- every value on both sides scored worse,
      including the smaller ones, so this is not merely "smaller is better"
      down to some floor. 0.4*sqrt(n) is calibrated to land on 30 at the
      5,000-point scale the sweep was run at (subsampled from the 32k
      export for tractable runtime) while still growing sublinearly for
      larger datasets, since the mechanism that made linear scaling wrong
      (neighbourhood outgrowing a single condition) does not stop existing
      just because 30 was measured at one particular n. (Was 1.5% of n,
      clamped to [15, 500]: linear growth meant two exports of the same
      experiment at 33k and 48k rows got the identical clamped value of 500,
      which is both too large and a parameter coincidence, not a considered
      choice.)
    * **UMAP min_dist** 0.1 for learned embeddings, not 1.0. 1.0 is close to
      umap-learn's ceiling and is documented as favouring a smooth, spread
      figure over resolving clusters -- exactly wrong for finding which of
      50-100 conditions are and are not distinct, which is the actual
      question here. (The AI4AB analysis script this recipe was copied from
      optimises for a presentation figure, not for this app's job.)
    * **t-SNE perplexity** = 4, flat, not scaled with n. This overrides the
      general literature guidance (Kobak & Berens 2019 and openTSNE's own
      docs suggest perplexity in the hundreds for tens of thousands of
      points) with a direct measurement on this project's actual embedding
      space: a silhouette-score sweep against the real 32,256-point "Aug26
      CRISPRi & ABx" DINO export (54 ground-truth perturbations,
      dose/guide collapsed) at FULL scale -- not subsampled -- tested
      perplexity in {3, 5, 7, 10, 15, 20} and found 3 and 5 tied for best
      (-0.2107 / -0.2110), both clearly ahead of 7 (-0.2275) and everything
      above. The same low-perplexity result held at a 5,000-point subsample
      first (optimum 4-5 there too), which argued for testing at full scale
      in case it was a density-per-cluster artefact of subsampling -- it
      was not; the result is stable across a 6x change in dataset size.
      The likely explanation is that these are FROZEN DINO features with no
      fine-tuning on this data, so there is no training signal that made
      the 54 perturbations linearly separable in the embedding at all
      (every score in the sweep was negative); a low perplexity finds
      whatever local structure exists without a large neighbourhood
      averaging it away first. If a future embedding is fine-tuned on this
      task and separates conditions more cleanly, this flat default should
      be re-measured rather than assumed to still hold -- it is a property
      of THIS embedding space, not a law about t-SNE.
    * **t-SNE iterations** scale with n instead of a flat 500 main / 250
      early-exaggeration: ``n_iter`` from n/25 clamped to [750, 1500],
      ``early_exaggeration_iter`` from n/50 clamped to [250, 500]. 500 total
      iterations is a documented floor for convergence, not a target
      (Belkina et al., 2019 measured under-converged defaults costing over
      30 points of 1-NN accuracy versus a properly converged run); a fixed
      count that never grows with the dataset increasingly under-converges
      as n rises, which looks identical on screen to genuine cluster
      collapse -- the one confound most worth ruling out before reading
      "these conditions look the same" as a biological conclusion.
    * **subsample** kicks in only above ~40k points, where a full UMAP starts
      costing more minutes than the first look is worth.
    * **PCA** is skipped for low-dimensional inputs -- reducing 31 image
      descriptors to 50 components does nothing but cost a fit.

    ``learned`` says whether the vectors are learned embeddings (cosine
    geometry, worth normalising) or hand-computed descriptors, which are
    already standardised per feature and live in a Euclidean space.
    """
    import math

    n_points = max(1, int(n_points))

    neighbours = int(round(0.4 * math.sqrt(n_points)))
    neighbours = max(15, min(100, neighbours))
    # Never at or above the sample size: UMAP clamps it anyway, but a value
    # that has to be clamped is not a sensible default to show the user.
    neighbours = min(neighbours, max(2, n_points - 1))

    # Flat 4, empirically measured (see docstring) -- not scaled with n. Must
    # still respect openTSNE's perplexity < n/3 for a genuinely tiny dataset,
    # which the runner enforces at use time regardless, but a SUGGESTED value
    # that already needs clamping the moment it is shown is not sensible.
    perplexity = min(4.0, max(2.0, (n_points - 1) / 3.0))
    n_iter = max(750, min(1500, round(n_points / 25)))
    early_exaggeration_iter = max(250, min(500, round(n_points / 50)))

    max_points = None if n_points <= 40_000 else 20_000

    return ProjectionParams(
        method=UMAP,
        normalize=learned,
        pca_components=DEFAULT_PCA_COMPONENTS if learned else 0,
        metric="cosine" if learned else "euclidean",
        n_neighbors=neighbours,
        min_dist=0.1,
        perplexity=perplexity,
        n_iter=n_iter,
        early_exaggeration_iter=early_exaggeration_iter,
        early_exaggeration=12.0,
        late_exaggeration=0.0,
        learning_rate=0.0,
        initialization="pca",
        seed=RANDOM_STATE,
        max_points=max_points,
        deterministic=False,
    )


# Iteration presets. The names describe what you get, not a number, because
# the right count depends on the dataset -- but the ordering is the point:
# a fast look is not a converged layout, and saying so beats a single default
# that is quietly one or the other.
#
# "Standard" raised from 500/250 to 750/250: 500 total iterations is
# openTSNE's documented convergence FLOOR, not a target, and this project's
# datasets (5k-48k points) sit above the size where that floor is enough --
# see suggest()'s n_iter formula, which starts at 750 for the same reason.
# Keeping "Standard" at the old value would mean the preset ladder's own
# middle rung under-converges the moment n_points scales past a few thousand.
TSNE_PRESETS: tuple[tuple[str, int, int], ...] = (
    ("Fast exploration", 250, 125),
    ("Standard", 750, 250),
    ("High quality", 1500, 400),
    ("Very high quality", 3000, 500),
)


def tsne_preset(name: str) -> tuple[int, int] | None:
    """(n_iter, early_exaggeration_iter) for a preset name."""
    for preset, iterations, early in TSNE_PRESETS:
        if preset == name:
            return iterations, early
    return None


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

    # openTSNE requires perplexity < n_samples / 3. Clamped rather than
    # rejected: a perplexity that is merely too large for a small subsample is
    # a reasonable thing to have typed, and failing the run would lose it.
    perplexity = float(max(5.0, min(params.perplexity, (data.shape[0] - 1) / 3.0)))
    n_iter = max(50, int(params.n_iter))
    early_iter = max(0, int(params.early_exaggeration_iter))

    # openTSNE takes a real callback. The neighbour search before
    # optimisation reports nothing, so it owns the first third of the bar and
    # the callback fills the rest.
    if progress is not None:
        progress(0.05, "finding nearest neighbours")

    # "auto" is openTSNE's own adaptive choice and is almost always right;
    # 0 in our params means "let it decide" rather than "use zero".
    learning_rate = params.learning_rate if params.learning_rate > 0 else "auto"
    exaggeration = params.late_exaggeration if params.late_exaggeration > 0 else None

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric=effective_metric(params),
        initialization=params.initialization,
        learning_rate=learning_rate,
        early_exaggeration=params.early_exaggeration,
        early_exaggeration_iter=early_iter,
        exaggeration=exaggeration,
        # Unlike UMAP, openTSNE parallelises with a seed set, so this stays
        # reproducible AND threaded.
        random_state=int(params.seed),
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

    def clear(self) -> int:
        """Delete every cached projection. Returns how many were removed.

        Counted in pairs (.npz + .json), not files, so the number reported
        means "projections" to whoever reads it rather than "files on disk".
        A missing counterpart (an interrupted save) is still removed -- this
        is a cleanup operation, not a consistency check.
        """
        if not self.directory.is_dir():
            return 0
        removed = 0
        for meta_path in self.directory.glob("*.json"):
            array_path = meta_path.with_suffix(".npz")
            if array_path.exists():
                array_path.unlink()
            meta_path.unlink()
            removed += 1
        # Any .npz left without a .json (a save interrupted after the array
        # write) counts too, since it is still disk space this is meant to
        # reclaim, and would otherwise linger forever.
        for array_path in self.directory.glob("*.npz"):
            array_path.unlink()
            removed += 1
        return removed


def project(
    vectors: np.ndarray,
    params: ProjectionParams,
    *,
    fingerprint: str = "",
    cache: ProjectionCache | None = None,
    progress=None,
    on_progress=None,
    is_cancelled=None,
) -> ProjectionResult:
    """Project ``vectors`` to 2-D, reusing a cached result when one exists.

    ``progress`` is an optional callable taking a status string.
    ``on_progress`` is an optional ``callable(fraction, message)`` driven by
    the backend's own reporting, for a determinate progress bar.

    ``is_cancelled`` is an optional ``callable() -> bool``, checked right
    after the (uninterruptible) fit returns. Neither UMAP nor openTSNE
    exposes a way to stop a fit already in progress, so a cancelled run still
    burns the CPU time -- but it must not look like it succeeded: skipping
    the cache write here is what stops a "cancelled" run from silently
    reappearing, fully computed, the next time the same parameters are asked
    for.
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
    if is_cancelled is not None and is_cancelled():
        # The work already happened -- there is no way to have stopped it --
        # but a cancelled run must not leave behind a cache entry that makes
        # the NEXT run of the same parameters silently return this result
        # instead of actually computing.
        return result
    if cache is not None and fingerprint:
        cache.save(fingerprint, result)
    return result
