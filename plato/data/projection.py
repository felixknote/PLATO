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
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

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

    def key(self) -> str:
        relevant = asdict(self)
        if self.method == UMAP:
            relevant.pop("perplexity")
        else:
            relevant.pop("n_neighbors")
            relevant.pop("min_dist")
        parts = [f"{k}={relevant[k]}" for k in sorted(relevant)]
        return "_".join(parts).replace(" ", "").replace("/", "-")


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


def _run_umap(data: np.ndarray, params: ProjectionParams) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise MissingDependency("umap-learn") from exc

    # n_neighbors must stay below the sample count; the tuned default of 500
    # is larger than some datasets are.
    n_neighbors = int(max(2, min(params.n_neighbors, data.shape[0] - 1)))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=params.min_dist,
        metric=params.metric,
        random_state=RANDOM_STATE,
    )
    return np.asarray(reducer.fit_transform(data), dtype=np.float32)


def _run_tsne(data: np.ndarray, params: ProjectionParams) -> np.ndarray:
    try:
        from openTSNE import TSNE
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise MissingDependency("openTSNE") from exc

    # openTSNE requires perplexity < n_samples / 3.
    perplexity = float(max(5.0, min(params.perplexity, (data.shape[0] - 1) / 3.0)))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric=params.metric,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return np.asarray(tsne.fit(data), dtype=np.float32)


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
) -> ProjectionResult:
    """Project ``vectors`` to 2-D, reusing a cached result when one exists.

    ``progress`` is an optional callable taking a status string; it is the only
    feedback available during a multi-minute fit.
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
    if params.method == UMAP:
        coords = _run_umap(data, params)
    elif params.method == TSNE:
        coords = _run_tsne(data, params)
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
