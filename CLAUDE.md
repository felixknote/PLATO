# PLATO — orientation for agents

PLATe Overview: a plate-map-aware image browser and embedding explorer for
high-content phenotypic screening. PySide6 (Qt6) desktop app.

This file is a **map**, not documentation. It says where things live and which
invariants matter, so a session can start work without re-deriving the layout.
Keep it to things that change slowly; do not put line numbers in it.

## Running it

- `Launch PLATO.bat` → `.venv\Scripts\python.exe -m plato.cli gui`
- The venv is **`.venv/` in the repo root**, not a conda env.
- Tests: `.venv\Scripts\python.exe -m pytest tests/ -q`
- Optional extras: `gui` (PySide6, pyqtgraph), `embed` (umap-learn, openTSNE,
  scikit-learn). The plate browser works without `embed`.

## The two arms

PLATO has two largely independent halves that share a theme and a main window:

1. **Plate Browser** — SQLite-indexed plate images, grid view, comparison.
2. **Embedding Explorer** — UMAP/t-SNE over precomputed feature vectors.

`plato/gui/main_window.py` owns the `QTabWidget` holding both.

## Layout

```
plato/
  cache/        thumbnail decode + SQLite thumbnail store
  data/         everything non-visual (the model layer)
  gui/          app shell: main window, theme, viewer, settings, branding
  views/        the widgets (both arms)
```

### data/ — where the model lives

| module | responsibility |
|---|---|
| `embeddings.py` | `EmbeddingDataset` (vectors + frame + run_info), loading a DINO export |
| `explorer_model.py` | the **join layer**: canonical column names, `build_frame`, `ImageResolver`, `FIELD_LABELS` |
| `image_lookup.py` | `ImageIndex`: resolve a metadata row to a file **by name**, never by row order |
| `projection.py` | UMAP/t-SNE, `ProjectionParams`, disk cache keyed by params |
| `annotations.py` | condition-string parsing, MoA/pathway tables, control labels |
| `cluster_stats.py` | lasso composition: rank fields by divergence from background |
| `image_stats.py` | raw per-image statistics (brightness/contrast/focus), opt-in |
| `image_features.py` | 31 descriptors used **as** an embedding when no export exists |
| `locations.py` | configured roots (env var → QSettings). **No hard-coded paths.** |
| `session.py`, `index/db.py` | the browser arm's SQLite index |

### views/ — the widgets

| module | responsibility |
|---|---|
| `explorer.py` | the Embedding Explorer: controls left, plot centre, selection right |
| `scatter.py` | `EmbeddingScatter`: pyqtgraph plot, hover, click/shift-click, lasso, density |
| `selection_panel.py` | right column: one cropped preview per selected point |
| `cluster_panel.py` | Lasso Analysis: ranked composition of a selection |
| `stats_panel.py` / `stats_worker.py` | opt-in image statistics + its progress |
| `compare_view.py` | `ComparisonView`/`ComparisonColumn`, shared locked `Viewport` |
| `palette.py` | categorical + continuous colour assignment |
| `browser.py`, `model.py`, `gallery.py`, `preview.py` | browser arm + shared decode |

## Invariants — break these and things go subtly wrong

- **Row order is not a contract.** The embedding row order is an artefact of
  the extractor's directory walk. Images are resolved through metadata
  columns via `ImageIndex`, never by index. A vectors/metadata length
  mismatch is fatal, not something to truncate.
- **Selection lives in frame-row space**, not plot positions, so it survives
  filtering and recolouring. Only a *new projection* clears it.
- **No hard-coded paths.** Roots come from `locations.py`. The software has to
  run on other people's machines.
- **Discover, don't assume.** Columns, plate layouts and category vocabularies
  vary between exports. `colour_fields`/`filter_fields`/`cluster_stats.compose`
  all derive their options from the frame.
- **Expensive work goes on `QThreadPool`**, never the GUI thread, and reports
  determinate progress. Reads are 266 MB files on a network share.
- **`read_plane` slices the memmap before materialising** — never materialise
  then discard.

## Theme

`gui/theme.py` holds the palette constants and one big `STYLESHEET`, applied
once to the `QApplication`.

**Caveat that matters:** many widgets import colour constants at module import
time and bake them into f-string stylesheets (`setStyleSheet(f"color: {TEXT}")`).
Reassigning the module constants therefore does **not** restyle anything already
built. A runtime theme switch has to re-apply the app stylesheet *and* have
widgets re-read their colours.

## Gotchas that have bitten before

- pyqtgraph reports a double click on the **second** press, after already
  delivering a normal click — so a click handler must tolerate the first
  click having already acted.
- Qt's SVG renderer **ignores `clipPath`** entirely.
- `histogram2d` returns `[x][y]`; `ImageItem` reads `[y][x]`. Transpose.
- UMAP silently drops threads if `random_state` is set; the two are exclusive.
- After L2-normalising, cosine and Euclidean give identical neighbours —
  ask for Euclidean, it has a fast path (measured 2.5× faster, ARI 1.000).
- `QSettings` defaults only apply when nothing was saved before.
