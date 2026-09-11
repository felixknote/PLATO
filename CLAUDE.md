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
| `grid_view.py` | `GridView`: one facet per group, shared axes/legend, page export |
| `tsne_panel.py` | `TsnePanel`: t-SNE's own parameters, presets, misread caveats |
| `selection_panel.py` | right column: one cropped preview per selected point |
| `cluster_panel.py` | Lasso Analysis: ranked composition of a selection |
| `stats_panel.py` / `stats_worker.py` | opt-in image statistics + its progress |
| `compare_view.py` | `ComparisonView`/`ComparisonColumn`, shared locked `Viewport` |
| `section.py` | `Accordion`/`Section`: the sidebar's one-open-at-a-time column |
| `palette.py` | value -> colour/shape assignment, stable across filters |
| `palettes.py` | the palette registry: every named categorical/sequential/diverging scale |
| `browser.py`, `model.py`, `preview.py` | browser arm + shared decode |

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

**Two hues are reserved and mean one thing each**, in both themes:

| hue | meaning | where |
|---|---|---|
| cyan | selected / active | `accent`; raw `#1af8fe` for the live lasso only |
| magenta | control wells, flagged | `flag` |

Both are held OUT of the categorical scales PLATO owns (`plato`, `deep`,
`bright` in `views/palettes.py`), so a coloured mark in a plot is never the
same hue as an interface state. `tests/test_section_accordion.py` enforces a
15-degree minimum. Okabe-Ito and Tableau are deliberately exempt: they are
external standards whose value is that a reader already knows them.

The chrome descends from the logo (`scripts/make_logo.py`) -- a deep navy
plate with wells glowing cyan and magenta -- rather than running a separate
generic blue alongside it.

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
- `QWidget.grab()` on a window that was just shown renders STALE geometry --
  observed painting a collapsed accordion section at its old expanded height,
  and clipping widgets that measured correctly. Verify layout by reading
  `geometry()`, and screenshot via `screen.grabWindow(win.winId())` after the
  event loop has settled, not `widget.grab()`.
- `QSizePolicy.Ignored` on a label lets the layout collapse it to width 0
  while `isVisible()` still returns True and `sizeHint()` still reports the
  full width -- it renders nothing and every measurement says it is fine.
  Use `Preferred` plus explicit `elidedText` when a label must yield space.
- An unstyled `QSlider` takes its groove colour from the QPalette Highlight,
  not the stylesheet, so it keeps the old accent after a theme change until
  a `QSlider::sub-page` rule exists.
- pyqtgraph draws same-z-value `ScatterPlotItem`s in ADD order, which for the
  explorer's colour groups was alphabetical by category label -- so
  whichever category sorted last always painted over every other one where
  they overlapped. `scatter._draw_order` fixes this by chunking each colour
  group and shuffling chunk order (not just group order) rather than one
  colour fully occluding another; per-point brushes on a single item would
  be correct too but cost ~25x more at 32k points (measured).
- `QWidget.width()`/`.height()` on a widget that is not currently shown are
  **not** zero or otherwise safe — Qt hands back a stale/default value
  (observed: 640x480) unrelated to its actual layout. Never `max()` that
  against `sizeHint()`; check `isVisible()` first and use live geometry only
  when it is true, `sizeHint()` only when it is not (see `GridView.export`).
- A `QComboBox`'s value->colour/shape mapping must be built from the FULL
  column (`distinct_values`/`_colour_universe`), never from whichever rows
  are currently visible after a filter — deriving it from the filtered
  subset reassigns other categories' colours the moment one category is
  filtered out, since assignment is by positional index into the palette.
- `QWidget.isVisible()` reflects the WHOLE ancestor chain being shown, so a
  widget built for a test and never added to a shown window reads `False`
  regardless of its own `setVisible(True)` — check `isVisibleTo(parent)`
  instead when testing a widget that is never actually put on screen.
- Connecting a second listener onto a signal a dialog already wired to its
  own handler (e.g. a `_EntryRow`'s `add_folder_requested`, which
  `LocateAllDialog` connects to `_add_folder_for`) fires BOTH on emit — if
  the dialog's own handler opens a real `QFileDialog`, that blocks
  headlessly with no visible error. Test such a row's own signal wiring on a
  bare instance, not one already owned by the dialog.
