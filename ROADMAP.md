# PLATO roadmap

Open work, in rough priority order. Each entry says what is wrong or missing
and what is already known about it, so picking one up does not mean
re-deriving the diagnosis.

## Image viewer in the Embedding Explorer

**Status: broken for the `Jul26 ABx` export. Diagnosed, not fixed.**

The preview column shows an empty frame. It is *not* a decode bug -- the
decode chain, the crop, the panel and the resize path all produce correct
pixmaps against real TIFFs. The cause is a data mismatch:

```
metadata image_name:  20260728_143951_318__WellA01_PointA01_0000_...
files under E:\Data\2026_07_28 AI4AMR Antibiotics:
                      20260730_130742_149__WellA01_PointA01_0000_...
```

Same wells, same points, **different acquisition session** (28 July vs 30
July). The name-based lookup correctly matches nothing.

The real defect is that `ImageResolver.detect` accepted that root anyway:
a root where *zero* rows resolve should be rejected, and the UI should say
"these images are not here" and point at Locate…, instead of showing a blank
frame that looks like a rendering failure.

To do:
* `detect` must require a minimum hit rate, not merely "a directory exists".
* Surface the count ("0 of 24,192 rows resolved here") in the source label.
* Then establish where the 28-July acquisition actually lives, or accept that
  this export has no images on this machine and say so.

Note: E: is a mechanical disk shared with a long-running training job
(measured 25.6 ms average read latency, queue length 39). Verification
against real images should wait for that job, or use a local copy.

## Full revision of usability and layout

The left sidebar has outgrown its column. The t-SNE panel alone is four
nested group boxes, and the wrapping fix applied to it is a patch over a
structural problem: the sidebar is now a long scroll of unrelated sections.

Wants collapsible sections, or tabs (Data / Embedding / Encoding / Analysis),
rather than more scrolling. Also worth revisiting:
* which controls deserve to be visible at all times vs. behind a disclosure;
* whether Display should be split into encoding vs. layout.

(The Projection / t-SNE settings split that used to duplicate perplexity in
two places -- a standalone box shown only for UMAP, where it was never read,
alongside the t-SNE panel's own control -- is fixed: the standalone box is
gone, and the t-SNE panel is now the sole authority on perplexity, seeded
from the dataset via `apply_suggested_params` -> `TsnePanel.load_from`.)

## 3D projections

`n_components=3` in `projection.py` is trivial (plus a `ProjectionParams`
field so the cache key distinguishes 2-D from 3-D). The cost is entirely in
the view: pyqtgraph's `GLViewWidget` is a separate OpenGL stack with no
`ScatterPlotItem`, no SVG export and its own hit-testing, so hover, lasso and
selection would need reimplementing rather than extending. 2-D remains the
default either way.

## Re-validate the empirical UMAP/t-SNE defaults if the embedding changes

`plato/data/projection.py`'s `suggest()` sets t-SNE perplexity to a flat 4
and UMAP `n_neighbors` to `0.4*sqrt(n)` (peaking at 30 around 5k points) not
from general literature guidance but from a direct silhouette-score sweep
against the real `Aug26 CRISPRi & ABx` DINO export, scored against 54
ground-truth perturbations. Both results are specific to *frozen, untuned*
DINO features -- the low optimum plausibly comes from there being no
training signal that made these particular conditions separable in the
embedding space at all. If a future embedding is fine-tuned on this task (or
a fundamentally different feature extractor is used), these defaults should
be re-measured rather than assumed to still hold. The methodology (subsample
sweep, then confirm at full scale) is worth reusing; see `suggest()`'s
docstring for the exact numbers found at each grid point.

## Smaller items

* The logo's outer arcs: 4 strands above and below the plate where the
  reference has 2.
* Light theme has not been checked widget by widget; the mechanism works and
  most labels now take their colour from the global stylesheet, but anything
  still baking colours at import will stay dark until rebuilt.

## Done, kept for context

* **Joint projection across embeddings.** `plato/data/joint_projection.py`
  concatenates several open embeddings' vectors and fits ONE projection over
  the union (`make_joint_entry`), so position is genuinely comparable across
  datasets rather than two independent UMAPs that happen to share a plot.
  Verified against real data: combining `Jul26 ABx` + `Jul26CRISPRi`
  (48,384 points) produced correct frame length, correct "Colour by Dataset"
  legend, and a t-SNE fit completing in ~16s. Reached via **Combine
  embeddings…**.
