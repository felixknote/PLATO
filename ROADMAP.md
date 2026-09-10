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
* the Projection / t-SNE settings split, which currently duplicates
  perplexity in two places depending on the method;
* whether Display should be split into encoding vs. layout.

## 3D projections

`n_components=3` in `projection.py` is trivial (plus a `ProjectionParams`
field so the cache key distinguishes 2-D from 3-D). The cost is entirely in
the view: pyqtgraph's `GLViewWidget` is a separate OpenGL stack with no
`ScatterPlotItem`, no SVG export and its own hit-testing, so hover, lasso and
selection would need reimplementing rather than extending. 2-D remains the
default either way.

## Joint projection across embeddings

The workspace holds several embeddings, but the grid facets one at a time.
"Display by dataset" across *open* embeddings needs their coordinate systems
reconciled -- two independent UMAPs are not comparable point for point.
`Workspace.combined_frame()` exists for the aggregate questions; the honest
version needs a projection fitted over the union, which is a design decision
rather than a wiring job.

## Smaller items

* The logo's outer arcs: 4 strands above and below the plate where the
  reference has 2.
* `views/gallery.py` is now unreferenced since the selection panel replaced
  the strip under the plot.
* Light theme has not been checked widget by widget; the mechanism works and
  most labels now take their colour from the global stylesheet, but anything
  still baking colours at import will stay dark until rebuilt.
