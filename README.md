# PLATO — PLATe Overview

Browse high-content screening images by experimental condition instead of by
filename. Point it at an image folder and a plate map; it joins them, validates
the join, and gives you a filterable thumbnail grid with a full-resolution
viewer behind it.

## Install

```bash
cd plato-browser
pip install -e ".[gui,dev]"
```

Python 3.11+. The indexing layer needs only numpy/pandas/tifffile/Pillow; the
GUI adds PySide6 and pyqtgraph.

## Try it on synthetic data

```bash
python scripts/make_demo_data.py /tmp/demo --defects
cd /tmp/demo
plato index      # parses, joins, validates
plato thumbs     # renders the thumbnail cache
plato gui
```

`--defects` injects an unparseable filename, an out-of-plate well and a
duplicate plate map row so you can see what the validation report does with them.

## Use it on your data

```bash
plato init plato.toml     # starter config
# edit [images].pattern and [platemap] to match your naming scheme
plato index -c plato.toml
plato thumbs -c plato.toml --workers 8
plato gui -c plato.toml
```

The only thing you normally have to change is the filename pattern. It is a
named-group regex matched against the file name:

```toml
pattern = '^(?P<plate>[^_]+)_(?P<well>[A-P]\d{1,2})_f(?P<field>\d+)_(?P<channel>[A-Za-z0-9]+)\.tif$'
```

`well` is required. `plate`, `field`, `channel` and `z` are optional — capture
what your files actually encode. Unknown group names are rejected at startup
rather than silently ignored.

## Plate map layouts

`[platemap].layout` takes two values.

**`long`** — one row per well, a `Well` column plus one column per field. What a
LIMS or picklist exports.

**`matrix`** — the plate drawn as a grid, one cell per well, as you would lay it
out by hand:

```csv
rplA_3,folP_3,parE_1,lpxC_2,ftsZ_3,ACE-1 NC_6,parE_3,rpsL_1,...
mrcB_1,lpxA_2,MG1655 NC_6,mrdA_3,lptC_2,dnaE_1,ACE-1 NC_4,...
```

Row and column labels are optional (`index_col`, `header_row`); without them
the position in the grid *is* the well. The grid shape is checked against
`plate_format`, so a missing row raises rather than shifting every well by
twelve — the failure mode that makes a whole plate quietly wrong.

Matrix cells usually pack several fields into one string. `split_pattern`
unpacks them:

```toml
value_column = "Condition"
split_pattern = '^(?P<gene>.+)_(?P<replicate>\d+)$'
```

Each named group becomes a filterable column, so `ftsZ_1`, `ftsZ_2` and
`ftsZ_3` collapse into one gene with three replicates instead of three
unrelated conditions. Cells that do not match are kept verbatim and reported.

`plate_pattern` pulls the plate name out of the map's file name
(`..._P1.csv` → `P1`), so one config serves a series of per-plate maps.

## Filename tokens

Capture groups: `well` (required), plus `plate`, `field`, `channel`, `z`,
`seq`, `point`. Anything else is rejected at startup rather than ignored.

NIS-Elements output, for example:

```
WellA01_PointA01_0000_ChannelCam-DIA DIC Master Screening_Seq0000.tiff
```

```toml
pattern = '^Well(?P<well>[A-P]\d{1,2})_Point(?P<point>[^_]+)_(?P<field>\d+)_Channel(?P<channel>.+?)_Seq(?P<seq>\d+)\.tiff?$'

[images.channel_aliases]
"Cam-DIA DIC Master Screening" = "DIC"
```

The channel group is lazy so the channel name can contain spaces *and*
underscores — export and upload tools swap one for the other, and both forms
parse. `channel_aliases` shortens the label for filters and captions; the raw
name stays in the file.

`point` is captured rather than discarded because NIS repeats the well there.
If the Well token and the Point name disagree, the acquisition was re-pointed
or renamed mid-run and the image may not be from the well it is filed under —
`plato index` reports every such file.

A worked config for this exact format is in
`examples/nis_elements_matrix/plato.toml`.

## Read the validation report

`plato index` prints, and writes to `.plato/index_report.json`:

| Check | Why it matters |
|---|---|
| filenames not parsed | Files invisible to the browser |
| images with no plate map row | Wells you will browse without knowing the condition |
| plate map wells with no image | Acquisition gaps — often the interesting ones |
| duplicate plate map keys | Would fan out the join; extra rows are dropped and counted |
| unparseable plate map wells | Typos, merged cells, stray footer rows |
| plate map cells not split by pattern | `split_pattern` does not fit every cell |
| filename Well/Point disagreements | Image possibly filed under the wrong well |
| wells with unusual field/channel count | Partial acquisitions, aborted runs |

Nothing here is fatal by design — `--strict` makes it exit non-zero if you want
to gate a pipeline on a clean join.

## Keyboard

| Key | Action |
|---|---|
| Arrows | move between thumbnails |
| Enter | open full resolution |
| Space | flag / unflag selection |
| 1–5 / 0 | rate / clear rating |
| R | random image |
| Ctrl+D | compare two conditions side by side |
| Ctrl+Shift+D | compare along a variable — one image per value, stepped together |
| ←/→ (comparison) | step only the selected column; click a column to select it |
| scroll (comparison) | zoom every column, centred on the cursor |
| drag (comparison) | pan every column |
| + / − / 0 (comparison) | zoom in / out / reset |
| Ctrl+Shift+W | close the comparison |
| Ctrl+B | blinded review |
| ←/→ (viewer) | step through the current filter |
| A (viewer) | per-image autoscale |
| S (viewer) | toggle scale bar |

## Comparing along a variable

`Ctrl+Shift+D` sets up the comparison a dose series or a timepoint course is
actually for: hold one condition constant, pick a variable to compare along,
and tick which of its values to show.

```
Hold constant   Antibiotic = Ciprofloxacin
Compare along   Concentration
Show            1/8x  1/4x  1/2x  1x
```

You get one column per value, each showing a single image at size rather than
a grid of thumbnails, so the only thing differing between columns is the
variable. Every column shares the same fixed per-channel contrast, which means
a brightness difference between columns is a difference in the sample and not
in scaling — tick **Autoscale each image** when you need faint detail out of
one slice and can accept that the columns stop being directly comparable.

`Previous`/`Next` step every column together, keeping position N of each slice
on screen at once. The `←`/`→` arrows step only the selected column — click a
column to select it, it takes a highlight frame — which is what you want when
the Nth field of one slice is out of focus and you need a comparable one.
Columns can then fall out of step, so the position label says so and
**Realign** puts them all back on the selected column's position.

Columns are ordered by the magnitude of their value, not as text, so a dose
series reads left to right as a dose series: `1/8x  1/4x  1/2x  1x`, not the
`1/2x  1/4x  1/8x  1x` a plain string sort gives. The same applies to numeric
concentrations (`2` before `10`) and to timepoints (`T2` before `T10`).
Controls that carry no number sort to the end rather than into the middle of
the series, and value lists that are names rather than a series — channels,
antibiotics — stay alphabetical.

Scroll to zoom, drag to pan; `+`/`−` zoom from the keyboard and `0` resets.
Zoom is **shared by every column**: the view exists so the only difference
between columns is the compared variable, and columns showing different
regions of their images would quietly break that. Zoomed in, the columns
re-read their images at proportionally more detail rather than magnifying the
pixels they already had.

**Export what's on screen** writes exactly the visible images, one per column
— the comparison you are looking at, not the hundred images behind it. Format
choice, scale-bar baking and shared contrast behave as they do in the grid.

If nothing is held constant, each column mixes every condition at that value,
which looks like a comparison and is not one; the dialog says so rather than
letting it through quietly.

## Working with several plates

**Data → Add Plate…** loads another image folder and plate map alongside the
ones already open. Each plate keeps its own index database and thumbnail
cache — nothing is re-indexed — and the grid, the filters, the search and the
comparison view all query the union.

**Data → Loaded Plates…** lists what is open, with the image count and source
folder of each, and removes one. Removing unloads it from the session only:
the images, flags and ratings stay on disk, and adding it again brings them
back.

Three things this has to get right, because a screen is rarely one plate:

**Telling two plates apart.** The `plate` column comes from your filenames or
plate map, and for a series of folders indexed with the same config it is the
same string in every one of them — every row says `Plate1`. Filtering on it
then selects all of them at once. So the session gives each loaded plate a
name of its own (the plate map's `plate_pattern` match, else the image folder
name), disambiguated to `name (2)` if two collide, and offers it as its own
**Loaded Plate** filter once more than one is open. That name also prefixes
the thumbnail captions, appears in the metadata panel and the viewer title,
and is matched by the search box.

**Contrast across plates.** Each plate estimates its own per-channel display
limits from its own images, so plates imaged on different days arrive with
different ones. Taking the first plate's and applying them to the rest makes
a brightness difference between plates a difference in *scaling*, in a view
whose whole premise is that it is a difference in the sample. The limits are
therefore pooled — lowest lo, highest hi — so one window contains every
plate's data and all of them are stretched identically. A dim plate stays
visibly dim rather than being levelled up to look like the bright one.

**Not loading the same plate twice.** Adding a plate whose index is already
open is refused rather than doubling the grid: every image would appear
twice, every count would be wrong, and flagging one copy would leave the
other unflagged.

Ordering keeps each plate contiguous rather than interleaving them by well,
so scrolling reads as one plate after another. Exported annotations carry
`session_plate`, `plate_source` and a `uid` keyed on the owning index
database, so a batch export from a multi-plate session can be traced back to
actual files — `image_id` alone is only unique within one plate.

If your plate folders are named `P13_T1`, `P13_T2` and so on, the trailing
`_T<n>` is also offered as a **timepoint** filter, which slices across plates
(all T1 wells, regardless of plate). Turn it off in Settings if your folder
names end in something that is not a timepoint.

## Two things that are deliberate, not oversights

**Contrast is fixed per channel across the whole screen.** Limits are estimated
once from a random sample and reused for every thumbnail and (by default) the
full-resolution view. Per-image autoscaling makes a dead well and a healthy
well look identical, which destroys exactly the comparison you are making by
eye. Press `A` in the viewer when you want per-image contrast for faint detail.

**Blinded review exists because flagging with the label visible is not scoring,
it is confirmation.** `Ctrl+B` hides the metadata panel and captions and
randomises the order. If your flags feed anything downstream — training labels,
hit calls, figure selection — score blind and reveal afterwards.

## Architecture

```
config.py     TOML -> dataclasses; nothing experiment-specific is hardcoded
wells.py      A1 / A01 / a1 -> canonical (row, col). Join failures start here.
ordering.py   filter values sorted by magnitude, so a dose series reads as one
index/
  filenames.py  regex -> ImageRecord
  platemap.py   Excel/CSV -> tidy frame, sanitised column names
  build.py      join + ValidationReport + write
  db.py         SQLite schema and the query API the GUI uses
cache/
  thumbnails.py offline PNG cache in SQLite, fixed per-channel levels
gui/
  session.py    several IndexDBs unioned behind one query API; plate identity
  model.py      QAbstractListModel, thumbnails fetched off-thread
  delegate.py   tile painting
  panel.py      filters + grid + metadata (one panel; two = compare mode)
  plates_dialog.py  what is loaded, and unloading one
  viewer.py     pyqtgraph full-resolution window
  main_window.py
```

Derived state (`images`, `platemap`, `display_limits`, thumbnails) is
regenerable — delete `.plato/` and rebuild. Annotations live in a separate table
keyed by path-relative `image_id` and survive reindexing. They do **not**
survive renaming your files.

The layers are separable on purpose: `plato.index` has no Qt dependency,
so you can use the same joined table in a notebook:

```python
from plato.config import load_config
from plato.index import load_dataframe

df = load_dataframe(load_config("plato.toml"))
```

## Measured behaviour

On a synthetic 60k-image index (SQLite, cold):

| Query | Time |
|---|---|
| filtered (gene + channel) | ~40 ms |
| free-text search | ~150 ms |
| unfiltered, all 60k | ~460 ms |

Filtered browsing — what you actually do — is fast. Clearing all filters at 60k
images blocks the UI for roughly half a second while the rows are materialised.
If that becomes annoying, move `IndexDB.query` onto a worker thread; the model
already handles asynchronous updates.

Thumbnail rendering is the slow part and runs once. Measured on the real
2720×2720 uint16 uncompressed DIC TIFFs: a 2016-image plate builds in ~11 s
on a 36-core box (default worker count), so roughly a minute for 10k images.
It is incremental after that (keyed on mtime and size), so re-running only
picks up what changed.

Two things make that fast, both in `cache/thumbnails.py`. Planes are
memory-mapped rather than read through `tifffile.imread` — these files are
uncompressed with one row per strip, so the OS can map the bytes instead of
reassembling 2720 strips in Python, and the result is bit-identical.
Thumbnails then read every second pixel (`THUMBNAIL_READ_STRIDE`), which is
still ~5× more data than a 256px tile needs; the final PNGs differ from a
full read by at most 12/255. Both are thumbnail-only — the viewer and every
export path read full resolution.

The side-by-side comparison decodes at ~1200px rather than full resolution
and prefetches the neighbouring image in each column, so stepping is a cache
lookup (~0 ms) rather than a ~33 ms decode per column.

## Known limits

- One plane per thumbnail. Multi-page TIFFs are reduced to the first plane for
  the grid; the viewer shows the whole stack.
- Channels are separate rows, not composited. A per-well RGB overlay would need
  a compositing step in the thumbnail builder.
- No pyramidal/tiled loading in the viewer. Fine to ~4k×4k; whole-slide images
  would need OME-Zarr and a different viewer.
- Ratings are per image, not per well. If you want well-level scoring, add a
  second annotations table keyed on `(plate, well)`.
- Display limits are pooled across loaded plates by taking the widest window
  (lowest lo, highest hi), not re-estimated from a sample drawn across all of
  them. A plate with one very bright outlier therefore widens the window for
  every plate. Re-sampling across the union would be more precise and would
  mean re-reading images at load time.
- Blinded review hides the metadata and randomises order, but does not clear
  the filters. Filtering to one condition and then entering blind mode leaves
  you scoring a set you already know the label of.

## Tests

```bash
pytest -q                                              # indexing + session layers
QT_QPA_PLATFORM=offscreen python tests/test_gui_smoke.py /tmp/demo/plato.toml
```

`tests/test_session.py` covers what only goes wrong once a second plate is
loaded — plate identity, pooled contrast, and merging two sorted result sets
— without needing Qt to be on screen.
