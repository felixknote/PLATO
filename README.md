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
| Ctrl+Shift+W | close the comparison |
| Ctrl+B | blinded review |
| ←/→ (viewer) | step through the current filter |
| A (viewer) | per-image autoscale |
| S (viewer) | toggle scale bar |

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
index/
  filenames.py  regex -> ImageRecord
  platemap.py   Excel/CSV -> tidy frame, sanitised column names
  build.py      join + ValidationReport + write
  db.py         SQLite schema and the query API the GUI uses
cache/
  thumbnails.py offline PNG cache in SQLite, fixed per-channel levels
gui/
  model.py      QAbstractListModel, thumbnails fetched off-thread
  delegate.py   tile painting
  panel.py      filters + grid + metadata (one panel; two = compare mode)
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
2720×2720 uint16 uncompressed DIC TIFFs: ~52 ms per image at 4 workers
(192 images in 10 s), so roughly 15 minutes for 10k images at 8 workers. It is incremental
after that (keyed on mtime and size).

## Known limits

- One plane per thumbnail. Multi-page TIFFs are reduced to the first plane for
  the grid; the viewer shows the whole stack.
- Channels are separate rows, not composited. A per-well RGB overlay would need
  a compositing step in the thumbnail builder.
- No pyramidal/tiled loading in the viewer. Fine to ~4k×4k; whole-slide images
  would need OME-Zarr and a different viewer.
- Ratings are per image, not per well. If you want well-level scoring, add a
  second annotations table keyed on `(plate, well)`.

## Tests

```bash
pytest -q                                              # indexing layer
QT_QPA_PLATFORM=offscreen python tests/test_gui_smoke.py /tmp/demo/plato.toml
```
