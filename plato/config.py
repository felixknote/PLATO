"""Configuration.

Everything that differs between experiments lives in one TOML file. Nothing
about your naming scheme, plate map layout or microscope is hardcoded in the
application, because those change more often than the code does.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_TEMPLATE = '''\
# PLATO configuration
# Paths may be absolute or relative to this file.

[project]
name = "my_screen"
# Where the index database and thumbnail cache are written.
work_dir = "./.plato"
plate_format = 96

[images]
dir = "./images"
glob = "**/*.tif*"
# Named capture groups. Required: well.
# Optional: plate, field, channel, z, seq, point. Unknown names are rejected.
pattern = '^(?P<plate>[^_]+)_(?P<well>[A-P]\\d{1,2})_f(?P<field>\\d+)_(?P<channel>[A-Za-z0-9]+)\\.tif$'
# "name" matches the file name only; "relpath" matches the path relative to
# [images].dir, so you can capture the plate from a parent folder.
match_on = "name"

# Optional: shorten unwieldy microscope channel names for filters and captions.
# [images.channel_aliases]
# "Cam-DIA DIC Master Screening" = "DIC"

[platemap]
path = "./platemap.xlsx"
sheet = 0                 # sheet index or name
layout = "long"           # "long" = one row per well, "matrix" = plate grid
default_plate = "Plate1"  # used when nothing else supplies a plate name
# Optional regex on the plate map FILE NAME; wins over default_plate.
# plate_pattern = '_(?P<plate>P\\d+)\\.'

# --- layout = "long" ---
well_column = "Well"
plate_column = ""         # leave empty for a single-plate experiment

# --- layout = "matrix" ---
# The plate drawn as a grid (8x12 for 96-well). The grid shape is validated
# against plate_format, so a stray row or column is an error, not a silent shift.
# header_row = false      # first row holds column numbers 1..12
# index_col = false       # first column holds row letters A..H
# value_column = "Condition"
# Cells often pack several fields into one string ("ftsZ_2"). Splitting them is
# what makes filtering by gene collapse replicates instead of treating
# ftsZ_1/ftsZ_2/ftsZ_3 as three unrelated conditions.
# split_pattern = '^(?P<gene>.+)_(?P<replicate>\\d+)$'

[thumbnails]
size = 384                # long edge, pixels
# Percentiles used to derive FIXED per-channel display limits across the whole
# screen. Per-image autoscaling would make a dead well look like a healthy one.
percentiles = [1.0, 99.5]
sample_size = 200         # images sampled per channel to estimate the limits

[masks]
# Optional segmentation mask overlay. Leave dir empty to disable.
dir = ""
# Format string; available fields: plate, well, row, col, field, channel, stem
pattern = "{stem}_mask.tif"

[gui]
# Plate map columns shown under each thumbnail (kept short).
caption_fields = ["Gene", "Antibiotic", "Concentration"]
# Plate map columns offered as filters. Empty list = offer every column.
filter_fields = []
thumbnail_size = 220
'''


@dataclass(slots=True)
class ProjectConfig:
    name: str = "screen"
    work_dir: Path = Path(".plato")
    plate_format: int = 96


@dataclass(slots=True)
class ImagesConfig:
    dir: Path = Path("images")
    glob: str = "**/*.tif"
    pattern: str = r"^(?P<well>[A-P]\d{1,2})\.tif$"
    match_on: str = "name"  # "name" or "relpath"
    channel_aliases: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class PlatemapConfig:
    path: Path = Path("platemap.xlsx")
    layout: str = "long"  # "long" (one row per well) or "matrix" (plate grid)
    sheet: int | str = 0
    default_plate: str = "Plate1"
    plate_pattern: str = ""  # regex on the plate map FILE NAME, e.g. r"_(?P<plate>P\d+)"
    # long layout
    well_column: str = "Well"
    plate_column: str = ""
    # matrix layout
    header_row: bool = False
    index_col: bool = False
    value_column: str = "condition"
    split_pattern: str = ""


@dataclass(slots=True)
class ThumbnailsConfig:
    size: int = 384
    percentiles: tuple[float, float] = (1.0, 99.5)
    sample_size: int = 200


@dataclass(slots=True)
class MasksConfig:
    dir: str = ""
    pattern: str = "{stem}_mask.tif"

    @property
    def enabled(self) -> bool:
        return bool(self.dir)


@dataclass(slots=True)
class GuiConfig:
    caption_fields: list[str] = field(default_factory=list)
    filter_fields: list[str] = field(default_factory=list)
    thumbnail_size: int = 220


@dataclass(slots=True)
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    images: ImagesConfig = field(default_factory=ImagesConfig)
    platemap: PlatemapConfig = field(default_factory=PlatemapConfig)
    thumbnails: ThumbnailsConfig = field(default_factory=ThumbnailsConfig)
    masks: MasksConfig = field(default_factory=MasksConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)
    source: Path | None = None

    @property
    def db_path(self) -> Path:
        return self.project.work_dir / "index.sqlite"

    @property
    def thumb_db_path(self) -> Path:
        return self.project.work_dir / "thumbnails.sqlite"

    @property
    def report_path(self) -> Path:
        return self.project.work_dir / "index_report.json"


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path) -> Config:
    """Read a TOML config. Relative paths resolve against the config file."""
    path = Path(path).expanduser().resolve()
    with path.open("rb") as handle:
        raw: dict[str, Any] = tomllib.load(handle)
    base = path.parent

    proj = raw.get("project", {})
    img = raw.get("images", {})
    pmap = raw.get("platemap", {})
    thumb = raw.get("thumbnails", {})
    masks = raw.get("masks", {})
    gui = raw.get("gui", {})

    cfg = Config(
        project=ProjectConfig(
            name=proj.get("name", "screen"),
            work_dir=_resolve(base, proj.get("work_dir", ".plato")),
            plate_format=int(proj.get("plate_format", 96)),
        ),
        images=ImagesConfig(
            dir=_resolve(base, img.get("dir", "images")),
            glob=img.get("glob", "**/*.tif"),
            pattern=img["pattern"] if "pattern" in img else ImagesConfig.pattern,
            match_on=img.get("match_on", "name"),
            channel_aliases=dict(img.get("channel_aliases", {})),
        ),
        platemap=PlatemapConfig(
            path=_resolve(base, pmap.get("path", "platemap.xlsx")),
            layout=pmap.get("layout", "long"),
            sheet=pmap.get("sheet", 0),
            default_plate=pmap.get("default_plate", "Plate1"),
            plate_pattern=pmap.get("plate_pattern", ""),
            well_column=pmap.get("well_column", "Well"),
            plate_column=pmap.get("plate_column", ""),
            header_row=bool(pmap.get("header_row", False)),
            index_col=bool(pmap.get("index_col", False)),
            value_column=pmap.get("value_column", "condition"),
            split_pattern=pmap.get("split_pattern", ""),
        ),
        thumbnails=ThumbnailsConfig(
            size=int(thumb.get("size", 384)),
            percentiles=tuple(thumb.get("percentiles", (1.0, 99.5))),  # type: ignore[arg-type]
            sample_size=int(thumb.get("sample_size", 200)),
        ),
        masks=MasksConfig(
            dir=str(_resolve(base, masks["dir"])) if masks.get("dir") else "",
            pattern=masks.get("pattern", "{stem}_mask.tif"),
        ),
        gui=GuiConfig(
            caption_fields=list(gui.get("caption_fields", [])),
            filter_fields=list(gui.get("filter_fields", [])),
            thumbnail_size=int(gui.get("thumbnail_size", 220)),
        ),
        source=path,
    )
    cfg.project.work_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def write_template(path: str | Path) -> Path:
    """Write a commented starter config. Refuses to overwrite."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    return path
