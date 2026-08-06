"""Generate a small synthetic screen so the app can be tried without real data.

Creates:
    <out>/images/Plate1_A01_f1_DAPI.tif ...
    <out>/platemap.xlsx
    <out>/plato.toml

The images are noise blobs, not biology. They exist to exercise the indexing,
caching and GUI paths, including a few deliberate defects (see --defects).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

GENES = ["ftsZ", "murA", "lpxC", "rpoB", "gyrA", "control"]
ANTIBIOTICS = ["Ampicillin", "Ciprofloxacin", "Polymyxin B", "DMSO"]
CONCENTRATIONS = [0.0, 0.5, 2.0, 8.0]
CHANNELS = ["DAPI", "FM464"]
FIELDS = [1, 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--size", type=int, default=512, help="image edge in pixels")
    parser.add_argument(
        "--defects",
        action="store_true",
        help="inject an unparseable filename, an orphan well and a duplicate map row",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    image_dir = args.out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row_idx, row_letter in enumerate("ABCDEFGH"):
        for col in range(1, 13):
            well = f"{row_letter}{col:02d}"
            gene = GENES[(row_idx + col) % len(GENES)]
            antibiotic = ANTIBIOTICS[col % len(ANTIBIOTICS)]
            conc = CONCENTRATIONS[row_idx % len(CONCENTRATIONS)]
            replicate = 1 + (col - 1) // 4
            rows.append(
                {
                    "Well": well if col % 2 else f"{row_letter}{col}",  # mixed padding
                    "Gene": gene,
                    "Antibiotic": antibiotic,
                    "Concentration (ug/mL)": conc,
                    "Replicate": replicate,
                }
            )
            for fld in FIELDS:
                for channel in CHANNELS:
                    intensity = 400 + 120 * conc + rng.normal(0, 40)
                    image = rng.poisson(intensity, size=(args.size, args.size))
                    yy, xx = np.mgrid[: args.size, : args.size]
                    for _ in range(12):
                        cy, cx = rng.integers(0, args.size, 2)
                        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 12**2))
                        image = image + blob * 3000
                    tifffile.imwrite(
                        image_dir / f"Plate1_{well}_f{fld}_{channel}.tif",
                        image.astype(np.uint16),
                    )

    platemap = pd.DataFrame(rows)

    if args.defects:
        tifffile.imwrite(image_dir / "stray_snapshot.tif", np.zeros((8, 8), np.uint16))
        platemap = pd.concat(
            [
                platemap,
                pd.DataFrame(
                    [
                        {
                            "Well": "H13",  # outside a 96-well plate
                            "Gene": "bogus",
                            "Antibiotic": "None",
                            "Concentration (ug/mL)": 0,
                            "Replicate": 1,
                        },
                        platemap.iloc[0].to_dict(),  # duplicate key
                    ]
                ),
            ],
            ignore_index=True,
        )

    platemap.to_excel(args.out / "platemap.xlsx", index=False)

    (args.out / "plato.toml").write_text(
        f"""[project]
name = "demo"
work_dir = "./.plato"
plate_format = 96

[images]
dir = "./images"
glob = "**/*.tif"
pattern = '^(?P<plate>[^_]+)_(?P<well>[A-P]\\d{{1,2}})_f(?P<field>\\d+)_(?P<channel>[A-Za-z0-9]+)\\.tif$'

[platemap]
path = "./platemap.xlsx"
sheet = 0
well_column = "Well"
plate_column = ""
default_plate = "Plate1"

[thumbnails]
size = 256
percentiles = [1.0, 99.5]
sample_size = 100

[masks]
dir = ""
pattern = "{{stem}}_mask.tif"

[gui]
caption_fields = ["gene", "antibiotic", "concentration_ug_ml"]
filter_fields = []
thumbnail_size = 180
""",
        encoding="utf-8",
    )
    print(f"demo screen written to {args.out}")


if __name__ == "__main__":
    main()
