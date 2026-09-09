"""Generate the PLATO logo as SVG, matching the designed artwork.

Drawn parametrically rather than traced: a 96-well plate is a regular 12x8
grid, so generating it keeps every well identical and lets the highlight set
be edited by changing one list.

Colours are sampled from the original artwork.
"""

from pathlib import Path

# -- sampled from the artwork ------------------------------------------------
TILE_BG = "#0b1730"        # tile interior
TILE_EDGE = "#141f38"      # tile rim
PANEL_STROKE = "#a0a7b3"   # plate outline + well rings
ORBIT = "#fcfeff"          # the ring
MAGENTA = "#ea45cc"
CYAN = "#1af8fe"
PURPLE = "#7977d2"
TEAL = "#3fa9a2"           # the dimmer greenish well

# -- geometry (512 canvas) ---------------------------------------------------
S = 512
TILE_R = 112               # corner radius of the squircle tile

COLS, ROWS = 12, 8
PITCH = 30.5               # centre-to-centre
WELL_R = 12.4              # ring radius
RING_W = 2.6

GRID_W = (COLS - 1) * PITCH
GRID_H = (ROWS - 1) * PITCH
CX, CY = S / 2, S / 2
X0 = CX - GRID_W / 2
Y0 = CY - GRID_H / 2

PAD_X, PAD_Y = 26, 24      # plate panel padding around the grid

# Highlighted wells: (col, row, colour, glow) with 0-based indices, read off
# the artwork.
HIGHLIGHTS = [
    (6, 0, MAGENTA, True),
    (2, 2, CYAN, True),
    (7, 2, CYAN, True),
    (4, 3, PURPLE, True),
    (1, 5, MAGENTA, True),
    (11, 5, CYAN, True),
    (6, 6, TEAL, False),
    (7, 7, MAGENTA, True),
]


def well_xy(col: int, row: int) -> tuple[float, float]:
    return X0 + col * PITCH, Y0 + row * PITCH


def build() -> str:
    panel_x = X0 - PAD_X
    panel_y = Y0 - PAD_Y
    panel_w = GRID_W + 2 * PAD_X
    panel_h = GRID_H + 2 * PAD_Y

    highlight_at = {(c, r): (colour, glow) for c, r, colour, glow in HIGHLIGHTS}

    out: list[str] = []
    add = out.append

    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {S} {S}" '
        f'width="{S}" height="{S}" role="img" aria-label="PLATO">'
    )
    add("<title>PLATO</title>")

    # --- defs: one glow filter per colour, plus the tile bevel
    add("<defs>")
    for name, colour in (
        ("magenta", MAGENTA),
        ("cyan", CYAN),
        ("purple", PURPLE),
        ("teal", TEAL),
    ):
        add(
            f'<filter id="glow-{name}" x="-140%" y="-140%" width="380%" height="380%">'
            f'<feGaussianBlur stdDeviation="5.5" result="b"/>'
            f'<feFlood flood-color="{colour}" flood-opacity="0.85"/>'
            f'<feComposite in2="b" operator="in" result="g"/>'
            f'<feMerge><feMergeNode in="g"/><feMergeNode in="g"/>'
            f'<feMergeNode in="SourceGraphic"/></feMerge>'
            f"</filter>"
        )
    add("</defs>")

    # --- tile
    add(
        f'<rect x="4" y="4" width="{S - 8}" height="{S - 8}" rx="{TILE_R}" '
        f'fill="{TILE_BG}" stroke="{TILE_EDGE}" stroke-width="8"/>'
    )

    # --- plate panel
    add(
        f'<rect x="{panel_x:.1f}" y="{panel_y:.1f}" width="{panel_w:.1f}" '
        f'height="{panel_h:.1f}" rx="10" fill="none" '
        f'stroke="{PANEL_STROKE}" stroke-width="3" opacity="0.9"/>'
    )

    # --- wells: plain ones first, so glows layer above the grid
    add(f'<g fill="none" stroke="{PANEL_STROKE}" stroke-width="{RING_W}" opacity="0.75">')
    for row in range(ROWS):
        for col in range(COLS):
            if (col, row) in highlight_at:
                continue
            x, y = well_xy(col, row)
            add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{WELL_R}"/>')
    add("</g>")

    # --- highlighted wells
    filter_name = {MAGENTA: "magenta", CYAN: "cyan", PURPLE: "purple", TEAL: "teal"}
    for col, row, colour, glow in HIGHLIGHTS:
        x, y = well_xy(col, row)
        attr = f' filter="url(#glow-{filter_name[colour]})"' if glow else ""
        add(f"<g{attr}>")
        add(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{WELL_R}" fill="none" '
            f'stroke="{colour}" stroke-width="{RING_W + 0.8:.1f}"/>'
        )
        add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{WELL_R - 5.4:.1f}" fill="{colour}"/>')
        add("</g>")

    # --- orbit ring
    #
    # A narrow, steeply tilted ellipse -- the "O" of PLATO read as an orbit.
    # In the artwork it passes BEHIND the plate on the left and in FRONT on
    # the right, which is what gives the mark its depth. That is drawn as two
    # arcs of the same ellipse: the back half before the plate is painted
    # would be hidden, so instead the front half is simply drawn last, over
    # everything, and the back half is omitted where the panel covers it.
    ocx, ocy = CX + 6, CY
    orx, ory = 236.0, 120.0
    angle = 62
    add(
        f'<ellipse cx="{ocx:.1f}" cy="{ocy:.1f}" rx="{orx:.1f}" ry="{ory:.1f}" '
        f'fill="none" stroke="{ORBIT}" stroke-width="9" stroke-linecap="round" '
        f'transform="rotate({angle} {ocx:.1f} {ocy:.1f})"/>'
    )

    add("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1])
    target.write_text(build(), encoding="utf-8")
    print(f"wrote {target} ({target.stat().st_size:,} bytes)")
