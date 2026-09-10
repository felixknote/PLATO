"""The palette registry: every colour any plot uses, in one place.

``palette.py`` decides *which* value gets a colour; this decides *what* the
available colours are. Split so that adding a palette does not mean touching
the assignment logic, and so nothing outside here hard-codes a scale.

Three kinds, because they answer different questions:

* **categorical** -- unordered classes (gene, antibiotic, dataset). Adjacent
  entries must be distinguishable; there is no meaningful order.
* **sequential** -- a quantity with a floor (entropy, concentration, focus).
  Must be monotonic in lightness so it still reads as ordered in greyscale
  and to a colour-blind viewer.
* **diverging** -- a quantity with a meaningful midpoint (a log ratio, a
  difference from control). Two hues away from a neutral centre.

Choosing a diverging palette for data with no natural midpoint is a way to
invent structure, so the two are kept separate and the UI says which is which.
"""

from __future__ import annotations

from dataclasses import dataclass

CATEGORICAL = "categorical"
SEQUENTIAL = "sequential"
DIVERGING = "diverging"


@dataclass(frozen=True, slots=True)
class Palette:
    """One named scale."""

    key: str
    name: str
    kind: str
    colours: tuple[str, ...]
    description: str = ""

    def __len__(self) -> int:
        return len(self.colours)


# -- categorical -------------------------------------------------------------

# The app's own qualitative scale. Ordered so consecutive hues never differ by
# red/green alone.
_PLATO = (
    "#4a90d9", "#e8a33d", "#4fc3a1", "#e2647a", "#a98bdc",
    "#8fc866", "#e07b53", "#5bc0d4", "#d98cc4", "#c2b34a",
    "#7f9cd4", "#d4735a", "#6cc39a", "#c98fb0", "#9ab84f",
    "#6fb3c9", "#d9a05b", "#8d9ee0", "#b5cc5e", "#df8f8f",
)

# Okabe & Ito (2008), the standard colour-blind-safe qualitative set. Eight
# colours chosen to stay distinct under deuteranopia, protanopia and
# tritanopia. Short, by design -- if a field needs more than eight classes,
# colour is the wrong encoding for it.
_OKABE_ITO = (
    "#0072b2", "#e69f00", "#009e73", "#cc79a7",
    "#56b4e9", "#d55e00", "#f0e442", "#000000",
)

# Tableau 10, familiar from most plotting tools; wider gamut than Okabe-Ito
# but not colour-blind safe.
_TABLEAU = (
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
)

# High-contrast set for dark backgrounds, brighter and more saturated.
_BRIGHT = (
    "#66c2ff", "#ffb347", "#5ee6a8", "#ff7b93", "#c9a0ff",
    "#a8e05f", "#ff9d6e", "#7ae0f0", "#f5a3d8", "#e0d24a",
)

# -- sequential --------------------------------------------------------------

# The app's own ramp: dark blue -> cyan -> amber, monotonic in lightness.
_PLATO_SEQ = (
    "#3b4cc0", "#4a7dd4", "#5ba7d4", "#6fc7bc",
    "#9dd693", "#d4d06a", "#e8a33d",
)

# Viridis, sampled. Perceptually uniform and the de facto standard for
# scientific figures.
_VIRIDIS = (
    "#440154", "#472d7b", "#3b528b", "#2c728e", "#21918c",
    "#28ae80", "#5ec962", "#addc30", "#fde725",
)

# Magma: dark background friendly, wide lightness range.
_MAGMA = (
    "#000004", "#1c1044", "#4f127b", "#812581", "#b5367a",
    "#e55c30", "#fba40a", "#fcffa4",
)

# Simple greyscale, for when colour is carrying another variable entirely.
_GREYS = ("#1a1a1a", "#404040", "#6b6b6b", "#999999", "#c9c9c9", "#f0f0f0")

# -- diverging ---------------------------------------------------------------

# Blue-white-red. The classic, and the one most readers already know how to
# read as "below / at / above the midpoint".
_COOLWARM = (
    "#3b4cc0", "#7396f5", "#aec0f7", "#dcdcdc",
    "#f6b8a4", "#ee7b62", "#b40426",
)

# Purple-green, colour-blind safe unlike blue-red.
_PRGN = (
    "#762a83", "#9970ab", "#c2a5cf", "#e7e7e7",
    "#a6dba0", "#5aae61", "#1b7837",
)


PALETTES: tuple[Palette, ...] = (
    Palette("plato", "PLATO", CATEGORICAL, _PLATO,
            "The app's own 20-colour qualitative scale."),
    Palette("okabe_ito", "Okabe–Ito (colour-blind safe)", CATEGORICAL, _OKABE_ITO,
            "Eight colours that stay distinct under all common forms of "
            "colour vision deficiency. Publication default."),
    Palette("tableau", "Tableau 10", CATEGORICAL, _TABLEAU,
            "Familiar ten-colour scale. Not colour-blind safe."),
    Palette("bright", "Bright (for dark backgrounds)", CATEGORICAL, _BRIGHT,
            "Higher saturation, for plots on a dark ground."),

    Palette("plato_seq", "PLATO ramp", SEQUENTIAL, _PLATO_SEQ,
            "Blue to amber, monotonic in lightness."),
    Palette("viridis", "Viridis", SEQUENTIAL, _VIRIDIS,
            "Perceptually uniform; the scientific default."),
    Palette("magma", "Magma", SEQUENTIAL, _MAGMA,
            "Wide lightness range, suits a dark background."),
    Palette("greys", "Greys", SEQUENTIAL, _GREYS,
            "No hue, for when colour encodes something else."),

    Palette("coolwarm", "Cool–warm", DIVERGING, _COOLWARM,
            "Blue–white–red about a midpoint. Only meaningful when the "
            "variable HAS a midpoint."),
    Palette("prgn", "Purple–green", DIVERGING, _PRGN,
            "Diverging and colour-blind safe, unlike blue–red."),
)

DEFAULTS = {
    CATEGORICAL: "plato",
    SEQUENTIAL: "plato_seq",
    DIVERGING: "coolwarm",
}


def get(key: str) -> Palette:
    """A palette by key, falling back to the default for its kind."""
    for palette in PALETTES:
        if palette.key == key:
            return palette
    return get(DEFAULTS[CATEGORICAL])


def of_kind(kind: str) -> list[Palette]:
    return [p for p in PALETTES if p.kind == kind]


def default_for(kind: str) -> Palette:
    return get(DEFAULTS.get(kind, DEFAULTS[CATEGORICAL]))
