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
#
# Six slots (2, 7, 8, 12, 13, 15) are deliberately held away from two hues the
# CHROME owns: the cyan accent (~185 deg) that means "selected/active", and the
# magenta (~311 deg) that means "control". Before this, index 7 sat 5 deg from
# the accent and index 15 sat 10 deg from it, so a plain condition was the same
# colour as a selection. Their replacements keep each original's lightness and
# saturation and move only hue, chosen to maximise the minimum separation from
# every other entry AND from both chrome hues; the worst case is now 19 deg.
#
# The rule this encodes: a hue the interface uses to mean something is not
# available to the data. Adding a colour here means re-checking that.
_PLATO = (
    "#4a90d9", "#e8a33d", "#4fc376", "#e2647a", "#a98bdc",
    "#8fc866", "#e07b53", "#b85bd4", "#90d98c", "#c2b34a",
    "#7f9cd4", "#d4735a", "#6cc3a9", "#c98fac", "#9ab84f",
    "#766fc9", "#d9a05b", "#8d9ee0", "#b5cc5e", "#df8f8f",
)

# Okabe & Ito (2008), the standard colour-blind-safe qualitative set. Eight
# colours chosen to stay distinct under deuteranopia, protanopia and
# tritanopia. Short, by design -- if a field needs more than eight classes,
# colour is the wrong encoding for it.
#
# EXEMPT from the chrome-hue rule that shapes _PLATO and _BRIGHT, as is
# _TABLEAU below. These two are external standards, and their whole value is
# that a reader already knows how to read them: #cc79a7 sits 16 deg from the
# magenta flag and Tableau's #b07aa1 only 6 deg, but silently editing a
# published scale would cost more than the collision does. A user who picks
# one is choosing familiarity over PLATO's own separation guarantee.
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
#
# Entries 7 and 8 were #7ae0f0 (3 deg from the cyan accent) and #f5a3d8
# (10 deg from the magenta flag) -- the same chrome collision _PLATO had, and
# worse for being more saturated. Moved on the same rule; the result keeps a
# 19 deg margin from both chrome hues and a 16 deg minimum inside the scale,
# which is what the scale already had between entries 1 and 6.
_BRIGHT = (
    "#66c2ff", "#ffb347", "#5ee6a8", "#ff7b93", "#c9a0ff",
    "#a8e05f", "#ff9d6e", "#7cf07a", "#a3aaf5", "#e0d24a",
)

# Deep-toned set for light/white backgrounds. _PLATO's mid-saturation hues
# are tuned for a dark ground (see palette.py's own docstring) and read
# washed out on white -- 17 of its 20 colours fall below WCAG's 3:1 minimum
# contrast for graphical objects against white. Same hue order and count as
# _PLATO so switching between them for a background change keeps each value
# in the same relative position, just darkened. The six chrome-avoiding hue
# shifts above are mirrored here for the same reason.
_DEEP = (
    "#1f6fb2", "#c9700a", "#1f8f44", "#c23a58", "#6b4a9e",
    "#4d7a1f", "#b1461f", "#741a8f", "#53a24f", "#8a7a1f",
    "#3f5fa8", "#a1462a", "#217a5f", "#93446c", "#5a7a1f",
    "#342c91", "#a5701f", "#4c5cab", "#6f8a1f", "#b1504f",
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
    Palette("deep", "Deep (for light backgrounds)", CATEGORICAL, _DEEP,
            "Darker, more saturated version of PLATO's scale -- legible on "
            "white where the default washes out."),

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


def categorical_for_background(mode: str | None, *, theme_is_dark: bool = True) -> Palette:
    """The categorical palette that reads well on ``mode``'s ground.

    Mirrors the light/dark resolution scatter.py's own set_background and
    _set_legend already use for axis ink and the legend panel: "light" is
    always light; "theme" (follow the app theme) is light exactly when the
    app theme itself is light; "transparent" has no ground of its own to key
    off, so it follows the app theme too, since a transparent plot is
    composited over the app most of the time it is actually being looked at.
    "dark" is the one explicit case that is never light.
    """
    is_light = mode == "light" or (mode in (None, "theme", "transparent") and not theme_is_dark)
    return get("deep") if is_light else get(DEFAULTS[CATEGORICAL])


def default_for(kind: str) -> Palette:
    return get(DEFAULTS.get(kind, DEFAULTS[CATEGORICAL]))
