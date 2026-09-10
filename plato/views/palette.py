"""Colour mapping for the embedding scatter.

Two mappings, because the two kinds of metadata want different things:

*Categorical* (gene, drug, MoA, plate) gets a fixed qualitative palette,
assigned in sorted order so the same value keeps the same colour between
sessions, between datasets and between the two experiment arms. That
stability is the point: if ``Gyrase`` is teal in the ABx arm it must be teal
in the CRISPRi arm too, or the comparison the plot exists to support cannot
be made by eye.

*Continuous* (concentration, once parsed to a number) gets a perceptually
ordered ramp, so 0.25x -> 2x reads as a progression rather than four
unrelated hues.

Colours are chosen for a dark background (#16181d): mid-to-light, saturated
enough to separate at 3 px, none so dark it disappears.
"""

from __future__ import annotations

import re

from PySide6.QtGui import QColor

from ..data.annotations import UNANNOTATED

# Qualitative palette. Ordered so that adjacent entries are distinguishable
# even for the most common forms of colour vision deficiency -- consecutive
# hues never differ by red/green alone.
CATEGORICAL = (
    "#4a90d9",  # blue
    "#e8a33d",  # amber
    "#4fc3a1",  # teal
    "#e2647a",  # rose
    "#a98bdc",  # violet
    "#8fc866",  # green
    "#e07b53",  # orange
    "#5bc0d4",  # cyan
    "#d98cc4",  # pink
    "#c2b34a",  # olive
    "#7f9cd4",  # slate blue
    "#d4735a",  # terracotta
    "#6cc39a",  # jade
    "#c98fb0",  # mauve
    "#9ab84f",  # lime
    "#6fb3c9",  # steel
    "#d9a05b",  # sand
    "#8d9ee0",  # periwinkle
    "#b5cc5e",  # chartreuse
    "#df8f8f",  # salmon
)

# Anything unannotated or empty is grey and sits visually behind the real
# categories -- present, countable, but never mistaken for a finding.
UNKNOWN_COLOUR = "#5a616e"

# Continuous ramp, dark blue -> cyan -> yellow. Monotonic in lightness so it
# still orders correctly when read as greyscale.
CONTINUOUS = (
    "#3b4cc0",
    "#4a7dd4",
    "#5ba7d4",
    "#6fc7bc",
    "#9dd693",
    "#d4d06a",
    "#e8a33d",
)

_UNKNOWN_KEYS = {"", UNANNOTATED, "none", "nan", "unknown", "n/a"}

# "0.25x" / "1x" / "2x" -> 0.25 / 1.0 / 2.0, so a dose series can be treated
# as continuous. Also matches a bare number.
_NUMERIC_RE = re.compile(r"^\s*([\d.]+)\s*x?\s*$", re.IGNORECASE)


def is_unknown(value: object) -> bool:
    return str(value).strip().lower() in _UNKNOWN_KEYS


def as_number(value: object) -> float | None:
    match = _NUMERIC_RE.match(str(value))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def looks_continuous(values: list[str]) -> bool:
    """True when every known value parses as a number and there are enough
    distinct ones for a ramp to say more than distinct hues would."""
    known = [v for v in values if not is_unknown(v)]
    if len(known) < 3:
        return False
    return all(as_number(v) is not None for v in known)


def categorical_colours(values: list[str]) -> dict[str, str]:
    """value -> hex colour, stable for a given sorted set of values.

    Values beyond the palette length wrap around. That is a real limitation at
    high cardinality (186 conditions), but a 186-colour palette would be
    unreadable anyway -- at that point the legend is the wrong tool and
    filtering is the right one, which the panel offers.
    """
    mapping: dict[str, str] = {}
    index = 0
    for value in values:
        if is_unknown(value):
            mapping[value] = UNKNOWN_COLOUR
            continue
        scale = active_categorical()
        mapping[value] = scale[index % len(scale)]
        index += 1
    return mapping


def _lerp(start: str, end: str, t: float) -> str:
    a, b = QColor(start), QColor(end)
    return QColor(
        round(a.red() + (b.red() - a.red()) * t),
        round(a.green() + (b.green() - a.green()) * t),
        round(a.blue() + (b.blue() - a.blue()) * t),
    ).name()


def continuous_colour(fraction: float) -> str:
    """Sample the ramp at ``fraction`` in [0, 1]."""
    fraction = max(0.0, min(1.0, fraction))
    scale = active_continuous()
    scaled = fraction * (len(scale) - 1)
    low = int(scaled)
    if low >= len(scale) - 1:
        return scale[-1]
    return _lerp(scale[low], scale[low + 1], scaled - low)


def continuous_colours(values: list[str]) -> dict[str, str]:
    """value -> hex colour along the ramp, ordered by numeric magnitude."""
    numbers = {v: as_number(v) for v in values if not is_unknown(v)}
    numbers = {v: n for v, n in numbers.items() if n is not None}
    mapping = {v: UNKNOWN_COLOUR for v in values if v not in numbers}
    if not numbers:
        return mapping
    low, high = min(numbers.values()), max(numbers.values())
    span = high - low
    for value, number in numbers.items():
        mapping[value] = continuous_colour(0.5 if span <= 0 else (number - low) / span)
    return mapping


def set_active_palette(key: str) -> None:
    """Choose which registered palette this module assigns from.

    Module-level rather than threaded through every call site: colour
    assignment happens in several places (the scatter, the facets, the legend,
    the cluster bars) and they must all agree, so the choice lives in one
    place they all read. See plato.views.palettes for the registry.
    """
    global _ACTIVE_CATEGORICAL, _ACTIVE_CONTINUOUS
    from . import palettes

    palette = palettes.get(key)
    if palette.kind == palettes.CATEGORICAL:
        _ACTIVE_CATEGORICAL = palette.colours
    else:
        _ACTIVE_CONTINUOUS = palette.colours


def active_categorical() -> tuple[str, ...]:
    return _ACTIVE_CATEGORICAL


def active_continuous() -> tuple[str, ...]:
    return _ACTIVE_CONTINUOUS


# The live scales. Start as the module's originals so behaviour is unchanged
# until something calls set_active_palette.
_ACTIVE_CATEGORICAL: tuple[str, ...] = CATEGORICAL
_ACTIVE_CONTINUOUS: tuple[str, ...] = CONTINUOUS


def ramp_over_array(values, low: float, high: float, *, steps: int = 24):
    """Colour a numeric array along the ramp, batched into draw groups.

    Returns ``(colours, groups)`` in the form ``EmbeddingScatter.set_points``
    wants: one hex colour per point, plus colour -> positional mask so the
    scatter can draw each shade in a single call.

    Quantised to ``steps`` bands rather than one colour per point, because the
    scatter batches by colour: 24k distinct colours would mean 24k draw calls
    and an unusable plot, while 24 bands are indistinguishable to the eye at
    the ramp's resolution and draw in 24. Non-finite values (an image that
    could not be read) become the unknown grey, which is honest -- they have
    no measurement rather than a low one.
    """
    import numpy as np

    values = np.asarray(values, dtype=np.float32)
    span = high - low
    finite = np.isfinite(values)
    # Compute the band only where there is a number. np.where would evaluate
    # the cast over the whole array first, and casting NaN to an integer is
    # undefined -- it happens to be discarded here, but it raises a warning
    # and the value it produces is platform-dependent.
    bands = np.full(values.shape, -1, dtype=np.int32)
    if finite.any():
        fractions = np.clip(
            (values[finite] - low) / (span if span > 0 else 1.0), 0.0, 1.0
        )
        bands[finite] = np.round(fractions * (steps - 1)).astype(np.int32)

    palette = {band: continuous_colour(band / (steps - 1)) for band in range(steps)}
    palette[-1] = UNKNOWN_COLOUR

    colours = [palette[int(b)] for b in bands]
    groups: dict[str, np.ndarray] = {}
    # Unknown first, so measured points draw over the grey rather than under.
    for band in sorted(set(bands.tolist())):
        colour = palette[int(band)]
        mask = np.flatnonzero(bands == band)
        if colour in groups:
            groups[colour] = np.concatenate([groups[colour], mask])
        else:
            groups[colour] = mask
    return colours, groups


def colours_for(values: list[str]) -> tuple[dict[str, str], bool]:
    """(value -> colour, is_continuous) for one column's distinct values."""
    if looks_continuous(values):
        return continuous_colours(values), True
    return categorical_colours(values), False


def sort_values(values: list[str]) -> list[str]:
    """Legend order: numerics by magnitude, everything else alphabetically,
    with the unknown bucket always last."""
    known = [v for v in values if not is_unknown(v)]
    unknown = [v for v in values if is_unknown(v)]
    if known and all(as_number(v) is not None for v in known):
        known.sort(key=lambda v: as_number(v) or 0.0)
    else:
        known.sort(key=str.casefold)
    return known + unknown
