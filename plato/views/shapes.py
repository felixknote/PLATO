"""Point shapes as a second encoding, independent of colour.

Colour carries one variable; shape can carry another, so "colour by
antibiotic, shape by dataset" reads both at once. That only works while the
shapes stay *distinguishable*, which is a much harder limit than colour: a
20-colour palette is legible, a 20-shape one is not, because at 6-10 px most
symbols collapse into "a blob".

So this offers a deliberately short list, ordered by how well each survives at
plot size, and refuses to go past it. A field with more categories than shapes
does not silently wrap -- wrapping would say two unrelated groups are the same
group, which is worse than not encoding the field at all. Instead the extra
categories fall to a single fallback shape and the caller is told, so the UI
can say so rather than quietly lying.
"""

from __future__ import annotations

# pyqtgraph symbol names, in preference order. Every one of these reads
# clearly at 6-10 px and differs from the others in silhouette, not merely in
# detail: circle, square, triangle, diamond, and so on. The plus/cross forms
# come last because they thin out badly against a busy background.
SHAPES: tuple[str, ...] = (
    "o",   # circle
    "s",   # square
    "t",   # triangle down
    "d",   # diamond
    "t1",  # triangle up
    "p",   # pentagon
    "h",   # hexagon
    "star",
    "+",
    "x",
)

SHAPE_NAMES = {
    "o": "circle",
    "s": "square",
    "t": "triangle",
    "d": "diamond",
    "t1": "triangle (up)",
    "p": "pentagon",
    "h": "hexagon",
    "star": "star",
    "+": "plus",
    "x": "cross",
}

# What everything gets when shape encodes nothing, and what categories beyond
# the limit fall back to.
DEFAULT_SHAPE = "o"

# The honest maximum. Past this, shapes stop being a code and start being
# noise; see the module docstring.
MAX_SHAPES = len(SHAPES)


def assign(values: list[str]) -> tuple[dict[str, str], list[str]]:
    """Map categories to shapes.

    Returns ``(mapping, overflow)`` where ``overflow`` lists the categories
    that could not be given a distinct shape. They are present in the mapping
    -- pointing at ``DEFAULT_SHAPE`` -- so rendering never has to special-case
    a missing key, but the caller can warn about them.

    Order is the caller's: pass values already sorted the way the legend will
    show them, so shape assignment and legend order agree.
    """
    mapping: dict[str, str] = {}
    overflow: list[str] = []
    for index, value in enumerate(values):
        if index < len(SHAPES):
            mapping[value] = SHAPES[index]
        else:
            mapping[value] = DEFAULT_SHAPE
            overflow.append(value)
    return mapping, overflow


def describe_overflow(overflow: list[str], total: int) -> str:
    """A sentence for the UI when a field has more categories than shapes."""
    if not overflow:
        return ""
    return (
        f"{total} categories, but only {MAX_SHAPES} shapes are visually "
        f"distinct — {len(overflow)} share the default circle. "
        f"Colour handles more categories than shape can."
    )
