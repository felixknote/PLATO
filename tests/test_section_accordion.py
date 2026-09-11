"""The sidebar accordion, and the colour rule the chrome accent depends on.

Two things are being protected here.

**The accordion invariant** -- at most one section open. If that breaks, the
sidebar silently goes back to being the long scroll it replaced, and nothing
else in the suite would notice.

**The reserved-hue rule** -- the interface owns cyan (selected/active) and
magenta (control/flagged), and those two hues are held out of PLATO's own
categorical scales so a mark in a plot can never be mistaken for an interface
state. That rule lives in a comment in palettes.py and in the choice of
accent in themes.py; without a test, adding one colour to a scale or nudging
the accent would break it invisibly.
"""

from __future__ import annotations

import colorsys

import pytest
from PySide6.QtWidgets import QApplication, QWidget

from plato.gui import themes
from plato.views import palettes
from plato.views.section import Accordion


# -- helpers -------------------------------------------------------------------


def _hue(hex_colour: str) -> float:
    """Hue in degrees."""
    r, g, b = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hls(r, g, b)[0] * 360


def _hue_gap(a: float, b: float) -> float:
    """Shortest distance between two hues, around the circle."""
    delta = abs(a - b) % 360
    return min(delta, 360 - delta)


def _relative_luminance(hex_colour: str) -> float:
    channels = []
    for i in (1, 3, 5):
        value = int(hex_colour[i : i + 2], 16) / 255
        channels.append(
            value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
        )
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a: str, b: str) -> float:
    la, lb = _relative_luminance(a), _relative_luminance(b)
    high, low = max(la, lb), min(la, lb)
    return (high + 0.05) / (low + 0.05)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def accordion(app):
    acc = Accordion()
    for key in ("one", "two", "three"):
        acc.add(key, key.title(), QWidget())
    return acc


# -- the accordion invariant ---------------------------------------------------


def test_starts_with_everything_closed(accordion):
    assert accordion.open_key() is None
    for key in accordion.keys():
        assert not accordion.section(key).body.isVisibleTo(accordion)


def test_opening_one_closes_the_others(accordion):
    accordion.open_section("one")
    assert accordion.open_key() == "one"
    accordion.open_section("three")
    assert accordion.open_key() == "three"
    assert [k for k in accordion.keys() if accordion.section(k).is_open()] == ["three"]


def test_invariant_holds_when_the_header_is_clicked(accordion):
    """Not just via open_section -- the header is what a user actually hits."""
    accordion.section("one").header.setChecked(True)
    accordion.section("two").header.setChecked(True)
    open_now = [k for k in accordion.keys() if accordion.section(k).is_open()]
    assert open_now == ["two"]


def test_a_section_can_be_closed_leaving_none_open(accordion):
    accordion.open_section("one")
    accordion.section("one").set_open(False)
    assert accordion.open_key() is None


def test_body_visibility_follows_the_header(accordion):
    section = accordion.section("two")
    accordion.open_section("two")
    assert section.body.isVisibleTo(section)
    accordion.open_section("one")
    assert not section.body.isVisibleTo(section)


# -- summaries: what makes hiding a section safe -------------------------------


def test_summary_survives_being_set_while_open(accordion):
    """The value is kept whole; only its *rendering* is elided."""
    accordion.open_section("one")
    accordion.set_summary("one", "UMAP · 5,000 pts")
    assert accordion.section("one").header.summary_text() == "UMAP · 5,000 pts"


def test_summary_is_hidden_while_the_section_is_open(accordion):
    """Header and body would otherwise say the same thing twice."""
    accordion.set_summary("one", "3 active")
    header = accordion.section("one").header
    accordion.open_section("one")
    assert not header._summary.isVisibleTo(header)
    accordion.section("one").set_open(False)
    assert header._summary.isVisibleTo(header)


def test_badge_is_a_styleable_property(accordion):
    """The accent dot is driven by a dynamic property, not a baked colour."""
    header = accordion.section("one").header
    assert not header.has_badge()
    accordion.set_badge("one", True)
    assert header.has_badge()
    assert header._summary.property("badge") == "on"
    accordion.set_badge("one", False)
    assert not header.has_badge()


def test_header_reports_its_state_for_accessibility(accordion):
    header = accordion.section("one").header
    assert "collapsed" in header.accessibleName()
    accordion.open_section("one")
    assert "expanded" in header.accessibleName()


def test_unknown_keys_are_ignored_rather_than_raising(accordion):
    """These run from a redraw path; a missing section must not abort it."""
    accordion.set_summary("nope", "x")
    accordion.set_badge("nope", True)
    accordion.open_section("nope")
    assert accordion.section("nope") is None


# -- the reserved-hue rule -----------------------------------------------------

# Scales PLATO owns and therefore guarantees separation for. Okabe-Ito and
# Tableau are deliberately exempt: they are external standards whose value is
# that a reader already knows them, so they are not edited to fit our chrome.
# See the comments in palettes.py.
_OWNED_SCALES = ("plato", "deep", "bright")

# The narrowest gap any owned scale currently has to a chrome hue. Tightening
# the accent or adding a colour that lands closer should fail here.
_MIN_CHROME_GAP = 15.0


@pytest.mark.parametrize("theme_key", [themes.DARK, themes.LIGHT])
@pytest.mark.parametrize("palette_key", _OWNED_SCALES)
def test_owned_palettes_avoid_the_chrome_hues(theme_key, palette_key):
    colours = themes.THEMES[theme_key]
    reserved = (_hue(colours.accent), _hue(colours.flag))
    for colour in palettes.get(palette_key).colours:
        gap = min(_hue_gap(_hue(colour), hue) for hue in reserved)
        assert gap >= _MIN_CHROME_GAP, (
            f"{palette_key} colour {colour} is {gap:.0f}° from a reserved "
            f"chrome hue in the {theme_key} theme; the accent means "
            f"'selected' and the flag means 'control', so a data colour that "
            f"close reads as an interface state"
        )


def test_the_two_reserved_hues_are_distinct_from_each_other():
    """Cyan and magenta must not converge, in either theme."""
    for colours in (themes.DARK_COLOURS, themes.LIGHT_COLOURS):
        assert _hue_gap(_hue(colours.accent), _hue(colours.flag)) > 60


def test_the_light_accent_keeps_enough_contrast_to_be_seen():
    """A state colour that fails 3:1 on its own ground cannot do its job."""
    light = themes.LIGHT_COLOURS
    for name in ("accent", "flag"):
        colour = getattr(light, name)
        assert _contrast(colour, light.background) >= 3.0
        assert _contrast(colour, light.surface) >= 3.0


def test_hint_text_clears_the_contrast_floor_in_both_themes():
    """text_faint carries hint text, so it is held to 3:1 like any graphic."""
    for colours in (themes.DARK_COLOURS, themes.LIGHT_COLOURS):
        assert _contrast(colours.text_faint, colours.background) >= 3.0


def test_body_text_is_comfortably_readable_in_both_themes():
    for colours in (themes.DARK_COLOURS, themes.LIGHT_COLOURS):
        assert _contrast(colours.text, colours.background) >= 7.0
        assert _contrast(colours.text_muted, colours.background) >= 4.5


# -- typography ----------------------------------------------------------------


def test_no_tracked_out_caps_in_the_stylesheet():
    """The doubled all-caps treatment on headings is not to come back.

    QGroupBox::title and QLabel#panelHeading were both uppercase with letter
    spacing, so the sidebar shouted the same label twice. Tracking belongs to
    the wordmark alone.
    """
    sheet = themes.stylesheet(themes.DARK_COLOURS)
    assert "text-transform: uppercase" not in sheet
