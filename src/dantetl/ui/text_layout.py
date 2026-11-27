"""Deterministic English layout in physical pixels, bounded to the client viewport."""

from dataclasses import dataclass, replace
from functools import lru_cache
from math import ceil
from unicodedata import bidirectional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontMetricsF, QTextLayout, QTextOption

from dantetl.domain import Rect
from dantetl.i18n import tr

PAD_X = 10
PAD_Y = 6


@dataclass(frozen=True, slots=True)
class EnglishLayout:
    bounds: Rect
    pixel_size: int
    line_height: int
    ascent: float
    lines: tuple[str, ...]
    truncated: bool
    right_to_left: bool = False


def _wrap(text: str, font: QFont, width: int) -> tuple[str, ...]:
    # Qt wraps shaped glyph clusters and supplies Unicode line breaks for unspaced text.
    text = " ".join(text.split())
    layout = QTextLayout(text, font)
    option = QTextOption()
    option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
    layout.setTextOption(option)
    encoded = text.encode("utf-16-le")  # QTextLine offsets count UTF-16 code units.
    lines: list[str] = []
    layout.beginLayout()
    while (line := layout.createLine()).isValid():
        line.setLineWidth(width)
        start = line.textStart() * 2
        lines.append(encoded[start : start + line.textLength() * 2].decode("utf-16-le").rstrip())
    layout.endLayout()
    return tuple(lines)


def _spaced(word: str) -> bool:
    """Words of scripts written with spaces; unspaced scripts may wrap between any characters."""
    return not any(0x0E00 <= ord(c) <= 0x0EFF or ord(c) >= 0x2E80 for c in word)


def _longest_word(text: str, metrics: QFontMetricsF) -> float:
    return max((metrics.horizontalAdvance(w) for w in text.split() if _spaced(w)), default=0.0)


def layout_english(
    text: str,
    source: Rect,
    viewport: tuple[int, int],
    *,
    automatic: bool = True,
    manual_pixels: int = 20,
    vertical: bool = False,
    source_line_height: int | None = None,
    wrap_width: int | None = None,
) -> EnglishLayout:
    # Moving a source must not repeat font measurement/wrapping. Geometry-dependent
    # placement is separate from the bounded cache of text/size measurements.
    measured = _measure_english(
        text,
        Rect(0, 0, *source.size),
        viewport,
        automatic=automatic,
        manual_pixels=manual_pixels,
        vertical=vertical,
        source_line_height=source_line_height,
        wrap_width=wrap_width,
    )
    box = measured.bounds
    # English keeps its source's reading edge; placement offers centered and trailing anchors
    # when the extra width would cover neighbouring text.
    return replace(
        measured,
        bounds=Rect(
            max(0, min(source.x, viewport[0] - box.width)),
            max(0, min(source.y, viewport[1] - box.height)),
            box.width,
            box.height,
        ),
    )


@lru_cache(maxsize=512)
def _measure_english(
    text: str,
    source: Rect,
    viewport: tuple[int, int],
    *,
    automatic: bool,
    manual_pixels: int,
    vertical: bool,
    source_line_height: int | None,
    wrap_width: int | None,
) -> EnglishLayout:
    view_width, view_height = viewport
    if min(viewport) <= 0:
        raise ValueError(tr("Translation layout needs a nonempty viewport."))

    glyph_height = source_line_height or (source.width if vertical else source.height)
    preferred = max(12, min(48, round(glyph_height * 0.72))) if automatic else manual_pixels
    preferred = max(10, min(60, preferred))
    floor = max(12, round(preferred * 0.75)) if automatic else preferred
    floor = min(preferred, floor)
    width = min(view_width, max(40, wrap_width or (source.width if not vertical else 240)))
    font = QFont("Segoe UI")
    lines: tuple[str, ...] = ()
    pixels = preferred
    requested = width
    while True:
        font.setPixelSize(pixels)
        metrics = QFontMetricsF(font)
        line_height = max(1, round(metrics.lineSpacing()))
        # Break between words, never inside one, while the viewport has room.
        width = min(view_width, max(requested, ceil(_longest_word(text, metrics)) + 1 + PAD_X * 2))
        lines = _wrap(text, font, max(1, width - PAD_X * 2))
        if len(lines) * line_height + PAD_Y * 2 <= source.height:
            break
        if pixels == floor:
            # A limited width expansion preserves readability for longer English.
            if wrap_width is None and len(lines) > 1:
                width = min(view_width, round(width * 1.5) + 2 * PAD_X)
                lines = _wrap(text, font, max(1, width - PAD_X * 2))
            break
        pixels -= 1
    max_height = min(view_height, max(source.height, line_height * 4 + PAD_Y * 2))
    line_count = max(1, (max_height - PAD_Y * 2) // line_height)
    truncated = len(lines) > line_count
    lines = lines[:line_count]
    if truncated and lines:
        tail = metrics.elidedText(
            lines[-1] + "…", Qt.TextElideMode.ElideRight, max(1, width - PAD_X * 2)
        )
        lines = (*lines[:-1], tail)
    height = min(view_height, max(source.height, len(lines) * line_height + PAD_Y * 2))
    x = max(0, min(source.x, view_width - width))
    y = max(0, min(source.y, view_height - height))
    return EnglishLayout(
        Rect(x, y, width, height),
        pixels,
        line_height,
        metrics.ascent(),
        lines,
        truncated,
        next((direction for c in text if (direction := bidirectional(c)) in ("L", "R", "AL")), "L")
        != "L",
    )
