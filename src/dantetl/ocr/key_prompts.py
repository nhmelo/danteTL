"""Remove button prompts, short Latin key names drawn inside their own outline, at line ends.

Games print controller and keyboard prompts beside labels (状態/効果 [L2]+[R2], [Esc] 戻る). OCR
reads them as weak letters and brackets (状態効果(L24(R). The prompt's letters are smaller than the
line's glyphs and its outline fits tightly around them: an edge map encloses them in a pocket about
a glyph high. Ordinary lettering leaves no such pocket around small marks except inside enclosing
kanji such as 回, which OCR reads as itself. Prompts joined by small marks (+, /) go together.
"""

from dataclasses import dataclass, replace
from statistics import median

import cv2
import numpy as np

from dantetl.domain import Rect
from dantetl.ocr.ink import Component, Ink, ink
from dantetl.ocr.values import TextCharacter, TextRegion

_GLYPH_PIXELS = 32  # Analyze at about this glyph height, whatever the capture scale.
_WORD_LETTER = 0.5  # Weaker letters are no evidence of a word; see text_content.
_MAX_KEYS = 3
# Canny hysteresis relative to the lettering's own contrast: outlines are drawn about as boldly
# as the text, while background texture is fainter.
_EDGE_HIGH, _EDGE_LOW = 1.2, 0.4


@dataclass(frozen=True, slots=True)
class _Key:
    left: float  # Analyzed pixels, outline included.
    right: float


def _prompt_text(char: TextCharacter) -> bool:
    """What OCR may read over a prompt: Latin key names, its outline, joining marks."""
    return char.text.isascii() or char.confidence < _WORD_LETTER


def _word(char: TextCharacter) -> bool:
    return char.text.isalpha() and char.confidence >= _WORD_LETTER


def _area(bounds: Rect, leading: bool) -> Rect:
    """The line's end, reaching outward past a prompt the detector cut through."""
    height = bounds.height
    width = min(bounds.width, 6 * height)
    reach, pad = round(1.5 * height), max(2, round(0.5 * height))
    x = bounds.x - reach if leading else bounds.x + bounds.width - width
    return Rect(x, bounds.y - pad, width + reach, height + 2 * pad)


def _plates(found: Ink, glyph: float) -> np.ndarray:
    """Areas enclosed by edges, filled, without thin lines such as borders passing by."""
    gray = cv2.cvtColor(found.image, cv2.COLOR_BGR2GRAY)
    high = _EDGE_HIGH * found.level
    edges = cv2.dilate(cv2.Canny(gray, _EDGE_LOW * high, high), np.ones((3, 3), np.uint8))
    count, labels = cv2.connectedComponents((edges == 0).astype(np.uint8), connectivity=4)
    border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    filled = (~np.isin(labels, border[border > 0])).astype(np.uint8)
    size = max(3, round(0.4 * glyph)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)


def _keys(found: Ink, glyph: float, line: float) -> list[_Key]:
    """Enclosed plates about a line high holding lettering that floats apart from their rim."""
    plates = _plates(found, glyph)
    count, _, stats, _ = cv2.connectedComponentsWithStats(plates, connectivity=8)
    height, width = plates.shape
    keys: list[_Key] = []
    for left, top, w, h, _ in stats[1:count].tolist():
        if left == 0 or top == 0 or left + w == width or top + h == height:
            continue
        if not (0.5 * line <= h <= 1.5 * line and 0.5 * line <= w <= 4 * line):
            continue
        right, bottom = left + w, top + h
        inside = [
            c
            for c in found.components
            if c.left >= left - 2
            and c.right <= right + 2
            and c.top >= top - 2
            and c.bottom <= bottom + 2
            and c.bottom - c.top >= 0.2 * line  # Specks and pieces of a broken outline.
            # An outline drawn in the lettering's own color spans the plate; it is not content.
            and not (c.right - c.left >= 0.85 * w and c.bottom - c.top >= 0.85 * h)
        ]
        if not inside or sum(c.right - c.left for c in inside) < 0.25 * w:
            continue
        # Lettering floats inside its outline; a glyph's own strokes reach the plate's edges.
        inset = 0.1 * h
        if (
            min(c.left for c in inside) < left + inset
            or max(c.right for c in inside) > right - inset
            or min(c.top for c in inside) < top + inset
            or max(c.bottom for c in inside) > bottom - inset
        ):
            continue
        keys.append(_Key(left, right))
    return keys


def _between(components: tuple[Component, ...], a: _Key, b: _Key, glyph: float) -> bool:
    """Two prompts joined only by small marks, about a glyph apart."""
    gap = (min(a.right, b.right), max(a.left, b.left))
    if gap[1] - gap[0] > 1.5 * glyph:
        return False
    return all(
        c.bottom - c.top < 0.8 * glyph for c in components if gap[0] < c.left and c.right < gap[1]
    )


def _prompt_span(bgr: np.ndarray, bounds: Rect, leading: bool) -> tuple[float, float] | None:
    """Source x span from the outermost prompt at this end to the innermost one joined to it."""
    found = ink(bgr, _area(bounds, leading), min(1.0, _GLYPH_PIXELS / bounds.height))
    if found is None:
        return None
    line = bounds.height * found.scale
    tall = [c.bottom - c.top for c in found.components if c.bottom - c.top >= 0.6 * line]
    if len(tall) < 2:
        return None
    glyph = median(tall)
    keys = sorted(_keys(found, glyph, line), key=lambda k: k.left, reverse=not leading)
    if not keys:
        return None
    # No glyph-height ink of the line lies beyond the outermost prompt.
    first = keys[0]
    end = (bounds.x + (0 if leading else bounds.width) - found.x) * found.scale
    beyond = [
        c
        for c in found.components
        if c.bottom - c.top >= 0.8 * glyph
        and (
            (end <= c.left and c.right <= first.left)
            if leading
            else (first.right <= c.left and c.right <= end)
        )
    ]
    if beyond:
        return None
    chain = [first]
    for key in keys[1:]:
        if len(chain) == _MAX_KEYS or not _between(found.components, chain[-1], key, glyph):
            break
        chain.append(key)
    # OCR boxes what it read over an outline a little beyond the outline itself.
    margin = 0.15 * line
    left = min(k.left for k in chain) - margin
    right = max(k.right for k in chain) + margin
    return found.source_x(left), found.source_x(right)


def trim_key_prompts(bgr: np.ndarray, regions: tuple[TextRegion, ...]) -> tuple[TextRegion, ...]:
    """Drop what OCR read over prompts at either end; detector bounds stay."""
    trimmed = list(regions)
    for index, region in enumerate(regions):
        chars = region.characters
        b = region.bounds
        if (
            region.vertical
            or len(chars) < 2
            or "".join(c.text for c in chars) != region.text
            or b.height < 8
            or not any(_word(c) and not c.text.isascii() for c in chars)
        ):
            continue
        for leading in (True, False):
            chars = trimmed[index].characters
            edge = chars[0] if leading else chars[-1]
            if not _prompt_text(edge):
                continue
            span = _prompt_span(bgr, b, leading)
            if span is None:
                continue
            over = [
                (c.bounds.x + c.bounds.width / 2 <= span[1])
                if leading
                else (c.bounds.x + c.bounds.width / 2 >= span[0])
                for c in chars
            ]
            dropped = [c for c, o in zip(chars, over, strict=True) if o]
            kept = tuple(c for c, o in zip(chars, over, strict=True) if not o)
            if not dropped or not all(_prompt_text(c) for c in dropped):
                continue
            trimmed[index] = replace(
                trimmed[index], text="".join(c.text for c in kept), characters=kept
            )
    return tuple(trimmed)
