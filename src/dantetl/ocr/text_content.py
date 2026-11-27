"""Conservative text-only observations for tracking; raw OCR remains untouched."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from statistics import median
from typing import TYPE_CHECKING

from dantetl.domain import Rect, union_bounds
from dantetl.languages.selection import DEFAULT_SOURCE
from dantetl.languages.source import SourceLanguage
from dantetl.ocr.values import MAX_OCR_REGIONS, OcrResult, TextCharacter, TextRegion

if TYPE_CHECKING:
    from dantetl.ocr.observation import MatchingEvidence

_STRUCTURAL = frozenset("（）()[]“”\"'，．：；？！!?…+-−±%％")


def _isolated_icon(
    language: SourceLanguage, edge: tuple[TextCharacter, ...], core: tuple[TextCharacter, ...]
) -> bool:
    if len(core) < 4 or not language.contains_text("".join(c.text for c in core)):
        return False
    if any(
        c in _STRUCTURAL or c in language.extra_edge_marks or c.isnumeric()
        for char in edge
        for c in char.text
    ):
        return False
    if not all(isfinite(c.confidence) and 0 <= c.confidence <= 1 for c in (*edge, *core)):
        return False
    strength = median(c.confidence for c in core)
    if not all(0 <= c.confidence <= 0.55 for c in edge) or strength < 0.75:
        return False
    if strength - max(c.confidence for c in edge) < 0.25:
        return False
    leading = edge[0].bounds.x < core[0].bounds.x
    left, right = (edge[-1], core[0]) if leading else (core[-1], edge[0])
    gap = right.bounds.x - (left.bounds.x + left.bounds.width)
    glyph = median(c.bounds.width for c in core)
    usual_gap = median(
        max(0, b.bounds.x - a.bounds.x - a.bounds.width)
        for a, b in zip(core, core[1:], strict=False)
    )
    return gap >= max(3, glyph * 0.45, usual_gap * 3)


_WEAK_LETTER = 0.5
_READ_NUMBER = 0.6


def _unsupported_letters(
    language: SourceLanguage, characters: tuple[TextCharacter, ...]
) -> tuple[TextCharacter, ...]:
    """Weak letters that are a row's only letter evidence, beside a confident number or alone.

    OCR reads a value's operator or an icon as a letter, e.g. 396ツ% for 396% or 十 for a
    crosshair. Words keep at least one confident letter, so any such row is untouched.
    """
    letters = tuple(c for c in characters if language.contains_text(c.text))
    if not letters or len(letters) > 2 or any(c.confidence >= _WEAK_LETTER for c in letters):
        return ()
    if len(characters) == 1:
        return letters
    digits = [c.confidence for c in characters if c.text.isdecimal()]
    return letters if digits and median(digits) >= _READ_NUMBER else ()


def content_region(region: TextRegion, language: SourceLanguage = DEFAULT_SOURCE) -> TextRegion:
    """Use character extents, dropping only isolated weak edge glyphs by strong text and
    weak letters that are a row's only letter evidence beside a confident number or alone.

    No global confidence threshold is raised. Quotes, punctuation, numbers, interior
    characters of words, and incomplete character metadata retain their recognized content.
    A row left without characters keeps its bounds and empty text, which tracking ignores.
    """
    characters = region.characters
    if not characters or "".join(c.text for c in characters) != region.text:
        return region
    if any(min(c.bounds.size) <= 0 for c in characters):
        return region
    scored = all(isfinite(c.confidence) and 0 <= c.confidence <= 1 for c in characters)
    if scored and (unsupported := _unsupported_letters(language, characters)):
        characters = tuple(c for c in characters if all(c is not u for u in unsupported))
        if not characters:
            return replace(region, text="", characters=())
    ordered = all(
        a.bounds.x <= b.bounds.x for a, b in zip(characters, characters[1:], strict=False)
    )
    if language.trim_edge_icons and not region.vertical and ordered:
        for count in (2, 1):
            if _isolated_icon(language, characters[:count], characters[count:]):
                characters = characters[count:]
                break
        for count in (2, 1):
            if _isolated_icon(language, characters[-count:], characters[:-count]):
                characters = characters[:-count]
                break
    return replace(
        region,
        text="".join(c.text for c in characters),
        characters=characters,
        bounds=union_bounds(tuple(c.bounds for c in characters)),
    )


def same_label_crop(a: Rect, b: Rect, old: str, new: str) -> bool:
    """Two horizontal reads of the same ink, with optional extra ink on the left."""
    h = min(a.height, b.height)
    if (
        max(a.height, b.height) > 2 * h
        or abs((a.y + a.height / 2) - (b.y + b.height / 2)) > h * 0.25
        or abs((a.x + a.width) - (b.x + b.width)) > min(h * 0.5, min(a.width, b.width) * 0.25)
    ):
        return False
    if old == new:
        return True
    if a.width > b.width:
        a, b, old, new = b, a, new, old
    if not 1.5 * h <= a.x - b.x <= 5 * h or len(old) < 2:
        return False
    if 0 < len(new) - len(old) <= 2 and new.endswith(old):
        return any(c.isalpha() for c in old)
    # The icon can also displace the first letter in recognition, e.g. *魂
    # over the pixels of 星魂. Require a symbol in its place and a word ending.
    ending = old[1:]
    prefix_size = len(new) - len(ending)
    return (
        1 <= prefix_size <= 2
        and old[0].isalpha()
        and ending.isalpha()
        and new.endswith(ending)
        and not any(c.isalnum() for c in new[:prefix_size])
    )


def cleaned_rows(
    result: OcrResult, language: SourceLanguage, matching: MatchingEvidence | None
) -> tuple[TextRegion, ...]:
    """Tracking input for the first rows, reusing the worker's preparation of this exact source."""
    if matching is not None and matching.describes(result, language):
        return matching.content
    return tuple(content_region(r, language) for r in result.regions[:MAX_OCR_REGIONS])
