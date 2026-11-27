import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from dantetl.domain import Rect  # noqa: E402
from dantetl.ocr.regions import intersects  # noqa: E402
from dantetl.ui.english_placement import EnglishLabel, EnglishPlacement  # noqa: E402
from dantetl.ui.text_layout import layout_english  # noqa: E402

pytestmark = pytest.mark.usefixtures("app")


def label(i, x=100, y=100, text="This is a longer English translation for the small label."):
    return EnglishLabel(((i, 1, "日本語"),), text, Rect(x, y, 150, 24), 24)


def test_wrapping_preserves_hindi_graphemes_and_arabic_paragraph_direction():
    source = Rect(0, 0, 48, 450)
    text = "कि" * 12
    layout = layout_english(text, source, (320, 480), automatic=False, manual_pixels=20)
    assert len(layout.lines) > 1 and "".join(layout.lines) == text
    assert all(line.startswith("क") and line.endswith("ि") for line in layout.lines)
    assert not layout.right_to_left
    arabic = layout_english("30 ثانية متبقية", source, (320, 480))
    assert arabic.right_to_left


def test_neighboring_expansions_avoid_collision_without_losing_english():
    labels = (label(1), label(2, y=130))
    independent = tuple(layout_english(item.text, item.source, (800, 600)) for item in labels)
    assert intersects(*(layout.bounds for layout in independent))
    placed = EnglishPlacement().arrange(labels, (800, 600), tuple(item.source for item in labels))
    assert not intersects(*(layout.bounds for layout in placed if layout is not None))
    for item, layout in zip(labels, placed, strict=True):
        assert layout is not None
        assert layout.bounds.contains(item.source)
        assert not layout.truncated
        assert " ".join(layout.lines) == item.text
        assert layout.pixel_size >= 12


def test_scene_reordering_repaints_and_unrelated_arrivals_do_not_move_labels():
    owner = EnglishPlacement()
    labels = (label(1), label(2, y=130))
    sources = tuple(item.source for item in labels)
    first = owner.arrange(labels, (1200, 800), sources)
    for _ in range(20):
        assert owner.arrange(labels[::-1], (1200, 800), sources[::-1]) == first[::-1]
    extra = label(3, x=900, y=600)
    assert owner.arrange((*labels, extra), (1200, 800), (*sources, extra.source))[:2] == first
    assert owner.arrange(labels, (1200, 800), sources) == first


def test_untranslated_sources_are_reserved_and_manual_size_is_preserved():
    item = label(1, y=200)
    obstacle = Rect(100, 230, 150, 24)
    (layout,) = EnglishPlacement().arrange(
        (item,), (800, 600), (item.source, obstacle), automatic=False, pixels=24
    )
    assert layout is not None and not intersects(layout.bounds, obstacle)
    assert layout.pixel_size == 24


def test_crowded_edge_layout_is_bounded_and_revisions_resize_and_reset_are_fresh():
    labels = tuple(label(i, x=(i % 4) * 35, y=(i // 4) * 8) for i in range(128))
    owner = EnglishPlacement()
    placed = owner.arrange(labels, (300, 280))
    boxes = [p.bounds for p in placed if p is not None]
    assert 0 < len(boxes) < len(labels)
    assert all(Rect(0, 0, 300, 280).contains(box) for box in boxes)
    assert all(not intersects(a, b) for i, a in enumerate(boxes) for b in boxes[:i])
    changed = (replace(labels[0], key=((0, 2, "新しい文章"),), text="Changed."),)
    assert owner.arrange(changed, (240, 180)) == EnglishPlacement().arrange(changed, (240, 180))
    owner.clear()
    assert owner.arrange(labels[:2], (600, 400)) == EnglishPlacement().arrange(
        labels[:2], (600, 400)
    )


def test_group_and_vertical_use_line_metrics_and_full_overflow_source_is_preserved():
    group = EnglishLabel(
        ((1, 1, "一"), (2, 1, "二")), "A complete sentence.", Rect(20, 20, 500, 150), 32
    )
    vertical = replace(label(3, x=600), source=Rect(600, 100, 24, 150), vertical=True)
    overflowing = label(4, y=300, text="Long translation " * 100)
    layouts = EnglishPlacement().arrange((group, vertical, overflowing), (900, 600))
    assert layouts[0] is not None and layouts[0].pixel_size == round(32 * 0.72)
    assert all(p and Rect(0, 0, 900, 600).contains(p.bounds) for p in layouts)
    assert layouts[-1] and layouts[-1].truncated and layouts[-1].lines[-1].endswith("…")
    assert overflowing.text == "Long translation " * 100


def test_ready_profile_paragraph_supersedes_fragment_in_any_arrival_order():
    fragment = EnglishLabel(
        ((1, 1, "その身をもって"),), "Fushuryū, its body", Rect(3416, 837, 325, 47), 47
    )
    paragraph = EnglishLabel(
        ((4, 1, "伏龍はその身をもって"), (5, 1, "大地を支え、千年の苦しみに耐える。")),
        "The Crouching Dragon supports the shattered earth with its body, "
        "enduring pain for a thousand years.",
        Rect(2979, 837, 764, 176),
        47,
    )
    neighbor = replace(label(8), source=Rect(2975, 1050, 760, 65))
    viewport = (3840, 2160)
    owner = EnglishPlacement()
    (first,) = owner.arrange((fragment,), viewport)
    assert first is not None
    full, short, separate = owner.arrange((paragraph, fragment, neighbor), viewport)
    assert full and short is None and separate
    assert not intersects(full.bounds, separate.bounds)
    assert EnglishPlacement().arrange((neighbor, fragment, paragraph), viewport) == (
        separate,
        None,
        full,
    )
    assert owner.arrange((fragment,), viewport) == (first,)


def test_competing_short_icon_crop_and_menu_preview_yield_to_live_word():
    clean = EnglishLabel(((6, 1, "詳細"),), "Details", Rect(434, 285, 99, 42), 42)
    wide = EnglishLabel(((3, 1, "具詳細"),), "Gore Details", Rect(335, 282, 194, 50), 50)
    preview = replace(clean, key=((-1, 1, "詳細"),))
    owner = EnglishPlacement()
    assert owner.arrange((wide,), (3840, 2160))[0] is not None
    broad, old, word = owner.arrange((wide, preview, clean), (3840, 2160))
    assert word and broad is None and old is None


def test_equal_coverage_uses_reading_order_even_if_track_ids_are_reallocated():
    for left_id, right_id in ((1, 2), (2, 1)):
        left = label(left_id, x=100, text="First")
        right = label(right_id, x=125, text="Second")
        owner = EnglishPlacement()
        owner.arrange((right,), (500, 300))
        hidden, winner = owner.arrange((right, left), (500, 300))
        assert hidden is None and winner is not None


def test_one_alternate_anchor_preserves_three_complete_readable_labels():
    labels = tuple(
        EnglishLabel(((i, 1, "設定"),), text, Rect(x, y, 120, 24), 24)
        for i, (text, x, y) in enumerate(
            (
                ("Examine the nearby equipment", 50, 240),
                ("Open your character inventory", 380, 250),
                ("Switch to the previous character", 190, 190),
            ),
            start=1,
        )
    )
    owner = EnglishPlacement()
    placed = owner.arrange(labels, (650, 420))
    assert owner.arrange(labels[::-1], (650, 420)) == placed[::-1]
    for i, (label, layout) in enumerate(zip(labels, placed, strict=True)):
        assert layout and not layout.truncated
        assert " ".join(layout.lines) == label.text
        assert layout.bounds.contains(label.source)
        assert all(other and not intersects(layout.bounds, other.bounds) for other in placed[:i])


def test_short_labels_keep_words_whole_and_stay_centered_on_their_source():
    source = Rect(3297, 142, 72, 36)  # A two-glyph menu label at 4K.
    layout = layout_english("Observation", source, (3840, 2160), source_line_height=36)
    assert layout.lines == ("Observation",)
    middle = layout.bounds.x + layout.bounds.width / 2
    assert abs(middle - (source.x + source.width / 2)) <= 1
    words = layout_english("Hoshigen no Niwa", Rect(2200, 2000, 60, 30), (3840, 2160))
    assert " ".join(words.lines).split() == ["Hoshigen", "no", "Niwa"]  # No word is broken.


def test_long_sources_keep_their_reading_edge_and_unspaced_text_wraps_anywhere():
    dialogue = Rect(100, 1500, 900, 80)
    sentence = "A long sentence of ordinary dialogue that should wrap at its words."
    assert layout_english(sentence, dialogue, (3840, 2160)).bounds.x == dialogue.x
    narrow = Rect(100, 100, 60, 30)
    japanese = layout_english("見出しのテキストです", narrow, (3840, 2160))
    assert len(japanese.lines) > 1 and "".join(japanese.lines) == "見出しのテキストです"
    assert japanese.bounds.width < 160  # Not widened to the whole unspaced text.
