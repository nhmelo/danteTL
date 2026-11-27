"""Owned icon, label and stat scenes with exact ink ground truth, plus their OCR scoring.

Scenes are drawn procedurally in generic families (button prompts, silhouettes touching titles,
icon grids, tab bars, stat rows, icon-only and text-only negatives) rather than copied from any
game. Ground truth is the drawn ink, so a segmentation change can be scored exactly.
"""

import json
import math
import random
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from math import ceil, floor
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QFontMetricsF,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)

from dantetl.domain import CapturedFrame, Rect, WindowGeometry
from dantetl.languages.source import SourceLanguage
from dantetl.ocr.values import TextRegion

TEXT, VALUE = "text", "value"


@dataclass(frozen=True, slots=True)
class Label:
    """One separately meaningful piece of text. Values are numbers or symbols without letters."""

    text: str
    bounds: Rect
    role: str = TEXT


@dataclass(frozen=True, slots=True)
class Icon:
    kind: str
    bounds: Rect


@dataclass(frozen=True, slots=True)
class Scene:
    name: str
    family: str
    frame: CapturedFrame
    labels: tuple[Label, ...]
    icons: tuple[Icon, ...]
    scope: tuple[Rect, ...]
    areas: tuple[Rect, ...] | None = None  # Regional scan areas; None scans the full frame.


@dataclass(frozen=True, slots=True)
class Spec:
    name: str
    family: str
    draw: Callable[["_Canvas"], None]
    size: tuple[int, int] = (1920, 1080)
    background: str = "dark"


def _rect(r: QRectF, pad: float = 0) -> Rect:
    left, top = floor(r.left() - pad), floor(r.top() - pad)
    return Rect(left, top, ceil(r.right() + pad) - left, ceil(r.bottom() + pad) - top)


def _star(center: QPointF, outer: float, inner: float, points: int) -> QPolygonF:
    polygon = QPolygonF()
    for i in range(points * 2):
        radius = outer if i % 2 == 0 else inner
        angle = math.pi * i / points - math.pi / 2
        polygon.append(center + QPointF(radius * math.cos(angle), radius * math.sin(angle)))
    return polygon


@dataclass
class _Canvas:
    image: QImage
    painter: QPainter
    factor: float
    rng: random.Random
    labels: list[Label] = field(default_factory=list)
    icons: list[Icon] = field(default_factory=list)

    def px(self, value: float) -> float:
        return value * self.factor

    def background(self, style: str) -> None:
        w, h = self.image.width(), self.image.height()
        if style == "light":
            self.image.fill(QColor(236, 232, 224))
            return
        self.image.fill(QColor(22, 26, 34))
        if style == "gradient":
            gradient = QLinearGradient(0, 0, w, h)
            gradient.setColorAt(0, QColor(18, 22, 30))
            gradient.setColorAt(1, QColor(70, 82, 96))
            self.painter.fillRect(0, 0, w, h, QBrush(gradient))
        elif style == "textured":
            self.painter.setPen(Qt.PenStyle.NoPen)
            for _ in range(160):
                shade = self.rng.randrange(30, 120)
                color = QColor(shade, shade + self.rng.randrange(0, 30), shade // 2, 90)
                self.painter.setBrush(color)
                rx, ry = self.rng.uniform(20, 260) * self.factor, self.rng.uniform(10, 120)
                self.painter.drawEllipse(
                    QPointF(self.rng.uniform(0, w), self.rng.uniform(0, h)), rx, ry
                )

    def font(self, pixels: float, family: str = "Yu Gothic UI", bold: bool = False) -> QFont:
        font = QFont(family)
        font.setPixelSize(max(1, round(self.px(pixels))))
        font.setBold(bold)
        return font

    def text(
        self,
        text: str,
        x: float,
        baseline: float,
        pixels: float,
        color: QColor,
        *,
        role: str = TEXT,
        bold: bool = False,
        outline: QColor | None = None,
        accent: tuple[int, int, QColor] | None = None,
        heavy: int = 0,
    ) -> Label:
        """Draw at design coordinates; ``accent`` recolors characters [start, end) and the first
        ``heavy`` characters are bold, like an emphasized word leading a sentence."""
        font = self.font(pixels, bold=bold)
        metrics = QFontMetricsF(font)
        x, baseline = self.px(x), self.px(baseline)
        if heavy:
            strong = self.font(pixels, bold=True)
            self.painter.setPen(color)
            self.painter.setFont(strong)
            self.painter.drawText(QPointF(x, baseline), text[:heavy])
            lead = QFontMetricsF(strong)
            rest = x + lead.horizontalAdvance(text[:heavy])
            self.painter.setFont(font)
            self.painter.drawText(QPointF(rest, baseline), text[heavy:])
            ink = (
                lead.tightBoundingRect(text[:heavy])
                .translated(x, baseline)
                .united(metrics.tightBoundingRect(text[heavy:]).translated(rest, baseline))
            )
            label = Label(text, _rect(ink, 1), role)
            self.labels.append(label)
            return label
        pad = 0.0
        if outline is not None:
            pad = max(1.0, self.px(pixels) * 0.08)
            path = QPainterPath()
            path.addText(QPointF(x, baseline), font, text)
            self.painter.strokePath(path, QPen(outline, pad * 2, Qt.PenStyle.SolidLine))
        self.painter.setFont(font)
        if accent is None:
            self.painter.setPen(color)
            self.painter.drawText(QPointF(x, baseline), text)
        else:
            start, end, other = accent
            cursor = x
            for i, part in enumerate((text[:start], text[start:end], text[end:])):
                self.painter.setPen(other if i == 1 else color)
                self.painter.drawText(QPointF(cursor, baseline), part)
                cursor += metrics.horizontalAdvance(part)
        ink = metrics.tightBoundingRect(text).translated(x, baseline)
        label = Label(text, _rect(ink, pad + 1), role)
        self.labels.append(label)
        return label

    def advance(self, text: str, pixels: float) -> float:
        """Design-unit width of text, for layouts that center or space labels."""
        return QFontMetricsF(self.font(pixels)).horizontalAdvance(text) / self.factor

    def icon(self, kind: str, x: float, y: float, size: float, color: QColor) -> Icon:
        """Draw a generic icon whose square box starts at design (x, y)."""
        p = self.painter
        s = self.px(size)
        box = QRectF(self.px(x), self.px(y), s, s)
        c = box.center()
        stroke = max(1.5, s * 0.11)
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(color, stroke))
        inner = box.adjusted(stroke / 2, stroke / 2, -stroke / 2, -stroke / 2)
        if kind == "ring":
            p.drawEllipse(inner)
        elif kind == "square":
            p.drawRoundedRect(
                inner.adjusted(s * 0.08, s * 0.08, -s * 0.08, -s * 0.08), s * 0.08, s * 0.08
            )
        elif kind == "triangle":
            p.drawPolygon(
                QPolygonF([QPointF(c.x(), inner.top()), inner.bottomRight(), inner.bottomLeft()])
            )
        elif kind == "cross":
            p.drawEllipse(inner)
            d = s * 0.2
            p.drawLine(QPointF(c.x() - d, c.y() - d), QPointF(c.x() + d, c.y() + d))
            p.drawLine(QPointF(c.x() + d, c.y() - d), QPointF(c.x() - d, c.y() + d))
        elif kind == "keycap":
            box = QRectF(box.left(), box.top() + s * 0.15, s * 1.3, s * 0.7)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(225, 228, 235))
            p.drawRoundedRect(box, s * 0.15, s * 0.15)
            p.setPen(QColor(30, 34, 44))
            p.setFont(self.font(size * 0.42, "Segoe UI", bold=True))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, self.rng.choice(("R1", "L1", "F2")))
        elif kind == "emoji":
            p.setFont(self.font(size * 0.8, "Segoe UI Emoji"))
            p.setPen(color)
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, self.rng.choice(("⚔", "⭐", "💎", "❤")))
        else:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            if kind == "star":
                p.drawPolygon(_star(c, s / 2, s * 0.2, 5))
            elif kind == "spark":
                p.drawPolygon(_star(c, s / 2, s * 0.1, 4))
            elif kind == "dot":
                p.setBrush(QColor(225, 50, 60))
                p.drawPolygon(_star(c, s / 2, s / 2 * 0.7071, 2))
            elif kind == "gear":
                path = QPainterPath()
                path.addPolygon(_star(c, s / 2, s * 0.38, 8))
                path.addEllipse(c, s * 0.14, s * 0.14)
                path.setFillRule(Qt.FillRule.OddEvenFill)
                p.drawPath(path)
            elif kind == "silhouette":
                path = QPainterPath()
                path.addEllipse(QPointF(c.x() - s * 0.18, c.y() - s * 0.12), s * 0.26, s * 0.3)
                path.addEllipse(QPointF(c.x() + s * 0.18, c.y() - s * 0.12), s * 0.26, s * 0.3)
                path.addRoundedRect(
                    QRectF(c.x() - s * 0.45, c.y(), s * 0.9, s * 0.48), s * 0.2, s * 0.2
                )
                path.setFillRule(Qt.FillRule.WindingFill)
                p.drawPath(path)
            elif kind == "badge":
                p.setBrush(QColor(196, 160, 92))
                p.drawRoundedRect(box, s * 0.18, s * 0.18)
                p.setBrush(QColor(40, 36, 52))
                p.drawPolygon(
                    QPolygonF(
                        [
                            QPointF(c.x() - s * 0.22, box.top() + s * 0.22),
                            QPointF(c.x() + s * 0.22, box.top() + s * 0.22),
                            QPointF(c.x() - s * 0.22, box.bottom() - s * 0.22),
                            QPointF(c.x() + s * 0.22, box.bottom() - s * 0.22),
                        ]
                    )
                )
            else:
                raise ValueError(f"Unknown icon kind: {kind}")
        p.restore()
        item = Icon(kind, _rect(box if kind == "keycap" else QRectF(self.px(x), self.px(y), s, s)))
        self.icons.append(item)
        return item


WHITE, INK, GOLD = QColor(240, 240, 236), QColor(32, 34, 40), QColor(236, 214, 150)
PROMPTS = ("ring", "square", "triangle", "cross")
FILLED = ("star", "gear", "silhouette", "spark", "badge")


def _prompts(gap: float, color: QColor) -> Callable[[_Canvas], None]:
    def draw(c: _Canvas) -> None:
        labels = ("戻る", "選択", "詳細", "一括装備", "スキップ", "メニュー")
        for i, (kind, text) in enumerate(zip(PROMPTS * 2, labels, strict=False)):
            x, y, size = 120 + (i % 2) * 820, 120 + (i // 2) * 260, 34.0
            c.icon(kind, x, y, size, color)
            c.text(text, x + size + gap * size, y + size * 0.86, size, color)

    return draw


def _titles(c: _Canvas) -> None:
    for i, (kind, text) in enumerate(
        zip(
            ("silhouette", "star", "gear", "emoji"),
            ("キャラクター", "ステータス", "装備", "アイテム"),
            strict=True,
        )
    ):
        x, y, size = 140 + (i % 2) * 900, 160 + (i // 2) * 360, 44.0
        c.icon(kind, x, y - size * 0.15, size * 1.3, WHITE)
        c.text(text, x + size * 1.3 + size * 0.12, y + size * 0.86, size, WHITE, bold=True)


def _grid(words: tuple[str, ...], spread: float) -> Callable[[_Canvas], None]:
    """Icons above centered labels; the space between labels is near the space inside words."""

    def draw(c: _Canvas) -> None:
        size, icon = 22.0, 50.0
        widths = [c.advance(word, size) for word in words]
        pitch = max(widths) + size * spread
        x0, y = 1880 - pitch * len(words), 70.0
        c.icon("keycap", x0 - 50, y + 10, 30, WHITE)
        kinds = (*FILLED, "ring", "silhouette", "star")
        for i, (word, width) in enumerate(zip(words, widths, strict=True)):
            center = x0 + pitch * (i + 0.5)
            c.icon(kinds[i % len(kinds)], center - icon / 2, y, icon, WHITE)
            if i % 3 == 1:
                c.icon("dot", center + icon * 0.35, y - 4, 12, WHITE)
            c.text(word, center - width / 2, y + icon + 12 + size * 0.86, size, WHITE)

    return draw


def _hotbar(c: _Canvas) -> None:
    """Sentences centered just under a row of skill icons; they must stay whole."""
    size, icon, pitch = 24.0, 50.0, 80.0
    for row, (x0, sentence) in enumerate(
        ((560.0, "仲間と一緒に森の奥へ進む"), (560.0, "森の奥は危険だ、気をつけて進もう"))
    ):
        y = 120.0 + row * 420
        for i in range(8):
            c.icon((*FILLED, "ring", "gear", "star")[i], x0 + i * pitch + 15, y, icon, WHITE)
        width = c.advance(sentence, size)
        c.text(sentence, x0 + 4 * pitch - width / 2, y + icon + 14 + size * 0.86, size, WHITE)


def _tabs(c: _Canvas) -> None:
    c.painter.fillRect(QRectF(c.px(40), c.px(900), c.px(1840), c.px(60)), QColor(40, 40, 46))
    size, y = 26.0, 900 + 30 + 26 * 0.36
    c.icon("keycap", 60, 912, 30, WHITE)
    x = 220.0
    for word in ("詳細", "星祝", "装備", "調律", "プロフィール"):
        if word == "装備":
            c.painter.setBrush(QColor(245, 245, 245))
            c.painter.setPen(Qt.PenStyle.NoPen)
            c.painter.drawRoundedRect(
                QRectF(c.px(x - 60), c.px(906), c.px(200), c.px(48)), c.px(24), c.px(24)
            )
            c.icon("spark", x - 30, y - size * 0.95, size, INK)
            c.text(word, x + 4, y, size, INK)
        else:
            c.text(word, x, y, size, QColor(200, 200, 205))
        x += 300
    c.icon("keycap", 1760, 912, 30, WHITE)


def _stats(c: _Canvas) -> None:
    rows = (("攻撃力", "1250"), ("会心率", "15.0%"), ("防御力", "+15%"), ("HP回復", "×3"))
    for i, (name, value) in enumerate(rows):
        y, size = 140 + i * 90, 28.0
        c.icon("badge", 160, y, size * 1.1, GOLD)
        c.text(name, 160 + size * 1.4, y + size * 0.95, size, WHITE)
        c.text(value, 820 - c.advance(value, size), y + size * 0.95, size, WHITE, role=VALUE)
    for i, value in enumerate(("+15%", "1200", "Lv.80/80", "×2")):
        c.text(value, 1200 + (i % 2) * 300, 200 + (i // 2) * 220, 30, GOLD, role=VALUE)


def _icons_only(c: _Canvas) -> None:
    kinds = (*PROMPTS, *FILLED, "keycap", "emoji", "dot")
    for i, kind in enumerate(kinds):
        size = 30.0 + (i % 4) * 14
        c.icon(kind, 160 + (i % 6) * 280, 160 + (i // 6) * 300, size, WHITE)


def _marked(color: QColor) -> Callable[[_Canvas], None]:
    """Text only, led by heavy brackets, bullets and enumerators, plus ruby over its base text."""

    def draw(c: _Canvas) -> None:
        rows = (
            "【アリス】こんにちは",
            "●選択肢を選ぶ",
            "①最初の選択",
            "〔重要〕報告書",
            "▶次へ進む",
        )
        for i, text in enumerate(rows):
            c.text(text, 140 + (i % 2) * 900, 160 + (i // 2) * 220, 34, color)
        c.text("警告：明日は危険です", 140, 1000, 34, color, heavy=2)
        c.text("新着メッセージ", 1040, 1000, 34, color, heavy=2)
        c.text("ぼうけんしゃ", 1050, 776, 14, color)
        c.text("冒険者ギルドへ行く", 1040, 820, 34, color)

    return draw


def _negatives(color: QColor) -> Callable[[_Canvas], None]:
    """Text only: box-like kanji, brackets, split radicals, small kana, accents and outlines."""

    def draw(c: _Canvas) -> None:
        rows = (
            "口回田日目",
            "「回復する」",
            "川の小道を歩く…",
            "ハートを集めよう",
            "ぁぃぅぇぉっゃ",
            "ー・ー",
        )
        for i, text in enumerate(rows):
            c.text(text, 140 + (i % 2) * 900, 160 + (i // 2) * 220, 34, color)
        c.text("必ず報告すること", 140, 820, 34, color, accent=(3, 5, QColor(230, 80, 70)))
        c.text("縁取りの文字", 1040, 820, 34, WHITE, bold=True, outline=QColor(10, 10, 10))

    return draw


def _badges(c: _Canvas) -> None:
    for i, text in enumerate(("通常攻撃", "戦闘スキル", "必殺技", "天賦", "秘技")):
        y, size = 300 + i * 130, 40.0
        c.icon("badge", 1500, y, size * 1.1, GOLD)
        c.text(text, 1500 + size * 1.1 + size * 0.25, y + size * 0.95, size, WHITE)


SPECS = (
    *(
        Spec(
            f"prompts-{bg}-{name}",
            "prompt",
            _prompts(gap, INK if bg == "light" else WHITE),
            background=bg,
        )
        for bg in ("dark", "light", "gradient")
        for name, gap in (("tight", 0.15), ("loose", 0.5))
    ),
    Spec("titles-textured", "title", _titles, background="textured"),
    Spec("titles-dark", "title", _titles),
    Spec(
        "grid-menu",
        "grid",
        _grid(("クエスト", "チーム", "イベント", "漫遊記", "観測所", "キボウ", "キャラ"), 0.45),
        background="gradient",
    ),
    Spec(
        "grid-menu-4k",
        "grid",
        _grid(("ショップ", "図鑑", "任務", "メール", "設定", "フレンド"), 0.35),
        size=(3840, 2160),
        background="gradient",
    ),
    Spec("tab-bar", "tabs", _tabs, background="textured"),
    Spec("hotbar-subtitles", "negatives", _hotbar, background="gradient"),
    Spec("stats", "stats", _stats),
    Spec("stats-4k", "stats", _stats, size=(3840, 2160), background="gradient"),
    Spec("icons-only", "icons", _icons_only, background="gradient"),
    Spec("negatives-dark", "negatives", _negatives(WHITE)),
    Spec("negatives-light", "negatives", _negatives(INK), background="light"),
    Spec("marked-dark", "negatives", _marked(WHITE), background="gradient"),
    Spec("marked-light", "negatives", _marked(INK), background="light"),
    Spec("badges-4k", "badges", _badges, size=(3840, 2160), background="textured"),
)


def _frame(image: QImage) -> CapturedFrame:
    image = image.convertToFormat(QImage.Format.Format_RGB32)
    bounds = Rect(0, 0, image.width(), image.height())
    return CapturedFrame(1, 0, WindowGeometry(bounds, bounds, bounds, 96), bytes(image.constBits()))


def render(spec: Spec) -> Scene:
    if not QFontDatabase.hasFamily("Yu Gothic UI"):
        raise RuntimeError(
            "The Japanese fixture font is unavailable. Use the Windows Qt platform, "
            "not QT_QPA_PLATFORM=offscreen; missing-glyph scans cannot validate OCR."
        )
    width, height = spec.size
    image = QImage(width, height, QImage.Format.Format_RGB32)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    canvas = _Canvas(image, painter, width / 1920, random.Random(spec.name))
    canvas.background(spec.background)
    spec.draw(canvas)
    painter.end()
    return Scene(
        spec.name,
        spec.family,
        _frame(image),
        tuple(canvas.labels),
        tuple(canvas.icons),
        (Rect(0, 0, width, height),),
    )


def load_holdout(path: Path) -> tuple[Scene, ...]:
    """Annotated real frames, kept outside the repository; see the benchmark's --help."""
    document = json.loads(path.read_text(encoding="utf-8"))
    scenes = []
    for entry in document["frames"]:
        image = QImage(str(path.parent / entry["image"]))
        if image.isNull():
            raise ValueError(f"Unreadable holdout image: {entry['image']}")
        scenes.append(
            Scene(
                entry["name"],
                entry.get("family", "holdout"),
                _frame(image),
                tuple(
                    Label(t["text"], Rect(*t["bounds"]), t.get("role", TEXT))
                    for t in entry["labels"]
                ),
                tuple(Icon(i.get("kind", "icon"), Rect(*i["bounds"])) for i in entry["icons"]),
                tuple(Rect(*r) for r in entry["scope"]),
                tuple(Rect(*r) for r in entry["areas"]) if "areas" in entry else None,
            )
        )
    return tuple(scenes)


def normalized(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, char in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def _contains(box: Rect, x: float, y: float, slack: float) -> bool:
    return (
        box.x - slack <= x <= box.x + box.width + slack
        and box.y - slack <= y <= box.y + box.height + slack
    )


def _center(box: Rect) -> tuple[float, float]:
    return box.x + box.width / 2, box.y + box.height / 2


def _covered(icon: Rect, row: Rect) -> float:
    w = min(icon.x + icon.width, row.x + row.width) - max(icon.x, row.x)
    h = min(icon.y + icon.height, row.y + row.height) - max(icon.y, row.y)
    return max(0, w) * max(0, h) / max(1, icon.width * icon.height)


@dataclass(frozen=True, slots=True)
class LabelScore:
    expected: str
    observed: str
    errors: int
    role: str
    state: str  # separate, merged, fragmented or missed
    merged_with: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Phantom:
    """A row carrying no label characters, and whether the translation gate would admit it."""

    text: str
    translatable: bool


@dataclass(frozen=True, slots=True)
class SceneScore:
    labels: tuple[LabelScore, ...]
    icon_chars_attached: int
    icon_chars_phantom: int
    phantoms: tuple[Phantom, ...]
    covered_icons: int
    icons: int


@dataclass(frozen=True, slots=True)
class _Owned:
    kind: str  # label, icon or stray
    index: int
    text: str
    x: int


def _owners(scene: Scene, row: TextRegion) -> tuple[_Owned, ...]:
    """Attribute each character center to a label, else an icon, else nothing."""
    owned = []
    for char in row.characters:
        cx, cy = _center(char.bounds)
        kind, index = "stray", -1
        for i, label in enumerate(scene.labels):
            if _contains(label.bounds, cx, cy, max(2.0, label.bounds.height * 0.1)):
                kind, index = "label", i
                break
        else:
            for i, icon in enumerate(scene.icons):
                if _contains(icon.bounds, cx, cy, 1.0):
                    kind, index = "icon", i
                    break
        owned.append(_Owned(kind, index, char.text, char.bounds.x))
    return tuple(owned)


def score(scene: Scene, rows: Iterable[TextRegion], language: SourceLanguage) -> SceneScore:
    """Score tracking rows (after content filtering) against drawn ink.

    Labels are scored by edit distance over their own characters and by whether one row carries
    them alone. Icon characters are contamination. Rows with no label characters are phantoms.
    Rows need character boxes; rows without them count as phantoms of their full text.
    """
    rows = [r for r in rows if any(_contains(a, *_center(r.bounds), 0) for a in scene.scope)]
    owners = [_owners(scene, row) for row in rows]
    labels = []
    for i, label in enumerate(scene.labels):
        carriers = [o for o in owners if any(u.kind == "label" and u.index == i for u in o)]
        chars = sorted(
            (u.x, u.text) for o in carriers for u in o if u.kind == "label" and u.index == i
        )
        others = {u.index for o in carriers for u in o if u.kind == "label" and u.index != i}
        expected, observed = normalized(label.text), normalized("".join(t for _, t in chars))
        state = (
            "missed"
            if not carriers
            else "fragmented"
            if len(carriers) > 1
            else "merged"
            if others
            else "separate"
        )
        labels.append(
            LabelScore(
                expected,
                observed,
                distance(expected, observed),
                label.role,
                state,
                tuple(sorted(normalized(scene.labels[n].text) for n in others)),
            )
        )
    attached = phantom = 0
    phantoms = []
    covered: set[int] = set()
    for row, owned in zip(rows, owners, strict=True):
        icon_chars = sum(u.kind == "icon" for u in owned)
        if any(u.kind == "label" for u in owned):
            attached += icon_chars
            covered.update(
                i for i, icon in enumerate(scene.icons) if _covered(icon.bounds, row.bounds) >= 0.3
            )
        else:
            phantom += icon_chars
            phantoms.append(Phantom(row.text, language.contains_text(row.text)))
    return SceneScore(
        tuple(labels), attached, phantom, tuple(phantoms), len(covered), len(scene.icons)
    )


@dataclass(frozen=True, slots=True)
class Totals:
    labels: int = 0
    exact: int = 0
    errors: int = 0
    separate: int = 0
    merged: int = 0
    fragmented: int = 0
    missed: int = 0
    icon_chars_attached: int = 0
    icon_chars_phantom: int = 0
    phantom_rows: int = 0
    phantom_translatable: int = 0
    covered_icons: int = 0
    icons: int = 0

    @classmethod
    def of(cls, scores: Iterable[SceneScore]) -> "Totals":
        scores = tuple(scores)
        labels = [label for s in scores for label in s.labels]
        phantoms = [p for s in scores for p in s.phantoms]
        return cls(
            labels=len(labels),
            exact=sum(label.errors == 0 for label in labels),
            errors=sum(label.errors for label in labels),
            separate=sum(label.state == "separate" for label in labels),
            merged=sum(label.state == "merged" for label in labels),
            fragmented=sum(label.state == "fragmented" for label in labels),
            missed=sum(label.state == "missed" for label in labels),
            icon_chars_attached=sum(s.icon_chars_attached for s in scores),
            icon_chars_phantom=sum(s.icon_chars_phantom for s in scores),
            phantom_rows=len(phantoms),
            phantom_translatable=sum(p.translatable for p in phantoms),
            covered_icons=sum(s.covered_icons for s in scores),
            icons=sum(s.icons for s in scores),
        )
