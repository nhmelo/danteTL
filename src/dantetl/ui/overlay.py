"""Qt overlay rendering; receives geometry and never owns capture state."""

from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QHideEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QRegion,
)
from PySide6.QtWidgets import QWidget

from dantetl.capture.backdrop import LabelBackdrops
from dantetl.capture.windows import Windows
from dantetl.domain import Rect, WindowTarget, sample_box
from dantetl.ocr.observation import TextTrack
from dantetl.translation.grouping import TextBlock
from dantetl.translation.values import TranslatedText, TranslationRequest
from dantetl.ui.diagnostics import DebugDisplay
from dantetl.ui.english_placement import EnglishLabel, EnglishPlacement
from dantetl.ui.text_layout import PAD_X, PAD_Y, EnglishLayout


class OverlayWindow(QWidget):
    def __init__(self, windows: Windows) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.setWindowTitle("DanteTL surface")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._windows = windows
        self._bounds = Rect(0, 0, 1, 1)
        self._anchor: WindowTarget | None = None
        self._opacity = 0.85
        self._configured = False
        self._configuration_failed = False
        self._include_in_capture = True
        self._tracks: tuple[TextTrack, ...] | None = None
        self._translations: tuple[TranslatedText, ...] = ()
        self._english_mode = False
        self._blocks: tuple[TextBlock, ...] | None = None
        self._automatic_size = True
        self._manual_size = 20
        self._debug: DebugDisplay | None = None
        self._placement = EnglishPlacement()
        self._layouts: dict[tuple[tuple[int, int, str], ...], EnglishLayout] = {}
        self._backdrops: LabelBackdrops | None = None
        self._backed: frozenset[Rect] = frozenset()
        self._painted_boxes: set[Rect] = set()
        self._backdrop_update = QTimer(self)
        self._backdrop_update.setSingleShot(True)
        self._backdrop_update.timeout.connect(self._sync_backdrops)

    def set_backdrop_blur(self, enabled: bool) -> None:
        if enabled == (self._backdrops is not None) or QGuiApplication.platformName() != "windows":
            return
        if enabled:
            self._backdrops = LabelBackdrops(self._windows)
            self.destroyed.connect(self._backdrops.clear)
        elif self._backdrops is not None:
            self._backdrop_update.stop()
            self.destroyed.disconnect(self._backdrops.clear)
            self._backdrops.clear()
            self._backdrops = None
            self._backed = frozenset()
            self._painted_boxes.clear()
        self.update()

    def hideEvent(self, event: QHideEvent) -> None:
        self._backdrop_update.stop()
        if self._backdrops is not None:
            self._backdrops.clear()
        self._backed = frozenset()
        self._painted_boxes.clear()
        super().hideEvent(event)

    def _sync_backdrops(self) -> None:
        self._place_backdrops(
            tuple(
                layout.bounds
                for layout in self._layouts.values()
                if layout.bounds in self._painted_boxes
            )
        )

    def _retire_backdrops(self) -> None:
        if not self._backed.issubset(self._painted_boxes):
            self._place_backdrops(tuple(self._backed & self._painted_boxes))

    def _place_backdrops(self, boxes: tuple[Rect, ...]) -> None:
        if self._backdrops is None or not self._configured or not self.isVisible():
            return
        ready = self._backdrops.place(
            int(self.winId()), self._bounds, boxes, included=self._include_in_capture
        )
        backed = ready or frozenset()
        for box in backed ^ self._backed:
            self.update(self._widget_box(box))
        self._backed = backed
        if ready is not None and any(box not in ready for box in boxes):
            self._backdrop_update.start(1)

    def set_debug(self, display: DebugDisplay | None) -> None:
        if display != self._debug:
            self._debug = display
            self.update()

    def set_scene(
        self,
        tracks: tuple[TextTrack, ...],
        blocks: tuple[TextBlock, ...] | None = None,
        translations: tuple[TranslatedText, ...] = (),
        *,
        english: bool = False,
    ) -> None:
        if (tracks, blocks, translations, english) == (
            self._tracks,
            self._blocks,
            self._translations,
            self._english_mode,
        ):
            return
        mode_changed = self._tracks is None or self._english_mode != english
        sources_changed = blocks != self._blocks or (blocks is None and tracks != self._tracks)
        results_changed = translations != self._translations
        self._tracks, self._blocks = tracks, blocks
        self._translations, self._english_mode = translations, english
        if not english:
            self._placement.clear()
            self._layouts.clear()
            self._painted_boxes.clear()
            self._retire_backdrops()
            self.update()
        elif mode_changed or sources_changed or results_changed:
            self._layout_english()
            if mode_changed:
                self.update()  # Clear the previous raw/debug surface.

    def _widget_box(self, bounds: Rect) -> QRect:
        ratio = self.devicePixelRatioF()
        return (
            QRectF(bounds.x / ratio, bounds.y / ratio, bounds.width / ratio, bounds.height / ratio)
            .toAlignedRect()
            .adjusted(-1, -1, 1, 1)
        )

    def _layout_english(self) -> None:
        blocks = self._blocks
        if blocks is None:
            blocks = tuple(TextBlock((track,)) for track in self._tracks or ())
        english = {value.request: value.english for value in self._translations}
        labels = tuple(
            EnglishLabel(
                block.key,
                text,
                block.bounds,
                block.bounds.width if block.lines[0].vertical else block.line_height,
                block.lines[0].vertical,
            )
            for block in blocks
            if (text := english.get(TranslationRequest.from_block(block)))
        )
        placed = self._placement.arrange(
            labels,
            self._bounds.size,
            tuple(block.bounds for block in blocks),
            automatic=self._automatic_size,
            pixels=self._manual_size,
        )
        current = {
            label.key: layout
            for label, layout in zip(labels, placed, strict=True)
            if layout is not None
        }
        dirty = QRegion()
        for key in self._layouts.keys() | current.keys():
            old, new = self._layouts.get(key), current.get(key)
            if old != new:
                for layout in (old, new):
                    if layout is not None:
                        dirty |= QRegion(self._widget_box(layout.bounds))
        if not dirty.isEmpty():
            self._painted_boxes.intersection_update(
                layout.bounds
                for key, layout in current.items()
                if self._layouts.get(key) == layout or layout.bounds in self._backed
            )
            self._retire_backdrops()
            self.update(dirty)
        self._layouts = current

    def set_text_style(self, automatic: bool, pixels: int) -> None:
        if (automatic, pixels) != (self._automatic_size, self._manual_size):
            self._automatic_size, self._manual_size = automatic, pixels
            if self._english_mode:
                self._layout_english()

    def set_opacity(self, value: int) -> None:
        opacity = max(0.2, min(value / 100, 1.0))
        if opacity != self._opacity:
            self._opacity = opacity
            self.update()

    def retry_configuration(self) -> None:
        self._configuration_failed = False

    def _configure(self, hwnd: int, included: bool) -> None:
        self._windows.configure_overlay(hwnd)
        self._windows.set_overlay_capture(hwnd, included)
        self._configured = True
        self._configuration_failed = False

    def align(self, bounds: Rect, *, anchor: WindowTarget | None = None) -> str | None:
        """Place the surface, reporting a first-show failure once per explicit attempt."""
        if self._configuration_failed:
            return None
        hwnd = int(self.winId())
        if not self._configured:
            try:
                self._configure(hwnd, self._include_in_capture)
            except (OSError, RuntimeError) as error:
                self._configuration_failed = True
                self.hide()
                return str(error)
        anchor_changed = self._anchor != anchor
        self._anchor = anchor
        changed = self._bounds != bounds
        resized = self._bounds.size != bounds.size
        if changed:
            self._painted_boxes.clear()
            self._retire_backdrops()
        self._bounds = bounds
        if resized and self._english_mode:
            self._layout_english()
        showing = not self.isVisible()

        def place() -> bool:
            if anchor is None and not (showing or changed or anchor_changed):
                return True
            handles = self._backdrops.visible_handles if self._backdrops else frozenset()
            area = bounds if changed or showing else None
            if anchor is not None:
                return self._windows.align_to_game(hwnd, anchor, area, handles)
            return self._windows.place_overlay(hwnd, area, handles)

        if not place():
            self.hide()
            return None
        if showing:
            self.show()
            # Qt may apply its pending show geometry; enforce physical bounds afterwards.
            if not place():
                self.hide()
                return None
        if changed or showing:
            self.update()
        return None

    def set_capture_included(self, included: bool) -> None:
        """Change recording visibility without hiding/repainting or resetting live work."""
        if included != self._include_in_capture:
            if self._backdrops is not None:
                self._backdrops.clear()
                self._backed = frozenset()
                self._painted_boxes.clear()
                self.update()
            if self._configured:
                self._windows.set_overlay_capture(int(self.winId()), included)
            elif self._configuration_failed:
                self._configure(int(self.winId()), included)
            self._include_in_capture = included

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        ratio = self.devicePixelRatioF()
        painter.scale(1 / ratio, 1 / ratio)
        width, height = self._bounds.size
        if self._tracks is not None:
            if self._english_mode:
                painted = len(self._painted_boxes)
                for layout in self._layouts.values():
                    if event.region().intersects(self._widget_box(layout.bounds)):
                        self._paint_english(painter, layout, blurred=layout.bounds in self._backed)
                        if self._backdrops is not None:
                            self._painted_boxes.add(layout.bounds)
                if len(self._painted_boxes) > painted and not self._backdrop_update.isActive():
                    self._backdrop_update.start(1)
                self._paint_debug(painter, ratio)
                return  # Pending regions leave the original game text visible.
            font = QFont("Segoe UI")
            font.setPixelSize(round(13 * ratio))
            painter.setFont(font)
            for track in self._tracks:
                box = track.bounds
                painter.setOpacity(self._opacity)
                painter.setPen(QPen(QColor("#5eead4"), max(2, ratio)))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(QRectF(box.x, box.y, box.width, box.height))
                label = str(track.track_id)
                badge_width = max(
                    28 * ratio, painter.fontMetrics().horizontalAdvance(label) + 8 * ratio
                )
                badge = QRectF(
                    max(0, box.x - badge_width - 2 * ratio), box.y, badge_width, 20 * ratio
                )
                painter.fillRect(badge, QColor("#123a35"))
                painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, label)
                painter.setOpacity(1.0)
            self._paint_debug(painter, ratio)
            return
        painter.setPen(QPen(QColor("#5eead4"), max(2, ratio)))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(2, 2, max(0, width - 4), max(0, height - 4)))

        box = sample_box(width, height)
        area = QRectF(box.x, box.y, box.width, box.height)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(12, 20, 31, round(255 * self._opacity)))
        painter.drawRoundedRect(area, 8 * ratio, 8 * ratio)
        font = QFont("Segoe UI")
        font.setPixelSize(max(12, round(min(height * 0.035, box.height * 0.24))))
        painter.setFont(font)
        painter.setPen(QColor("#f1f5f9"))
        painter.drawText(
            area.adjusted(12 * ratio, 4 * ratio, -12 * ratio, -4 * ratio),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            "Sample overlay · clicks pass through\nNative capture and alignment check",
        )
        self._paint_debug(painter, ratio)

    def _paint_debug(self, painter: QPainter, ratio: float) -> None:
        if self._debug is None:
            return
        painter.save()
        painter.setOpacity(1)
        painter.setPen(Qt.PenStyle.NoPen)
        for box in self._debug.changed:
            painter.fillRect(QRectF(box.x, box.y, box.width, box.height), QColor(255, 180, 50, 55))
        painter.setPen(QPen(QColor("#22d3ee"), max(2, ratio), Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for box in self._debug.scanned:
            painter.drawRect(
                QRectF(box.x + 1, box.y + 1, max(0, box.width - 2), max(0, box.height - 2))
            )
        font = QFont("Consolas")
        font.setPixelSize(round(12 * ratio))
        painter.setFont(font)
        metrics = painter.fontMetrics()
        lines = self._debug.text.splitlines()
        width = min(self._bounds.width - 16, max(metrics.horizontalAdvance(s) for s in lines) + 16)
        area = QRectF(8, 8, max(1, width), len(lines) * metrics.height() + 12)
        painter.fillRect(area, QColor(8, 14, 23, 230))
        painter.setPen(QColor("#e7edf5"))
        painter.setClipRect(area)
        painter.drawText(area.adjusted(8, 6, -8, -6), Qt.AlignmentFlag.AlignLeft, self._debug.text)
        painter.restore()

    def _paint_english(self, painter: QPainter, layout: EnglishLayout, *, blurred: bool) -> None:
        area = QRectF(layout.bounds.x, layout.bounds.y, layout.bounds.width, layout.bounds.height)
        painter.save()
        painter.setBrush(
            Qt.BrushStyle.NoBrush if blurred else QColor(12, 20, 31, round(255 * self._opacity))
        )
        painter.setPen(QPen(QColor(255, 255, 255, 112), 1))
        radius = min(4 * self.devicePixelRatioF(), area.width() / 2, area.height() / 2)
        painter.drawRoundedRect(area.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
        english_font = QFont("Segoe UI")
        english_font.setPixelSize(layout.pixel_size)
        painter.setFont(english_font)
        if layout.right_to_left:
            painter.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        painter.setClipRect(area)
        shadow = QPointF(1, min(2, layout.pixel_size / 12))
        align = Qt.AlignmentFlag.AlignRight if layout.right_to_left else Qt.AlignmentFlag.AlignLeft
        flags = (
            align
            | Qt.AlignmentFlag.AlignAbsolute
            | Qt.AlignmentFlag.AlignVCenter
            | Qt.TextFlag.TextSingleLine
        )
        for index, line in enumerate(layout.lines):
            bounds = QRectF(
                area.x() + PAD_X,
                area.y() + PAD_Y + index * layout.line_height,
                area.width() - 2 * PAD_X,
                layout.line_height,
            )
            painter.setPen(QColor(0, 0, 0, 160))
            painter.drawText(bounds.translated(shadow), flags, line)
            painter.setPen(QColor("#f1f5f9"))
            painter.drawText(bounds, flags, line)
        painter.restore()
