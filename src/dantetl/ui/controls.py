"""Accessible native Qt controls with a small, consistent presentation."""

from collections.abc import Callable
from html import escape

from PySide6.QtCore import QPoint, QRect, QRectF, QSignalBlocker, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFocusEvent,
    QIcon,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QStandardItem,
    QStandardItemModel,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QProgressBar,
    QPushButton,
    QStyle,
    QStyleOptionFocusRect,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from dantetl.i18n import mark, tr
from dantetl.languages.selection import SOURCE_LANGUAGES
from dantetl.languages.target import TARGET_LANGUAGES
from dantetl.ui.localization import localized, localized_item


def add_combo_section(combo: QComboBox, label: str) -> None:
    """A consistent, non-selectable category in a picker."""
    model = combo.model()
    assert isinstance(model, QStandardItemModel)
    heading = QStandardItem(label)
    heading.setFlags(Qt.ItemFlag.NoItemFlags)
    font = combo.font()
    font.setBold(True)
    heading.setFont(font)
    heading.setForeground(QColor("#9baabc"))
    heading.setBackground(QColor("#101922"))
    heading.setSizeHint(QSize(0, max(32, combo.fontMetrics().height() + 14)))
    heading.setAccessibleDescription(tr("Category heading"))
    model.appendRow(heading)


def set_tone(widget: QWidget, tone: str) -> None:
    if widget.property("tone") != tone:
        widget.setProperty("tone", tone)
        widget.style().unpolish(widget)
        widget.style().polish(widget)


class SourceLanguageCombo(QComboBox):
    """Choose a source language from the installed application's catalog."""

    def __init__(self, active: str) -> None:
        super().__init__()
        localized(self, accessibleName=mark("Game text language"))
        for language in SOURCE_LANGUAGES.values():
            localized_item(self, language.name, language.code, capitalize=True)
        self.setCurrentIndex(self.findData(active))


class OutputLanguageCombo(QComboBox):
    language_changed = Signal(str)

    def __init__(self, selected: str) -> None:
        super().__init__()
        localized(self, accessibleName=mark("Output language"))
        for target in TARGET_LANGUAGES.values():
            localized_item(self, target.name, target.code, capitalize=True)
        self.set_language(selected)
        self.currentIndexChanged.connect(lambda: self.language_changed.emit(self.currentData()))

    def set_language(self, code: str) -> None:
        if code != self.currentData():
            with QSignalBlocker(self):
                self.setCurrentIndex(self.findData(code))


class StatusBanner(QWidget):
    """A visible symbol, short explanation and action; color is never the only cue."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("statusBanner")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        body = QVBoxLayout(self)
        body.setContentsMargins(12, 12, 12, 12)
        layout = QHBoxLayout()
        body.addLayout(layout)
        self.label = QLabel()
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        self.label.setWordWrap(True)
        layout.addWidget(self.label, 1)
        self.action = QPushButton()
        layout.addWidget(self.action)
        self.detail = QLabel()
        self.detail.setTextFormat(Qt.TextFormat.PlainText)
        self.detail.setWordWrap(True)
        body.addWidget(self.detail)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        localized(self.progress, accessibleName=mark("Loading progress"))
        body.addWidget(self.progress)
        self.detail.hide()
        self.progress.hide()
        self.hide()

    def set_message(
        self,
        text: str,
        *,
        tone: str = "warning",
        action: str = "",
        detail: str = "",
        busy: bool = False,
    ) -> None:
        symbol = "⚠" if tone in ("warning", "error") else "ⓘ"
        self.label.setText(f"{symbol}  {text[:400]}" if text else "")
        self.label.setToolTip(text)
        self.action.setText(action)
        self.action.setVisible(bool(action))
        self.detail.setText(detail)
        self.detail.setVisible(bool(detail))
        self.progress.setVisible(busy)
        set_tone(self, tone)
        self.setVisible(bool(text))


class Switch(QCheckBox):
    """Keep Qt's checkbox semantics, keyboard handling and signals; paint a switch indicator."""

    def sizeHint(self) -> QSize:
        return QSize(
            48 + self.fontMetrics().horizontalAdvance(self.text()),
            max(28, self.fontMetrics().height() + 8),
        )

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def hitButton(self, pos: QPoint) -> bool:
        return self.rect().contains(pos)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.isEnabled():
            painter.setOpacity(0.45)
        y = (self.height() - 20) / 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#4fd1bd" if self.isChecked() else "#58677a"))
        painter.drawRoundedRect(QRectF(2, y, 36, 20), 10, 10)
        painter.setBrush(QColor("#11232b" if self.isChecked() else "#edf2f7"))
        painter.drawEllipse(QRectF(20 if self.isChecked() else 4, y + 2, 16, 16))
        painter.setPen(self.palette().windowText().color())
        painter.drawText(
            QRect(48, 0, self.width() - 48, self.height()),
            Qt.AlignmentFlag.AlignVCenter,
            self.text(),
        )
        if self.hasFocus():
            option = QStyleOptionFocusRect()
            option.initFrom(self)
            option.rect = self.rect().adjusted(0, 1, -1, -1)
            self.style().drawPrimitive(
                QStyle.PrimitiveElement.PE_FrameFocusRect, option, painter, self
            )


class CaptureButton(QPushButton):
    """One capture action, with matching text, icon and running color."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("captureButton")
        self._icons = {}
        for running in (False, True):
            pixmap = QPixmap(36, 36)
            pixmap.setDevicePixelRatio(2)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#ffffff"))
            if running:
                painter.drawRoundedRect(QRectF(4, 4, 10, 10), 1, 1)
            else:
                painter.drawEllipse(QRectF(3, 3, 12, 12))
            painter.end()
            self._icons[running] = QIcon(pixmap)
        self.setIconSize(QSize(18, 18))
        self.set_capturing(False)

    def set_capturing(self, active: bool) -> None:
        self.setText(tr("Stop capture") if active else tr("Start capture"))
        self.setIcon(self._icons[active])
        set_tone(self, "recording" if active else "idle")


class WrappingLayout(QLayout):
    """Pack widgets at their preferred widths and move overflowing items to the next row."""

    def __init__(self, *widgets: QWidget, spacing: int = 18, row_spacing: int = 6) -> None:
        super().__init__()
        self._items: list[QLayoutItem] = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)
        self._row_spacing = row_spacing
        for widget in widgets:
            self.addWidget(widget)

    def addItem(self, item: QLayoutItem) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:
        return self._items[index] if 0 <= index < self.count() else None

    def takeAt(self, index: int) -> QLayoutItem | None:
        return self._items.pop(index) if 0 <= index < self.count() else None

    def minimumSize(self) -> QSize:
        size = QSize(0, 0)
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._arrange(width)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._arrange(rect.width(), rect.topLeft())

    def _arrange(self, width: int, origin: QPoint | None = None) -> int:
        x = y = row_height = 0
        for item in self._items:
            size = item.sizeHint()
            size.setWidth(min(size.width(), max(0, width)))
            if x and x + size.width() > width:
                x = 0
                y += row_height + self._row_spacing
                row_height = 0
            if origin is not None:
                item.setGeometry(QRect(origin + QPoint(x, y), size))
            x += size.width() + self.spacing()
            row_height = max(row_height, size.height())
        return y + row_height


class InfoButton(QToolButton):
    def __init__(
        self, name: str, explanation: str | Callable[[], str], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setText("ⓘ")
        self.setObjectName("info")
        localized(
            self,
            accessibleName=lambda: tr("About {name}", name=tr(name)),
            accessibleDescription=explanation,
            toolTip=lambda: f"<qt>{escape(self.accessibleDescription())}</qt>",
        )
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.clicked.connect(self._show_help)

    def _show_help(self) -> None:
        QToolTip.showText(self.mapToGlobal(QPoint(0, self.height())), self.toolTip(), self)

    def focusInEvent(self, event: QFocusEvent) -> None:
        super().focusInEvent(event)
        self._show_help()

    def focusOutEvent(self, event: QFocusEvent) -> None:
        QToolTip.hideText()
        super().focusOutEvent(event)


class CopyTextButton(QToolButton):
    """Icon-only clipboard action with keyboard/tooltip semantics and brief feedback."""

    def __init__(self, label: str, text: str, parent: QWidget) -> None:
        super().__init__(parent)
        self._label, self._text = label, text
        self.setAccessibleName(label)
        self.setToolTip(label)
        self.setAutoRaise(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(28, 28)
        self.setIconSize(QSize(18, 18))
        self.setStyleSheet(
            "QToolButton { border: 0; padding: 4px; background: transparent; }"
            "QToolButton:hover, QToolButton:focus { background: #304259; border-radius: 4px; }"
        )
        self._icons = {}
        for copied in (False, True):
            pixmap = QPixmap(36, 36)
            pixmap.setDevicePixelRatio(2)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(QColor("#5eead4" if copied else "#9baabc"), 1.5))
            if copied:
                painter.drawLine(3, 9, 7, 13)
                painter.drawLine(7, 13, 15, 4)
            else:
                painter.drawRoundedRect(QRectF(3, 2, 9, 11), 1, 1)
                painter.setBrush(QColor("#172230"))
                painter.drawRoundedRect(QRectF(7, 6, 8, 10), 1, 1)
            painter.end()
            self._icons[copied] = QIcon(pixmap)
        self.setIcon(self._icons[False])
        self._feedback = QTimer(self)
        self._feedback.setSingleShot(True)
        self._feedback.timeout.connect(self._reset)
        self.clicked.connect(self._copy)

    def set_label(self, label: str) -> None:
        if label != self._label:
            self._label = label
            self.setAccessibleName(label)
            self._reset()

    def set_copy_text(self, text: str) -> None:
        if text != self._text:
            self._text = text
            self._reset()

    def _copy(self) -> None:
        QApplication.clipboard().setText(self._text)
        self.setIcon(self._icons[True])
        self.setToolTip(self._label.replace(tr("Copy"), tr("Copied"), 1))
        self._feedback.start(1200)

    def _reset(self) -> None:
        self._feedback.stop()
        self.setIcon(self._icons[False])
        self.setToolTip(self._label)
