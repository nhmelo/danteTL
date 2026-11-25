"""Inline Ollama server discovery and model selection for Settings and the welcome guide."""

from PySide6.QtCore import QModelIndex, QPersistentModelIndex, QRect, QSize, Qt
from PySide6.QtGui import QColor, QFontMetrics, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from dantetl.application.services import OllamaCommands
from dantetl.i18n import mark, tr
from dantetl.translation.ollama.values import OllamaListing, OllamaModel
from dantetl.ui.localization import localized, on_language_change

_DETAIL_ROLE = Qt.ItemDataRole.UserRole + 1


class _ModelRow(QStyledItemDelegate):
    """Use the native checkbox and selection with a title and a secondary metadata line."""

    def sizeHint(
        self, option: QStyleOptionViewItem, index: QModelIndex | QPersistentModelIndex
    ) -> QSize:
        return QSize(0, 2 * option.fontMetrics.height() + 16)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        style = opt.widget.style() if opt.widget else QApplication.style()
        area = style.subElementRect(QStyle.SubElement.SE_ItemViewItemText, opt, opt.widget)
        title, opt.text = opt.text, ""
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)
        area.adjust(5, 0, -5, 0)
        line = opt.fontMetrics.height()
        text_box = QRect(area.left(), area.center().y() - line, area.width(), line)
        align = Qt.AlignmentFlag.AlignLeading | Qt.AlignmentFlag.AlignVCenter
        painter.save()
        painter.setLayoutDirection(opt.direction)
        font = opt.font
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(opt.palette.text().color())
        painter.drawText(
            text_box,
            align,
            QFontMetrics(font).elidedText(title, Qt.TextElideMode.ElideRight, area.width()),
        )
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor("#aabacc"))
        painter.drawText(
            text_box.translated(0, line),
            align,
            QFontMetrics(font).elidedText(
                index.data(_DETAIL_ROLE), Qt.TextElideMode.ElideRight, area.width()
            ),
        )
        painter.restore()


def _details(entry: OllamaListing) -> str:
    parts = [entry.model.name]
    if entry.parameter_size:
        parts.append(entry.parameter_size)
    if entry.quantization:
        parts.append(entry.quantization)
    if entry.size_bytes:
        parts.append(
            f"{entry.size_bytes / 1e9:.2f} GB"
            if entry.size_bytes >= 10**9
            else f"{entry.size_bytes / 1e6:.0f} MB"
        )
    return "   ·   ".join(parts)


class OllamaSettings(QWidget):
    def __init__(self, service: OllamaCommands, *, guide: bool = False):
        super().__init__()
        self.service, self._guide = service, guide
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 4)
        layout.setSpacing(8)
        label = localized(QLabel(), text=mark("Ollama server"))
        label.setObjectName("sectionTitle")
        layout.addWidget(label)
        intro = localized(
            QLabel(),
            text=mark(
                "Find translation models installed in Ollama on this PC or another computer."
            ),
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        row = QHBoxLayout()
        self.address = QLineEdit(service.snapshot().config.server)
        self.address.setObjectName("ollamaAddress")
        localized(self.address, accessibleName=mark("Server address"))
        self.address.setMaxLength(512)
        label.setBuddy(self.address)
        self.find_button = localized(QPushButton(), text=mark("Find models"))
        self.find_button.setObjectName("primary")
        self.find_button.clicked.connect(self._find)
        row.addWidget(self.address, 1)
        row.addWidget(self.find_button)
        layout.addLayout(row)
        self.results = QWidget()
        box = QVBoxLayout(self.results)
        box.setContentsMargins(0, 10, 0, 0)
        box.setSpacing(8)
        heading = localized(QLabel(), text=mark("Models"))
        heading.setObjectName("sectionTitle")
        box.addWidget(heading)
        hint = localized(
            QLabel(), text=mark("Choose which models appear in the translation picker.")
        )
        hint.setWordWrap(True)
        box.addWidget(hint)
        self.list = QListWidget()
        self.list.setObjectName("ollamaModels")
        self.list.setItemDelegate(_ModelRow(self.list))
        self.list.setUniformItemSizes(True)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        localized(self.list, accessibleName=mark("Models"))
        heading.setBuddy(self.list)
        box.addWidget(self.list)
        layout.addWidget(self.results)
        self.status = QLabel()
        self.status.setObjectName("muted")
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.save_button = localized(QPushButton(self), text=mark("Save"))
        self.save_button.clicked.connect(self.save)
        layout.addWidget(self.save_button, 0, Qt.AlignmentFlag.AlignLeading)
        layout.addStretch()
        self._last = None
        self.list.itemChanged.connect(self._update_save)
        self.address.textEdited.connect(self._edited)
        self.address.returnPressed.connect(self._find)
        on_language_change(self, self._status)
        self.refresh_view()

    def cancel_pending(self) -> None:
        if not self.service.snapshot().busy:
            return
        self.service.begin()
        self.address.setText(self.service.snapshot().config.server)
        self.refresh_view()

    def _edited(self) -> None:
        self.service.discard()
        self.refresh_view()

    def _find(self) -> None:
        self.service.find(self.address.text())
        self.refresh_view()

    def _selection(self) -> tuple[OllamaModel, ...]:
        return tuple(
            self.list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.list.count())
            if self.list.item(i).checkState() == Qt.CheckState.Checked
        )

    def _update_save(self) -> None:
        view = self.service.snapshot()
        selected = set(self._selection())
        enabled = tuple(entry for entry in view.listings if entry.model in selected)
        changed = view.config.server != view.server or view.config.listings != enabled
        self.save_button.setVisible(
            not self._guide and bool(view.server and (view.listings or view.config.models))
        )
        self.save_button.setEnabled(
            bool(view.server)
            and not view.busy
            and changed
            and (not self._guide or bool(selected) or view.server == view.config.server)
        )
        self._status()

    def _status(self) -> None:
        view = self.service.snapshot()
        message = view.error
        if view.busy:
            message = tr("Working…")
        elif not message and view.server and not view.listings:
            message = tr(
                "No models found. Install a text generation model in Ollama, "
                "then choose Find models again."
            )
        elif view.saved and self._selection() == view.config.models:
            message = tr("Model selection saved.")
        self.status.setText(message)
        self.status.setVisible(bool(message))

    @property
    def pending_selection(self) -> bool:
        return self.save_button.isEnabled()

    def save(self) -> None:
        self.service.save(self.service.snapshot().server, self._selection())
        self.refresh_view()

    def refresh_view(self) -> None:
        view = self.service.snapshot()
        previous = self._last
        if view is previous:
            return
        self._last = view
        self.find_button.setEnabled(not view.busy)
        self.address.setEnabled(not view.busy)
        self.list.setEnabled(not view.busy)
        if view.server:
            self.address.setText(view.server)
        if previous is None or (view.server, view.listings, view.config) != (
            previous.server,
            previous.listings,
            previous.config,
        ):
            self.list.blockSignals(True)
            self.list.clear()
            checked = (
                {m.name for m in view.config.models} if view.server == view.config.server else set()
            )
            for entry in view.listings:
                item = QListWidgetItem(entry.model.display_name, self.list)
                item.setData(Qt.ItemDataRole.UserRole, entry.model)
                detail = _details(entry)
                item.setData(_DETAIL_ROLE, detail)
                item.setData(Qt.ItemDataRole.AccessibleDescriptionRole, detail)
                item.setToolTip(detail)
                item.setCheckState(
                    Qt.CheckState.Checked
                    if entry.model.name in checked
                    else Qt.CheckState.Unchecked
                )
            self.list.blockSignals(False)
            self.results.setVisible(bool(view.listings))
            if view.listings:
                self.list.setFixedHeight(
                    min(3 if self._guide else 5, self.list.count()) * self.list.sizeHintForRow(0)
                    + 2 * self.list.frameWidth()
                    + 4
                )
        self._update_save()
