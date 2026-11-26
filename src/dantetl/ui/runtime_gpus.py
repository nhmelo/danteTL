"""Graphics hardware presentation shared by Acceleration and the setup guide."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QResizeEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QWidget,
)

from dantetl.i18n import tr
from dantetl.runtime.hardware_values import Adapter
from dantetl.ui.controls import WrappingLayout
from dantetl.ui.icons import ICON_DIR


def _icon(adapter: Adapter) -> QIcon:
    vendor = {0x10DE: "nvidia", 0x1002: "amd", 0x8086: "intel"}.get(adapter.vendor)
    return QIcon(str(ICON_DIR / f"{vendor}.svg")) if vendor else QIcon()


def _adapter_name(adapter: Adapter, adapters: tuple[Adapter, ...]) -> str:
    if sum(item.name == adapter.name for item in adapters) > 1:
        return f"{adapter.name} · GPU {adapters.index(adapter)}"
    return adapter.name


class GraphicsList(QScrollArea):
    """Show compact adapter entries, with vertical scrolling for large inventories."""

    def __init__(self) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.set_adapters((), tr("Checking graphics hardware…"))

    def set_adapters(self, adapters: tuple[Adapter, ...], message: str = "") -> None:
        if not adapters:
            label = QLabel(message)
            label.setWordWrap(True)
            self.setAccessibleDescription(message)
            self.setWidget(label)
            self._fit_height()
            return
        content = QWidget()
        flow = WrappingLayout(spacing=6, row_spacing=4)
        content.setLayout(flow)
        descriptions = []
        for adapter in adapters:
            chip = QFrame()
            chip.setObjectName("gpuChip")
            line = QHBoxLayout(chip)
            line.setContentsMargins(6, 3, 6, 3)
            line.setSpacing(4)
            vendor = _icon(adapter)
            if not vendor.isNull():
                icon = QLabel()
                icon.setPixmap(vendor.pixmap(14, 14))
                line.addWidget(icon)
            name = QLabel(_adapter_name(adapter, adapters))
            name.setTextFormat(Qt.TextFormat.PlainText)
            name.setMinimumWidth(1)
            line.addWidget(name, 1)
            details = [name.text()]
            if adapter.dedicated_bytes:
                memory = QLabel(tr("{value:.1f} GB", value=adapter.dedicated_bytes / 2**30))
                memory.setObjectName("muted")
                line.addWidget(memory)
                details.append(memory.text())
            if adapter.primary:
                primary = QLabel(tr("Primary"))
                primary.setObjectName("primaryGpu")
                line.addWidget(primary)
                details.append(primary.text())
            description = " · ".join(details)
            chip.setToolTip(description)
            descriptions.append(description)
            flow.addWidget(chip)
        self.setAccessibleDescription("\n".join(descriptions))
        self.setWidget(content)
        self._fit_height()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width():
            self._fit_height()

    def _fit_height(self) -> None:
        if content := self.widget():
            height = content.heightForWidth(self.contentsRect().width())
            limit = 5 * (self.fontMetrics().height() + 12)
            self.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
                if height > limit
                else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self.setFixedHeight(max(0, min(height, limit)))
