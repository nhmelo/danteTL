"""Game-scoped translation cache controls and removal of unused runtime files."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dantetl.application.services import CacheCommands
from dantetl.i18n import mark, tr
from dantetl.runtime.setup_values import RuntimeFilesView
from dantetl.translation.cache_values import CacheView
from dantetl.ui.localization import localized, on_language_change
from dantetl.ui.model_summary import file_size


class CacheSettings(QWidget):
    def __init__(self, cache: CacheCommands) -> None:
        super().__init__()
        self._cache = cache
        self._shown: CacheView | None = None
        self._shown_runtime: RuntimeFilesView | None = None
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        self.status = QLabel()
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        header.addWidget(self.status, 1)
        self.refresh = localized(QPushButton(), text=mark("Refresh"))
        header.addWidget(self.refresh)
        layout.addLayout(header)
        self.games = QListWidget()
        localized(self.games, accessibleName=mark("Game name"))
        layout.addWidget(self.games, 1)
        actions = QHBoxLayout()
        self.clear_game = localized(QPushButton(), text=mark("Clear selected game…"))
        self.clear_all = localized(QPushButton(), text=mark("Clear entire cache…"))
        actions.addWidget(self.clear_game)
        actions.addStretch()
        actions.addWidget(self.clear_all)
        layout.addLayout(actions)
        self.runtime_section = QWidget()
        runtime = QVBoxLayout(self.runtime_section)
        runtime.setContentsMargins(0, 8, 0, 0)
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        runtime.addWidget(divider)
        runtime_row = QHBoxLayout()
        self.runtime_status = QLabel()
        self.runtime_status.setTextFormat(Qt.TextFormat.PlainText)
        self.runtime_status.setWordWrap(True)
        runtime_row.addWidget(self.runtime_status, 1)
        self.clear_runtime = localized(QPushButton(), text=mark("Remove unused runtime files…"))
        runtime_row.addWidget(self.clear_runtime)
        runtime.addLayout(runtime_row)
        layout.addWidget(self.runtime_section)
        self.clear_runtime.clicked.connect(self._confirm_runtime)
        self.games.currentRowChanged.connect(self._enable_actions)
        self.refresh.clicked.connect(self._reload)
        self.clear_game.clicked.connect(lambda: self._confirm(False))
        self.clear_all.clicked.connect(lambda: self._confirm(True))
        on_language_change(self, self._retranslate)
        self.refresh_view()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._reload()

    def _reload(self) -> None:
        self._cache.refresh()
        self.refresh_view()

    def _retranslate(self) -> None:
        self._shown = self._shown_runtime = None
        self.refresh_view()

    def _enable_actions(self) -> None:
        inventory = self._shown.inventory if self._shown and not self._shown.busy else None
        self.clear_game.setEnabled(inventory is not None and self.games.currentItem() is not None)
        self.clear_all.setEnabled(bool(inventory and inventory.total))

    def refresh_view(self) -> None:
        self._refresh_runtime()
        view = self._cache.snapshot()
        if self._shown is view:
            return
        self._shown = view
        inventory = view.inventory
        current = self.games.currentItem()
        selected = current.data(Qt.ItemDataRole.UserRole) if current else None
        self.games.clear()
        if inventory is not None:
            for name, count in inventory.games:
                item = QListWidgetItem(f"{name or tr('Unassigned')}  ·  {count}")
                item.setData(Qt.ItemDataRole.UserRole, name)
                self.games.addItem(item)
                if name == selected:
                    self.games.setCurrentItem(item)
        self.status.setText(
            view.error
            or (
                tr("Working…")
                if view.busy
                else tr("Cached translations: {count}", count=inventory.total if inventory else 0)
            )
        )
        self.refresh.setEnabled(not view.busy)
        self._enable_actions()

    def _confirm(self, entire: bool) -> None:
        current = self.games.currentItem()
        if not entire and current is None:
            return
        name = (
            current.data(Qt.ItemDataRole.UserRole) if current is not None and not entire else None
        )
        prompt = QMessageBox(self)
        prompt.setWindowTitle(tr("Cache"))
        prompt.setTextFormat(Qt.TextFormat.PlainText)
        prompt.setText(
            tr("Permanently clear all cached translations?")
            if entire
            else tr(
                "Clear cached translations for {game}? Entries shared with other games are kept.",
                game=name or tr("Unassigned"),
            )
        )
        prompt.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        )
        prompt.setDefaultButton(QMessageBox.StandardButton.Cancel)
        answer = prompt.exec()
        prompt.deleteLater()
        if answer == QMessageBox.StandardButton.Yes:
            self._cache.clear(name)
            self.refresh_view()

    def _refresh_runtime(self) -> None:
        view = self._cache.runtime_files()
        if self._shown_runtime is view:
            return
        self._shown_runtime = view
        files = view.files
        self.runtime_section.setVisible(bool(files or view.busy or view.error))
        if view.busy:
            text = tr("Working…")
        elif view.error or files is None:
            text = view.error
        elif files.removed is None:
            text = tr("Unused runtime files: {size}", size=file_size(files.unused))
        elif files.unused:
            text = tr(
                "Removed {size}. Files in use by a running DanteTL window were kept.",
                size=file_size(files.removed),
            )
        else:
            text = tr("Removed {size}.", size=file_size(files.removed))
        self.runtime_status.setText(text)
        self.clear_runtime.setEnabled(not view.busy and bool(files and files.unused))

    def _confirm_runtime(self) -> None:
        prompt = QMessageBox(self)
        prompt.setWindowTitle(tr("Cache"))
        prompt.setTextFormat(Qt.TextFormat.PlainText)
        prompt.setText(
            tr(
                "Remove older runtimes and runtime downloads that DanteTL no longer needs? "
                "The selected runtime, the newest runtime for each acceleration type and all "
                "models are kept."
            )
        )
        prompt.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        )
        prompt.setDefaultButton(QMessageBox.StandardButton.Cancel)
        answer = prompt.exec()
        prompt.deleteLater()
        if answer == QMessageBox.StandardButton.Yes:
            self._cache.clear_runtime_files()
            self.refresh_view()
