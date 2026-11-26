"""Game window picker with executable favorites and asynchronous process details."""

import logging
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto

from PySide6.QtCore import (
    QObject,
    QPersistentModelIndex,
    QSignalBlocker,
    Qt,
    QThreadPool,
    Signal,
    Slot,
)
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import QComboBox

from dantetl.application.preference_values import valid_executable_id
from dantetl.domain import WindowTarget
from dantetl.i18n import tr
from dantetl.ui.controls import add_combo_section
from dantetl.ui.localization import on_language_change

logger = logging.getLogger(__name__)

_EXECUTABLE = Qt.ItemDataRole.UserRole + 1


class _ProcessResult(QObject):
    ready = Signal(object)


class _Selection(Enum):
    INITIAL = auto()
    REFRESH = auto()
    KEEP = auto()


@dataclass
class _WindowRow:
    target: WindowTarget
    executable: str | None = None
    icon: QIcon = field(default_factory=QIcon)


class GameWindowCombo(QComboBox):
    opening = Signal()
    favorite_state_changed = Signal()

    def __init__(
        self,
        read_details: Callable[[WindowTarget], tuple[str, QImage | None] | None],
        favorites: tuple[str, ...],
    ) -> None:
        super().__init__()
        self._read_details = read_details
        self._favorites = set(favorites)
        self._selection = _Selection.INITIAL
        self.currentIndexChanged.connect(self.keep_selection)
        self.activated.connect(self.keep_selection)
        self._next_row = 0
        self._needs_regroup = False
        self._loading: QPersistentModelIndex | None = None
        # The outstanding job can outlive the picker. Qt disconnects the deleted receiver.
        self._result = _ProcessResult()
        self._result.ready.connect(self._details_ready)
        on_language_change(self, self._regroup)

    def showPopup(self) -> None:
        if self.currentData() is not None:
            self.keep_selection()
        self.opening.emit()
        super().showPopup()

    @Slot()
    def keep_selection(self) -> None:
        self._selection = _Selection.KEEP

    @property
    def selected_executable(self) -> str:
        return self.currentData(_EXECUTABLE) or ""

    def set_favorites(self, favorites: tuple[str, ...]) -> None:
        self.keep_selection()
        self._favorites = set(favorites)
        self._regroup()

    def _rows(self) -> list[_WindowRow]:
        return [
            _WindowRow(target, self.itemData(i, _EXECUTABLE), self.itemIcon(i))
            for i in range(self.count())
            if isinstance(target := self.itemData(i), WindowTarget)
        ]

    def set_targets(self, targets: Sequence[WindowTarget], *, refresh: bool) -> None:
        # On popup open, reuse details for windows still listed with the same HWND/PID.
        existing = {(r.target.hwnd, r.target.process_id): r for r in self._rows()}
        if not refresh and {r.target for r in existing.values()} == set(targets):
            return
        rows = []
        for target in targets:
            old = existing.get((target.hwnd, target.process_id))
            row = _WindowRow(target)
            if old and not refresh:
                row.executable, row.icon = old.executable, old.icon
            rows.append(row)
        was_favorite = self.selected_executable in self._favorites
        if refresh and (existing or self._selection is not _Selection.INITIAL):
            self._selection = _Selection.REFRESH
        self._render_rows(rows, self.currentData())
        if was_favorite and self.currentData() is not None:
            self.keep_selection()
        self._load_next()

    def select_target(self, target: WindowTarget) -> None:
        self.keep_selection()
        rows = self._rows()
        for row in rows:
            if (row.target.hwnd, row.target.process_id) == (target.hwnd, target.process_id):
                row.target = target
                break
        else:
            rows.append(_WindowRow(target))
        self._render_rows(rows, target)
        self._load_next()

    def _regroup(self) -> None:
        self._render_rows(self._rows(), self.currentData())
        self._load_next()

    def _render_rows(
        self, rows: list[_WindowRow], selected: WindowTarget | None, *, plain: bool = False
    ) -> None:
        rows.sort(key=lambda r: (r.target.title.casefold(), r.target.hwnd))
        counts = Counter((r.target.title, r.target.process_id) for r in rows)
        labels: dict[WindowTarget, str] = {}
        for row in rows:
            target = row.target
            label = tr(
                "{title}  ·  PID {process_id}", title=target.title, process_id=target.process_id
            )
            if counts[target.title, target.process_id] > 1:
                label += tr("  ·  HWND {hwnd:#x}", hwnd=target.hwnd)
            labels[target] = label

        popup = self.view()
        highlighted = (
            popup.currentIndex().data(Qt.ItemDataRole.UserRole) if popup.isVisible() else None
        )
        with QSignalBlocker(self):
            self.clear()
            if not plain:
                try:
                    self._add_grouped_rows(rows, labels)
                except Exception:
                    logger.debug("Using the plain window list", exc_info=True)
                    self.clear()  # A heading or some items may have already been added.
                    plain = True
            if plain:
                for row in rows:
                    self.addItem(labels[row.target], row.target)
            if self._selection is _Selection.INITIAL and selected is None and rows:
                selected = rows[0].target
            index = self._find_window(selected)
            if plain:
                self.keep_selection()
            self.setCurrentIndex(index)
            highlighted_index = self._find_window(highlighted)
            if highlighted_index >= 0:
                popup.setCurrentIndex(self.model().index(highlighted_index, 0))
        self._next_row = self.count() if plain else 0
        self._needs_regroup = False
        self.favorite_state_changed.emit()

    def _add_grouped_rows(self, rows: list[_WindowRow], labels: dict[WindowTarget, str]) -> None:
        favorites = [r for r in rows if r.executable in self._favorites]
        others = [r for r in rows if r.executable not in self._favorites]
        for heading, group in ((tr("Favorites"), favorites), (tr("Other windows"), others)):
            if group and favorites:
                add_combo_section(self, heading)
            for row in group:
                self.addItem(row.icon, labels[row.target], row.target)
                self.setItemData(self.count() - 1, row.executable, _EXECUTABLE)

    def _find_window(self, target: WindowTarget | None) -> int:
        if isinstance(target, WindowTarget):
            for i in range(self.count()):
                row = self.itemData(i)
                if isinstance(row, WindowTarget) and (row.hwnd, row.process_id) == (
                    target.hwnd,
                    target.process_id,
                ):
                    return i
        return -1

    def _load_next(self) -> None:
        if self._loading is not None:
            return
        while self._next_row < self.count():
            row = self._next_row
            self._next_row += 1
            target = self.itemData(row)
            if isinstance(target, WindowTarget) and self.itemData(row, _EXECUTABLE) is None:
                self._loading = QPersistentModelIndex(self.model().index(row, 0))
                break
        else:
            prefer_favorite = (
                self._selection is _Selection.REFRESH
                and self.selected_executable not in self._favorites
            )
            if self.count():
                self.keep_selection()
            if self._needs_regroup:
                # Publish the newly discovered favorites together, preserving popup focus.
                self._render_rows(self._rows(), self.currentData())
                if prefer_favorite:
                    for row in range(self.count()):
                        if self.itemData(row, _EXECUTABLE) in self._favorites:
                            self.setCurrentIndex(row)
                            break
            return

        read_details, result = self._read_details, self._result

        def load() -> None:
            details = None
            try:
                details = read_details(target)
            except Exception:
                logger.debug("Window process details unavailable", exc_info=True)
            result.ready.emit(details)

        try:
            QThreadPool.globalInstance().start(load)
        except Exception:
            logger.debug("Window details worker unavailable", exc_info=True)
            self._loading = None
            self._render_rows(self._rows(), self.currentData(), plain=True)

    @Slot(object)
    def _details_ready(self, details: tuple[str, QImage | None] | None) -> None:
        index = self._loading
        self._loading = None
        if index is not None and index.isValid():
            row = index.row()
            try:
                executable, image = details or ("", None)
                executable = executable if valid_executable_id(executable) else ""
                self.setItemData(row, executable, _EXECUTABLE)
                if executable in self._favorites:
                    self._needs_regroup = True
                    if self._selection is _Selection.INITIAL:
                        # Lookups follow the same alphabetical order as the Favorites section.
                        self.keep_selection()
                        self.setCurrentIndex(row)
                if image is not None and not image.isNull():
                    self.setItemIcon(row, QIcon(QPixmap.fromImage(image)))
            except Exception:
                logger.debug("Could not apply optional window details", exc_info=True)
            if row == self.currentIndex():
                self.favorite_state_changed.emit()
        self._load_next()
