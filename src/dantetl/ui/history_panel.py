"""Non-modal reader for a bounded transcript; no processing or prompt policy."""

from collections.abc import Callable
from html import escape

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QResizeEvent, QStandardItemModel, QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from dantetl.i18n import mark, tr
from dantetl.translation.dialogue.values import HistoryTurn
from dantetl.ui.controls import CopyTextButton, add_combo_section
from dantetl.ui.localization import localized, on_language_change


class HistoryReader(QTextBrowser):
    """Selectable transcript with real accessible copy controls beside its text blocks."""

    def __init__(self) -> None:
        super().__init__()
        self._copies: dict[str, CopyTextButton] = {}
        self._positions: dict[str, int] = {}
        self.verticalScrollBar().valueChanged.connect(self._place_copies)

    def set_copies(self, turns: tuple[HistoryTurn, ...]) -> None:
        wanted = {
            f"copy-{turn.number}-{code}": (language, text)
            for turn in turns
            for code, language, text in (
                (f"source-{turn.language.code}", tr(turn.language.name), turn.source),
                (f"target-{turn.target.code}", tr(turn.target.name), turn.english),
            )
        }
        for key in set(self._copies) - wanted.keys():
            button = self._copies.pop(key)
            button.hide()
            button.deleteLater()
        for key, (language, text) in wanted.items():
            label = tr("Copy {language}", language=language)
            if key not in self._copies:
                self._copies[key] = CopyTextButton(label, text, self.viewport())
            self._copies[key].set_label(label)
            self._copies[key].set_copy_text(text)
        self._positions = {}
        block = self.document().begin()
        while block.isValid():
            fragment = block.begin()
            if not fragment.atEnd():
                for anchor in fragment.fragment().charFormat().anchorNames():
                    if anchor in self._copies:
                        self._positions[anchor] = block.position()
            block = block.next()
        self._place_copies()

    def _place_copies(self) -> None:
        for key, button in self._copies.items():
            position = self._positions.get(key)
            if position is None:
                button.hide()
                continue
            cursor = QTextCursor(self.document())
            cursor.setPosition(position)
            rect = self.cursorRect(cursor)
            button.move(self.viewport().width() - button.width() - 4, rect.top() - 4)
            button.show()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._place_copies()


class HistoryPanel(QDialog):
    changed = Signal()

    def __init__(
        self,
        read: Callable[[], tuple[HistoryTurn, ...]],
        clear: Callable[[], None],
        parent: QWidget,
    ) -> None:
        super().__init__(parent)
        self._read, self._clear = read, clear
        self._shown: tuple[HistoryTurn, ...] | None = None
        self._model_choices: tuple[tuple[str, str, bool], ...] | None = None
        localized(self, windowTitle=mark("Dialogue history (Experimental) · DanteTL"))
        self.setModal(False)
        available = parent.screen().availableGeometry()
        self.resize(min(660, available.width() - 40), min(620, available.height() - 60))
        self.setMinimumSize(380, 300)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        header = QHBoxLayout()
        self.count = localized(QLabel(), text=mark("Dialogue history"))
        self.count.setObjectName("sectionTitle")
        header.addWidget(self.count, 1)
        self.clear_button = localized(QPushButton(), text=mark("Clear"))
        self.clear_button.clicked.connect(self._clear_history)
        header.addWidget(self.clear_button)
        layout.addLayout(header)
        experimental = localized(
            QLabel(), text=mark("Experimental · Dialogue and speaker detection may be inaccurate.")
        )
        experimental.setWordWrap(True)
        experimental.setObjectName("muted")
        layout.addWidget(experimental)
        self.model = QComboBox()
        localized(self.model, accessibleName=mark("Dialogue classifier model"))
        layout.addWidget(localized(QLabel(), text=mark("Classifier model")))
        layout.addWidget(self.model)
        self.model_notice = QLabel()
        self.model_notice.setWordWrap(True)
        self.model_notice.setObjectName("muted")
        layout.addWidget(self.model_notice)
        self.status = localized(
            QLabel(), text=mark("Disabled · No history collection or classification.")
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.model.currentIndexChanged.connect(self._controls_changed)
        note = localized(
            QLabel(), text=mark("Latest first · Last 20 turns · Never added to model context")
        )
        note.setObjectName("muted")
        note.setWordWrap(True)
        localized(
            note,
            toolTip=mark(
                "Confident speaker labels include their translation when it is available "
                "under your "
                "current model routing. History does not make additional translation requests."
            ),
        )
        layout.addWidget(note)
        self.reader = HistoryReader()
        localized(self.reader, accessibleName=mark("Recent dialogue and translations"))
        self.reader.setOpenExternalLinks(False)
        self.reader.setOpenLinks(False)
        self.reader.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        layout.addWidget(self.reader, 1)
        footer = QHBoxLayout()
        footer.addStretch(1)
        close = localized(QPushButton(), text=mark("Close"))
        close.clicked.connect(self.hide)
        footer.addWidget(close)
        layout.addLayout(footer)
        on_language_change(self, self._refresh_language)

    def _refresh_language(self) -> None:
        if not self.isVisible():
            self._shown = None
        if self._model_choices is not None:
            choices, self._model_choices = self._model_choices, None
            self.set_models(choices, self.model_key)
        self.refresh(force=True)

    def open_reader(self) -> None:
        self.show()
        self.refresh()
        self.raise_()
        self.activateWindow()

    def refresh(self, *, force: bool = False) -> None:
        if not self.isVisible():
            return
        turns = self._read()
        if turns == self._shown and not force:
            return
        bar = self.reader.verticalScrollBar()
        position = bar.value()
        follow = self._shown is None or position <= 4
        anchor = next(
            ((name, position - y) for name, y in reversed(self._turn_offsets()) if y <= position),
            None,
        )
        self._shown = turns
        self.count.setText(tr("Dialogue history · {count}/20", count=len(turns)))
        self.clear_button.setEnabled(bool(turns))
        parts = []
        mixed_targets = len({turn.target for turn in turns}) > 1
        for turn in reversed(turns):
            speaker = turn.speaker or tr("Unknown speaker")
            if turn.speaker and turn.speaker_english:
                speaker = f"({turn.speaker_english}) {turn.speaker}"
            parts.append(
                f'<p style="color:#5eead4; margin-top:18px"><b><a name="turn-{turn.number}">'
                + escape(speaker)
                + "</a></b>"
                + (f" · {escape(tr(turn.target.name))}" if mixed_targets else "")
                + "</p>"
                + '<p style="color:#9baabc; margin-right:36px">'
                + f'<a name="copy-{turn.number}-source-{turn.language.code}">'
                + escape(turn.source).replace("\n", "<br>")
                + "</a></p>"
                + '<p style="margin-right:36px">'
                + f'<a name="copy-{turn.number}-target-{turn.target.code}">'
                + escape(turn.english).replace("\n", "<br>")
                + '</a></p><hr style="background-color:#304259">'
            )
        self.reader.setHtml(
            "".join(parts)
            if parts
            else (
                tr(
                    "<h3>Your conversation goes here.</h3>"
                    "<p>Enable history in Settings → Experimental, choose a classifier, "
                    "then start capture with translation enabled.</p>"
                    "<p>The classifier checks dialogue and possible speakers together. "
                    "A speaker is optional. Uncertain text is skipped. "
                    "Layout-only mode uses conservative subtitle heuristics.</p>"
                    "<p>You can close this window; new turns will still be collected. "
                    "Starting a new capture clears this session’s history.</p>"
                )
            )
        )
        self.reader.set_copies(turns)
        # Preserve the turn being read, even when new turns prepend and the oldest
        # falls out of the bounded history (total document height may not change).
        offsets = dict(self._turn_offsets())
        if anchor and anchor[0] in offsets:
            position = round(offsets[anchor[0]] + anchor[1])
        bar.setValue(0 if follow else position)

    @property
    def model_key(self) -> str:
        return self.model.currentData() or ""

    def set_models(self, choices: tuple[tuple[str, str, bool], ...], selected: str) -> None:
        if choices == self._model_choices and selected == self.model_key:
            return  # Keep an open popup's rows and keyboard focus intact during GUI refresh.
        self._model_choices = choices
        self.model.blockSignals(True)
        self.model.clear()
        model = self.model.model()
        assert isinstance(model, QStandardItemModel)
        self.model.addItem(tr("Choose classifier…"), "")
        self.model.addItem(tr("Layout heuristics only"), "layout")
        group = ""
        for key, label, available in choices:
            section = (
                tr("Local instruction models") if key.startswith("local:") else tr("Cloud models")
            )
            if section != group:
                group = section
                add_combo_section(self.model, section)
            self.model.addItem(label if available else label + tr(" · Setup needed"), key)
            model.item(self.model.count() - 1).setEnabled(available)
        if selected and self.model.findData(selected) < 0:
            self.model.addItem(tr("Selected classifier unavailable"), selected)
            model.item(self.model.count() - 1).setEnabled(False)
        self.model.setCurrentIndex(max(0, self.model.findData(selected)))
        self.model.blockSignals(False)
        self._controls_changed(emit=False)

    def _controls_changed(self, *_: object, emit: bool = True) -> None:
        key = self.model_key
        self.model_notice.setText(
            tr(
                "Loads a separate local model. Uses extra RAM/VRAM and can compete "
                "with translation for hardware."
            )
            if key.startswith("local:")
            else tr(
                "Sends candidate text and nearby labels to this provider. Uses your plan/API "
                "allowance. No screenshots or transcript are sent."
            )
            if key.startswith("cloud:")
            else tr(
                "Classification runs independently of translation. Choose a configured "
                "cloud model or a prepared instruction model."
            )
        )
        if emit:
            self.changed.emit()

    def _turn_offsets(self) -> list[tuple[str, float]]:
        document = self.reader.document()
        layout = document.documentLayout()
        offsets = []
        block = document.begin()
        while block.isValid():
            fragment = block.begin()
            if not fragment.atEnd():
                names = fragment.fragment().charFormat().anchorNames()
                if names and names[0].startswith("turn-"):
                    offsets.append((names[0], layout.blockBoundingRect(block).top()))
            block = block.next()
        return offsets

    def _clear_history(self) -> None:
        self._clear()
        self.refresh()
