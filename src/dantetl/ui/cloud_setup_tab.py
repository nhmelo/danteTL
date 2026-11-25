"""Cloud model choice in the welcome guide, using the Settings connection editor inline."""

from PySide6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from dantetl.i18n import mark, tr
from dantetl.ui.cloud_editor import CloudModelDialog
from dantetl.ui.cloud_settings import CloudSettings
from dantetl.ui.localization import localized, on_language_change


class CloudSetupTab(QWidget):
    def __init__(self, settings: CloudSettings) -> None:
        super().__init__()
        self.settings = settings
        self.editor: CloudModelDialog | None = None
        self._profiles = None
        self._saved_key: str | None = None
        self._continuing = False
        self.body = QVBoxLayout(self)
        self.choice = localized(QComboBox(), accessibleName=mark("Cloud models"))
        self.body.addWidget(self.choice)
        self.note = localized(
            QLabel(), text=mark("Model saved. Select it on the main screen to use it.")
        )
        self.note.setWordWrap(True)
        self.body.addWidget(self.note)
        self.choice.currentIndexChanged.connect(self._choose)
        on_language_change(self, self._language)

    def _language(self) -> None:
        if self.choice.count():
            self.choice.setItemText(self.choice.count() - 1, tr("New cloud model…"))

    def present(self) -> None:
        view = self.settings.service.snapshot()
        self.choice.setEnabled(not view.busy)
        if view.busy:
            return
        if self._profiles != view.profiles or self._saved_key is not None:
            key = self.choice.currentData() if self._saved_key is None else self._saved_key
            self._saved_key = None
            self.choice.blockSignals(True)
            self.choice.clear()
            for profile in view.profiles:
                self.choice.addItem(profile.name, profile.key)
            self.choice.addItem(tr("New cloud model…"), "")
            index = self.choice.findData(key)
            self.choice.setCurrentIndex(index if index >= 0 else 0)
            self.choice.setVisible(bool(view.profiles))
            self.choice.blockSignals(False)
            self._profiles = view.profiles
        key = self.choice.currentData()
        ready = bool(key and key in view.available)
        self.note.setVisible(ready)
        if ready or self.editor or self.settings.editor:
            return
        profile = next((p for p in view.profiles if p.key == key), None)
        self.editor = self.settings.open_editor(profile, parent=self, inline=True)
        if self.editor:
            self.body.addWidget(self.editor, 1)
            self.editor.save.clicked.connect(self._choose_another)
            self.editor.accepted.connect(self._saved)
            self.editor.finished.connect(self._finished)
            self.editor.show()

    def _choose_another(self) -> None:
        self._continuing = False

    def save_draft(self) -> None:
        if self.editor:
            self._continuing = True
            self.editor.save_draft()

    def _saved(self) -> None:
        self._saved_key = self.settings.service.snapshot().selected if self._continuing else ""

    def _finished(self) -> None:
        self.editor = None
        self._continuing = False

    def _choose(self) -> None:
        self.discard()
        self.present()

    @property
    def has_draft(self) -> bool:
        editor = self.editor
        return bool(
            editor and (editor.model.currentData() or editor.key.text() or editor.name.text())
        )

    def discard(self) -> None:
        if self.editor:
            self.editor.reject()
