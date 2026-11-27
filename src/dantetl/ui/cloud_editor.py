"""Draft-only cloud model editor. The owner performs all persistence and network work."""

from uuid import uuid4

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from dantetl.i18n import mark, tr
from dantetl.translation.cloud.state import CloudDraft
from dantetl.translation.cloud.values import (
    MODELS,
    PROVIDERS,
    ApiConnection,
    CatalogModel,
    ChatGPTAccount,
    CloudProfile,
    CloudProvider,
    recommended,
)
from dantetl.ui.chatgpt_controls import ChatGPTControls
from dantetl.ui.controls import set_tone
from dantetl.ui.localization import localized, localized_item


class CloudModelDialog(QDialog):
    save_requested = Signal(object)
    test_requested = Signal(object)
    refresh_models_requested = Signal(object)
    forget_requested = Signal(str)
    sign_in_requested = Signal(str)
    sign_out_requested = Signal(str)
    refresh_requested = Signal(str)
    cancel_requested = Signal()

    def __init__(
        self, profile: CloudProfile | None, parent: QWidget, *, inline: bool = False
    ) -> None:
        super().__init__(parent)
        localized(
            self, windowTitle=mark("Edit cloud model") if profile else mark("Add cloud model")
        )
        if inline:
            self.setWindowFlags(Qt.WindowType.Widget)
        else:
            self.setWindowModality(Qt.WindowModality.WindowModal)
            self.resize(600, 640)
            self.setMinimumSize(480, 420)
        self.profile = profile
        self._id = profile.key if profile else "cloud-" + uuid4().hex
        self._model_label = ""
        self._api: tuple[ApiConnection, ...] = ()
        self._ready: set[str] = set()
        self._busy = False
        self._closing = False
        self._initialized = False
        self._last_provider = ""
        self._catalogs: dict[str, tuple[CatalogModel, ...]] = {}
        self._discovered: tuple[CatalogModel, ...] | None = None
        self._models: dict[str, CatalogModel] = {}
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.form = QWidget()
        body = QVBoxLayout(self.form)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.provider = QComboBox()
        self.provider.setMinimumContentsLength(20)
        self.provider.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        for key, label in PROVIDERS.items():
            localized_item(self.provider, label, key.value)
        if profile:
            self.provider.setCurrentIndex(self.provider.findData(profile.provider.value))
        self.provider.setEnabled(profile is None)
        form.addRow(localized(QLabel(), text=mark("Provider")), self.provider)
        self.chatgpt = ChatGPTControls()
        form.addRow(self.chatgpt)
        self.api_fields = QWidget()
        api_row = QHBoxLayout(self.api_fields)
        api_row.setContentsMargins(0, 0, 0, 0)
        self.connection = QComboBox()
        self.connection.setMinimumContentsLength(16)
        self.connection.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        localized(self.connection, accessibleName=mark("Saved API connection"))
        api_row.addWidget(self.connection, 1)
        self.forget = localized(QPushButton(), text=mark("Forget key"))
        localized(
            self.forget,
            toolTip=mark("Remove this saved key for every model using this connection."),
        )
        api_row.addWidget(self.forget)
        form.addRow(localized(QLabel(), text=mark("Connection")), self.api_fields)
        self._api_label = form.labelForField(self.api_fields)
        self.model = QComboBox()
        self.model.setMinimumContentsLength(16)
        self.model.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        model_row = QHBoxLayout()
        model_row.addWidget(self.model, 1)
        self.refresh_models = localized(QPushButton(), text=mark("Refresh models"))
        localized(
            self.refresh_models,
            toolTip=mark(
                "Ask this provider for its model catalog. No translation request is sent."
            ),
        )
        model_row.addWidget(self.refresh_models)
        model_label = localized(QLabel(), text=mark("Model"))
        if inline:
            form.insertRow(1, model_label, model_row)
        else:
            form.addRow(model_label, model_row)
        self.model_note = QLabel()
        self.model_note.setWordWrap(True)
        self.model_note.setObjectName("muted")
        form.addRow(self.model_note)
        self.name = QLineEdit(profile.name if profile else "")
        self.name.setMaxLength(80)
        form.addRow(localized(QLabel(), text=mark("Display name")), self.name)
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.setMaxLength(2560)
        localized(
            self.key,
            toolTip=mark(
                "Saved in Windows Credential Manager only when you save. Replacing a saved key "
                "updates every model using that connection. Leave blank to reuse it."
            ),
        )
        form.addRow(localized(QLabel(), text=mark("API key")), self.key)
        self._key_label = form.labelForField(self.key)
        body.addLayout(form)
        self.badge = QLabel()
        self.badge.setObjectName("modelBadge")
        self.badge.setWordWrap(True)
        body.addWidget(self.badge)
        self.advanced = QToolButton()
        localized(self.advanced, text=mark("Advanced"))
        self.advanced.setCheckable(True)
        self.advanced.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        body.addWidget(self.advanced, alignment=Qt.AlignmentFlag.AlignLeft)
        self.limits = QWidget()
        limits = QFormLayout(self.limits)
        limits.setContentsMargins(0, 0, 0, 0)
        self.rate = QSpinBox()
        self.rate.setRange(10, 600)
        self.rate.setValue(profile.requests_per_minute if profile else 120)
        localized(self.rate, suffix=mark(" / minute"))
        localized(
            self.rate,
            toolTip=mark("Maximum request starts, including retries. Cache hits use no requests."),
        )
        limits.addRow(localized(QLabel(), text=mark("Request rate")), self.rate)
        self.limit = QSpinBox()
        self.limit.setRange(1, 100_000)
        self.limit.setValue(profile.request_limit if profile else 1000)
        localized(
            self.limit,
            toolTip=mark(
                "Maximum reserved attempts for this model until DanteTL exits. Not a spending cap."
            ),
        )
        limits.addRow(localized(QLabel(), text=mark("Limit per app run")), self.limit)
        body.addWidget(self.limits)
        self.disclosure = QLabel()
        self.disclosure.setWordWrap(True)
        body.addWidget(self.disclosure)
        body.addStretch()
        scroll.setWidget(self.form)
        layout.addWidget(scroll, 1)
        self.status = QLabel()
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)
        footer = QHBoxLayout()
        self.test = localized(QPushButton(), text=mark("Test connection"))
        self.test.setAutoDefault(False)
        footer.addWidget(self.test)
        footer.addStretch()
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.save = self.buttons.button(QDialogButtonBox.StandardButton.Save)
        if inline:
            localized(self.save, text=mark("Save and choose another"))
        self.save.setDefault(not inline)
        self.save.setAutoDefault(not inline)
        self.cancel_button = self.buttons.button(QDialogButtonBox.StandardButton.Cancel)
        self.cancel_button.setVisible(not inline)
        footer.addWidget(self.buttons)
        layout.addLayout(footer)
        self.buttons.accepted.connect(self.save_draft)
        self.buttons.rejected.connect(self.reject)
        self.test.clicked.connect(lambda: self._submit(test=True))
        self.refresh_models.clicked.connect(lambda: self._submit(refresh=True))
        self.provider.currentIndexChanged.connect(self._provider_changed)
        self.connection.currentIndexChanged.connect(self._api_selected)
        self.model.currentTextChanged.connect(self._model_changed)
        self.chatgpt.account_changed.connect(self._account_changed)
        self.chatgpt.sign_in_requested.connect(
            lambda: self.sign_in_requested.emit(self.account_key)
        )
        self.chatgpt.sign_out_requested.connect(
            lambda: self.sign_out_requested.emit(self.account_key)
        )
        self.chatgpt.refresh_requested.connect(
            lambda: self.refresh_requested.emit(self.account_key)
        )
        self.forget.clicked.connect(
            lambda: self.forget_requested.emit(self.connection.currentData() or "")
        )
        self.advanced.toggled.connect(self._disclose)
        for text_input in (self.name, self.key):
            text_input.textChanged.connect(self.status.clear)
        self.key.textEdited.connect(self._key_edited)
        for limit in (self.rate, self.limit):
            limit.valueChanged.connect(self.status.clear)
        self.connection.currentIndexChanged.connect(self.status.clear)
        self._disclose(False)

    @property
    def account_key(self) -> str:
        account = self.chatgpt.selected
        return account.key if account else ""

    def refresh_connections(
        self,
        accounts: tuple[ChatGPTAccount, ...],
        api: tuple[ApiConnection, ...],
        ready: set[str],
        selected_account: str = "",
        catalogs: dict[str, tuple[CatalogModel, ...]] | None = None,
    ) -> None:
        """Update connection facts after setup, preserving the unsaved model draft."""
        self._api, self._ready = api, ready
        if catalogs is not None:
            self._catalogs = catalogs
        first = not self._initialized
        self.chatgpt.refresh(
            accounts,
            selected_account
            or (self.profile.account if first and self.profile else self.account_key),
        )
        self._provider_changed()
        if first and self.profile:
            index = self.model.findData(self.profile.model)
            if index < 0:
                self.model.addItem(self.profile.model + tr(" · unavailable"), self.profile.model)
                index = self.model.count() - 1
            self.model.setCurrentIndex(index)
            self.name.setText(self.profile.name)
        self._initialized = True

    def _disclose(self, expanded: bool) -> None:
        self.limits.setVisible(expanded)
        self.advanced.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)

    def _provider_changed(self) -> None:
        self.key.clear()
        chatgpt = self.provider.currentData() == CloudProvider.CHATGPT
        if self._last_provider != self.provider.currentData():
            self._models = {}
            self._discovered = None
            if self.isWindow():
                self.resize(self.width(), 600 if chatgpt else 480)
        self.refresh_models.setVisible(not chatgpt)
        for control in (self.api_fields, self._api_label, self.key, self._key_label):
            if control:
                control.setVisible(not chatgpt)
        self.chatgpt.setVisible(chatgpt)
        requested = (
            self.connection.currentData()
            if self._last_provider == self.provider.currentData()
            else None
        )
        self._last_provider = self.provider.currentData()
        if not self._initialized and self.profile:
            requested = self.profile.credential_key
        candidates = tuple(c for c in self._api if c.provider == self.provider.currentData())
        self.connection.blockSignals(True)
        self.connection.clear()
        for connection in candidates:
            self.connection.addItem(
                ("✓ " if connection.key in self._ready else "○ ") + connection.name, connection.key
            )
        self.connection.addItem(tr("New API key…"), "")
        index = self.connection.findData(requested)
        self.connection.setCurrentIndex(
            index
            if index >= 0
            else 0
            if len(candidates) == 1 and self.profile is None
            else self.connection.count() - 1
        )
        self.connection.blockSignals(False)
        self._account_changed()
        self.disclosure.setText(
            tr("Sends recognized text and dialogue context, never screenshots. ")
            + (
                tr("Uses ChatGPT plan permissions; no API-key fallback.")
                if chatgpt
                else tr("Uses separate API billing, not a chat subscription.")
            )
            + (
                tr(" Use a workspace-scoped Claude API key.")
                if self.provider.currentData() == CloudProvider.CLAUDE
                else ""
            )
        )

    def _account_changed(self) -> None:
        self._populate_models()
        self._api_selected()

    def _populate_models(self) -> None:
        previous = self.model.currentData()
        prior = self._models.get(previous)
        account = self.chatgpt.selected
        provider = self.provider.currentData()
        catalog = self._discovered
        if catalog is None and not self.key.text():
            catalog = self._catalogs.get(self.connection.currentData())
        entries = catalog if catalog is not None else recommended(provider)
        self._models = {m.model: m for m in entries}
        if prior and prior.model not in self._models:
            self._models[prior.model] = prior  # Refresh never silently replaces a selected model.
        if (
            self.profile
            and self.profile.provider == provider
            and self.profile.model not in self._models
        ):
            p = self.profile
            self._models[p.model] = CatalogModel(p.model, p.name, p.decoding, p.model_revision)
        models = (
            (account.models if account else ())
            if provider == CloudProvider.CHATGPT
            else tuple(
                (
                    m.model,
                    m.label
                    + (
                        tr(" · retired")
                        if m.lifecycle == "retired"
                        else tr(" · deprecated")
                        if m.lifecycle == "deprecated"
                        else ""
                    ),
                )
                for m in self._models.values()
            )
        )
        self.model.blockSignals(True)
        self.model.clear()
        self.model.addItem(tr("Choose a model…"), None)
        for key, label in models:
            self.model.addItem(label, key)
        self.model.setCurrentIndex(max(0, self.model.findData(previous)))
        self.model.blockSignals(False)
        self._model_changed(self.model.currentText())

    def _key_edited(self) -> None:
        self._discovered = None
        self._populate_models()

    def set_discovered(self, models: tuple[CatalogModel, ...]) -> None:
        self._discovered = models
        self._populate_models()

    def _api_selected(self) -> None:
        self.key.clear()
        self._discovered = None
        self._populate_models()
        saved = self.connection.currentData() in self._ready
        self.key.setPlaceholderText(
            tr("Saved key · leave blank to reuse") if saved else tr("Paste your API key")
        )
        self.forget.setEnabled(bool(self.connection.currentData()))
        account = self.chatgpt.selected
        chatgpt = self.provider.currentData() == CloudProvider.CHATGPT
        ready = bool(account and account.enabled) if chatgpt else saved
        self.badge.setText(
            (tr("✓ Signed in") if ready else tr("Sign in to choose models"))
            if chatgpt
            else (tr("✓ Key saved · not a connection test") if saved else tr("API key required"))
        )
        set_tone(self.badge, "success" if ready else "warning")

    def _model_changed(self, label: str) -> None:
        if not self.model.currentData():
            label = ""
        if self.name.text() in ("", self._model_label):
            self.name.setText(label[:80])
        self._model_label = label[:80]
        self.status.clear()
        selected = self._models.get(self.model.currentData())
        if self.provider.currentData() == CloudProvider.CHATGPT:
            self.model_note.setText(tr("Account models · list order is not a speed ranking."))
        else:
            source = (
                tr("Provider catalog")
                if self._discovered is not None
                else tr("Cached catalog")
                if self.connection.currentData() in self._catalogs and not self.key.text()
                else tr("Suggested models · refresh for the provider's current list")
            )
            self.model_note.setText(
                source
                + (
                    tr(" · Provider defaults; use Test connection to check compatibility")
                    if selected and selected.decoding != "preset"
                    else ""
                )
            )

    def save_draft(self) -> None:
        self._submit()

    def _submit(self, *, test: bool = False, refresh: bool = False) -> None:
        if self._busy or self._closing:
            return
        try:
            provider = self.provider.currentData()
            if not refresh and not self.model.currentData():
                raise ValueError(tr("Choose a translation model first."))
            selected = self._models.get(self.model.currentData())
            if selected and selected.lifecycle == "retired" and not refresh:
                raise ValueError(tr("This provider has retired the model. Choose another model."))
            profile = CloudProfile(
                self._id,
                self.name.text().strip() or self.model.currentText() or tr("Catalog refresh"),
                provider,
                self.model.currentData() or (MODELS[provider][0][0] if refresh else ""),
                self.rate.value(),
                self.limit.value(),
                account=self.account_key if provider == CloudProvider.CHATGPT else "",
                decoding=selected.decoding if selected else "preset",
                model_revision=selected.revision if selected else "",
            )
            draft = CloudDraft(
                profile,
                self.connection.currentData() or "",
                self.key.text().strip() if provider != CloudProvider.CHATGPT else "",
                self._discovered,
            )
        except ValueError as error:
            self.status.setText(str(error))
            return
        (
            self.refresh_models_requested
            if refresh
            else self.test_requested
            if test
            else self.save_requested
        ).emit(draft)

    def begin_operation(self, message: str) -> None:
        self._busy = True
        self.form.setEnabled(False)
        self.save.setEnabled(False)
        self.test.setEnabled(False)
        self.status.setText(message)
        self.progress.show()

    def finish_operation(self, message: str, *, saved: bool = False) -> None:
        self._busy = False
        self.progress.hide()
        self.form.setEnabled(True)
        self.save.setEnabled(True)
        self.test.setEnabled(True)
        self.status.setText(message)
        if self._closing:
            self.reject()
        elif saved:
            self.key.clear()
            self.accept()

    def reject(self) -> None:
        self.key.clear()
        if self._busy:
            self._closing = True
            self.cancel_button.setEnabled(False)
            self.status.setText(tr("Cancelling… waiting for the current operation to finish."))
            self.cancel_requested.emit()
        else:
            super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.reject()
        if self._busy:
            event.ignore()
        else:
            event.accept()
