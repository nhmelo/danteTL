"""Cloud model presentation. Application-owned services perform every operation."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from dantetl.i18n import mark, tr
from dantetl.translation.cloud.state import CloudDraft
from dantetl.translation.cloud.values import PROVIDERS, CloudProfile, CloudProvider
from dantetl.ui.cloud_editor import CloudModelDialog
from dantetl.ui.localization import localized, on_language_change

if TYPE_CHECKING:
    from dantetl.application.services import CloudCommands


class CloudSettings(QWidget):
    changed = Signal()

    def __init__(self, service: "CloudCommands", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self.editor: CloudModelDialog | None = None
        self._version = -1
        self._completion = service.snapshot().completion
        self._view = service.snapshot()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        header.addWidget(localized(QLabel(), text=mark("Cloud translation")))
        header.addStretch()
        self.add = localized(QPushButton(), text=mark("Add cloud model…"))
        self.add.clicked.connect(lambda: self.open_editor())
        header.addWidget(self.add)
        layout.addLayout(header)
        self.empty = localized(
            QLabel(),
            text=mark("No cloud models yet. Add a model using your ChatGPT account or an API key."),
        )
        self.empty.setWordWrap(True)
        layout.addWidget(self.empty)
        self.table = QTableWidget(0, 3)
        self.table.setObjectName("cloudModels")
        localized(self.table, accessibleName=mark("Cloud models"))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(150)
        self.table.setMaximumHeight(330)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.cellDoubleClicked.connect(lambda *_: self._edit_selected())
        layout.addWidget(self.table)
        actions = QHBoxLayout()
        self.edit = localized(QPushButton(), text=mark("Edit…"))
        self.remove = localized(QPushButton(), text=mark("Remove"))
        self.edit.clicked.connect(self._edit_selected)
        self.remove.clicked.connect(self._remove)
        actions.addWidget(self.edit)
        actions.addWidget(self.remove)
        actions.addStretch()
        layout.addLayout(actions)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.usage = QLabel()
        self.usage.setWordWrap(True)
        self.usage.setObjectName("muted")
        self.usage.setTextFormat(Qt.TextFormat.PlainText)
        localized(
            self.usage,
            toolTip=mark(
                "Counts belong to the selected saved model, including translation and "
                "classifier requests. Cache hits make no requests. Token totals include only "
                "usage reported by the provider; "
                "these are not billing totals. Counts reset when DanteTL exits."
            ),
        )
        layout.addWidget(self.usage)
        layout.addStretch()
        self._refresh_language()
        on_language_change(self, self._refresh_language)

    def _refresh_language(self) -> None:
        self.table.setHorizontalHeaderLabels((tr("Model"), tr("Connection"), tr("Status")))
        profile = self.selected_profile()
        self._render(profile.key if profile else "")

    @property
    def busy(self) -> bool:
        return self._view.busy

    @property
    def problem(self) -> str:
        return self._view.problem

    @property
    def profiles(self) -> tuple[CloudProfile, ...]:
        return self._view.profiles

    def cancel(self) -> None:
        self.service.cancel()

    def selected_profile(self) -> CloudProfile | None:
        item = self.table.item(self.table.currentRow(), 0)
        key = item.data(Qt.ItemDataRole.UserRole) if item else ""
        return next((p for p in self.profiles if p.key == key), None)

    def _selection_changed(self) -> None:
        selected = self.selected_profile()
        self.add.setEnabled(not self.busy and self.editor is None)
        self.edit.setEnabled(bool(selected) and not self.busy and self.editor is None)
        self.remove.setEnabled(self.edit.isEnabled())
        self._refresh_usage()

    def _refresh_usage(self) -> None:
        profile = self.selected_profile()
        if profile:
            usage = self.service.usage(profile.key)
            self.usage.setText(
                tr(
                    "{name} · This app run: {requests}/{request_limit} reserved requests · "
                    "{input_tokens:,} input / {output_tokens:,} output tokens reported",
                    name=profile.name,
                    requests=usage.requests,
                    request_limit=profile.request_limit,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                )
            )
        else:
            self.usage.clear()

    def _render(self, selected: str = "") -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.profiles))
        self.table.setVisible(bool(self.profiles))
        self.empty.setVisible(not self.profiles)
        self.table.setFixedHeight(
            min(
                330,
                self.table.horizontalHeader().sizeHint().height()
                + 56 * len(self.profiles)
                + 2 * self.table.frameWidth()
                + 2,
            )
        )
        for row, profile in enumerate(self.profiles):
            account = next((a for a in self._view.accounts if a.key == profile.account), None)
            connection = next(
                (
                    c
                    for c in self._view.api
                    if c.key == profile.credential_key and c.provider == profile.provider
                ),
                None,
            )
            label = (
                account.label if account else connection.name if connection else tr("Not connected")
            )
            ready = profile.key in self._view.available
            status = (
                (
                    tr("✓ Signed in")
                    if profile.provider == CloudProvider.CHATGPT
                    else tr("✓ Key saved")
                )
                if ready
                else tr("⚠ Connection needed")
            )
            for column, value in enumerate(
                (profile.name + "\n" + tr(PROVIDERS[profile.provider]), label, status)
            ):
                item = QTableWidgetItem(value)
                item.setToolTip(
                    tr(PROVIDERS[profile.provider]) + "\n" + value
                    if column != 2
                    else tr("Saved credentials are not a connection test. Edit to test explicitly.")
                )
                item.setData(Qt.ItemDataRole.UserRole, profile.key)
                if column == 2:
                    item.setForeground(QColor("#5eead4" if ready else "#fbbf24"))
                self.table.setItem(row, column, item)
            self.table.setRowHeight(row, 56)
            if profile.key == selected:
                self.table.selectRow(row)
        if self.profiles and not self.table.selectedItems():
            self.table.selectRow(0)
        self.table.blockSignals(False)
        self._selection_changed()

    def _edit_selected(self) -> None:
        profile = self.selected_profile()
        if profile:
            self.open_editor(profile)

    def open_editor(
        self,
        profile: CloudProfile | None = None,
        *,
        parent: QWidget | None = None,
        inline: bool = False,
    ) -> CloudModelDialog | None:
        if self.busy:
            return
        if self.editor:
            self.editor.raise_()
            return self.editor
        editor = self.editor = CloudModelDialog(profile, parent or self.window(), inline=inline)
        editor.refresh_connections(
            self._view.accounts,
            self._view.api,
            set(self._view.api_ready),
            catalogs=dict(self._view.catalogs),
        )
        editor.save_requested.connect(lambda value: self._command(self.service.save, value))
        editor.test_requested.connect(self._test)
        editor.refresh_models_requested.connect(
            lambda value: self._command(self.service.refresh_models, value)
        )
        editor.forget_requested.connect(self._forget_key)
        editor.sign_in_requested.connect(lambda value: self._command(self.service.sign_in, value))
        editor.sign_out_requested.connect(lambda value: self._command(self.service.sign_out, value))
        editor.refresh_requested.connect(
            lambda value: self._command(self.service.refresh_account, value)
        )
        editor.cancel_requested.connect(self.cancel)
        editor.finished.connect(self._editor_finished)
        self._selection_changed()
        if not inline:
            editor.open()
        return editor

    def _editor_finished(self, _: int) -> None:
        editor, self.editor = self.editor, None
        self.service.discard_draft()
        if editor:
            editor.deleteLater()
        self._selection_changed()

    def _command(self, operation: Callable, value: object) -> None:
        operation(value)
        # Closing the editor immediately after a click must still cancel/drain work.
        self.refresh_view()

    def refresh_view(self) -> None:
        view = self.service.snapshot()
        old = self._view
        self._view = view
        if view.version != self._version:
            self._version = view.version
            self.progress.setVisible(view.busy)
            self.status.setText(view.message)
            selected = self.selected_profile()
            self._render(view.selected or (selected.key if selected else ""))
            if self.editor and view.busy and not old.busy:
                self.editor.begin_operation(view.message)
            if view.completion != self._completion:
                self._completion = view.completion
                editor = self.editor
                if editor:
                    if view.mutating and not view.saved:
                        editor.refresh_connections(
                            view.accounts,
                            view.api,
                            set(view.api_ready),
                            view.new_account,
                            dict(view.catalogs),
                        )
                    if view.discovered is not None and not view.problem:
                        editor.set_discovered(view.discovered)
                    editor.finish_operation(view.message, saved=view.saved)
            if (old.profiles, old.available, old.revision) != (
                view.profiles,
                view.available,
                view.revision,
            ):
                self.changed.emit()
            self._selection_changed()
        self._refresh_usage()

    def handle_desktop_actions(self) -> None:
        url = self.service.take_browser()
        if url:
            QDesktopServices.openUrl(QUrl(url))
        account = self.service.take_welcome()
        if account:
            dialog = QMessageBox(
                QMessageBox.Icon.Information,
                tr("You're using your ChatGPT plan"),
                tr(
                    "Eligible translations use your ChatGPT plan. Manage allowance and credit "
                    "permissions in ChatGPT Settings → Usage. "
                    "DanteTL never falls back to an API key."
                ),
                parent=self.editor or self,
            )
            dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            dialog.finished.connect(lambda _: self.service.acknowledge_welcome(account))
            dialog.open()

    def _test(self, draft: CloudDraft) -> None:
        profile = draft.profile
        if (
            QMessageBox.question(
                self.editor or self,
                tr("Test cloud translation?"),
                tr(
                    "Send the sample label {test_sample} to this provider for translation "
                    "into your selected output language? ",
                    test_sample=self.service.test_sample,
                )
                + (
                    tr("This uses your ChatGPT plan permissions; review Manage usage first.")
                    if profile.provider == CloudProvider.CHATGPT
                    else tr("This makes one small API request and may incur a charge.")
                )
                + tr(" Model edits and new keys are not saved by this test."),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return

        self._command(self.service.test, draft)

    def _remove(self) -> None:
        profile = self.selected_profile()
        if not profile or self.busy or self.editor:
            return
        if (
            QMessageBox.question(
                self,
                tr("Remove cloud model?"),
                tr("Remove “{name}”? Its saved connection remains available.", name=profile.name),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return

        self._command(self.service.remove, profile)

    def _forget_key(self, key: str) -> None:
        if not key:
            return
        count = sum(
            p.credential_key == key for p in self.profiles if p.provider != CloudProvider.CHATGPT
        )
        if (
            QMessageBox.question(
                self.editor or self,
                tr("Forget API key?"),
                tr(
                    "Remove this saved API key? {count} model entries use it and will need "
                    "a connection again. This does not revoke the key at the provider.",
                    count=count,
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self._command(self.service.forget_key, key)
