"""About dialog and presentation of portable release update commands."""

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dantetl.application.updates import Updates, UpdateView
from dantetl.i18n import mark, tr
from dantetl.runtime.release_values import Release
from dantetl.ui.bug_reports import bug_report_link
from dantetl.ui.localization import localized, on_language_change


class AboutDialog(QDialog):
    restart = Signal()

    def __init__(self, installed: str, updates: Updates | None, parent: QWidget) -> None:
        super().__init__(parent)
        self.updates = updates
        self._view: UpdateView | None = None
        self._prompt: QMessageBox | None = None
        localized(self, windowTitle=mark("About"))
        self.setMinimumWidth(390)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)
        title = QLabel("DanteTL")
        title.setObjectName("heading")
        layout.addWidget(title)
        credits = QLabel(
            '<p>© 2026 Nuno Melo · <a href="https://github.com/nhmelo" '
            'style="color: #5eead4">GitHub @nhmelo</a></p>'
            '<p><a href="https://www.apache.org/licenses/LICENSE-2.0" '
            'style="color: #5eead4">Apache License 2.0</a></p>'
        )
        credits.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        credits.setOpenExternalLinks(True)
        layout.addWidget(credits)
        report = localized(QLabel(), text=bug_report_link)
        report.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        report.setOpenExternalLinks(True)
        layout.addWidget(report)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        actions = QHBoxLayout()
        self.check = localized(QPushButton(), text=mark("Check for updates"))
        self.check.clicked.connect(self._check)
        actions.addWidget(self.check)
        self.update_button = QPushButton()
        self.update_button.clicked.connect(self._update)
        actions.addWidget(self.update_button)
        self.notes = localized(QPushButton(), text=mark("Release notes"))
        self.notes.clicked.connect(self._notes)
        actions.addWidget(self.notes)
        actions.addStretch()
        actions.addWidget(QLabel(f"v{installed}"))
        layout.addLayout(actions)
        self.show_update(UpdateView())
        on_language_change(self, lambda: self.show_update(self._view or UpdateView(), refresh=True))

    def open_about(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def _check(self) -> None:
        if self.updates:
            self.updates.check()

    def _update(self) -> None:
        if self.updates and self._view and self._view.release:
            self.updates.decide(self._view.release, "yes")

    def _notes(self) -> None:
        if self._view and self._view.release:
            QDesktopServices.openUrl(QUrl(self._view.release.notes))

    def show_update(self, view: UpdateView, *, refresh: bool = False) -> None:
        if view == self._view and not refresh:
            return
        self._view = view
        release = view.release
        messages = {
            "checking": tr("Checking for updates…"),
            "current": tr("DanteTL is up to date."),
            "error": tr("Update failed: {detail}", detail=view.error),
            "preparing": tr("Preparing update…"),
            "downloading": tr("Downloading update…"),
            "verifying": tr("Verifying update…"),
            "unpacking": tr("Unpacking update…"),
            "preparing_restart": tr("Preparing to restart…"),
            "restart": tr("Restarting DanteTL…"),
        }
        status = messages.get(view.progress.stage if view.progress else view.phase, "")
        if release and view.phase == "available":
            status = tr("DanteTL {version} is available.", version=release.version)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setText(status)
        self.status.setVisible(bool(status))
        percent = view.progress.percent if view.progress else None
        self.progress.setRange(0, 100 if percent is not None else 0)
        if percent is not None:
            self.progress.setValue(percent)
        self.progress.setVisible(view.phase in ("checking", "installing", "restart"))
        self.progress.setAccessibleName(status)
        available = bool(release and self.updates and self.updates.can_install(release))
        self.update_button.setVisible(available)
        if release:
            self.update_button.setText(tr("Update to {version}", version=release.version))
        self.notes.setVisible(release is not None)
        idle = view.phase in ("idle", "current", "available", "error")
        self.check.setEnabled(idle and self.updates is not None)
        self.update_button.setEnabled(idle)
        if view.phase == "restart" and not refresh:
            self.restart.emit()

    def offer(self, release: Release) -> None:
        assert self.updates is not None
        updates = self.updates
        box = QMessageBox(self if self.isVisible() else self.parentWidget())
        self._prompt = box
        available = tr("DanteTL {version} is available.", version=release.version)
        can_install = updates.can_install(release)
        if can_install:
            box.setWindowTitle(available)
            box.setText(
                tr(
                    "Update from {current} to {version} and restart?",
                    current=updates.installed,
                    version=release.version,
                )
            )
            box.setTextFormat(Qt.TextFormat.RichText)
            box.setInformativeText(f'<a href="{release.notes}">{tr("Release notes")}</a>')
            box.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
            primary = box.addButton(tr("Yes"), QMessageBox.ButtonRole.YesRole)
        else:
            box.setWindowTitle("DanteTL")
            box.setText(available)
            primary = box.addButton(tr("Release notes"), QMessageBox.ButtonRole.ActionRole)
        later = box.addButton(tr("Ask me later"), QMessageBox.ButtonRole.RejectRole)
        no = box.addButton(tr("No"), QMessageBox.ButtonRole.NoRole)
        box.setDefaultButton(later)
        box.setEscapeButton(later)

        def answered() -> None:
            choice = box.clickedButton()
            if can_install and choice is primary:
                updates.decide(release, "yes")
                self.open_about()
            else:
                updates.decide(release, "no" if choice is no else "later")
                if choice is primary:
                    QDesktopServices.openUrl(QUrl(release.notes))
            self._prompt = None
            box.deleteLater()

        box.finished.connect(answered)
        box.open()
