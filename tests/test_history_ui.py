"""Reader presentation is independent of processing and safe for arbitrary OCR text."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shiboken6  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from dantetl.languages.target import TARGET_LANGUAGES  # noqa: E402
from dantetl.translation.dialogue.values import HistoryTurn  # noqa: E402
from dantetl.ui.controls import (  # noqa: E402
    CaptureButton,
    CopyTextButton,
    Switch,
    WrappingLayout,
)
from dantetl.ui.history_panel import HistoryPanel  # noqa: E402
from dantetl.ui.theme import STYLE  # noqa: E402


def test_overlay_controls_wrap_and_capture_state_keeps_icons_and_labels_consistent(app):
    first, second, third, fourth = (
        Switch("Show game overlay"),
        Switch("Translation overlay"),
        Switch("Show border"),
        Switch("Include in screen capture"),
    )
    row = QWidget()
    toggles = WrappingLayout(first, second, third, fourth)
    row.setLayout(toggles)
    button = CaptureButton()
    try:
        full_width = sum(c.sizeHint().width() for c in (first, second, third, fourth))
        full_width += toggles.spacing() * 3
        row.resize(full_width, 160)
        row.show()
        app.processEvents()
        assert first.y() == second.y() and first.x() < second.x()
        assert second.y() == third.y() and second.x() < third.x()
        assert third.y() == fourth.y() and third.x() < fourth.x()
        row.resize(full_width - 1, 160)
        app.processEvents()
        assert first.y() == second.y() == third.y() < fourth.y()
        assert fourth.x() == first.x()
        assert row.heightForWidth(row.width()) == fourth.geometry().bottom() + 1
        row.resize(max(c.sizeHint().width() for c in (first, second, third, fourth)), 160)
        app.processEvents()
        assert first.x() == second.x() and first.y() < second.y()
        assert second.x() == third.x() and second.y() < third.y()
        assert third.x() == fourth.x() and third.y() < fourth.y()
        row.resize(full_width, 160)
        app.processEvents()
        assert first.y() == fourth.y()
        assert button.text() == "Start capture" and not button.icon().isNull()
        record = button.icon().cacheKey()
        button.set_capturing(True)
        assert button.text() == "Stop capture" and button.property("tone") == "recording"
        assert button.icon().cacheKey() != record
        button.set_capturing(False)
        assert button.property("tone") == "idle" and button.icon().cacheKey() == record
    finally:
        shiboken6.delete(row)
        shiboken6.delete(button)


def test_reader_hides_without_losing_turns_escapes_text_and_preserves_scroll(app):
    parent = QWidget()
    turns = [
        HistoryTurn(i, f"第{i}話。" * 15, f"Turn {i}. " * 30, "旅人", f"Traveler {i}")
        for i in range(1, 21)
    ]
    panel = HistoryPanel(lambda: tuple(turns), turns.clear, parent)
    try:
        panel.open_reader()
        app.processEvents()
        assert panel.count.text().endswith("20/20")
        bar = panel.reader.verticalScrollBar()
        assert bar.maximum() > 0 and bar.value() == 0
        text = panel.reader.toPlainText()
        assert text.index("Turn 20.") < text.index("Turn 19.") < text.index("Turn 1.")
        assert "(Traveler 20) 旅人" in text
        bar.setValue(bar.maximum() // 2)
        visible = panel.reader.cursorForPosition(QPoint(12, 12)).block().text()
        turns[-1] = HistoryTurn(
            20, "<img src='https://invalid/track'>", "<b>literal</b>", "名前", "<i>Name</i>"
        )
        panel.refresh()
        assert panel.reader.cursorForPosition(QPoint(12, 12)).block().text() == visible
        assert "<b>literal</b>" in panel.reader.toPlainText()
        assert "(<i>Name</i>) 名前" in panel.reader.toPlainText()
        panel.hide()
        turns.append(HistoryTurn(21, "続き。", "A seguir.", target=TARGET_LANGUAGES["pt-PT"]))
        turns.pop(0)
        panel.refresh()
        assert "A seguir." not in panel.reader.toPlainText()
        panel.open_reader()
        assert "A seguir." in panel.reader.toPlainText()
        assert "Portuguese (Portugal)" in panel.reader.toPlainText()
        assert "English" in panel.reader.toPlainText()
        assert "Unknown speaker" in panel.reader.toPlainText()
        current = panel.reader.cursorForPosition(QPoint(12, 12)).block().text()
        assert current.removesuffix(" · English") == visible
        bar.setValue(0)
        turns.append(HistoryTurn(22, "新しい会話。", "Newest."))
        turns.pop(0)
        panel.refresh()
        assert bar.value() == 0
        assert panel.reader.toPlainText().index("Newest.") < panel.reader.toPlainText().index(
            "A seguir."
        )
        panel.clear_button.click()
        assert turns == [] and "Your conversation goes here." in panel.reader.toPlainText()
    finally:
        shiboken6.delete(parent)


def test_history_copy_icons_copy_exact_text_update_and_evict(app, monkeypatch):
    from unittest.mock import Mock

    clipboard = Mock()
    monkeypatch.setattr(QApplication, "clipboard", lambda: clipboard)
    parent = QWidget()
    turns = [HistoryTurn(1, "日本語。\n<原文>", "English.\n<source>", "旅人", "Traveler")]
    panel = HistoryPanel(lambda: tuple(turns), turns.clear, parent)
    try:
        panel.open_reader()
        app.processEvents()
        buttons = panel.reader.findChildren(CopyTextButton)
        assert len(buttons) == 2 and all(b.text() == "" for b in buttons)
        by_language = {b.accessibleName(): b for b in buttons}
        for language, expected in (("Japanese", turns[0].source), ("English", turns[0].english)):
            button = by_language[f"Copy {language}"]
            assert button.isVisible() and panel.reader.viewport().rect().contains(button.geometry())
            original_icon = button.icon().cacheKey()
            button.click()
            clipboard.setText.assert_called_with(expected)
            assert button.icon().cacheKey() != original_icon
            assert button.toolTip() == f"Copied {language}"
        turns[0] = HistoryTurn(1, "改訂。", "Revised.")
        panel.refresh()
        by_language["Copy English"].click()
        clipboard.setText.assert_called_with("Revised.")
        turns[:] = [HistoryTurn(2, "新しい。", "New.")]
        panel.refresh()
        app.processEvents()
        assert all(not b.isVisible() for b in buttons)
        visible = [b for b in panel.reader.findChildren(CopyTextButton) if b.isVisible()]
        assert len(visible) == 2
        panel.clear_button.click()
        assert not any(b.isVisible() for b in visible)
    finally:
        shiboken6.delete(parent)


def test_classifier_categories_cannot_be_selected_and_refresh_preserves_popup(app):
    parent = QWidget()
    parent.setStyleSheet(STYLE)
    panel = HistoryPanel(lambda: (), lambda: None, parent)
    choices = (
        ("local:ready", "Local fixture", True),
        ("local:missing", "Missing fixture", False),
        ("cloud:ready", "Cloud fixture", True),
    )
    changes = []
    panel.changed.connect(lambda: changes.append(panel.model_key))
    try:
        panel.set_models(choices, "layout")
        panel.open_reader()
        assert QTest.qWaitForWindowActive(panel, 1000)
        combo = panel.model
        for heading in ("Local instruction models", "Cloud models"):
            row = combo.findText(heading)
            flags = combo.model().index(row, 0).flags()
            assert not flags & (Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            index = combo.model().index(row, 0)
            assert index.data(Qt.ItemDataRole.BackgroundRole).color().name() == "#101922"
            assert index.data(Qt.ItemDataRole.ForegroundRole).color().name() == "#9baabc"
            assert index.data(Qt.ItemDataRole.FontRole).bold()
        QTest.keyClick(combo, Qt.Key.Key_Down)
        assert panel.model_key == "local:ready"
        QTest.keyClick(combo, Qt.Key.Key_Down)
        assert panel.model_key == "cloud:ready"
        assert changes == ["local:ready", "cloud:ready"]

        combo.showPopup()
        view = combo.view()
        assert QTest.qWaitForWindowExposed(view.window(), 1000)
        app.processEvents()
        assert view.isVisible()
        resets = []
        combo.model().modelReset.connect(lambda: resets.append(True))
        panel.set_models(choices, panel.model_key)  # Normal GUI refresh while the popup is open.
        assert not resets and view.isVisible()
        heading = combo.model().index(combo.findText("Cloud models"), 0)
        QTest.mouseClick(
            view.viewport(), Qt.MouseButton.LeftButton, pos=view.visualRect(heading).center()
        )
        assert panel.model_key == "cloud:ready" and len(changes) == 2
        combo.hidePopup()
        # Real availability changes still refresh the list and preserve the selected ID.
        panel.set_models(choices[:2], panel.model_key)
        assert panel.model_key == "cloud:ready"
        assert combo.currentText() == "Selected classifier unavailable"
        assert len(changes) == 2
    finally:
        shiboken6.delete(parent)
