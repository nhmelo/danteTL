"""Recreated views observe existing application operations without restarting them."""

import os
import threading
from dataclasses import replace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
import shiboken6  # noqa: E402
from PySide6.QtWidgets import QMessageBox, QWidget  # noqa: E402
from test_setup_ownership import configured  # noqa: E402

from dantetl.runtime.hardware_values import Adapter, CudaDevice, HardwareStatus  # noqa: E402
from dantetl.runtime.setup_values import DownloadProgress, ModelProgress, RuntimeStatus, SetupLane  # noqa: E402
from dantetl.translation.models.values import ImportedModel, ProbeResult  # noqa: E402
from dantetl.ui.cloud_settings import CloudSettings  # noqa: E402
from dantetl.ui.model_import_dialog import ModelImportDialog  # noqa: E402
from dantetl.ui.runtime_dialog import RuntimeSetupDialog  # noqa: E402
from dantetl.ui.setup_assistant import SetupAssistant  # noqa: E402
from dantetl.ui.setup_progress import SetupProgressView  # noqa: E402


def test_unmeasured_setup_step_animates_until_bytes_arrive_or_work_stops(app):
    view = SetupProgressView()
    try:
        step = ModelProgress("Preparing Python", stage=1, stages=4)
        view.update_progress(step)
        assert view.overall.value() == 1 and view.overall.maximum() == 4
        assert not view.download.isHidden() and view.download.maximum() == 0
        view.update_progress(replace(step, download=DownloadProgress("wheel", 2, 8)))
        assert view.download.maximum() == 1000 and view.download.value() == 250
        view.update_progress(replace(step, done=True, cancelled=True))
        assert view.download.isHidden() and view.overall.format() == "Stopped"
    finally:
        shiboken6.delete(view)


def test_runtime_picker_opens_on_active_profile_and_surfaces_verified_switch(app, tmp_path):
    replay = configured(tmp_path)
    target = tmp_path / "vulkan"
    status = RuntimeStatus(
        adapters=(Adapter("GPU", 0x1002, vulkan=True),),
        current=tmp_path / "cpu",
        recommendation="amd",
        active_profile="cpu",
    )
    detect = replay.dependencies.runtime_status
    select = replay.dependencies.select_runtime
    assert isinstance(detect, Mock) and isinstance(select, Mock)
    detect.return_value = status
    view = RuntimeSetupDialog(replay.setup)

    def finish():
        operation = replay.operations[-1]
        operation.work(lambda _: None, threading.Event())
        operation.finish()
        replay.setup.advance()
        view.refresh_view()

    try:
        finish()
        assert view.choice.currentData().key == "cpu" and "In use" in view.status.text()
        assert app.activeModalWidget() is None
        index = next(i for i in range(view.choice.count()) if view.choice.itemData(i).key == "amd")
        view.choice.setCurrentIndex(index)
        assert replay.setup.download("model")
        model_download = replay.operations[-1]
        view.install.click()
        detect.return_value = replace(status, targets=(("amd", target),))
        finish()
        assert not view.restart.isHidden() and "Vulkan" in view.status.text()
        assert replay.setup.installed_runtime == (view.choice.currentData().name, target)
        assert app.activeModalWidget() is None
        model_download.finish()
        replay.setup.advance()
        view.refresh_view()
        prompt = app.activeModalWidget()
        assert isinstance(prompt, QMessageBox) and "Vulkan" in prompt.text()
        prompt.button(QMessageBox.StandardButton.Cancel).click()
        select.assert_not_called()
        emitted = []
        view.restart_requested.connect(emitted.append)
        view.choice.activated.emit(index)
        prompt = app.activeModalWidget()
        assert isinstance(prompt, QMessageBox)
        prompt.defaultButton().click()
        assert emitted == [target]
        view.later.click()
        finish()
        select.assert_called_once_with(target)
        assert replay.setup.installed_runtime is None
    finally:
        shiboken6.delete(view)
        replay.close()


@pytest.mark.parametrize("vulkan,expected", [(True, "amd"), (False, "cpu")])
def test_pascal_guide_recommends_supported_runtime_and_explains_cuda(
    app, tmp_path, vulkan, expected
):
    replay = configured(tmp_path)
    parent = QWidget()
    reader = Mock()
    reader.snapshot.return_value = HardwareStatus(
        (Adapter("GTX 1050", 0x10DE, 4 * 2**30, CudaDevice(13000, 6, 1), vulkan),),
        llama_device="cpu",
    )
    try:
        view = SetupAssistant(
            replay.setup, reader, parent, cloud=CloudSettings(replay.cloud, parent)
        )
        view.next.click()
        view.next.click()
        assert view.runtime.currentData().key == expected and view.next.isEnabled()
        cuda = next(
            i for i in range(view.runtime.count()) if view.runtime.itemData(i).key == "nvidia"
        )
        view.runtime.setCurrentIndex(cuda)
        assert not view.next.isEnabled() and "6.1" in view.runtime_help.text()
        assert not replay.operations
    finally:
        shiboken6.delete(parent)
        replay.close()


def test_local_download_cancel_only_signals_the_model_lane(app, tmp_path, monkeypatch):
    from dantetl.ui.model_manager import ModelManager

    replay = configured(tmp_path)
    cloud_cancel = Mock()
    monkeypatch.setattr(replay.cloud, "cancel", cloud_cancel)
    view = ModelManager(replay.setup, cloud=replay.cloud)
    try:
        view.download.click()
        assert replay.setup.job(SetupLane.MODELS).busy
        view.cancel_button.click()
        assert replay.operations[-1].cancelled
        cloud_cancel.assert_not_called()
    finally:
        replay.close()
        shiboken6.delete(view)


def test_recreated_runtime_view_keeps_active_install_and_cancels_explicitly(app, tmp_path):
    replay = configured(tmp_path)
    try:
        assert replay.setup.install_runtime("cpu")
        for _ in range(2):
            view = RuntimeSetupDialog(replay.setup)
            try:
                view.refresh_view()
                assert view.choice.currentData().key == "cpu"
                assert not view.install.isEnabled() and not view.restart.isEnabled()
                view.hide()
                assert len(replay.operations) == 1
                assert not replay.operations[0].cancelled
            finally:
                shiboken6.delete(view)
        view = RuntimeSetupDialog(replay.setup)
        try:
            view.cancel_operation()
            assert replay.operations[0].cancelled and replay.setup.busy
            replay.operations[0].finish(cancelled=True)
            replay.setup.advance()
            view.refresh_view()
            assert "cancelled" in view.status.text().lower()
            assert not view.restart.isEnabled()
        finally:
            shiboken6.delete(view)
    finally:
        replay.close()


def test_recreated_guide_observes_same_review_and_completion(app, tmp_path):
    replay = configured(tmp_path)
    parent = QWidget()
    try:
        review = replay.setup.review("model", "cpu", HardwareStatus(llama_device="cpu"))
        assert replay.setup.start_review(review.token)
        detected = Mock()
        detected.snapshot.return_value = HardwareStatus((Adapter("GPU", 0x1002, vulkan=True),))
        for _ in range(2):
            view = SetupAssistant(
                replay.setup, detected, parent, cloud=CloudSettings(replay.cloud, parent)
            )
            try:
                assert view.pages.currentIndex() == 4
                assert view.model.currentData() == "model"
                assert view.runtime.currentData().key == "cpu"
                assert not view.next.isEnabled()
                view.hide()
                assert len(replay.operations) == 1 and not replay.operations[0].cancelled
            finally:
                shiboken6.delete(view)
        replay.operations[0].finish()
        replay.setup.advance()
        view = SetupAssistant(replay.setup, None, parent, cloud=CloudSettings(replay.cloud, parent))
        try:
            assert view.pages.currentIndex() == 4 and view.next.isEnabled()
            assert len(replay.operations) == 1
            assert not replay.setup.start_review(review.token)
            completed = []
            view.completed.connect(lambda key, target: completed.append((key, target)))
            view.next.click()
            assert completed == [("model", None)]
            reopened = SetupAssistant(
                replay.setup, None, parent, cloud=CloudSettings(replay.cloud, parent)
            )
            try:
                assert reopened.pages.currentIndex() == 0
                assert len(replay.operations) == 1
            finally:
                shiboken6.delete(reopened)
        finally:
            shiboken6.delete(view)
    finally:
        shiboken6.delete(parent)
        replay.close()


def test_guide_offers_only_models_it_can_review_and_keeps_imports_elsewhere(app, tmp_path):
    replay = configured(tmp_path)
    parent = QWidget()
    try:
        data = replay.inventory.snapshot().data
        imported = replace(data.files[-1], key="imported:user", name="User GGUF")
        replay.discover(replace(data, files=(*data.files, imported)))
        view = SetupAssistant(replay.setup, None, parent, cloud=CloudSettings(replay.cloud, parent))
        try:
            assert imported in replay.setup.files
            assert [view.model.itemData(i) for i in range(view.model.count())] == ["model"]
            view.hardware_status = HardwareStatus(llama_device="cpu")
            view._show(3)
            assert replay.setup.reviewed and replay.setup.reviewed.model == "model"
            with pytest.raises(ValueError, match="setup guide"):
                replay.setup.review(imported.key, "cpu", view.hardware_status)
            assert not replay.operations
        finally:
            shiboken6.delete(view)
    finally:
        shiboken6.delete(parent)
        replay.close()


def test_recreated_import_preserves_successful_probe_until_registration(app, tmp_path):
    replay = configured(tmp_path)
    try:
        model = ImportedModel(
            "imported:x",
            str(tmp_path / "file.gguf"),
            "Model",
            "hash",
            20,
            "llama",
            "embedded",
            "template",
        )
        assert isinstance(replay.dependencies.probe, Mock)
        replay.dependencies.probe.return_value.result = ProbeResult(
            model, "cpu", (("設定", "Settings", 12),), 20
        )
        assert replay.setup.test_import(model.path, model.profile)
        replay.operations[0].finish()
        replay.setup.advance()
        for _ in range(2):
            view = ModelImportDialog(replay.setup)
            try:
                assert view.file.text() == model.path and view.name.text() == model.name
                assert view.profile.currentData() == model.profile
                assert view.add_button.isEnabled() and "Settings" in view.results.toPlainText()
                assert len(replay.operations) == 1
            finally:
                shiboken6.delete(view)
        assert replay.setup.save_import("Renamed")
        assert not replay.setup.save_import("Duplicate")
        replay.operations[-1].finish()
        replay.setup.advance()
        assert replay.setup.probe_result is None
        assert not replay.setup.job(SetupLane.MODELS).busy
    finally:
        replay.close()
