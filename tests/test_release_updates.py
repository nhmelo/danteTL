import hashlib
import io
import json
import threading
import time
from dataclasses import replace
from queue import Queue
from unittest.mock import Mock
from zipfile import ZipFile

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from dantetl.application.updates import WEEK, Updates
from dantetl.runtime.package import PORTABLE_FILES, bundled_application
from dantetl.runtime.release_feed import parse_release
from dantetl.runtime.release_values import ARCHIVE, MANIFEST, SIGNATURE, Release, UpdateProgress
from dantetl.runtime.update_install import stage_release
from dantetl.runtime.update_swap import bundle_lease, swap


def settled(service):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        view = service.snapshot()
        if view.phase not in ("checking", "installing", "saving"):
            return view
        time.sleep(0.005)
    pytest.fail("Update command did not settle")


def test_startup_choices_across_launches_and_manual_override(tmp_path):
    release = Release("1.10.0", 100, 80, 64)
    now = 1000
    installs = []

    def launch():
        service = Updates(
            "1.9.0",
            tmp_path,
            fetch=lambda cancel: release,
            install=lambda candidate, cancel, progress: installs.append(candidate),
            clock=lambda: now,
        )
        service.check(automatic=True)
        assert settled(service).phase == "available"
        return service

    service = launch()
    assert service.take_offer() == release
    assert service.take_offer() is None
    service.decide(release, "later")
    assert settled(service).phase == "available"
    service.finish()
    now += WEEK - 1
    service = launch()
    assert service.take_offer() is None
    service.finish()
    now += 1
    service = launch()
    assert service.take_offer() == release
    service.decide(release, "no")
    settled(service)
    service.finish()
    now += 2 * WEEK
    service = launch()
    assert service.take_offer() is None
    service.check()
    assert settled(service).release == release
    assert service.take_offer() == release
    service.decide(release, "yes")
    assert settled(service).phase == "restart"
    assert installs == [release]
    service.finish()
    release = replace(release, version="1.11.0")
    service = launch()
    assert service.take_offer() == release
    service.finish()


def test_feed_ignores_prereleases_and_binds_names_to_one_repository():
    data = dict(tag_name="v1.0.0", draft=False, prerelease=False, assets=[])
    release = parse_release(data)
    assert release and not release.signed_assets
    assert release.notes == "https://github.com/nhmelo/dantejp/releases/tag/v1.0.0"
    assert parse_release(data | {"prerelease": True}) is None
    assert parse_release(data | {"tag_name": "../other"}) is None
    assert parse_release(data | {"draft": True}) is None


@pytest.mark.parametrize(
    "button,phase", [("Yes", "restart"), ("Ask me later", "available"), ("No", "available")]
)
def test_about_prompt_buttons(app, tmp_path, button, phase):
    from PySide6.QtWidgets import QMessageBox, QWidget

    from dantetl.ui.about import AboutDialog

    release = Release("1.2.0", 100, 90, 64)
    installs = []
    service = Updates(
        "1.1.0",
        tmp_path,
        fetch=lambda _: release,
        install=lambda candidate, *_: installs.append(candidate),
    )
    window = QWidget()
    dialog = AboutDialog("1.1.0", service, window)
    try:
        service.check(automatic=True)
        dialog.show_update(settled(service))
        assert service.take_offer() == release
        dialog.offer(release)
        question = window.findChild(QMessageBox)
        assert question and "1.1.0" in question.text() and "1.2.0" in question.text()
        assert release.notes in question.informativeText()
        next(item for item in question.buttons() if item.text().replace("&", "") == button).click()
        assert settled(service).phase == phase
        assert installs == ([release] if button == "Yes" else [])
    finally:
        window.close()
        window.deleteLater()
        service.finish()


def test_failed_check_can_retry_and_never_offers_an_older_version(tmp_path):
    responses = iter((OSError("offline"), Release("1.9.0", 100, 90, 64)))

    def fetch(_):
        response = next(responses)
        if isinstance(response, OSError):
            raise response
        return response

    service = Updates("1.10.0", tmp_path, fetch=fetch, install=None)
    try:
        service.check(automatic=True)
        assert settled(service).phase == "error"
        assert service.take_offer() is None
        service.check()
        assert settled(service).phase == "current"
    finally:
        service.finish()


def test_about_tracks_measured_and_busy_update_stages(app, tmp_path):
    from PySide6.QtWidgets import QWidget

    from dantetl.ui.about import AboutDialog

    release = Release("1.2.0", 100, 90, 64)
    steps = Queue()
    published = threading.Event()

    def install(candidate, cancel, progress):
        while (step := steps.get(timeout=5)) is not None:
            progress(step)
            published.set()
        raise OSError("fixture error")

    service = Updates("1.1.0", tmp_path, fetch=lambda _: release, install=install)
    window = QWidget()
    dialog = AboutDialog("1.1.0", service, window)
    dialog.open_about()
    try:
        dialog.check.click()
        dialog.show_update(settled(service))
        dialog.update_button.click()
        dialog.show_update(service.snapshot())
        assert dialog.status.text() == "Preparing update…"
        assert dialog.progress.isVisible() and dialog.progress.maximum() == 0
        for step, label in (
            (UpdateProgress("downloading", 43), "Downloading update…"),
            (UpdateProgress("downloading", 100), "Downloading update…"),
            (UpdateProgress("verifying", 0), "Verifying update…"),
            (UpdateProgress("unpacking", 36), "Unpacking update…"),
            (UpdateProgress("preparing_restart"), "Preparing to restart…"),
        ):
            steps.put(step)
            assert published.wait(5)
            published.clear()
            dialog.show_update(service.snapshot())
            assert dialog.status.text() == label
            assert dialog.progress.isVisible()
            if step.percent is None:
                assert dialog.progress.maximum() == 0 and dialog.progress.text() == ""
            else:
                assert dialog.progress.maximum() == 100 and dialog.progress.value() == step.percent
                assert dialog.progress.text() == f"{step.percent}%"
        steps.put(None)
        dialog.show_update(settled(service))
        assert dialog.progress.isHidden() and dialog.update_button.isEnabled()
        assert "fixture error" in dialog.status.text()
    finally:
        steps.put(None)
        service.finish()
        window.close()


@pytest.mark.parametrize(
    "installer,signature_size,primary",
    [(True, 64, "Yes"), (False, 64, "Release notes"), (True, 0, "Release notes")],
)
def test_window_prompts_at_startup_and_on_manual_recheck(
    app, tmp_path, monkeypatch, installer, signature_size, primary
):
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QMessageBox
    from ui_fixtures import create_window

    release = Release("1.2.0", 100, 90, signature_size)
    install, open_url = Mock(), Mock()
    monkeypatch.setattr(QDesktopServices, "openUrl", open_url)
    service = Updates(
        "1.1.0", tmp_path, fetch=lambda _: release, install=install if installer else None
    )
    windows = Mock()
    windows.list_targets.return_value = []
    window = create_window(windows, Mock(), updates=service)
    window.show()

    def prompt(parent):
        deadline = time.monotonic() + 3
        while not isinstance(box := app.activeModalWidget(), QMessageBox):
            assert time.monotonic() < deadline, "Expected a visible update prompt"
            QTest.qWait(10)
        assert box.isVisible() and box.parentWidget() is parent
        assert release.version in box.text()
        buttons = {b.text().replace("&", ""): b for b in box.buttons()}
        assert buttons.keys() == {primary, "Ask me later", "No"}
        return buttons

    try:
        service.check(automatic=True)
        prompt(window)["No"].click()
        settled(service)
        window.about_button.click()
        window.about.check.click()
        buttons = prompt(window.about)
        if primary == "Release notes":
            buttons[primary].click()
            assert open_url.call_args.args[0].toString() == release.notes
        else:
            buttons["Ask me later"].click()
        settled(service)
        install.assert_not_called()
    finally:
        window.close()
        service.finish()


def bundle_files(version):
    wheel = f"dantetl-{version}-py3-none-any.whl"
    data = version.encode()
    checksum = f"{hashlib.sha256(data).hexdigest()}  {wheel}"
    return {name: b"fixture" for name in PORTABLE_FILES} | {
        "_internal/app/standalone.json": b"{}",
        f"_internal/app/application/{wheel}": data,
        f"_internal/app/application/{wheel}.sha256": checksum.encode(),
    }


def installed(folder):
    return bundled_application(folder / "_internal/app").version


@pytest.mark.parametrize(
    "case", ["install", "rollback", "locked", "signature", "digest", "path", "interrupt"]
)
def test_portable_transaction(tmp_path, case):
    cancel = threading.Event()
    old, state = tmp_path / "Portable", tmp_path / "state"
    state.mkdir()
    preferences = state / "preferences.json"
    preferences.write_text("keep me")
    for name, data in bundle_files("1.0.0").items():
        path = old / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        for name, data in bundle_files("1.1.0").items():
            archive.writestr("DanteTL/" + name, data)
        if case == "path":
            archive.writestr("DanteTL/_internal/../../outside", b"fixture")
    payload = stream.getvalue()
    key = Ed25519PrivateKey.generate()
    manifest = json.dumps(
        dict(
            file=ARCHIVE,
            version="1.1.0",
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    ).encode()
    files = {ARCHIVE: payload, MANIFEST: manifest, SIGNATURE: key.sign(manifest)}
    release = Release("1.1.0", len(payload), len(manifest), 64)
    if case == "signature":
        files[SIGNATURE] = bytes(64)
    if case == "digest":
        files[ARCHIVE] = bytes(len(payload))

    def receive(url, destination, size, progress, cancel):
        if case == "interrupt" and destination.name == ARCHIVE:
            raise OSError("connection interrupted")
        destination.write_bytes(files[destination.name])
        assert destination.stat().st_size == size
        progress(size // 2, 0)  # A response without Content-Length still has a signed ZIP size.
        progress(size, 0)

    reports = []

    def stage():
        return stage_release(
            release, old, key.public_key(), cancel, reports.append, receive=receive
        )

    if case in ("signature", "digest", "path", "interrupt"):
        with pytest.raises((ValueError, OSError)):
            stage()
        assert not list(tmp_path.glob(".dantetl-update-*"))
    else:
        work = stage()
        assert list(dict.fromkeys(r.stage for r in reports)) == [
            "downloading",
            "verifying",
            "unpacking",
            "preparing_restart",
        ]
        for stage_name in ("downloading", "verifying", "unpacking"):
            values = [r.percent for r in reports if r.stage == stage_name and r.percent is not None]
            assert values == sorted(values) and values[0] == 0 and values[-1] == 100
            if stage_name != "verifying":
                assert any(0 < value < 100 for value in values)
        assert reports[-1] == UpdateProgress("preparing_restart")

        def launch(path, data, expected):
            assert installed(path) == expected == "1.1.0"
            if case == "rollback":
                raise RuntimeError("Startup failed")

        if case == "locked":
            with bundle_lease(old, state), pytest.raises(OSError):
                swap(old, work, state, release.version, launch=launch)
        elif case == "rollback":
            with pytest.raises(RuntimeError, match="Startup failed"):
                swap(old, work, state, release.version, launch=launch)
        else:
            swap(old, work, state, release.version, launch=launch)
            assert not work.exists()
    assert preferences.read_text() == "keep me"
    assert installed(old) == ("1.1.0" if case == "install" else "1.0.0")
    assert not (tmp_path / "outside").exists()
