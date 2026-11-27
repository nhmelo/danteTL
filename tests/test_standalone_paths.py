import hashlib
import json
import sys
import threading
from pathlib import Path

import pytest

from dantetl import paths
from dantetl.application.model_discovery import inspect_models
from dantetl.languages.selection import source_language
from dantetl.ocr import models
from dantetl.runtime import application as runtime_application
from dantetl.runtime import model_store
from dantetl.runtime import setup as runtime_setup
from dantetl.runtime.assets import ModelAsset
from dantetl.runtime.package import bundled_application
from dantetl.runtime.requirements import expected_wheels
from dantetl.translation.backends import gemma


def app_wheel(resources, content=b"app version one"):
    directory = resources / "application"
    directory.mkdir(exist_ok=True)
    wheel = directory / "dantetl-0.1.0-py3-none-any.whl"
    wheel.write_bytes(content)
    wheel.with_suffix(".whl.sha256").write_text(
        f"{hashlib.sha256(content).hexdigest()}  {wheel.name}\n", encoding="ascii"
    )


def install_app_metadata(resources, site):
    application = bundled_application(resources)
    metadata = site / f"dantetl-{application.version}.dist-info"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "direct_url.json").write_text(
        json.dumps({"archive_info": {"hashes": {"sha256": application.sha256}}}), "utf-8"
    )
    (site / "dantetl").mkdir(exist_ok=True)
    (site / "dantetl/__main__.py").touch()


@pytest.fixture
def package(tmp_path, monkeypatch):
    resources = tmp_path / "Read only app/_internal/app"
    resources.mkdir(parents=True)
    for name in ("standalone.json", "uv.lock", "pyproject.toml"):
        (resources / name).write_text("fixture")
    app_wheel(resources)
    (resources / ".python-version").write_text("3.12.16")
    (resources / "runtime-locks").mkdir()
    (resources / "cpu-runtime").mkdir()
    (resources / "cpu-runtime/llama_cpp_python-0.3.37-py3-none-win_amd64.whl.sha256").write_text(
        "fixture"
    )
    for profile in runtime_setup.PROFILES:
        (resources / "runtime-locks" / f"pylock.{profile.key}.toml").write_text(
            'lock-version = "1.0"\n[[packages]]\nname = "example"\n'
            f'wheels = [{{url = "https://example.test/example_{profile.key}-1.0-py3-none-any.whl", '
            'hashes = {sha256 = "' + "a" * 64 + '"}}]\n'
        )
    (resources.parent / "uv.exe").touch()
    monkeypatch.setattr(paths, "resource_root", lambda: resources)
    monkeypatch.setattr(runtime_application, "resource_root", lambda: paths.resource_root())
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "User AppData"))
    monkeypatch.delenv("DANTETL_DATA_DIR", raising=False)
    for name in ("DANTETL_MODEL_DIR", "DANTETL_GEMMA_DIR"):
        monkeypatch.delenv(name, raising=False)
    return resources


def test_standalone_models_and_state_share_appdata_without_writing_beside_executable(package):
    before = set(package.rglob("*"))
    state = paths.state_root()
    assert state.name == "DanteTL" and state.parent.name == "User AppData"
    assert models.model_directory() == state / "models/meiki"
    assert gemma.model_directory() == state / "models/gemma-4-e2b"
    assert set(package.rglob("*")) == before
    assert not state.exists()  # Path resolution alone does not install/download anything.


def test_source_checkout_preserves_existing_model_paths(package, monkeypatch):
    (package / "standalone.json").unlink()
    monkeypatch.setenv("DANTETL_DATA_DIR", "ignored-in-a-source-checkout")
    assert paths.state_root() == package
    assert models.model_directory() == package / "models/meiki"
    assert paths.module_command(Path("python.exe"), "dantetl") == [
        "python.exe",
        "-m",
        "dantetl",
    ]


def test_bundled_ocr_is_available_offline_without_installing_or_copying(package, monkeypatch):
    asset = ModelAsset("fixture", "pinned", "ocr.onnx", hashlib.sha256(b"good").hexdigest(), 4)
    monkeypatch.setattr(models, "ASSETS", (asset,))
    bundled = package / "models/meiki"
    bundled.mkdir(parents=True)
    (bundled / asset.filename).write_bytes(b"good")
    monkeypatch.setattr(
        model_store, "download_file", lambda *a, **kw: pytest.fail("No network allowed")
    )
    catalog = next(b for b in model_store.model_catalog() if b.key == "ocr")
    assert catalog.directory == paths.state_root() / "models/meiki"
    assert catalog.included and catalog.present() and catalog.local_directory == bundled
    assert models.model_directory() == bundled
    assert models.local_model_paths(models.model_directory()) == (bundled / asset.filename,)
    model_store.install_model(catalog, lambda p: None, threading.Event())
    assert not paths.state_root().exists()
    assert "Included" in inspect_models((catalog,), include_imports=False).files[0].status

    # Relocation follows the current package, not the old install folder.
    moved = package.with_name("relocated 日本語")
    package.rename(moved)
    monkeypatch.setattr(paths, "resource_root", lambda: moved)
    assert models.model_directory() == moved / "models/meiki"
    explicit = moved.parent / "explicit models"
    monkeypatch.setenv("DANTETL_MODEL_DIR", str(explicit))
    assert models.model_directory() == explicit
    assert not next(b for b in model_store.model_catalog() if b.key == "ocr").present()


@pytest.mark.parametrize(
    "code,folder,assets,resolve,override",
    [
        ("ja", "meiki", "ASSETS", models.model_directory, "DANTETL_MODEL_DIR"),
        (
            "en",
            "paddle-latin-v5",
            "PADDLE_ASSETS",
            models.paddle_directory,
            "DANTETL_PADDLE_MODEL_DIR",
        ),
    ],
)
def test_complete_appdata_models_take_precedence_but_partial_files_do_not_mask_bundle(
    package, monkeypatch, code, folder, assets, resolve, override
):
    asset = ModelAsset("fixture", "pinned", "ocr.onnx", hashlib.sha256(b"good").hexdigest(), 4)
    monkeypatch.setattr(models, assets, (asset,))
    bundled = package / "models" / folder
    bundled.mkdir(parents=True)
    (bundled / asset.filename).write_bytes(b"good")
    bundle = next(b for b in model_store.model_catalog(source_language(code)) if b.key == "ocr")
    assert bundle.included and bundle.present() and bundle.local_directory == resolve() == bundled
    primary = bundle.directory
    primary.mkdir(parents=True)
    (primary / asset.filename).write_bytes(b"bad")
    assert resolve() == bundled
    (primary / asset.filename).write_bytes(b"good")
    assert resolve() == primary
    explicit = package.parent / "explicit"
    monkeypatch.setenv(override, str(explicit))
    assert resolve() == explicit


def test_repair_cannot_write_to_standalone_resources(package):
    directory = package / "custom"
    asset = ModelAsset("fixture", "pinned", "ocr.onnx", "a" * 64, 4)
    bundle = model_store.ModelBundle("ocr", "OCR", directory, (asset,), "")
    with pytest.raises(RuntimeError, match="read-only"):
        model_store.install_model(bundle, lambda p: None, threading.Event())
    assert not directory.exists()


def test_explicit_state_override_requires_absolute_path(package, monkeypatch, tmp_path):
    monkeypatch.setenv("DANTETL_DATA_DIR", "relative")
    with pytest.raises(ValueError, match="absolute"):
        paths.state_root()
    monkeypatch.setenv("DANTETL_DATA_DIR", str(tmp_path / "test data"))
    assert paths.state_root() == tmp_path / "test data"


@pytest.fixture
def installer(package, monkeypatch):
    from dantetl.runtime.hardware_values import Adapter, CudaDevice, VulkanDevice

    monkeypatch.setattr(
        "dantetl.runtime.hardware.detect_adapters",
        lambda *args: (
            Adapter(
                "Fixture",
                0x10DE,
                cuda=CudaDevice(13000, 8, 6, "a" * 32),
                vulkan=VulkanDevice("b" * 32, 1),
            ),
        ),
    )
    calls = []

    def command(argv, cwd, environment, *args, **kwargs):
        calls.append(argv)
        assert cwd == paths.resource_root()
        assert environment["DANTETL_DATA_DIR"] == str(paths.state_root())
        if "venv" in argv:
            target = Path(environment["UV_PROJECT_ENVIRONMENT"])
            (target / "Scripts").mkdir(parents=True)
            for name in ("python.exe", "pythonw.exe"):
                (target / "Scripts" / name).touch()
            (target / "pyvenv.cfg").write_text("version_info = 3.12.16\nimplementation = CPython")
        elif "--target" in argv:
            install_app_metadata(paths.resource_root(), Path(argv[argv.index("--target") + 1]))

    def prepare(export, *args, **kwargs):
        profile = export.parent.name.split("-", 1)[0]
        requirement = export.parent / "verified-wheels.txt"
        requirement.write_text(
            "".join(
                f"file:///C:/old%20cache/{name} --hash=sha256:{digest}\n"
                for name, digest in expected_wheels(paths.resource_root(), profile)
            ),
            "utf-8",
        )
        return requirement

    monkeypatch.setattr(runtime_setup, "run_setup_command", command)
    monkeypatch.setattr(runtime_application, "run_setup_command", command)
    monkeypatch.setattr(runtime_setup, "prepare_wheels", prepare)
    return calls, command


@pytest.mark.parametrize("profile", runtime_setup.PROFILES, ids=lambda p: p.key)
def test_app_update_and_relocation_reuse_legacy_dependencies(
    package, installer, monkeypatch, profile
):
    state = paths.state_root()
    calls, _ = installer
    runtime_setup.install_runtime(state, profile, lambda _: None, threading.Event())
    ready = runtime_setup.ready_runtime(state, profile.key)
    assert ready
    runtime_setup.select_runtime(state, ready)
    assert all("--editable" not in call and "export" not in call for call in calls)
    # Old releases put the app hash into this receipt; migration must leave it unchanged.
    receipt = {"environment": ready.name, "fingerprint": "f" * 64}
    (ready / "ready.json").write_text(json.dumps(receipt))
    (state / "data/runtimes" / f"{profile.key}.json").write_text(json.dumps(receipt))
    install_app_metadata(package, ready / "Lib/site-packages")
    previous_native = {
        str(p.relative_to(ready)): p.read_bytes() for p in ready.rglob("*") if p.is_file()
    }
    previous_app = runtime_application.prepare_application(
        state, ready / "Scripts/python.exe", threading.Event()
    )["PYTHONPATH"]
    selection = (state / "data/runtime.json").read_bytes()
    moved = package.with_name("Updated 日本語")
    package.rename(moved)
    monkeypatch.setattr(paths, "resource_root", lambda: moved)
    app_wheel(moved, b"app version two")
    (moved / "pyproject.toml").write_text("new version and description")
    (moved / "uv.lock").write_text("changed application metadata")
    assert runtime_setup.selected_runtime(state) == ready
    assert runtime_setup.ready_runtime(state, profile.key) == ready
    assert not runtime_setup.runtime_selection(state).cuda_update
    before = len(calls)
    current_app = runtime_application.prepare_application(
        state, ready / "Scripts/python.exe", threading.Event()
    )["PYTHONPATH"]
    assert current_app != previous_app and Path(previous_app).exists()
    assert all("venv" not in call for call in calls[before:])
    assert (state / "data/runtime.json").read_bytes() == selection
    assert previous_native == {
        str(p.relative_to(ready)): p.read_bytes() for p in ready.rglob("*") if p.is_file()
    }
    assert bundled_application(moved).installed_at(Path(current_app))
    # Hash identity, not only version labels; switching back reuses the older app install.
    app_wheel(moved)
    assert (
        runtime_application.prepare_application(
            state, ready / "Scripts/python.exe", threading.Event()
        )["PYTHONPATH"]
        == previous_app
    )
    # A genuinely changed native wheel must still reject reuse.
    lock = moved / "runtime-locks" / f"pylock.{profile.key}.toml"
    lock.write_text(lock.read_text().replace("a" * 64, "b" * 64))
    assert runtime_setup.selected_runtime(state) is None
    assert runtime_setup.runtime_selection(state).cuda_update == (profile.key == "nvidia")
    from dantetl import bootstrap, desktop

    launches = []
    monkeypatch.setattr(bootstrap, "main", lambda *, cuda_update: launches.append(cuda_update))
    monkeypatch.setenv("DANTETL_STARTUP_NOTICE", "")
    monkeypatch.setattr(sys, "argv", ["DanteTL.exe"])
    desktop.main()
    assert launches == [profile.key == "nvidia"]
    assert (state / "data/runtime.json").read_bytes() == selection
    (ready / "verified-wheels.txt").write_text("damaged record")
    assert not runtime_setup.runtime_selection(state).cuda_update


@pytest.mark.parametrize("failure", ["install", "cancel", "release_changed"])
def test_failed_app_update_never_reinstalls_dependencies_or_overwrites_previous_code(
    package, installer, monkeypatch, failure
):
    from dantetl.runtime.setup_values import Cancelled

    calls, command = installer
    state = paths.state_root()
    profile = runtime_setup.PROFILES[0]
    runtime_setup.install_runtime(state, profile, lambda _: None, threading.Event())
    ready = runtime_setup.ready_runtime(state, profile.key)
    assert ready
    runtime_setup.select_runtime(state, ready)
    old_apps = list((state / ".applications").iterdir())
    app_wheel(package, b"update")
    cancel = threading.Event()

    def fail(argv, *args, **kwargs):
        if "--target" in argv and failure == "install":
            raise RuntimeError("Wheel install failed")
        command(argv, *args, **kwargs)
        if "--application-only" in argv:
            if failure == "cancel":
                cancel.set()
            if failure == "release_changed":
                app_wheel(package, b"changed during update")

    monkeypatch.setattr(runtime_application, "run_setup_command", fail)
    with pytest.raises(Cancelled if failure == "cancel" else RuntimeError):
        runtime_setup.install_runtime(state, profile, lambda _: None, cancel)
    assert list((state / ".applications").iterdir()) == old_apps
    assert list((state / ".runtimes").iterdir()) == [ready]
    assert runtime_setup.selected_runtime(state) == ready
    assert len([c for c in calls if "venv" in c]) == 1


def test_incomplete_or_ambiguous_release_never_supplies_an_application_wheel(package):
    wheel = bundled_application(package)
    (wheel.path.parent / "dantetl-0.2.0-py3-none-any.whl").touch()
    with pytest.raises(ValueError, match="ambiguous"):
        bundled_application(package)
    (wheel.path.parent / "dantetl-0.2.0-py3-none-any.whl").unlink()
    wheel.path.with_suffix(".whl.sha256").write_text("not a checksum")
    with pytest.raises(ValueError, match="checksum"):
        bundled_application(package)


def test_installed_local_wheel_provenance_survives_relocation_but_requires_exact_hash(package):
    wheel = bundled_application(package)
    target = paths.state_root() / "example-runtime"
    install_app_metadata(package, target)
    metadata = target / "dantetl-0.1.0.dist-info/direct_url.json"
    # Actual uv local-wheel provenance: the original URL retains its integrity fragment.
    metadata.write_text(
        json.dumps(
            {
                "url": "file:///C:/old-release/application/"
                + wheel.path.name
                + "#sha256="
                + wheel.sha256,
                "archive_info": {},
            }
        ),
        "utf-8",
    )
    assert wheel.installed_at(target)
    metadata.write_text(json.dumps({"url": wheel.path.as_uri(), "archive_info": {}}), "utf-8")
    assert not wheel.installed_at(target)
    metadata.write_text("[]", "utf-8")
    assert not wheel.installed_at(target)


def test_concurrent_launches_install_once_and_cancel_waiter_without_touching_owner(
    package, installer, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor

    from dantetl.runtime.setup_values import Cancelled

    calls, command = installer
    root = paths.state_root()
    python = root / ".runtimes" / ("cpu-" + "a" * 32) / "Scripts/python.exe"
    entered, release, cancel = threading.Event(), threading.Event(), threading.Event()

    def delayed(argv, *args, **kwargs):
        if "--target" in argv:
            entered.set()
            assert release.wait(5)
        command(argv, *args, **kwargs)

    monkeypatch.setattr(runtime_application, "run_setup_command", delayed)
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(
            runtime_application.prepare_application, root, python, threading.Event()
        )
        try:
            assert entered.wait(5)
            second = pool.submit(
                runtime_application.prepare_application, root, python, threading.Event()
            )
            cancelled = pool.submit(runtime_application.prepare_application, root, python, cancel)
            cancel.set()
            with pytest.raises(Cancelled):
                cancelled.result(timeout=5)
        finally:
            release.set()
        assert first.result(timeout=5)["PYTHONPATH"] == second.result(timeout=5)["PYTHONPATH"]
    assert len([c for c in calls if "--target" in c]) == 1


def test_launch_environment_uses_only_current_app_and_retains_native_environment(
    package, installer, monkeypatch
):
    from unittest.mock import Mock

    from dantetl.runtime import restart as runtime_restart
    from dantetl.runtime.setup_process import run_setup_command

    root = paths.state_root()
    profile = runtime_setup.PROFILES[0]
    runtime_setup.install_runtime(root, profile, lambda _: None, threading.Event())
    ready = runtime_setup.ready_runtime(root, profile.key)
    assert ready
    monkeypatch.setenv("PYTHONPATH", "unrelated-code")
    monkeypatch.setenv("PYTHONHOME", "unrelated-python")
    monkeypatch.setenv("PYTHONINSPECT", "1")
    monkeypatch.setenv("PYTHONHASHSEED", "invalid-seed")
    monkeypatch.setenv("PYTHONUTF8", "invalid-utf8")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "unrelated-venv")
    monkeypatch.setenv("VIRTUAL_ENV", "unrelated-venv")
    process = Mock()
    launch = Mock(return_value=process)
    monkeypatch.setattr(runtime_restart, "OwnedProcess", launch)
    restart = runtime_restart.RuntimeRestart(root, ready)
    argv, cwd, env = launch.call_args.args
    assert argv == [str(ready / "Scripts/pythonw.exe"), "-P", "-s", "-B", "-m", "dantetl"]
    assert cwd == root
    assert Path(env["PYTHONPATH"]).parent == root / ".applications"
    assert not any(
        k in env for k in ("PYTHONHOME", "PYTHONINSPECT", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV")
    )
    # App launch must also start a real interpreter with the clean baseline, preserving
    # its intentionally supplied wheel path and UTF-8 setting.
    try:
        run_setup_command(
            [
                sys.executable,
                *argv[1 : argv.index("-m")],
                "-c",
                "import os,sys; assert sys.flags.utf8_mode == 1; "
                "assert not sys.flags.inspect; assert os.environ['PYTHONPATH'] in sys.path",
            ],
            cwd,
            env,
            threading.Event(),
            lambda _: None,
            timeout=5,
        )
    finally:
        restart.cancel()


@pytest.mark.parametrize("profile", runtime_setup.PROFILES, ids=lambda p: p.key)
@pytest.mark.parametrize("reuse", [False, True], ids=["fresh", "reuse"])
def test_python_environment_cannot_break_probes_or_trigger_dependency_repair(
    package, installer, monkeypatch, profile, reuse
):
    from dantetl.runtime.setup_process import run_setup_command

    root = paths.state_root()
    calls, command = installer
    cancel = threading.Event()
    if reuse:
        runtime_setup.install_runtime(root, profile, lambda _: None, cancel)
    previous = runtime_setup.ready_runtime(root, profile.key)
    if previous:
        runtime_setup.select_runtime(root, previous)
    for key, value in {
        "PYTHONHASHSEED": "invalid-seed",
        "PYTHONUTF8": "invalid-utf8",
        "PYTHONINSPECT": "1",
        "PYTHONOPTIMIZE": "2",
        "PYTHONHOME": "unrelated-python",
        "PYTHONPATH": "unrelated-code",
    }.items():
        monkeypatch.setenv(key, value)
    probes = []

    def probe(argv, cwd, environment, *args, **kwargs):
        command(argv, cwd, environment, *args, **kwargs)
        if "dantetl.runtime.probe" in argv:
            # Exercise real interpreter startup with the exact environment and flags passed
            # to either probe, without needing native inference packages in fast tests.
            script = (
                "import sys; assert sys.flags.utf8_mode == 1; "
                "assert not sys.flags.inspect and not sys.flags.optimize; print('started')"
            )
            run_setup_command(
                [sys.executable, *argv[1 : argv.index("-m")], "-c", script],
                cwd,
                environment,
                cancel,
                lambda _: None,
                timeout=5,
            )
            probes.append("app" if "--application-only" in argv else "native")

    monkeypatch.setattr(runtime_setup, "run_setup_command", probe)
    monkeypatch.setattr(runtime_application, "run_setup_command", probe)
    runtime_setup.install_runtime(root, profile, lambda _: None, cancel)
    ready = runtime_setup.ready_runtime(root, profile.key)
    assert ready
    assert probes == (["native"] if reuse else ["app", "native"])
    assert len([call for call in calls if "venv" in call]) == 1
    assert list((root / ".runtimes").iterdir()) == [ready]
    if reuse:
        assert ready == previous == runtime_setup.selected_runtime(root)


@pytest.mark.parametrize("damage", ["python", "wheels", "profile", "receipt"])
def test_incompatible_or_incomplete_runtime_cannot_be_reused(package, installer, damage):
    root = paths.state_root()
    runtime_setup.install_runtime(
        root, runtime_setup.PROFILES[0], lambda _: None, threading.Event()
    )
    ready = runtime_setup.ready_runtime(root, "nvidia")
    assert ready
    runtime_setup.select_runtime(root, ready)
    if damage == "python":
        (ready / "pyvenv.cfg").write_text("version_info = 3.13.0")
    elif damage == "wheels":
        (ready / "verified-wheels.txt").write_text("")
    elif damage == "receipt":
        (ready / "ready.json").write_text("[]")
    else:
        (root / "data/runtime.json").write_text(
            json.dumps({"profile": "amd", "environment": ready.name})
        )
    assert runtime_setup.selected_runtime(root) is None
