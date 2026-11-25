"""Verify a signed portable release and stage it beside the running installation."""

import json
import re
import shutil
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path, PureWindowsPath
from zipfile import ZipFile

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from packaging.version import Version

from dantetl.runtime.download_transport import download_file
from dantetl.runtime.package import PORTABLE_FILES, bundled_application, check_portable_folder
from dantetl.runtime.release_values import ARCHIVE, MANIFEST, SIGNATURE, Release, UpdateProgress
from dantetl.runtime.verified_file import check_cancel, matches


def extract_bundle(
    archive: Path,
    directory: Path,
    cancel: threading.Event,
    progress: Callable[[UpdateProgress], None],
) -> None:
    progress(UpdateProgress("unpacking"))
    with ZipFile(archive) as package:
        entries = package.infolist()
        total = sum(entry.file_size for entry in entries)
        if len(entries) > 20_000 or total > 4 * 2**30:
            raise ValueError("Release exceeds extraction limits.")
        written = 0
        percent = 0
        if total:
            progress(UpdateProgress("unpacking", percent))
        for entry in entries:
            check_cancel(cancel)
            parts = entry.filename.replace("\\", "/").rstrip("/").split("/")
            if parts[0] != "DanteTL" or any(
                not part
                or part.endswith((" ", "."))
                or re.search(r'[<>:"|?*\x00-\x1f]', part)
                or PureWindowsPath(part).is_reserved()
                for part in parts
            ):
                raise ValueError("Invalid release archive path.")
            if len(parts) == 1:
                if entry.is_dir():
                    continue
                raise ValueError("Invalid release archive root.")
            if parts[1] != "_internal" and not (len(parts) == 2 and parts[1] in PORTABLE_FILES):
                raise ValueError("Unexpected release archive member.")
            target = directory.joinpath(*parts[1:])
            if entry.filename.endswith(("/", "\\")):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.open(entry) as source, target.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    check_cancel(cancel)
                    output.write(chunk)
                    written += len(chunk)
                    if (updated := 100 * written // total) != percent:
                        percent = updated
                        progress(UpdateProgress("unpacking", percent))


def stage_release(
    release: Release,
    install: Path,
    key: Ed25519PublicKey,
    cancel: threading.Event,
    progress: Callable[[UpdateProgress], None],
    *,
    receive=download_file,
) -> Path:
    if not release.signed_assets:
        raise ValueError("Release is missing its signed manifest.")
    check_portable_folder(install)
    current = bundled_application(install / "_internal/app")
    if Version(current.version) >= Version(release.version):
        raise ValueError("The installed release is already current or newer.")
    work = Path(tempfile.mkdtemp(prefix=".dantetl-update-", dir=install.parent))
    try:

        def quiet(received: int, total: int) -> None:
            pass

        for name, size in ((MANIFEST, release.manifest_size), (SIGNATURE, 64)):
            receive(release.asset(name), work / name, size, quiet, cancel)
        document = (work / MANIFEST).read_bytes()
        try:
            key.verify((work / SIGNATURE).read_bytes(), document)
        except InvalidSignature as error:
            raise ValueError("Release signature verification failed.") from error
        manifest = json.loads(document)
        digest = manifest["sha256"]
        if (
            manifest["version"] != release.version
            or manifest["file"] != ARCHIVE
            or manifest["size"] != release.archive_size
            or not re.fullmatch("[0-9a-f]{64}", digest)
        ):
            raise ValueError("Signed manifest does not match the offered release.")
        archive = work / ARCHIVE
        size = release.archive_size
        progress(UpdateProgress("downloading", 0))

        def downloaded(received: int, total: int) -> None:
            progress(UpdateProgress("downloading", 100 * received // size))

        receive(release.asset(ARCHIVE), archive, size, downloaded, cancel)
        progress(UpdateProgress("verifying", 0))
        if not matches(
            archive,
            digest,
            size,
            cancel,
            progress=lambda checked: progress(UpdateProgress("verifying", 100 * checked // size)),
        ):
            raise ValueError("Release checksum mismatch.")
        candidate = work / "new"
        extract_bundle(archive, candidate, cancel, progress)
        progress(UpdateProgress("preparing_restart"))
        check_portable_folder(candidate)
        if bundled_application(candidate / "_internal/app").version != release.version:
            raise ValueError("Staged application version mismatch.")
        archive.unlink()
        # Capture the exact installed revision for the helper to compare under its install lock.
        (work / "previous.sha256").write_text(current.sha256, "ascii")
        return work
    except BaseException:
        shutil.rmtree(work)
        raise


def publisher_key(project: Path) -> Ed25519PublicKey | None:
    try:
        key = load_pem_public_key((project / "update-public.pem").read_bytes())
    except (FileNotFoundError, ValueError):
        return None
    return key if isinstance(key, Ed25519PublicKey) else None
