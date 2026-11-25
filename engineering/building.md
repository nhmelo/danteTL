# Standard Windows build

Install **uv 0.13.8** and **Visual Studio C++ Build Tools** (Desktop development with C++, including
the Windows SDK) on Windows x64. The GitHub Windows runner already supplies the compiler.
From a checkout or extracted source distribution:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build.ps1 -DownloadOcr
```

This explicit build action downloads dependencies, managed CPython 3.12.16 and the pinned
MeikiOCR and PaddleOCR assets. Omit `-DownloadOcr` if models are already present. `-Offline` requires all assets,
Python and dependencies to be cached, plus a previously built CPU wheel and checksum in
`build/cpu-runtime`. No translation weights are bundled/downloaded.

The script only orchestrates established tools:

1. `uv sync --locked` prepares `.venv-build` with CPU OCR, common LLM dependencies and build tools.
2. `pip wheel --require-hashes --no-build-isolation` builds the pinned upstream CPU source through
   its scikit-build-core/CMake backend. `delvewheel repair` bundles required external DLLs. Both
   tools and their dependencies are locked; their standard code owns wheel metadata and packaging.
3. `uv build --no-build-isolation` uses locked Hatchling to make the wheel and source archive.
   The build imports the wheel's CLI entry point with the working directory excluded, checking
   that nested packages are present without relying on the editable source installation.
4. `uv export --locked --no-emit-project` produces dependency-only PEP 751 locks for each profile.
5. `python -m PyInstaller packaging/DanteTL.spec` collects the app using upstream hooks/bootloader,
   including that same app wheel and SHA-256 sidecar for optional runtime installation.
6. PowerShell `Compress-Archive` makes the portable ZIP; `Get-FileHash` creates SHA-256 sidecars.

Outputs are `dist/DanteTL-windows-x64.zip`, `dist/dantetl-<version>-py3-none-any.whl`,
`dist/dantetl-<version>.tar.gz` and `.sha256` sidecars. Distribute the portable ZIP to players;
The Python wheel also ships inside the portable app for private runtime setup. Source distributions include the locked project,
build spec, verification fixtures and engineering docs, without local models, caches or settings.
For only Python artifacts, ordinary `uv build` works through the pinned PEP 517 backend.

The build environment is separate from `.venv`, `.venv-gpu`, `.venv-amd` and AppData runtimes.
Native dependency discovery restricts PATH to the build tools and Windows directories; unrelated
applications must not supply bundled DLLs. Runtime libraries retain their own pinned versions.
App-only updates automatically reuse the installed CPU/CUDA/Vulkan dependencies in AppData,
including compatible runtimes created by earlier releases. Validation compares exact dependency
wheel hashes and the pinned Python version, independently of application metadata or version.
Actual dependency changes still require explicit setup into a separate environment with CPU fallback.
When a previously selected NVIDIA runtime has outdated dependencies and the GPU and driver
qualify for the new runtime, startup offers **Update CUDA**
and **Continue on CPU**. Update opens Acceleration with NVIDIA selected and starts the installation.
Setup asks before restarting. Continuing uses CPU for
this launch and preserves the saved selection.
Models, preferences and selection are kept. Moving an unchanged release requires no installation.

For a fresh native runtime, uv creates the environment. The installer uses the pip bundled with its
pinned CPython to unpack verified wheels directly into that environment at background priority. The
release includes the standalone installer script so a fresh interpreter can run it before setup.
Optional runtimes use standard offline `uv pip install --target --no-deps --require-hashes` to
prepare only the bundled app wheel in an immutable, hash-specific AppData directory. This is shared
across native profiles and takes no network access. A process-local `PYTHONPATH` selects that app,
with `python -P -s -B -m dantetl` excluding the working directory and user site. No loaded files
are replaced, so concurrent old instances and rollback keep their app code. The launcher passes
the current bundle's read-only resources separately; no duplicate `app/src` tree, editable standalone
install, custom bootstrap or dependency on the frozen interpreter is needed.
Source-development runtime installs remain editable so checkout edits apply immediately.

## Verification

Run `uv run python tools/localize.py extract` to collect new or changed interface strings.
Complete unfinished entries in `translations/*.ts` with Qt Linguist, then run
`uv run python tools/localize.py compile` to update the bundled QM files. The fast checks
verify coverage, placeholders and compiled translations offline.

```powershell
uv run python tools/check.py
uv run python tools/verify_standalone_windows.py --headless --output artifacts/release-headless
uv run python tools/verify_standalone_windows.py --invalid-selection --output artifacts/release-desktop
```

The headless check extracts to a Unicode path, removes Python from PATH, uses disposable AppData,
loads the packaged engines, and recognizes both Japanese and Latin-text fixtures offline.
It hashes the distribution before/after to catch writes into packaged resources. The desktop
check additionally observes Qt welcome/startup acknowledgement and clean shutdown. Model tests
and real capture checks are separate; see [verification](verification.md).

## GitHub Actions and releases

Official Windows releases embed an Ed25519 public key. Before the first updater-enabled tag,
generate a signing key pair on a trusted machine outside the checkout. With OpenSSL and GitHub CLI:

```sh
openssl genpkey -algorithm Ed25519 -out update-private.pem
openssl pkey -in update-private.pem -pubout -out update-public.pem
gh secret set DANTETL_UPDATE_PRIVATE_KEY --repo nhmelo/dantejp < update-private.pem
gh variable set DANTETL_UPDATE_PUBLIC_KEY --repo nhmelo/dantejp < update-public.pem
```

Keep the private key in protected storage. Use the same key for subsequent tags so installed
copies can verify newer releases. The tag job signs `DanteTL-update.json`, binding the stable
version, Windows ZIP name, byte count and SHA-256 digest. Only the release job reads the
private key. For a local build, set the `DANTETL_UPDATE_PUBLIC_KEY` environment variable to
the public PEM text to enable in-place updates. Without it, About offers release notes and
manual release downloads. Existing versions predating the updater require one manual upgrade.

`check.yml` runs the same fast gate on pushes and pull requests: Ruff formatting/lint, Pyright,
Import Linter ownership contracts and model-free tests. `security.yml` adds dependency and secret
scans on pushes/PRs and weekly. `build.yml` requires both workflows, builds on
`windows-2025`, then verifies the relocated EXE and bundled offline OCR. Packaging or app changes
on pull requests also exercise the build; manual runs produce downloadable Actions artifacts.
The workflows need no provider credentials, personal tokens or preinstalled models.

After creating the GitHub repository, push this checkout and enable Actions. Set the default
branch's required check to **Windows fast checks / check** (select the actual reported check name
in GitHub). Run **Windows build and release** manually once to verify the clean hosted build.
For a release:

1. Update `project.version` in `pyproject.toml`, run `uv lock`, and commit both.
2. Complete the relevant native/hardware acceptance in [verification](verification.md).
3. Create and push a matching tag, for example `git tag v0.1.0` and `git push origin v0.1.0`.

A tag must exactly equal `v` plus the project version. Only a successful tagged build publishes a
GitHub Release, with the portable ZIP, wheel, source archive and SHA-256 sidecars. Extract the ZIP
and launch `DanteTL.exe`; keep its `_internal` directory beside it. Failed checks/build/smoke tests
prevent publication. The release job downloads only that run's artifact, checks its hashes and
uses GitHub CLI with `--verify-tag`; it cannot silently create a tag. No project code or checkout
runs with its write token. PR/manual runs cannot publish. Existing releases are never overwritten;
a failed upload can leave a draft to inspect before retrying. This initial recipe uses stable
version tags; review release notes before sharing a release widely.

Hosted CI does not prove live WGC, global hotkeys, interactive Qt behavior, CUDA/Vulkan inference,
game compatibility or app-update/restart behavior on real PCs. Those remain explicit checks on an
unlocked Windows desktop with the appropriate installed runtimes/models. Do not attach privileged
self-hosted desktop/GPU runners to untrusted PRs. The packaged smoke test supplies real CPU OCR
coverage without pretending that it covers those hardware paths.

Action revisions, uv and Python/build dependencies are pinned. This is a repeatable build recipe,
not a claim of byte-identical ZIPs: Windows images, compiler versions, signing and archive timestamps
can differ. The portable executable is currently unsigned. See [security maintenance](security.md)
for the scoped dependency exception and remaining repository/signing setup.

Workflow references: [reusable checks](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows),
[GitHub CLI releases](https://cli.github.com/manual/gh_release_create).

The normal source shortcut remains `Run DanteTL.cmd`; `--cpu` bypasses a saved runtime for recovery.
Model and optional runtime setup live in the app. `tools/setup.ps1` remains a developer environment
helper, while the old top-level setup/hardware shortcuts and custom packagers have been removed.

CPU compilation is needed only on developer/CI machines. `tools/build.ps1 -CpuWheelOnly` prepares
just the wheel used by source-checkout in-app CPU setup. Standalone releases include it and install
it offline with its checksum, without a compiler. NVIDIA setup downloads the JamePeng CUDA 13.0
wheel for the managed Python ABI. The wheel includes CPU backends ranging from baseline x64 to
AVX-512. ggml selects a supported backend at model startup. Vulkan uses abetlen's Windows wheel.
Both downloads use the URLs and hashes in `uv.lock`. CPU cannot use the older published
0.3.36 wheel because it lacks PLaMo's tokenizer. The deleted `native_runtime.py` assembler and
`setup_native_runtime.py` helper are not used anywhere.

A preinstalled LLVM-MinGW toolchain also works through the same upstream backend: add its `bin`
to PATH, set `CC=clang`, `CXX=clang++`, `CMAKE_GENERATOR=Ninja`, and
`CXXFLAGS=-static-libstdc++ -D_WIN32_WINNT=0x0A00`. Keep the toolchain available to delvewheel during
repair. The app build then restricts PATH before PyInstaller so unrelated apps cannot supply DLLs.
`-Offline` reuses the verified finished CPU wheel; it does not compile a replacement.
Online CPU builds use pip's normal build cache. Clear that cache when deliberately changing the
compiler or build flags; the published CPU wheel/checksum must then be regenerated together.

References: [uv builds](https://docs.astral.sh/uv/guides/package/),
[PyInstaller specs](https://pyinstaller.org/en/stable/spec-files.html),
[Qt deployment](https://doc.qt.io/qtforpython-6/deployment/deployment-pyinstaller.html).
