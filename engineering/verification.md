# Verification

Run checks for the changed boundary. Keep outcomes, environment and limitations in ignored
`artifacts/`; this document contains current procedures, not historical results.

## Fast checks

Run `uv run python tools/check.py` after code changes. It runs Ruff format/lint, Pyright and
model-free pytest, plus direct/transitive ownership contracts using Import Linter. Public view ports
and application owners have strict Pyright checks (the `strict` list in `pyproject.toml`).
Dependencies must already be installed; no models or provider accounts are required. It also
checks the bundled UI catalogs against the English source and verifies
format placeholders and native dialog labels. Tests use temporary SQLite and disposable
fake credentials/local HTTP transports.
Qt tests share one session-owned QApplication. Tests stop their workers, and the app fixture releases
remaining test-created top-level widgets and flushes deferred deletes after each Qt test.
Windows CI runs this entry point; interactive capture/GPU acceptance is separate.
Release builds use [the standard build entry point](building.md). The distribution workflow also
runs `verify_standalone_windows.py --headless` against the relocated EXE with offline Japanese OCR.

Use explicit-time controller replays for processing/lifecycle policy. Keep GUI checks for commands,
visible state, pause/disclosure, progress and responsiveness. Exercise both startup discovery orders and reopen setup after acknowledging its result. Switch models with a paused inspector after the live scene empties. Keep worker integration tests for
thread ownership, teardown, publication races and cancellation. New extraction tests should use
public commands/snapshots, not mirror private fields or implementation helper order.
`test_application_transitions.py` drives `start_capture`, `configure`, `assign`, `consent`, `stop`
and `advance` against controlled worker completions. Cover frozen devices, cleanup gates, invalid
geometry, consent failure, import suspension and model removal/relocation there. UI fixtures retain
their injected owners explicitly; change inventory through its injected source and invalidation,
then drive `DesktopDriver.refresh`. Use widget actions/signals and observable snapshots instead of
calling individual coordinator phases or private window renderers. Scripted scene output is for
view wiring only; source/acknowledgement/retention policy uses the real `LiveSession` in headless replays.

Download/runtime loading changes also run `tests/test_download_transport.py` and
`tests/test_llama_runtime.py`: local trickling headers/body, blocked connection setup, deadlines,
cancel and retry from a saved prefix, range validation, full-response fallback, concurrent
initialization and failed-load retry. Check the windowed helper
in a rebuilt EXE. Model-free tests must not contact real model hosts.

## Native fixture rules

Run native checks on an unlocked Windows desktop. They create/capture only our own fixture windows;
synthetic input must target only those fixtures, never games or unrelated applications. Some checks
need normal desktop/local-socket permissions outside a restricted sandbox. That is an environment
requirement, not evidence of game compatibility. Each tool owns a deadline and drains its resources.
Read its `--help` before selecting options. Use separate task-owned `--output artifacts/<slice>/...`
paths and disposable state/cache roots; do not overwrite user preferences or preparation receipts.
Use the native Windows Qt platform for Japanese image fixtures. The offscreen platform can lack system
fonts and render missing-glyph boxes; the regional corpus rejects that configuration explicitly.

Model files must already be installed explicitly. Never download automatically to satisfy a check.
Do not run real cloud provider calls, use personal tokens, or trigger charges in automated checks.
The cloud UI fixture uses fake connections and responses. Native local UI fixtures use
`verification_isolation.run_isolated`: empty disposable cloud catalogs/accounts, a credential store
that refuses access, and blocked Python sockets. Model-free MainWindow tests also replace the
default account/catalog dependencies. Select models by identity, never by a presumed last-row index.

## Choose checks by changed boundary

Commands below use `uv run python` unless an isolated interpreter is specified. Run appropriate
checks once after changes settle; repeat only for new changes, failures or unresolved concerns.

| Change | Required additional verification |
| --- | --- |
| Capture/coordinates/overlay | `tools/verify_windows.py`; `tools/verify_capture_controls.py --output artifacts/<slice>/controls` (add `--menu-blur` for native backdrop changes); manual click-through, focus, move, resize, minimize and source closure |
| English overlay placement | `tools/verify_english_layout.py --output artifacts/<slice>/layout`; replay a translated fragment followed by a containing paragraph with separate track IDs. Check the paint and backdrop handoff, reversed arrival order and disjoint labels in crowded layouts. |
| OCR adapter/provider | `tools/verify_ocr.py`, then `--dense`; `tools/verify_ocr_windows.py`. For the Latin reader run `tools/verify_paddle.py` on CPU and with `--device cuda` in the GPU environment |
| Tracking/content/presence/visual/regional OCR | Dense OCR; `tools/verify_retention_windows.py --appearance`; live translation/dialogue with `--ocr-delay-ms 900 --typewriter --transitions` |
| Grouping/readiness/English publication | `tools/verify_translation.py`; `tools/verify_ocr_windows.py --translation --dialogue --typewriter --transitions --ocr-delay-ms 900` |
| Translation scheduling/cache/swapping | Offline translation; live `--translation --dialogue --gemma --debug`; retention/slow-OCR checks above |
| Cache management | Replay shared game ownership, unnamed captures, eviction and clears during live work. Exercise cancellation and confirmation for a game and the full cache using disposable state. |
| Model consent/preparation/probe/shutdown | `tools/verify_setup_windows.py` and `--missing-runtime`; live translation/model swaps; import lifecycle check when probe changes |
| Main/Settings/Advanced/pause/preferences | `tools/verify_ui_windows.py`; native setup and live translation/pause checks. For source changes, check silent idle and same-reader switches, then cancel and confirm a cross-reader switch from the two pickers on an owned window. Check that both native windows and the active Settings tab persist. Verify same-model OCR reuse, cross-model replacement and exact-target capture resumption. |
| Interface language and catalog packaging | Run the fast checks and `tools/verify_ocr_windows.py --translation --dialogue` to exercise a live language switch during capture. Run `tools/verify_standalone_windows.py` on the rebuilt distribution to check the English, European Portuguese and Japanese welcome screens with disposable preferences. |
| Dialogue reader/classification | Explicit-time history replays; live `--translation --dialogue --gemma --typewriter --ocr-delay-ms 900`; confirm whole-turn deduplication, separate speaker and unchanged model work |
| Ollama server and picker | Fake Ollama transport and shared welcome/Settings editor tests. Exercise an empty catalog, endpoint changes, stale tags, interrupted requests, mixed provider choices, and switching tabs before Next. Verify bundled OCR, missing OCR, and optional CUDA runtime-only setup without a local translator. |
| Cloud/routing/editor/credentials | Fast fake-transport/credential tests; `tools/verify_cloud_windows.py`; revise a dialogue while a menu call is active, and verify shared and retained calls through the translation owner |
| Two assigned local models | `tools/verify_category_models.py --device cpu` in a CPU runtime, `--device cuda` in NVIDIA and `--device vulkan` in Vulkan; verifies inference/switching without upstream disk caches; live swaps and retention checks |
| Runtime installer/launcher/restart | `tools/verify_runtime_windows.py` in a disposable extracted distribution; standalone relocation check |
| First-run/progress | `tools/verify_first_run_windows.py`, native setup and packaged runtime/restart check |
| Global shortcuts | `tools/verify_hotkeys_windows.py` with only our foreground fixture |
| Imported GGUF | `tools/verify_import_models.py`, `tools/verify_import_windows.py`, built-in Gemma and packaged lifecycle checks |
| Runtime dependency upgrade | Isolated candidate; `tools/verify_shared_cuda.py` in both initialization orders, strict real OCR, live Gemma switching and CPU fallback |
| Distribution/bundled OCR | `tools/verify_standalone_windows.py`, fresh offline bundled OCR with no AppData models, packaged setup/restart |

## Processing and model acceptance

For pre-translation latency, replay a high-confidence sentence appearing on a blank panel.
Compare its first eligible request with ordinary settling on the same captured sequence.
Include a typewriter pause at a comma, a delayed second row, a terminal OCR comma and a Latin
abbreviation. For menu reuse, retire the original track and verify an exact reappearance before
another OCR result. Change one native pixel, the DPI and the backend generation independently.
Measure visual readiness separately from model inference and time to visible English.
For first-reading translation, compare fresh launches of `tools/profile_pipeline_windows.py`
with and without `--confirmed-only` at the same capture rate and model. Alternate the order.
Compare first and complete English publication for cold and cached labels, menus and dialogue.
Check actual model-call counts, preservation of parallel menu decoding, cache-only remote misses
and withdrawal when a previously cached prefix grows or wraps.

`tools/benchmark_tracking.py --output artifacts/<slice>/tracking.json` measures completed dense
4K OCR observations, including long text, reordering, edits, same-pixel misses, removals and partial
coverage. It reports worker preparation separately from serialized tracking. Optional `--baseline`
loads a trusted local copy of the preceding `tracking.py` and asserts identical assignments, visible
tracks and translation readiness. This complements the Qt regression that blocks worker scoring after
OCR completion while checking event processing, retained English and delayed acknowledgement.
The long/repetitive cases contain the stated number of characters, up to 512. The baseline adapter supports the previous keyword evidence interface and the typed review interface.
Both sides receive evidence for the same source frames.

Small-crop preprocessing changes run `tools/verify_ocr_crops.py --device cpu --output
artifacts/<slice>/crops-cpu.json` and strict CUDA with the GPU interpreter. Preserve its accuracy
assertions. Before rebuilding, save the preceding local app wheel; then run
`tools/benchmark_crop_recovery.py --baseline-wheel <saved-wheel> --device cpu --output
artifacts/<slice>/differential-cpu.json` and the CUDA equivalent. This tool executes the Meiki adapter
from that trusted local wheel, not downloaded code. It compares per-target edits, extra blank
detections and exact actual coverage on identical source frames, including all first observations.
Raw outputs/scores and timings are retained. Do not credit confidence alone as correctness or
interpret first-use compilation as warm latency. Native retention checks below remain required.

Regional planner changes also run `tools/benchmark_scan_selection.py --device cpu --repeats 5
--output artifacts/<slice>/cpu.json` and the same command with `.venv-gpu/Scripts/python.exe`
and `--device cuda`. The tool blocks sockets, asserts CUDA, rotates scan order, separates first
passes from warm measurements, and fails on any per-target edit-error or demand-coverage regression
against the original crops (including first transition observations). It exercises bounded planner
hints across repeated observations. Preserve JSON raw regions, character scores, actual coverage,
selected plans and phase timings. This corpus does not replace the live retention/slow-worker checks
above or establish general game accuracy. `benchmark_regions.py` retains the original four-case
separate/paired/merged/full comparison. OCR consensus remains a separate, non-production experiment.
The scan corpus includes five/eight compact demands and five widely separated demands. Use repeatable
`--case` filters for investigation; candidate merged timings do not authorize relaxing the production
crop limit. Compare individual target errors and actual coverage as well as total time.

`tools/verify_ocr.py --dense` uses synthetic 4K labels and reports actual detection count, raw
text/boxes, OCR and visual costs. It alternates full scans and a padded regional crop, checking
coordinates and first/warm/after-full timings. Preserve raw accuracy evidence; installed providers
alone do not prove successful inference. Regional tests cover crop offsets/edges, whole-line closure,
unscanned-track retention, coalesced coverage, per-line sequences and independent discovery.

For discovery changes replay an isolated dim digit, equal-luminance colored lettering, a slow fade,
and moving scenery without text at the same capture timestamps. Record time to the first regional
request separately from correct OCR and visible English. Count extra OCR on moving scenery. For
fixed-line edits compare real CPU and strict CUDA reads against ordinary regional detection on
identical pixels, including growth and removal. Exercise uncertain reads and source invalidation
through the worker, and verify that a successful direct read reports only its sampled coverage.

`tools/verify_translation.py` defaults to Gemma E2B. Use `--backend gemma --device cpu` or
`--device cuda` with the correct interpreter. It reports model/cache identity, load/warm/inference
and memory; sample phrases are a smoke test, not a quality ranking. Imports and curated models must
retain their prompts/identities and reject incomplete/reasoning output. Verify each advertised
CPU/CUDA/Vulkan path; label unavailable hardware as untested. Compare download size and working
memory separately. Recheck exact imported content on load; never delete user-owned GGUF files.

For parallel menu decoding, compare a page of distinct uncached labels with individual calls
on the same model and target. Report preparation separately from page time, count exact wording
matches, and repeat with the labels reordered. Include a seeded sampler and a hybrid or recurrent
model when verifying slot reuse, alongside a greedy transformer.

`tools/verify_category_models.py` accepts `--short` and `--long` model IDs (defaults: Gemma E2B/E4B).
It uses two local owners, temporary SQLite, no network, and repeated unchanged requests. Require
actual offload in the CUDA log; an installed GPU wheel alone is insufficient. These fixture timings
include loading and are not comparative speed or quality benchmarks.
It also replaces the long-text lane with two offline cloud fixtures while translating new labels
through the unchanged native worker; its native load count must remain unchanged.

`tools/verify_ocr_windows.py --translation --dialogue --gemma --debug --typewriter --transitions
--ocr-delay-ms 900` (one command) exercises real capture, source-matched inspection, grouping,
Gemma E2B → E4B → E2B, debug disclosure, pause, resize/minimize, CPU restart and clean shutdown.
It requires complete grouped requests after the fixture's reveal and checks already-published
English while OCR is delayed. `--typewriter` uses a fresh synthetic translation cache. Transition
checks briefly remove the lettering while its panel stays in place and require continuous published English.
With debug drawing off, the fixture also compares the painted dialogue pixels during visual loss.
At 8 FPS removing the entire panel must hide the dialogue on the first analyzed capture showing the
exposed background and within 400 ms of closure, even with delayed OCR. Record the
elapsed time and whether it hid before OCR observed the closure. Audit overlay publications for
at least one second after dismissal. Reopening must restore English.
Repeat with `--populated-transition` to switch to a textured menu with new Japanese labels.
It brings back one original row and changes the wording of the other in the same position.
Record the first menu capture, visibility of the old dialogue at the first visual analysis,
elapsed time to hide, and time to the new English. Old dialogue stays hidden throughout inference.
These fixtures exercise one reveal cadence; they do not establish every game's behavior.
Grouping replays also cover long profile paragraphs and full/regional OCR quote readings, including
damaged brackets and hanging marks, through tracking/publication as well as scaled/reordered boxes.
Replay identical quote pixels with the leading dash recognized and omitted. Check exact source text
and bounds, adjacent choices and single-row rescans with the stroke still present or removed.
Replay first- and second-row splits across scales, including partial confirmation and an intact
row splitting between scans. Check compact buttons above prose and adjacent columns independently.
Equipment-panel replays keep headings and item labels separate, include numeric percentage tails,
cover full-frame and cropped readings where OCR turns a plus into a colon,
and preserve group membership across full/cropped icon readings. Replay a noisy numeric operator
both on and off the expected row, and check the paragraph opening with an unreadable tail.
Exercise pending ordinary edits,
unchanged-pixel noisy scans and real value changes through English publication, not just static boxes.
Keep paragraph boundaries, headings/choices, exact member readiness and bounded requests intact.
Line count and trailing dashes must not cut a paragraph. Reject distant rows before text analysis
and avoid repeated member scans when extending long paragraphs in dense layouts.
The delayed-OCR continuity stage also changes Show border while preserving the capture/OCR/model
owners, monotonic frame sequence and published English. `verify_capture_controls.py` samples only
the owned topmost fixture's desktop rectangle: the overlay must appear in screen composition while
remaining absent from the raw WGC source. It measures border off/on and drains capture afterward;
an OS-enforced border is a visible unsupported result, not proof of application state loss.
With dialogue enabled, the live check also verifies the history reader records a complete group,
does not duplicate it through grouping/model switches, and opens without replacing the translator.
The fixture may display Unknown speaker when its small name text scores below the reader threshold.
When confident, its bilingual label must match the current published name translation; the reader
opens at the newest entry. Model-free GUI tests also cover scroll anchoring through prepends/eviction.
The fixture explicitly enables history (normal default is off). Add `--classifier gemma` to exercise
a separate local classifier concurrently with translation and its stop/drain lifecycle. Without it,
the fixture explicitly chooses layout-only admission. `verify_history_classifier.py --model gemma
--device cuda --output artifacts/<slice>/classifier.json` evaluates bounded text-only judgments with
an installed local model and sockets blocked. This is a small smoke sample, not an accuracy benchmark.
Its panel fixtures include simultaneous response alternatives, identical words spoken alone,
loading headings, continuous story paragraphs, stacked name/title headers and adjacent responses.
Single-response fixtures include the supplied OCR bounds for short greetings, the same question
without panel context, mirrored/scaled layouts, stale choice recovery and ambiguous adjacent panels.
Record both false admissions and missed dialogue;
do not equate the model-free admission tests with semantic model accuracy.

`tools/verify_retention_windows.py --appearance` requires unchanged blocks and English through
repeated hostile/noisy/missing OCR over a moving background, then discovers a new regional label
without disturbing the old labels. Add `--low-contrast` to exercise faint text over smooth moving
bright light. Also run `--colored` without `--appearance` for long green descriptions over dense
moving white glints; this audits the six descriptions while allowing unrelated tiny detections.
Run all three variants when changing visual retention. The fixture records every English
publication and saves only its owned test window. Explicit-time tests additionally cover detector
splits, faint digit edits/typing/removal, stale generations,
static WGC frames, small continuation growth, polarity/brightness changes, failures and resets.
Long horizontal/vertical digit-change replays must both retain unchanged text through lighting/OCR noise
and accept genuine edits after ordinary confirmation. Include exhausted fine-reference budgets.
For quest switches, replay a two-line paragraph disappearing against its original gradient while
OCR is delayed. A new menu layout must hide its old English on the first analyzed capture and
keep independent labels visible. Also replay a panel closing onto a different color of equal
luminance, isolated lettering loss, distant animation, surviving color strokes, and OCR arriving
after a switch to a static page. Replay similar glyph outlines and delayed revisions after
dismissal, including an already-cached translation and a partially returning group.
Check restoration on the first capture that verifies the returning source.
Repeated polls of one capture count once.
Colored-paragraph replays cover partial scans/misses, appended false detections, mixed-color digits,
real growth/removal and preservation of raw OCR. Keep supplied frame pairs with their actual OCR in
ignored artifacts when diagnosing background-dependent disagreement.
Only committed display changes should request repaint; Qt may still repaint on native expose events.

## Runtime and packaging acceptance

For portable updater changes, run `tests/test_release_updates.py`, rebuild the portable ZIP
and run `tools/verify_update_windows.py --output artifacts/<slice>/update`. The native check
uses a disposable signing key, a loopback release server, an old fixture installation and the
actual helper and replacement GUI. Check skip, seven-day reminders, manual override, interrupted
transfers, bad signatures, extraction limits, busy installations and rollback. Set
`DANTETL_DISABLE_UPDATE_CHECK=1` in native fixtures that launch the packaged GUI.

Use `.venv-gpu/Scripts/python.exe` for strict NVIDIA checks with `--device cuda`; fail rather than
accept fallback. Normal automatic mode must report CPU fallback. Keep `.venv` CPU, `.venv-gpu`
CUDA and `.venv-amd` Vulkan/experimental DirectML isolated; never co-install ONNX distributions or
native llama wheels. The NVIDIA profile pins ONNX 1.30 and shared CUDA 13 libraries; preserve the
legacy ONNX 1.26/CUDA 12 loader. Test both import/initialization orders and loaded DLL identities
when changing dependencies. For NVIDIA translation, verify Gemma context creation and the
selected CPU backend on both AVX2-only and AVX-512 systems. For CPU wheel changes, run a local
translation on both CPU classes and record the selected ggml DLL. The relocated standalone check
compares the frozen CPU variants with those in the offline installer wheel. `tools/verify_shared_cuda.py` records
the loaded CPU variant alongside the CUDA libraries. First-use compilation and warm inference are
separate measurements.

AMD in-app setup intentionally uses Vulkan translation with CPU OCR. Strict experimental DirectML
must run the numerical compatibility probe and fail on mismatch; `auto-directml` reports fallback.
Do not describe provider initialization as validated Radeon OCR acceleration. Hardware detection
and actual inference success are separate evidence.

`tools/verify_setup_windows.py --missing-runtime` checks downloaded Gemma stays visible with an
explanation and cannot load without an engine. Ordinary setup checks explicit consent, decline,
failure rollback, reuse of the prepared worker and closing during setup. First-run checks require
explicit confirmation, skippable setup, concurrent model and runtime progress, cancellation of both,
retry from a saved prefix and postponement of the restart prompt. The packaged download helper must
recover a truncated local HTTP response and request the remaining bytes.

`tools/verify_runtime_windows.py --profile cpu|nvidia|amd --output ...` operates only in a disposable
extracted distribution/state root. Use `--offline --cache-root <verified-cache>` when available.
Use `--gpu <device UUID>` to select a specific compatible adapter. Check the saved device and
exercise a secondary GPU on multi-adapter hardware. The fixture installs or reuses a verified
runtime, exercises restart acknowledgement and the normal launcher,
and drains owned descendants on failure/cancel. Validate saved environment A after preparation of
B and after failed/postponed B activation. Never repair/replace an active loaded environment.
Containment tests must cover a child attempting to spawn before user code is allowed to run.
Standalone runtime checks must confirm imports originate in the installed app wheel, while OCR
assets follow the relocated bundle. Check CPU, CUDA and Vulkan; simulate a same-version app-only
wheel update and a version bump, reuse the same native environment, and import the new app from
its immutable AppData directory. Test legacy receipt migration, concurrent launches and an older
running process, rollback, checksum rejection, cancellation and a real dependency mismatch. App
failures must never trigger dependency reinstall. Verify ordinary EXE handoff and a fresh setup,
with no Python on PATH and native files unchanged.

`tools/verify_standalone_windows.py --output ... --invalid-selection` relocates the real EXE to a
Unicode path, removes Python from PATH, and checks fallback/fresh OCR. Distribution resources must
remain unchanged/read-only. Models/preferences/cache/runtimes use per-user state; managed Python
must be independent of the movable bundle. Pinned MeikiOCR and PaddleOCR assets ship with notices/hashes;
never include translation weights, personal caches/preferences, or development/GPU environments.
After package checks, remove verified task-owned extraction/build environments; retain small
reports/screenshots and the current distribution. Inspect paths/reparse points and live processes
before recursive cleanup. Artifact size must not grow without a deliberate reason.

## Interactive acceptance

1. Open the demo, select it and start capture. Inspect raw Japanese/boxes on their matching frame.
2. Enable Inspection and diagnostics in Settings → Experimental. Expand/collapse the section,
   pause/resume inspection, resize and maximize. Processing continues;
   frozen inspection stays source-matched and main controls remain compact/top-left.
3. Check English placement, automatic/manual text size, opacity, overflow/tooltips and click-through.
4. Toggle Translation overlay and Show OCR boxes. Change model; old English clears immediately,
   including paused inspection, while backend-specific disk caches remain reusable when returning.
5. Move, minimize/restore, change focus, stop/restart, then close the source and app. Nothing stale
   should reappear; progress remains responsive while workers drain. Test failed restart recovery.
6. Exercise both configured shortcuts against the fixture. Hiding and restoring the overlay preserves
   capture and translation. Registration conflicts preserve previous bindings.
7. Separately test the intended game and record resolution, DPI, HDR and window mode. Demo results
   do not establish game approval, anti-cheat, exclusive fullscreen or mixed-DPI compatibility.
