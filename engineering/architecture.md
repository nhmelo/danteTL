# Current architecture

This document owns cross-module contracts. Read the affected sections; code/tests own exact thresholds,
prompts and implementation details. See [building](building.md) and [verification](verification.md).

## Package layout

The distribution, Python package and CLI are `dantetl`; app-specific environment variables use
`DANTETL_*`. `DanteTL` is the display name. There are no aliases for previous names or module paths.

| Package | Owns |
| --- | --- |
| `application` | Commands, snapshots, preferences and coordination of capture, models, setup and shutdown |
| `translation` | Workers, scheduling, routing, publication; `backends`, `cloud`, `models` and `dialogue` hold their specific integrations |
| `ocr` | Recognition, scan planning, text tracking and visual retention |
| `languages` | Source-language selection, word boundaries and linguistic evidence for game text |
| `capture` | Native window identity, geometry and capture |
| `ui` | Qt views, controls, layout and desktop event delivery |
| `runtime` | Shared model assets, native library selection, downloads, contained processes and isolated runtime installation |

The root contains shared physical-coordinate values, resource paths, worker wakeups and application
entry/assembly modules. Package initializers do not import or re-export implementations. Consumers
import the module that owns each API. Import Linter checks whole application/processing packages for
Qt/view dependencies, all views for backend/service dependencies, and runtime services for UI coupling.
Shared asset validation belongs in `runtime.assets`; OCR owns only its pinned model definitions.

## Application ownership

| Owner | Responsibility |
| --- | --- |
| `composition` | Construct services, dependencies and desktop adapter |
| `ApplicationSession` | Serialize commands/completions; coordinate capture, setup, translation and shutdown |
| `CaptureLifecycle` | Freeze device/rate choices, validate capture targets/geometry, and gate capture teardown |
| `TranslationCoordinator` | Resolve selections, model identities, backend recipes and classifier configuration |
| `ModelLifecycle` | Commit assignments; own consent/preparation, route commitment, worker availability and adoption |
| `application.presentation` | Pure scene-reset/status projection; no worker operations |
| `CloudServices` | Backend/account/catalog services and credentials; no widgets |
| `SetupCoordinator` | Download/import, runtime and guide operations, cancellation and probe permits |
| `ModelInventory` | Cached model/runtime discovery and preparation receipts |
| `LiveSession` | Visual/OCR reconciliation and acknowledgement order, scene resets, tracking, grouping, English publication |
| `ShutdownCoordinator` | Cancel/drain owners, flush preferences, release hotkeys, then restart/close |
| `DesktopDriver`, Qt views | Native events, presentation and commands through typed view ports |

Views may retain drafts and last-rendered values, never backend factories, credentials or model policy.
Rendering must not start work or scan files. `ApplicationCommands`, `CloudCommands`, `SetupCommands`
expose commands and immutable snapshots. Import Linter enforces transitive view/service separation;
public ports have strict type checks. Each settings gesture submits only the editable fields it owns.
The session applies those fields to current preferences. The settings writer merges changes since its
last successful save under a cross-process lock, preserving unrelated edits from other instances. Model assignments, source changes and setup
intent use separate commands. On a source change the application applies one source-specific set of
OCR choices, setup bundles and inventory inspector, then waits for discovery before resuming capture.
Prefer headless explicit-time replays for coordination policy.

`advance` adopts inventory/cloud/setup completions before integrating live owners. Nested Qt events
request a coalesced follow-up, not reentrant coordination. Worker hints wake the same path; a 33 ms
fallback services geometry/lifecycle. Dialogs opened from this path return immediately and dispatch
consent or acknowledgement when the user responds. Image analysis and native/file work never run on that thread.
Capture polling returns an explicit phase; the application does not infer it from several worker
flags. Translation returns a typed activity (probe, preparation, standby, inactive, switching or
active), with a publication worker only when usable. The application joins probe readiness with
classifier cleanup, and sends current scene requests through `LiveSession` only. Capture teardown
waits for translation before draining OCR/visual owners. Status formatting cannot advance workers.
Controls and notices reuse the published OCR and translation state for the current refresh.
Resetting the scene releases its OCR snapshot, including the source image, as soon as capture stops.
`TranslationCoordinator` owns a `ModelLifecycle` through composition. Lifecycle state is private;
commands change it and immutable snapshots expose it. Assignment/readiness revisions can be read
without polling workers. Starting preparation and committing its result both require the consented
model reference and weights/runtime identity in the current inventory. Revocation drains the worker
and preserves the previous assignment and receipts. Handoff checks the same identity against the
available route. Rediscovery can update the file location or display name for that identity.
`ModelLifecycle` owns each one-shot consent token and adopts prepared workers by exact identity during
route reconciliation. It gates route activation, consent, probing and
shutdown. The coordinator tracks availability for each assigned model. Revoking one connection
withdraws its recipe on the next reconciliation even when another assignment is ready.
The routed owner retains unchanged workers during committed
assignment changes. It drains replaced workers before starting replacements and coalesces rapid edits
to the latest plan. Unchanged lanes keep processing; retired replies cannot publish. Active plus retiring
owners remain bounded to two. One assignment-plan constructor owns runtime routing; older saved
preferences are decoded only at the input boundary. Identity includes local weights/runtime preparation
or cloud inference settings, limits, game hint and connection/account credential revision. Display names
are presentation only: renaming an active model or editing an unrelated model preserves workers and
published results. Consent/preparation and import probes retain their full-drain safety
boundary. The publication port exposes
update/invalidate/snapshot, without load/stop/routing operations. A separate import contract keeps
status projection independent of worker implementations. Confirmed disappearance of the selected
HWND ends capture normally during startup, a frame callback or a border change. Other geometry
failures remain visible.
Capture Stop intent remains authoritative
while the native owner is still reporting its last running snapshot.
Inventory uses one bounded worker, explicit invalidation and a 30-second fallback scheduled from
completed scan adoption. Invalidation during a scan discards stale results and requests one follow-up.
A discovery pass samples bundle locations, runtime capabilities and the imported registry once.
The picker and setup view derive their choices and status from those same observations.
Unchanged revisions do not rebuild routes/options. Cached badges never authorize native model loading.
Imported-registry discovery has its own failure boundary. A malformed/unreadable registry is preserved
and reported while healthy built-in choices and OCR assets remain available; a later refresh can recover.
Registration, relocation and removal hold the registry lock from the initial read through replacement.
The complete candidate catalog must pass the reader’s schema and byte limits before replacement.

Consent/review replies carry monotonic tokens. Decline/failure preserves committed assignments;
stale replies cannot adopt new proposals. Setup has three bounded lanes: models/imports, runtime,
and guide. Import probes wait for both translator and classifier cleanup. The skippable guide offers
downloadable bundles, not imported GGUF registrations. Its saved choice survives delayed discovery
and goes through the ordinary assignment and consent flow even if the picker already displays it.
The guide preserves cloud assignments. Postponing a runtime switch defers guide-model preparation.
The guide opens the Settings cloud editor as its own child. The same cloud service owns drafts,
credentials and operations while local downloads continue. Saving a profile makes it available
for explicit selection on the main screen. The guide waits for editing to finish before offering restart.
Restored assignments use exact preparation receipts. A missing receipt offers preparation directly
on the assigned model. Discovery must settle before the UI reports a missing assignment.
The application acknowledges the choice on assignment or when the user answers the preparation prompt.
After review, the guide runs one model installer and one runtime installer concurrently and publishes
progress for each. The guide owns both until they drain. Settings can also install a model and a runtime
concurrently. Choosing a verified inactive runtime or finishing its install offers a restart once
other setup work finishes. Detecting hardware or restoring a picker never triggers the prompt.
A cancellation request leaves the lane busy until its worker reports an outcome. Committed changes
refresh inventory even when cancellation arrives before the application observes completion. Discovery
also reconciles verified files kept by a partially completed setup.
The first-run offer stays pending until setup is available. A recreated guide can observe the same
operation and recover its result. Finishing or closing a successful guide acknowledges its exact
review token once, then a subsequent guide opens at the welcome page.
Hiding Settings never cancels an operation implicitly; app close cancels/drains all owners.
Shutdown waits for actual resource release and owner exit, not just a published result/failure.
Preferences and hardware work drain and hotkeys release before restart; failed restart resumes services.
Close and restart release the published scene immediately. Draining workers cannot republish its
frames or observations. A persistently invalid capture clears its scene once per loss of validity.

## UI, identity and coordinates

The public app name is DanteTL.
About and error dialogs link to the project's GitHub issue tracker. Opening the tracker is an
explicit user action. Native startup and update error boxes show the address as plain text.
The interface locale is a separate preference from game source and translation target.
Changing the interface language replaces the app-owned Qt catalogs and updates text on existing widgets. Capture, model workers and editable values keep their state. English is the fallback. Mark app-authored copy
with `tr` and static metadata with `mark`, then translate that metadata for display.
Keep IDs, OCR samples, user content and inference prompts independent of the interface locale.
Main/Settings/Advanced are views of one live session. Controls remain top-left and width-bounded;
Settings → Experimental → Inspection and diagnostics defaults off and remembers the selection.
Enabling it reveals a collapsed section. Disabling it closes inspection and clears pause and
diagnostic overlays. Inspection expands without changing maximized state. Simple/advanced routing preserves assignments.
Model selectors share styled nonselectable category headings. Unchanged lists preserve popup focus;
missing selections never choose replacements. Persist validated IDs/values, never HWNDs, pause or debug.
Badges distinguish files-present from engine-available. Device/rate edits apply next capture, including
when models arrive mid-session. Worker-based hardware advice separates detection from verified inference.
Capture status lives in Game & overlay. Its indeterminate loading bar and stage text project actual
capture/OCR/per-model readiness, including a partially loaded route. Ordinary translation requests
do not activate the loading bar. Startup indicators
remain visible with inspection collapsed and clear on readiness, stop or failure; no estimated
percentage or extra polling is introduced.

Native rectangles use physical pixels, including negative desktop origins; OCR is client-relative.
Enable per-monitor V2 DPI awareness before windows. Validate/crop only dimension-matched containing
frames. Position the overlay in physical coordinates; compensate its device pixel ratio when painting.
Moves may reuse a compatible frame. After an observed size mismatch or minimize, wait for a
newer, size-matched frame even if the window returns to its previous dimensions. Capture fixes HWND plus PID
until explicit new selection. Never substitute windows/monitors after failure. Copy borrowed native
pixels before callbacks return; use bounded mailboxes and generations to reject late work.
Static WGC windows need not deliver repeated frames; frame age alone is not failure evidence.
Users can set a sampling target or select Unlimited to accept each native callback.
At a numeric target, the rate guard admits slightly early callbacks in their scheduled slot.
When WGC rejects the optional native update interval, the session retries the same target once
using Windows' default cadence and retains that choice across border edits. The rate guard still
enforces numeric targets. Unlimited accepts every delivered callback. Both modes replace the latest
frame in one mailbox, and OCR keeps its independent request cadence and coverage policy.
Capture copies packed BGRA rows from the mapped client area. The owned frame excludes row
padding and window chrome and remains valid after the native callback returns.

The overlay is nonactivating/input-transparent; hide on focus loss, minimize, stop, source failure or
size mismatch. Border edits reopen only the same native source while retaining processing owners and
sequence. Screen-capture inclusion changes overlay affinity only; report failure and restore settings.
If first-show native configuration fails, keep the surface hidden and report the failure once.
Retry on request, when the overlay is re-enabled, or when the recording preference changes.
Raw game capture must exclude our overlay. Affinity is not a universal recording/security guarantee.

English labels have lightly rounded borders, measured text padding and a translucent text shadow.
Windows rounds each native backdrop where the compositor supports the small-corner preference.
Backdrop blur defaults on, with the same preference in the welcome guide and Appearance.
It uses one click-through native surface per placed English label, up to
the live placement limit. Each new label paints with its regular background before a later event-loop
turn starts its native blur. Only boxes whose blur coverage changes need a follow-up repaint.
Retired surfaces hide immediately and can be reused until
the overlay hides or blur is disabled. Surfaces share the overlay's recording preference.
Native setup failures use the configured background opacity.

Overlay and capture shortcuts use RegisterHotKey on Qt without keyboard hooks or polling.
Rebinding is transactional and rejects stale messages. Only the explicit capture shortcut may choose
an eligible foreground HWND/PID; Start uses the selected target. Both honor modal/shutdown guards.

## OCR, visual evidence and retention

OCR owns imports, loading, inference and cleanup on its worker. Keep one active job, one replaceable
pending frame with coalesced coverage and one unacknowledged result. Native calls are not forcibly
interrupted; `finished` requires owner cleanup/exit. Failure leaves capture available. Raw `OcrResult`
retains its exact frame, text/character bounds, scores, coverage and timing. Never overlay old raw rows
on new pixels. `text_content` derives character-bounded tracking input without changing raw OCR;
only isolated weak edge glyphs beside strong Japanese runs may be filtered, never low confidence globally.

`LiveSession` waits for source-matched visual evidence, reconciles each completed OCR sequence once,
then acknowledges and publishes. The worker cannot overwrite unconsumed observations. Accepted OCR
lives outside inspection. Hidden inspection keeps the latest exact pair without conversion/table work;
pause freezes both displayed languages only. Resume shows the newest pair. Model changes revoke frozen
English; resize/minimize/stop also clear frozen source presentation.

Tracking accepts a live observation through `observe_reviewed`. A `CandidateReview` carries
checks and matching scores for the completed OCR source. `TrackPresence` carries the capture
sequence on which anchors were observed. The tracker validates the candidate and its prior track
state before mutation. Missing or stale scores leave the observation pending for a fresh review.

`ScanCadence` owns follow-up deadlines, periodic full scans and consumption of dirty tiles.
Probe analysis supplies growing, ready and settling regions. The OCR scan planner chooses how
to execute the resulting coverage on the selected provider.

`VisualSession` owns bounded thumbnails, tiles, templates/probes and one pending job, never frame history.
Discovery compares adjacent luminance thumbnails and localized clusters of changing color edges.
Each tile also keeps stroke detail from its last requested scan so slowly fading text can trigger
regional OCR. Widespread fine motion contributes no extra discovery demand. Independent luminance
checks and periodic full discovery still run. Supplemental edge evidence can request recognition.
The presence and tracking policies decide whether to hide or retire a translation.
Between OCR passes, a painted block can hide when every current member loses its glyph pattern.
A vacant line can hide on the first detailed capture if its backing or surrounding area changes
and native color edges confirm the space is clear. Samples on both sides of the line can detect
a backing color change at equal luminance. For menu switches, the worker compares tile-level
corner density with the accepted OCR source. Changes must extend beyond the line and its
neighbors and include part of the surrounding layout. When old stroke patches vanish and
separate parts of the line are clear at native color resolution, the English can hide on that
capture even if the new page reuses the old backing. Other vacant lines require three distinct
captures spanning at least 160 ms. A shared background gradient can correlate across a switch even
when all lettering has vanished. Very low surviving edge coverage and loss of separate
stroke patches across the line can corroborate vacancy in that case. For a textured replacement,
compare separate stroke patches across the old line with positional tolerance. Strong surrounding
change or a corroborated layout switch confirms the loss immediately.
An isolated replacement needs a second capture with the same new pattern. Weak or surviving glyph
evidence defers to OCR. After dismissal, matching glyphs can restore the line on the
next capture. Similar outlines on the new menu only provide evidence for retaining an
undismissed track. A new OCR template for a dismissed line must also match the display
capture. Live reconciliation preserves dismissal across pending bounds or text revisions
until the worker verifies the current line. During visual failure a later covered OCR
reading can restore it. Retirement requires a covered OCR miss
with corroborating pattern loss, or repeated covered misses across distinct captures. On a static
source, a single miss can settle only with strong pattern loss on that exact capture. Matching pixels
or a surviving stroke pattern at least as recent as a miss preserve the track. Positive OCR edits
still use their own confirmation rules. Coarse thumbnails screen changes; compressed long/vertical lines need overlapping,
aspect-preserving glyph-scale tiles before they can be declared unchanged. Reference pixels have
per-line and aggregate budgets. Exhausted or ambiguous fine evidence allows ordinary OCR confirmation;
it must not freeze a confident old reading. Fine comparison checks a new source-matched OCR review
and verifies suspected clearing of a compressed line.
Fine evidence may confirm unchanged glyphs despite a changed coarse thumbnail. Uniformly colored
rows with complete character locations can also use a foreground-color projection to reject neutral
background glints. Every letter/digit and reliable punctuation must contain that color; mixed text
falls back to grayscale. Local stroke-strength comparisons tolerate subpixel color fringes while
still rejecting numeric edits. Corrections may have two pixels of character-box jitter; sample only
inside accepted geometry. Optional color references share the pixel budget and cannot displace basic
references in the same observation. Beside such rows, small neutral glints are not growth; colored
additions and substantial strokes in another color still unlock OCR. Continuation checks cover the
nearby line end rather than extending into a distant column.
Visual failure visibly falls back to ordinary OCR. New translations wait for visual examination of their
OCR source; existing valid English is independent of readiness.
If optional crop evidence fails, matching can still accept the OCR result. A matching failure
acknowledges that observation once and requests a new full scan, preserving accepted text.

Partial OCR has explicit coverage; unscanned tracks are not misses. Pending areas union while pixels
are replaced. Retain per-line probe source sequences, whole-line closure and single coordinate offsets.
Within 120 ms the idle OCR worker can admit up to four requested areas if each additional
batch includes a disjoint area. Fully overlapping or full-frame demand waits on the worker
deadline, including when capture stops producing frames. The latest
pending image and accumulated coverage still require acknowledgement of the previous result.
An artificial-edge detection discards the provisional crops. An optimized plan retries the original
demand directly. An edge in the original crops triggers one full-frame fallback.
Independent full-discovery deadlines cannot be delayed
by typewriter activity elsewhere. An unsettled local edit carries a one-shot quiet deadline and
closed coverage on its exact capture. The desktop refresh releases that work if captures stop.
When another area needs OCR on the same capture, the next capture releases the independent
area alone; reaching the deadline first releases both together. Each capture supplies at most
one observation, including on a static window. `ScanPlanner` evaluates partitions of at most four areas using provider
costs and bounded geometry/confidence hints. Unknown/weak/small/vertical cases retain the baseline.
Changed plans need savings and glyph/coverage safeguards. Preservation requires a one-to-one matching
covering expected lines, using geometry/orientation/confidence without old-text equality. One oversized
detection cannot prove several lines survived. If expected lines are not preserved, discard
the result and scan original demand on the same frame, temporarily reverting to baseline. Full discovery
stays full. Publish only final raw output/actual coverage; reset hints with generation/provider/size.

Empty small crops may receive one background-padding retry without magnifying glyphs. Severed-glyph
recovery requires unambiguous adjacent detections and connecting ink; accept only strong complete
characters and a unique complete assignment preserving unrelated lines, independent of row order.
Otherwise keep the raw original. At most one extra
pass; subtract padding from both box types. CUDA recognition pads to batch four then slices actual
outputs; dummy detections never publish.

For unusually wide horizontal lines, retry recognition in overlapping pieces from the original
pixels. Join pieces where adjacent characters agree in text and position. Keep the original read
if any join is uncertain, the result is shorter, or previously observed glyph positions lose
coverage. Recovered text and character boxes belong to one original detection. Bound the extra
work per line and per scan.

On a single regional request the OCR worker can read an edited character directly from a
previously detected horizontal line. It compares every pixel of the requested area with the
previous covered OCR source, requires an isolated change within one character, and verifies
the new reading and confidence. Readers without character boxes use this only for numeric
fields. A successful read reports just its sampled line as coverage. Uncertain results,
changed surroundings, growth and subsequent reads after a direct read use the detector.

Tracking is deterministic and bounded (128 tracks, 512 candidate regions). Before assignment,
the matching worker keeps the highest-confidence short horizontal reading when competing
detections occupy the same ink bounds. At a common right edge and row center, a wider read
may include a leading icon or substitute its symbol for the label's first letter. A matching
ending and space for the icon link these crops. The tighter reading wins when both are present.
Across scans, a more confident text-only crop can correct a stationary icon-inclusive reading.
A later, taller crop with a weak or symbolic prefix outside the established word can retain
that reading through background changes. Confirmed visual loss allows ordinary edits.
Raw OCR rows retain their original indices and content.
Existing matches precede new admission; truncated input cannot vote for absence.
Spatially gated bipartite components use optimal
assignment with explicit unmatched costs, stable IDs and canonical tie ordering. Distant identical labels
cannot match on text alone. Candidate scores use normalized bit-parallel LCS/Indel with per-source masks
reused only within that matching pass. Bounded edit distance identifies nearby spelling variants;
short choices, numerical changes and typewriter growth have separate confirmation rules.
Confidence belongs to live content, not a screen-position cache.

The visual worker prepares immutable crop digests and geometric/text match scores for the exact OCR
source and an immutable snapshot of prior track identities, text and observed bounds. Results carry
source sequence/time/geometry and are discarded after worker invalidation. `LiveSession` rejects stale
matching state before reconciliation; assignment, confirmation, retention and acknowledgement remain
serialized on the application owner. After adopting evidence, reconciliation advances static settling
once and publishes an immutable tracking snapshot. Visual submission, grouping, history and display
consume that same revision. Reading a snapshot never advances the tracker. No pixel hashing or text-pair scoring runs on Qt. Direct policy
replays can calculate scores inline. Preparation caches only one bounded result, including on visual
pattern-analysis failure. At most 1,024 proposed crops are hashed, each capped at 65,536 pixels; the
application consumes digests under the original one-million-pixel observation budget in the original
query order. Unused crop proposals cannot spend missed-track retention evidence.

Observed and committed text/geometry are separate. Worker delay, skipped frames and repeated polls are
not absence evidence. Bounded crop digests retain text through same-pixel misses; changed pixels alone
never remove text. Negative OCR needs coverage and elapsed evidence. Source-time settling works on static
WGC frames. New content/reset cannot inherit confidence. Accepted unchanged glyphs retain identity,
geometry and English. For a matched line with a minor spelling or bracket difference, only a
better OCR score can advance the reading. Repeated weaker or unscored variants cannot win by
waiting. Strong visual loss of the old pattern on the OCR source allows a similar new line to
start its own confidence baseline. Larger disagreements on visually unchanged glyphs have a
bounded low-confidence review.
A very weak divergent reading at an established label needs another matching OCR observation
before it can resize or replace the label. A quiet image cannot confirm its spelling.
`EnglishPresentation` owns the painted scene. Each publication keeps its exact translated source
and follows its live member bounds through edits. Automatic splits and merges swap together when
the replacement translations for their shared members are ready. Independent boxes can update
immediately. Retired members, model changes, source resets and explicit grouping changes clear
the affected publications. A visually concealed publication can return for the same live content
once all of its dismissed members have returned.
When its track IDs acquire different text, only a ready translation can restore English there.

For a known standalone menu word, the visual worker can retain its exact native pixels with
two text-heights of surrounding context. It keeps at most eight patches of 65,536 pixels each.
Only a confident covered reading paired with a published single-label translation can paint
a preview after retirement. The latest capture must match the patch at its original location
and DPI. A preview is a display-only block within the same source, target and backend generation.
Normal OCR discovery continues and the preview never contributes a track, request or history turn.

Tracking, crop hashing and visual references share the cleaned detector rows for an exact OCR
observation. The row cache keeps detector order for track assignments and releases the source
frame independently of the derived evidence.
Tracker evidence has independent optional records for pending text, pending geometry, crop presence
and completed absence. Each record owns its associated counts, time or source bounds; clearing a
candidate clears that evidence together. Accepted display and confidence review state survive pending
edits. Partial scans affect only covered tracks; delayed confirmation still validates each track's
own source sequence and reading. These records do not change confirmation or retirement policy.

## Translation, grouping and layout

Each source selection has an immutable language policy. The catalog in
`languages.selection` registers Japanese and the 34 sources listed for Paddle’s PP-OCRv5 Latin recognizer. OCR evidence preparation, tracking, dialogue
grouping, translation prompts and cache keys use that selection. Blocks carry it so reconstructed
text, dialogue history and publication use the same word boundaries and script checks. The base
policy accepts alphabetic text and preserves word spaces. The 34 Latin sources use
`PaddleOCRLanguage` for inline quoted terms and closing quotes misread as apostrophes.
A source with additional text rules can register a subclass in `LATIN_SOURCES` and inherit
the same OCR model selection. Japanese supplies script and punctuation
cues, OCR cleanup, and evidence for quoted speech, equipment text and speaker names. The grouper
combines that evidence with paragraph position, row geometry and competing choices. The reader and
grouper use the same list syntax, with different evidence thresholds for possible and marked choices.
Pixel measurements, ambiguity checks, settling and request limits belong to the shared pipeline.
An isolated horizontal sentence may use a shorter dispatch delay when a confident first reading
appeared across a previously blank panel between consecutive captures. The ending must be
unambiguous for the source language. A separate capture at least 60 ms later must preserve
the exact glyphs and surrounding continuation space. Wrapped rows, typewriter growth, ambiguous
periods and closing quotes alone use ordinary confirmation. With grouping enabled, a revealing
or single-row dialogue clause ending in a comma allows a 1.5-second quiet period for a
continuation. This accommodates OCR that reads a terminal period as a comma. Unpunctuated turns keep their
normal quiet-period fallback.
Changing the source clears dialogue history, translation demand, tracks and visual evidence.
If the same recognizer and device can serve the new language, capture and OCR continue on their
existing workers with a new evidence generation. Both pickers ask for confirmation when active
capture must restart to change OCR model or device. Capture then resumes only on its original
window. An idle or stopping capture stays idle. Model inventory discards previous-source scans.
The translation route reconciles each assignment against the new inventory. Compatible local
backends switch their prompt and cache identity on the inference thread between calls, retaining
their loaded weights. Incompatible lanes release their workers. The selected source determines
both the setup bundle and the OCR factory.
Japanese uses bundled MeikiOCR. The other registered sources share one bundled pair
of PP-OCRv5 mobile ONNX graphs. The Latin reader uses the pinned model's CTC vocabulary and a 2 GiB
CUDA arena limit. The detector keeps its 512 MiB limit. Regional scans receive the full client's
dimensions so Paddle detects at the same scale across scans. Its recognition crops include context
around each line. Paddle reads overlapping fragments on a common baseline together to avoid decoding
the same edge glyph twice. Tracking receives tighter horizontal text boxes and recognition confidence, with
stronger evidence required for isolated short labels that resemble icons. Automatic NVIDIA fallback
reruns the entire observation on CPU and reports it. Output language is a separate saved
preference, with English as the default. Each translation batch freezes the target for its prompt,
cache hash and published result. Changing it clears pending translations and dialogue context while
retaining OCR and loaded models. A completed reader entry keeps its original target even if the same
line remains on screen during a switch. Speaker translations must match the entry's target. Regional
variants have distinct prompt instructions and cache keys. PLaMo accepts only its documented PFML3
output codes. Explicit cloud profile tests capture the selected output language when submitted.
The Translation card owns the output language picker. Before capture, it and Appearance settings show
documented target restrictions from enabled model assignments. An explicit Gemma switch uses the
existing assignment and preparation flow, preserving compatible assignments and category toggles.

`DialogueGrouper` forms bounded stable-line blocks with font-relative alignment/spacing and unambiguous
continuity. Keep speakers, choices, vertical text and unrelated columns separate. When ordinary
grouping finds no successor, a long unfinished row can pair with one tightly centered, shorter
sentence ending. A split word or a quoted term followed by a copula supplies the continuation
evidence. A short centered final word can also close a quotation embedded in a space-separated label.
This pair ends at its second row.
For left-aligned dialogue with a short final row and no terminal punctuation, the matching worker
can measure the left, top and bottom edges of a flat panel in the gutter beside each OCR line.
Uniquely adjacent rows of matching lettering can form one request when those edges agree and the
panel closely fits the text vertically. This evidence works across source languages, including
Japanese particles and space-separated word wraps. Source-language spacing still controls the join.
A sentence can end on a full-width row within a paragraph. A shortened sentence-ending row followed
by a full-width row can separate paragraphs within an established column. A trailing dash remains
continuation. Prose evidence may occur after the opening row. Quoted wraps tolerate hanging
marks/indentation and damaged brackets after a
typographic dash. A long hyphen run before an opening bracket can introduce a quote when a unique
aligned prose row follows. For an omitted dash, the matching worker checks a bounded strip of the
OCR source at the detector or recurring column margin for a thin, text-colored horizontal stroke.
The grouper uses the recovered edge only for that opening and continuation. A supported quote
ends at its outer closing bracket, keeping a button directly below it separate. Requests and
display retain the committed OCR text, member identities and ink bounds. Keep gap/font/ambiguity checks
intact. A quoted term followed by more words can continue an unfinished row on the established column pitch.
On a plain menu, grouping skips the paragraph search when no row supplies textual evidence,
no visual evidence supports a join, and no centered subtitle can fit. An edit rechecks those
conditions. Paragraphs have no separate line-count cap. Work is bounded by the scene's track limit and the
backend's 512-character source budget. Short inline
labels start independent items and allow aligned/hanging continuations; bare choices stay separate. Compact
percentage endings can complete established unfinished prose or a line that explicitly introduces
a value. A colon immediately before a percentage can represent a misread plus in a wrapped
Japanese clause. In a column of wrapped prose, two successive row centers predict the
position of the next row. When two wide lines continue unfinished prose on that pitch and margin,
their ink heights can vary if both are consistent with the preceding body row.
The predicted spacing also supports shorter percentage glyphs and noisy operators at
the body margin. The same spacing can support a hanging labelled opening even when the
final value is unreadable. Group using that evidence and keep the exact OCR text in the request.
Before grouping paragraphs, reconstruct split horizontal rows in reading order. Close fragments
need an unambiguous baseline and support from neighboring prose or a previously confirmed intact
row with matching text, position and a surviving track ID. Recurring column margins and choice
markers separate neighbors. Use reconstructed geometry for layout and retain the original fragments
for readiness, request identity and publication.
The dialogue graph links mutually nearest rows by pitch, overlap, alignment and relative width.
Row ownership defaults to adjacency. A reading path can override those groups only when it covers
every group it touches. Admitted prose segments claim their rows. Unclaimed runs keep their adjacency.
Item markers, short endings and changes in lettering can stop a passage before a button.
A short second row can complete a prose passage when it contains a Japanese comma (`、`)
and ends in an ellipsis (`…`).
Repeated section headings and highlighted effect sentences followed by an activation condition
support independent instructions, including in a two-row crop. The OCR matching worker samples
lettering on the source image. Tracking retains compact samples for the corresponding reading
and location across partial scans. Grouping also works with geometry and text alone.
Ordinary pending OCR cannot repartition committed blocks. Pending growth can reserve a continuation
but cannot become translation source before commitment. Readiness uses source time. Short labels and
confirmed single-line status counters such as `マイレージ425/79710` skip the dialogue delay when
they have stopped growing. Group sizing uses individual line heights, not whole paragraph height.

Requests identify combined text and every member's ID/source/revision. Moving a box cannot retranslate;
geometry is not cache identity. Validate all members at publication/rendering. Splits/edits/removal/reset
reject stale replies. Keep full English in inspector/clipboard on overlay overflow. `EnglishPlacement`
assigns nonintersecting, source-anchored rectangles to ready translations. Live readings precede
menu previews. A text-only menu reading takes precedence over its icon-inclusive alternative.
Otherwise the block with more recognized letters and numbers has first choice, with line count,
compact source bounds, reading order, source text and the stable block key resolving ties.
Each label can rewrap or shift within its available anchors. A bounded local search can move
one earlier winner to another anchor with identical wrapping and font, while preserving
its clearance of reserved source text. Unresolved conflicts omit the lower-priority label for
this scene. A fragment can remain visible until its paragraph has a translation.
Spatial grids index source boxes, default layouts and accepted placements. Only intersecting
rectangles contribute to overlap costs. Edge contact is allowed. A horizontal label whose
on-screen source intersects another accepted horizontal on-screen source can skip all its anchors.
The current scene determines placement independently of arrival order. Qt invalidates the old
and new painted rectangles and retires backdrops for hidden labels.

`TranslationPlan` assigns short and long text independently. Categories are structural/length-based,
not semantic dialogue. Missing models/connections never switch provider or billing route. Unchanged
requests stay pinned to their route. Switches revoke live/frozen English and drain before loading.
Only retained blocks with a configured route may hold old English while awaiting a successor.
Adopt the exact prepared worker; same model in both categories shares one owner. Different local models
may coexist and consume additional RAM/VRAM.

`TranslationSession` owns the inference thread, SQLite and publication. `BackendExecution` owns the
loaded backend from the moment its factory returns, including cleanup if capability discovery fails.
It owns the optional menu decoder, cancellation, source switching and recovery to individual calls
on the same model. `TranslationSchedule` owns bounded desired/retained work, exact source/context
deduplication, cache-first lookup, serial dialogue and fair waiting order. Outstanding sources
reuse their hashed keys and cache answers. Newly admitted
sources are checked even at deferred inference capacity. On a scheduling pass, another SQLite
connection's commit invalidates remembered misses; the owner's own writes update them directly.
Revisions join the tail. The owner cancels an active cloud call when its exact source and frozen
context have no remaining desired or retained box. A retained box can keep an existing call alive
through readiness changes. Only ready boxes initiate inference or receive newly published answers.
Diagnostics report the age of the oldest active translation (including cloud admission waits) and
count obsolete calls cancelled within the active workers. Futures only wake the owner.
Revalidate after cache I/O and before dispatch/publication.
Local llama.cpp translators, including imported GGUF profiles, can prepare a second, bounded
context over the same weights on the first qualifying backlog of distinct short-label misses. It starts independent
sequences. Each label uses the model's normal prompt, sampling settings, stop rules and answer
validation. Slots reset their own sampler for each request. The owner evaluates the common
tokenized prefix once in bounded chunks and copies its sequence state to the independent
slots. Each submission validates its own token prefix. The owner checks for demand changes and publishes completions between decode
steps. Padding uses disposable per-slot history to keep GPU kernel shapes stable. Wider
decoders require more ready work. The active runtime adapter determines capacity.
Sparse work, dialogue and prompts that need more context use the primary decoder. If the optional
context cannot run, the owner releases it and retries outstanding work on the same model.
Known `RequestRejected` failures are bounded to exact request/revision/context; fatal failures stop that
translator while capture/OCR continue. Backend generation also scopes retained English.

Cache keys include weights/tokenizers, runtime, languages, decoding, adapter and frozen context; future
glossaries must extend identity. Serialize SQLite WAL/schema initialization, keeping connections on owners.
Memory hits participate in bounded, batched disk-recency updates, flushed on writes and close. Recency
across owners is approximate until flushing; capacity accounting and eviction are transactional.
Older databases initialize the counter from existing translations once in the schema transaction. Evict
only the indexed oldest excess rows, without scanning an offset on every write. Keep the O(1) memory LRU.
Disk failure preserves files and falls back visibly to bounded memory. English caches are not encrypted;
source text appears only through hashed keys. No routine logging/saving of frames or recognized text.
Settings → Cache shows unique translations and counts by game. A separate indexed relation records
the normalized game hint (or capture title) on successful lookup and insertion. A capture with
no name appears as Unassigned. Game names are local metadata. Clearing a game removes its ownership
and exclusive translations. Clearing the entire cache also removes shared and unassigned entries.
Both actions require confirmation. A miss takes an in-memory scope at the current clear revision.
Hits and writes establish ownership transactionally. Repeated hits for the same game reuse its
membership without a disk change apart from batched recency updates. Eviction retires empty game
names. A bounded journal of clear revisions and game-name digests rejects writes that predate a
relevant clear. A scope older than the journal cannot authorize a disk write. Clearing the entire
cache retires all prior scopes and digests. Maintenance runs off the UI thread.
`DialogueMemory` is two turns/512 characters for one unambiguous panel, separate from the reader. Freeze
and hash context per request; only valid published English seeds it. Menus remain independent; reset on
scene/session/layout/age. Turn identity follows exact members/growth/correction lineage, not similar words
or reused position.

## Experimental dialogue reader

Settings → Experimental → Dialogue history defaults off and remembers opt-in. Off hides the reader/button
and stops collection/classification. Turning translation off also suspends collection and drains any
classifier, preserving the opt-in and existing turns. Opening/closing the reader never alters translation. Its separate selector
offers prepared instruction models, configured cloud models and explicit layout-only mode; never silently
select, download or prepare a model. Text-only semantic judgments remain fallible.

`DialogueHistory` observes ready exact-current blocks/English. `PanelLayout` supplies bounded nearby text
and relative layout: stacked name/title metadata, headed dialogue and response groups/single options.
No screenshots, transcript, game-specific rules or additional name translation. Speakers require current
OCR confidence. Headers are metadata; choices cannot borrow a neighboring panel's speaker. New evidence
may retract only an exact still-live false entry, never rewrite older turns.

`ClassifierSession` has one owner/call, at most 128 candidates and bounded exact-input outcome caching,
including invalid replies. FIFO waiting survives unchanged observations. Identity includes body/speaker/
relevant panel text and structure, not coordinates, IDs, detector order or timestamps. Unchanged panels
are not reclassified. Cached scene analysis still advances readiness/time. Withdrawn active calls may
cache but cannot publish without current demand. Uncertain/failed model answers never fall back to
layout admission. Choice exclusions last only for live content/panel; no permanent text blacklist.

Translation never awaits classification. Disable/switch/stop/reset/shutdown cancels and drains its owner;
import probes wait for it too. Native loading/prefill may finish before cancel. Separate local models
consume resources; cloud checks share selected-profile limits. The reader holds at most 20 bounded turns
in memory, newest first with scroll anchoring and exact copy controls. Clear suppresses unchanged current
turns; growth/corrections update a turn. A new capture or source language starts fresh turn identities. Reader history never enters prompts.

## Ollama API

One user-configured HTTP(S) origin serves translation through Ollama’s native API. Discovery can present the exact tag,
parameter count, quantization and model size. Discovery reads `/api/tags` on request. Save verifies
selected models through `/api/show` and retains their display metadata for the next launch. Requests have deadlines and response limits, preserve TLS
verification, and disable redirects and environment proxies. DanteTL contacts the selected origin only.
Models must support text completion on that server. Model pulls and cloud-backed entries are excluded.

Preferences store the server and selected tags and digests together. Assignments identify the server
and tag. Cache entries also depend on the model digest, game hint, source and target languages,
prompt version, exact source text and any dialogue context sent with it. Display metadata does not
affect reuse. The translation worker verifies the model before opening its cache scope. An outstanding chat can be
cancelled on a scene or route change. A server change requires a new model assignment.

## Cloud and credential boundary

Cloud discovery reports unreadable account records and API credentials and disables their dependent
profiles. Healthy connections remain usable. Cloud model, connection and API credential changes
share a per-catalog interprocess lock from the initial read through commit or rollback.
Each JSON catalog applies the same byte limit on read and before atomic replacement. These limits
accommodate full version-1 catalogs with JSON-escaped Unicode names and model revisions.

Cloud use needs explicit configuration/route selection. Send bounded recognized text/context and optional
game hint, never frames. Editing/testing does not commit routing. Validate complete answers and bounded
streams; preserve per-backend concurrency/attempt/request budgets/deadlines without billing fallback.
API catalog refresh shares one deadline across all pages and can cancel while waiting for headers or
body bytes. Auth exchange/refresh/discovery/rotation waits carry cancellation; successful rotation persists even if
later discovery fails. `CloudServices` publishes discovered models and new accounts together with
the operation's completion on the application thread. Model removal retains
connections; Forget/sign-out are explicit. Catalog failure preserves saved data and never selects a model.
Draft Save commits; Cancel discards edits. Credentials never enter prompts, ordinary JSON, logs or fixtures.
Usage is per saved model for the current app run, shared across backend recreation, translation and
classification. The selected model's label updates independently of configuration revisions. Cache
hits reserve no requests. A bounded FIFO admission queue paces shared dispatch across backend owners;
cancelled or expired waits consume no request allowance or future slot. Count requests at dispatch,
including retries and calls cancelled after dispatch. Token totals count only provider-reported usage
and are not billing totals.

API keys use Windows Credential Manager; OAuth uses user-bound DPAPI/account-bound encrypted envelopes.
OAuth uses the system browser, exclusive random loopback port, PKCE S256, state/nonce and validated
issuer/signature/audience/expiry/identity. Rotation is atomic under an account lock; temporary failure
preserves tokens. Logout clears local tokens even if remote revocation fails and reports the distinction.
Native credential buffers are wiped; Python copy zeroization is not guaranteed. Provider HTTPS verifies
certificates, disables redirects and ignores proxy environment; OAuth callbacks use loopback HTTP. The separate Ollama integration accepts an explicit HTTP(S) origin.
OS protection is a Windows-user boundary, not an application sandbox. Same-user programs, malware,
administrators and memory dumps are outside it. Forget does not revoke provider API keys; request limits
are not monetary limits. Automated tests use disposable fake credentials and transports, never paid calls.

## Models, downloads and native runtimes

Missing models do not stop capture. Inference never downloads; setup is explicit, with pinned hashes/sizes,
byte progress and atomic publication. Valid files survive retry/cancel; bundled repairs use writable state.
Both installers share `download_transport`: one contained child per transfer, parent-enforced 30-minute
deadline across DNS/connection/headers/body and cancellation polling at 100 ms. A worker lock
protects each digest-named partial through transfer, verification and publication. Transient errors
get bounded retries. Later attempts resume from the saved byte offset, including after app exit.
Transfer and checksum reads request Windows background I/O priority. The receiver coalesces network
fragments, flushes its user-space buffer before reporting progress and syncs at the end of a response.
Runtime setup uses uv to create the private interpreter environment. Its bundled pip installs the
pinned local wheels directly into that environment, with Windows background processing priority,
binary-only hash checking, and index, dependency resolution and package caching disabled.
Appending requires a matching HTTP range. A full response replaces the partial. The parent verifies
the complete file against the pinned SHA-256 before atomic replacement.
Bundled restoration and verified cache reuse stay offline without a downloader child.

`OwnedProcess` creates suspended, assigns its Windows job, then resumes. Never launch uncontained.
Cancellation/deadlines drain descendants; output is bounded. Windowed download helpers attach their
inherited output pipe; the application never redirects stdout. `llama_runtime` serializes DLL setup,
import and legacy plugin registration, caching only successful initialization. Failure may retry.
Loading stays on model workers/probes; preserve old fork APIs and current upstream PLaMo support.

CPU/CUDA/Vulkan engines are complete upstream packages; CPU uses upstream PEP 517 build and standard
DLL repair at release time, no player compiler. Never mix CPU/GPU ONNX or different llama wheels.
NVIDIA OCR/Gemma share pinned CUDA 13 runtime/cuBLAS; preserve legacy ONNX 1.26/CUDA 12 loading. Select
acceleration from local driver capabilities. Match devices to DXGI by Windows LUID. Query the
visible CUDA device zero for compute capability 7.5+ and a CUDA 13+ driver. Vulkan requires a
hardware GPU with Vulkan 1.2, a compute queue and 16-bit storage buffers. Run each driver query
in a bounded child. Recommend the supported adapter with the most dedicated memory, preferring
CUDA when it qualifies, then Vulkan, then CPU. Recheck before GPU installation and managed
runtime handoff. The isolated installer probe checks numerical output on the selected llama GPU
backend and, for NVIDIA, the CUDA OCR provider before accepting the environment. Model preparation
still checks the actual weights. Select DLLs for the actual OCR build.
Strict GPU checks fail on fallback;
automatic OCR reports fallback. Separate compilation from warm inference and keep the bounded CUDA cache.
In-app AMD is Vulkan translation plus CPU OCR. Experimental DirectML retains its numerical compatibility
probe; successful provider initialization is not validation.

Imported GGUFs use a separate bounded adapter/profile and explicit review/register. Inspect/hash on a
worker, test in a contained child after translator/classifier drain. Require actual GPU offload, complete
outputs, supported templates and empty reasoning. Cache identity includes content/profile/template/device/
decoding; recheck files on load. Never delete user-owned model files. Curated models reuse the adapter
without import registration. Preserve built-in prompts and distinct IDs; separate download size from RAM.

## Packaging, updates and persistence

The main window's About button shows the installed version and submits update commands.
On portable startup a bounded worker checks the public GitHub release metadata. AppData stores
the latest checked version and the offered version, choice and reminder deadline under a
cross-process lock. No skips one version. Ask me later allows the same version after seven days.
Manual checks show the prompt for any newer stable release, including a skipped or snoozed version.
Source checkouts support manual discovery. When the running copy lacks the publisher key or the
release lacks signed assets, the prompt opens the release page for a manual download.
Only an explicit update downloads the signed manifest and ZIP. The updater verifies them with
the installed publisher key, bounds extraction and stages a sibling folder. A separate helper
waits for shutdown, validates the old revision and swaps a dedicated installation. It restores
the previous folder if the replacement GUI fails to acknowledge its version. Settings, runtime
receipts and models stay in per-user state. See [release signing](building.md#github-actions-and-releases).

PyInstaller owns the folder release/EXE; uv/Hatchling own wheel/source archives. See [building](building.md).
Bundle verified MeikiOCR and PaddleOCR assets/notices, CPU ONNX/llama and the app wheel, never translation
weights, preferences/caches or development/GPU environments. `resource_root()` is read-only; `state_root()`
is per-user AppData in releases, project-local in source. Optional environments use AppData-managed Python
independent of movable bundle resources. Validate the selected environment's own receipt, not latest-prepared.
Dependency identity compares exact wheel hashes/Python version independently of app version.

App updates install the verified bundled wheel offline without resolution into immutable hash-specific
AppData directories. Lock, probe and atomically publish the pointer. CPU/CUDA/Vulkan share app code while
reusing dependencies. Old running instances/rollback retain old code; never overwrite loaded files.
App preparation failure must not trigger dependency repair. Actual dependency changes need explicit
isolated setup preserving models/selection. Source remains editable; no duplicated source/custom bootstrap.

Setup and launch share `clean_runtime_environment`: clear all inherited `PYTHON*`, `UV_*`, `VIRTUAL_ENV`,
then add intended values. Managed helpers use `-P -s -B -m` with the selected app/resource/state paths.
Frozen helpers dispatch through the EXE's `--module`. Owned external launches temporarily clear the frozen
DLL directory under serialization. Restart waits for new Qt acknowledgement from the actual GUI PID,
not the Windows venv launcher. Normal launch honors saved selection and retains CPU fallback. Never bind
managed runtimes to the bundled interpreter or repair loaded native libraries.
