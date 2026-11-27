DanteTL - Windows x64

1. Extract this entire ZIP to a folder. Do not run it inside the ZIP preview.
2. Double-click DanteTL.exe. Keep the _internal folder beside the EXE.
3. MeikiOCR and PaddleOCR are included. All 35 source languages work immediately on CPU.
   Follow the first-run Setup assistant to choose a translation model
   and review optional GPU support. Gemma 4 E2B is the recommended starting choice;
   E4B is the higher-resource recommendation. Other models remain available.
   Nothing downloads until you confirm. You can skip it and reopen
   Settings > Setup assistant, or use the Models and Acceleration tabs directly.
   Models has a Model type menu separating translation from text recognition (OCR).
4. CPU support is included: translation works without installing a GPU runtime.
   For faster processing, optionally use the Acceleration tab to install
   NVIDIA GPU - CUDA or GPU - Vulkan, then use Restart now. No Python installation,
   setup scripts, system CUDA toolkit or administrator rights are required.
   A compatible graphics driver must already be installed.

Runtime setup shows the current stage plus a second bar for the current download,
with real received bytes and a percentage when the total is known. Private Python
preparation uses an activity indicator. Cancel/retry keeps verified files.

CPU: recognition and translation; larger language models may be slower.
NVIDIA GPU - CUDA: accelerates supported recognition and translation engines.
GPU - Vulkan: accelerates supported translation engines; OCR stays on CPU.
Vulkan inference on Radeon still needs testing. Models are selected separately.
Setup checks the installed driver before recommending acceleration. The NVIDIA runtime needs
compute capability 7.5 or newer and driver support for CUDA 13. Older NVIDIA cards can use
Vulkan translation if their driver supports Vulkan 1.2 with compute and 16-bit storage.
NVIDIA OCR and Gemma share CUDA 13 libraries: about 1.29 GB of NVIDIA/ONNX downloads,
plus other app dependencies. Model downloads are separate. If upgrading from the
older dual-CUDA package, install the new runtime in Acceleration and restart once;
your existing models are reused.
Choose the game text language in Translation or Settings > Recognition. Japanese uses the included
MeikiOCR. The 34 Latin-script sources share the included PaddleOCR model.
Switching between these sources keeps capture running.
Changing the OCR model or device during play asks to restart capture on the same game window.
Compatible translation models stay loaded. Changing source clears the current dialogue history.
Translation needs a chosen translator.
Enable Dialogue history in Settings > Experimental, then open Dialogue beside the Settings gear.
The reader keeps the last 20 completed conversation turns, independently
of model context. Close the reader without stopping collection; Clear empties it. A new capture
starts a fresh history. Nothing is written to disk. Lower-screen dialogue with a nearby speaker
label is recognized conservatively; uncertain names show Unknown speaker and ambiguous UI is skipped.
Choose the output language under Translation in the main window. English is the default.
Each original or translated paragraph has a copy icon.
Show border controls the yellow Windows capture indicator without resetting translations.
Windows policy or another capturing app can still require the indicator.
To record or share the translations, use display capture/entire-screen sharing, such as OBS
Display Capture. Game-only/window-only recording can omit the separate overlay window.
Settings > Models lists translation models first, with tested strengths and limitations.
Text recognition (OCR) shows the included reader for the selected game text language.
Expand Manage OCR files when needed. Verify / repair restores
damaged or missing OCR files into AppData, reusing bundled files where possible.
Language models ask before first-use preparation. Local inference works offline after setup.

Optional cloud translation: Settings > Models > Model type > Cloud translation.
The compact list has Add, Edit and Remove. The editor saves model/key changes only
with Save; Cancel discards them. Advanced holds request limits. Test connection
sends a sample without saving the draft and may consume usage or incur API charges.
Key saved is distinct from a successful test. Connection sign-in/out and Forget key
apply immediately to every model using that connection.
Continue with ChatGPT uses eligible plan permissions; Manage usage controls whether
credits may be used. Disable credit use there to test with included allowance only.
OpenAI, DeepSeek, Gemini and Claude API keys use API quota/billing, not a chat subscription.
Gemini choices: 3.5 Flash-Lite, 3.8 Flash, 4 Hydrogen (Google AI Studio API key).
Claude choices: Haiku 5.5 and Sonnet 5.5 (workspace-scoped Claude Console API key).
Save a model, then choose it in the main Translation model picker. Advanced routing
offers separate choices for dialogue and labels. To inspect recognized text, enable
Inspection and diagnostics under Settings → Experimental, then expand it in the main window.
Model dropdowns group local models, ChatGPT plan models and each API provider. Section headings
are not selectable. Use the translation/category switches to turn work off without losing model
choices. Empty or unavailable selections do not automatically switch to another model.
Only text,
the optional game name and bounded dialogue context are sent; never screenshots.
No automatic fallback to API-key billing. Request limits are not dollar spending caps.
API keys use Windows Credential Manager; ChatGPT tokens use user-bound encryption.
Saved connections work across models and app restarts; leave the key blank to reuse
one. Removing a model keeps its connection. Forget key removes a saved API key;
it does not revoke it at the provider. Sign out disconnects a ChatGPT account.
OS protection cannot stop malware running as your user or an administrator.
Real-account cloud testing is still needed.

Downloads and settings are kept per Windows user at:
  %LOCALAPPDATA%\DanteTL
  models\       translation weights and any repaired/downloaded OCR files
  data\         preferences, translation cache, preparation/runtime receipts
  .runtimes\    optional CPU/GPU dependencies
  .applications\ versioned app wheels shared by optional runtimes
  .python\      private Python used by those optional environments
  .uv-cache\    dependency download cache
  logs\         diagnostic logs (no captured game images)

Open About (the info button beside Settings) to check GitHub for a stable release.
The portable app also checks at startup. Yes downloads the signed update and restarts.
Ask me later reminds you after a week. No skips that version until you check manually.
For a manual update, close DanteTL, extract a new version into a new folder, and run its EXE.
Your AppData is reused. App-only updates automatically install the small bundled app
wheel offline and keep your existing CPU/CUDA/Vulkan dependencies. No setup is needed.
Only a change to the required dependencies needs setup under Settings > Acceleration;
the bundled CPU runtime works meanwhile. Existing environments, models and preferences
are preserved. Prior app versions are retained for rollback and running instances.
Settings > Cache > Remove unused runtime files deletes older runtimes and runtime
downloads no longer needed; the selected runtime, the newest one per acceleration
type and all models are kept. Installing a runtime also removes them automatically.
Settings > Local models > Delete model files removes a downloaded translation model.
Moving the extracted app folder does not move or delete your downloads.
Uninstall: first use Forget key / Sign out for saved cloud connections. Deleting
folders does not remove Windows Credential Manager keys or revoke provider access.
Close the app and delete its extracted folder. Optionally delete the
AppData\Local\DanteTL folder too if you want to remove all models and settings.

This is a portable folder, not an installer or a single self-contained EXE.
The executable is unsigned. Only run a package obtained from a source you trust.
Use a borderless/windowed game. Compatibility with a specific game/anti-cheat is not guaranteed.
See THIRD-PARTY.txt and _internal/third-party/meiki for library/model notices and source references.
The release includes the three MeikiOCR and two PaddleOCR model files.
