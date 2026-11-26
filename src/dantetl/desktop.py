"""Desktop startup, selected runtime handoff and internal helper dispatch."""

import ctypes
import os
import runpy
import sys
import time
import traceback

from dantetl.i18n import tr
from dantetl.paths import state_root
from dantetl.runtime.release_values import ISSUES_URL


def main() -> None:
    if sys.argv[1:2] == ["--module"]:
        module = sys.argv[2]
        if module != "dantetl" and not module.startswith("dantetl."):
            raise ValueError(tr("Only DanteTL modules may be launched by this entry point."))
        sys.argv = [module, *sys.argv[3:]]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return

    from dantetl.runtime.restart import RuntimeRestart
    from dantetl.runtime.setup import runtime_selection

    state = state_root()
    force_cpu = "--cpu" in sys.argv[1:]
    cuda_update = False
    if force_cpu:
        sys.argv.remove("--cpu")
    if not force_cpu:
        selection = runtime_selection(state)
        cuda_update = selection.cuda_update
        if target := selection.target:
            try:
                restart = RuntimeRestart(state, target, gpu=selection.gpu)
                while not restart.poll():
                    time.sleep(0.1)
                return
            except (OSError, RuntimeError, ValueError) as error:
                os.environ["DANTETL_STARTUP_NOTICE"] = (
                    f"The selected runtime could not start. Using CPU fallback. {error}"
                )
        elif cuda_update:
            from dantetl.runtime.hardware import require_profile

            try:
                require_profile("nvidia")
            except (OSError, RuntimeError) as error:
                cuda_update = False
                os.environ["DANTETL_STARTUP_NOTICE"] = str(error)
        elif (state / "data/runtime.json").is_file():
            os.environ["DANTETL_STARTUP_NOTICE"] = (
                "The saved runtime is unavailable for this app version. Using CPU fallback. "
                "Open Settings → Acceleration to set it up again."
            )
    from dantetl.bootstrap import main as app_main

    app_main(cuda_update=cuda_update)


def run() -> None:
    helper = sys.argv[1:2] == ["--module"]
    try:
        main()
    except Exception:
        log = state_root() / "logs/startup-error.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(traceback.format_exc(), encoding="utf-8")
        if not helper:
            ctypes.windll.user32.MessageBoxW(
                None,
                f"DanteTL could not start. Details were saved to:\n{log}\n\n{ISSUES_URL}",
                "DanteTL",
                0x10,
            )
        raise SystemExit(1) from None


if __name__ == "__main__":
    run()
