"""Normal launcher respects the app's verified runtime selection; CPU remains a fallback."""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dantetl.runtime.restart import RuntimeRestart  # noqa: E402
from dantetl.runtime.setup import runtime_selection  # noqa: E402


def main() -> None:
    force_cpu = "--cpu" in sys.argv[1:]
    selection = None if force_cpu else runtime_selection(ROOT)
    if selection and (target := selection.target):
        try:
            pending = RuntimeRestart(ROOT, target, gpu=selection.gpu)
            while not pending.poll():
                time.sleep(0.1)
            return
        except (OSError, ValueError, RuntimeError) as error:
            os.environ["DANTETL_STARTUP_NOTICE"] = f"Opened the fallback runtime. {error}"
    python = ROOT / ".venv/Scripts/pythonw.exe"
    # Preserve existing users' verified CUDA launcher until they choose an in-app runtime.
    if (
        not force_cpu
        and not (ROOT / "data/runtime.json").exists()
        and (ROOT / ".venv-gpu/ocr-verified").is_file()
    ):
        candidate = ROOT / ".venv-gpu/Scripts/pythonw.exe"
        if candidate.is_file():
            python = candidate
    subprocess.Popen([str(python), "-m", "dantetl"], cwd=ROOT)


if __name__ == "__main__":
    main()
