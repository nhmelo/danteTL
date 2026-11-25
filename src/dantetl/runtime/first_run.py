"""Explicit first-run setup plan, using the existing bounded model/runtime installers."""

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from dantetl.i18n import display, tr
from dantetl.runtime.hardware_values import HardwareStatus
from dantetl.runtime.model_store import ModelBundle, install_model
from dantetl.runtime.setup import install_runtime, ready_runtime
from dantetl.runtime.setup_values import Cancelled, ModelProgress, Progress
from dantetl.runtime.values import RuntimeProfile


@dataclass(frozen=True, slots=True)
class SetupPlan:
    models: tuple[ModelBundle, ...]
    runtime: RuntimeProfile | None


def needs_runtime(profile: RuntimeProfile, hardware: HardwareStatus, *, upgrade: bool) -> bool:
    if upgrade:
        return True
    if profile.key == "nvidia":
        return not (hardware.cuda_ocr and hardware.llama_device == "cuda")
    device = "vulkan" if profile.key == "amd" else "cpu"
    return hardware.cuda_ocr or hardware.llama_device != device


def run_setup(
    plan: SetupPlan, root: Path, progress: Progress, cancel: threading.Event
) -> Path | None:
    def models(report: Progress) -> None:
        for index, model in enumerate(plan.models, 1):
            if cancel.is_set():
                raise Cancelled()
            install_model(
                model,
                lambda value, model=model, index=index: report(
                    replace(
                        value,
                        message=f"{display(model.name)}\n{value.message}",
                        stage=index,
                        stages=len(plan.models),
                    )
                ),
                cancel,
            )

    def runtime(report: Progress) -> Path:
        profile = plan.runtime
        assert profile is not None
        install_runtime(
            root,
            profile,
            lambda value: report(replace(value, message=f"{tr(profile.name)}\n{value.message}")),
            cancel,
        )
        target = ready_runtime(root, profile.key)
        if target is None:
            raise RuntimeError(tr("The installed runtime did not pass verification. Retry setup."))
        return target

    if cancel.is_set():
        raise Cancelled()
    target = None
    if plan.runtime is None:
        models(progress)
    elif not plan.models:
        target = runtime(progress)
    else:
        lock = threading.Lock()
        lanes = [ModelProgress(tr("Models")), ModelProgress(tr(plan.runtime.name))]

        def lane(index: int, work: Callable[[Progress], Path | None]) -> Path | None:
            def report(value: ModelProgress) -> None:
                with lock:
                    lanes[index] = value
                    progress(ModelProgress(parallel=(lanes[0], lanes[1])))

            try:
                result = work(report)
            except Cancelled:
                raise
            except Exception as error:
                report(replace(lanes[index], done=True, error=str(error) or type(error).__name__))
                return None
            report(replace(lanes[index], done=True))
            return result

        # The guide worker owns the model lane. Only the runtime needs another thread.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="guide-runtime") as pool:
            installing = pool.submit(lane, 1, runtime)
            lane(0, models)
            target = installing.result()
        if errors := "\n".join(value.error for value in lanes if value.error):
            raise RuntimeError(errors)
    if cancel.is_set():
        raise Cancelled()
    progress(
        ModelProgress(tr("Setup files are ready. Model preparation happens before first use."))
    )
    return target
