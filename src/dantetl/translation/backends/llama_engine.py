"""One loaded llama.cpp model with the context, batching and completion policy that every local
translation backend shares. Prompts, stop tokens, sampling and answer checks stay with each backend.
"""

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from dantetl.languages.target import TargetLanguage
from dantetl.runtime.llama import flash_attention_options, menu_sequence_width
from dantetl.translation.backends.label_decoder import LlamaLabelDecoder, Sampling
from dantetl.translation.backends.prompt_budget import fit_prompt
from dantetl.translation.dialogue.context import DialogueTurn

CONTEXT = 2048
OUTPUT_LIMIT = 384


class LlamaEngine:
    """Owns one native model on one device; callers own the thread it runs on."""

    def __init__(
        self,
        runtime: Any,
        model_path: Path,
        device: str,
        *,
        verbose: bool = False,
        bound_batch_threads: bool = True,
    ) -> None:
        # CPU prompt evaluation benefits from batching. On GPUs, short, changing prompts otherwise
        # repeatedly compile shape-specific CUDA kernels on Blackwell with this wheel; single-token
        # prefill keeps one shape and bounded scratch memory, and prefix reuse amortizes it.
        self.batch = 128 if device == "cpu" else 1
        threads = min(4, max(1, (os.cpu_count() or 2) // 2)) if device == "cpu" else 2
        # Imported models keep llama.cpp's default prompt-evaluation threads.
        batch_threads = {"n_threads_batch": threads} if bound_batch_threads else {}
        self.runtime, self.device = runtime, device
        self.model: Any = runtime.Llama(
            model_path=str(model_path),
            n_ctx=CONTEXT,
            n_batch=self.batch,
            n_ubatch=self.batch,
            n_threads=threads,
            **batch_threads,
            n_gpu_layers=0 if device == "cpu" else -1,
            # Keep one model on one adapter, including PCs with a discrete GPU + iGPU.
            split_mode=0,  # LLAMA_SPLIT_MODE_NONE
            main_gpu=0,
            **flash_attention_options(runtime, device),
            verbose=verbose,
        )

    def fit(
        self,
        text: str,
        context: tuple[DialogueTurn, ...],
        render: Callable[[str, tuple[DialogueTurn, ...]], str],
        **messages: str,
    ) -> list[int]:
        """Tokenize a complete source and as much recent dialogue as the context leaves room for."""
        return fit_prompt(
            self.model, text, context, render, max_tokens=CONTEXT - OUTPUT_LIMIT, **messages
        )

    def complete(self, tokens: list[int], stops: Sequence[str], sampling: Sampling) -> Any:
        return self.model.create_completion(
            prompt=tokens,
            max_tokens=OUTPUT_LIMIT,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            seed=sampling.seed,
            repeat_penalty=1.0,
            stop=list(stops),
            stream=False,
        )

    def label_decoder(
        self,
        render: Callable[[str, TargetLanguage], str],
        validate: Callable[[Any], str],
        stops: Sequence[str],
        target: TargetLanguage,
        sampling: Sampling,
    ) -> LlamaLabelDecoder:
        return LlamaLabelDecoder(
            self.runtime,
            self.model,
            menu_sequence_width(self.device),
            render,
            validate,
            stops,
            OUTPUT_LIMIT,
            target,
            sampling,
        )

    def close(self) -> None:
        if self.model is not None:
            self.model.close()
            self.model = None
