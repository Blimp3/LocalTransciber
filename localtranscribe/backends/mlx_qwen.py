"""Apple Silicon backend: Qwen3-ASR through `mlx-audio` (MLX, 8-bit weights, Metal GPU).

This module is only ever imported on an Apple Silicon Mac; on Windows/Linux nothing imports mlx.

Everything below was checked against the source of the pinned mlx-audio 0.5.7
(mlx_audio/stt/models/qwen3_asr/qwen3_asr.py):
  * `mlx_audio.stt.load(repo)` returns a `Model` whose attributes delegate to `Qwen3ASRModel`.
  * `Qwen3ASRModel.generate(audio, *, max_tokens, batch_size, language, chunk_duration,
    system_prompt, verbose, ...)` accepts a numpy array. Its own splitter only runs when the
    audio is longer than `chunk_duration`; we always pass a value above the piece length, so
    OUR pieces (identical to the NVIDIA path) are what the model sees, never mlx-audio's.
  * `system_prompt` becomes the chat "system" message, exactly what `--context` is for in
    qwen-asr. The only difference: mlx-audio appends one "\\n" after a non-empty system prompt.
  * With batch_size > 1 we call `_generate_chunks_batched` (private, present in 0.5.7) so that
    several of OUR pieces decode together; any error there drops back to the public API.
"""
import sys
from typing import List, Optional

import numpy as np

from .. import config
from ..textutil import detect_and_fix_repetitions, float_range_normalize
from .base import Backend

SR = config.SAMPLE_RATE


def metal_available(mx):
    try:
        return bool(mx.metal.is_available())
    except Exception:
        try:
            return bool(mx.is_available(mx.gpu))
        except Exception:
            return False


class MlxQwenBackend(Backend):
    name = "mlx"

    def __init__(self, repo, device="mps", batch_size=1):
        import mlx.core as mx
        from mlx_audio.stt import load

        self.mx = mx
        self.repo = repo
        self.batch_size = max(1, int(batch_size))
        self.dtype = next((t for t in ("4bit", "5bit", "6bit", "8bit", "bf16", "fp16") if t in repo.lower()), "mlx")
        self.notes = []

        if device == "cpu":
            mx.set_default_device(mx.cpu)
            self.device = "cpu"
        elif metal_available(mx):
            mx.set_default_device(mx.gpu)
            self.device = "mps"
        else:
            # e.g. a virtual machine without GPU passthrough: MLX still works on the CPU, just slowly.
            mx.set_default_device(mx.cpu)
            self.device = "cpu"
            self.notes.append("Metal GPU not available here: MLX runs on the CPU (slow).")
        self._warned_batch = False
        self.model = load(repo)

    # ------------------------------------------------------------------ transcription

    def transcribe(self, pieces: List[np.ndarray], language: Optional[str], context: str = "") -> List[str]:
        out = [""] * len(pieces)
        order = list(range(len(pieces)))
        step = self.batch_size  # fixed for this call: a fallback inside _run_group may lower self.batch_size
        if step > 1:
            order.sort(key=lambda i: -len(pieces[i]))  # similar lengths together = less padding
        for g in range(0, len(order), step):
            idx = order[g:g + step]
            group = [float_range_normalize(pieces[i]) for i in idx]
            for i, text in zip(idx, self._run_group(group, language, context)):
                out[i] = text
            self.mx.clear_cache()
        return out

    def _run_group(self, group, language, context):
        if len(group) > 1 and self.batch_size > 1:
            try:
                return self._batched(group, language, context)
            except Exception as e:  # API drift or memory pressure: fall back to one piece at a time
                if not self._warned_batch:
                    print(f"  [note] batched decoding failed ({type(e).__name__}: {e}); "
                          "continuing one piece at a time.", file=sys.stderr, flush=True)
                    self._warned_batch = True
                self.batch_size = 1
                self.mx.clear_cache()
        return [self._single(p, language, context) for p in group]

    def _single(self, piece, language, context):
        res = self.model.generate(
            piece,
            language=language,
            system_prompt=context or None,
            chunk_duration=len(piece) / SR + 1.0,  # above the piece length: mlx-audio never re-splits it
            max_tokens=config.MAX_NEW_TOKENS,
            batch_size=1,
            verbose=False,
        )
        return detect_and_fix_repetitions((res.text or "").strip())

    def _batched(self, group, language, context):
        from mlx_audio.lm.sample_utils import make_sampler

        sampler = make_sampler(0.0)  # greedy, like the NVIDIA path
        texts, _gen, _prompt, processed = self.model._generate_chunks_batched(
            [(p, 0.0) for p in group],
            max_tokens=config.MAX_NEW_TOKENS * len(group),
            sampler=sampler,
            logits_processors=None,
            language=language,
            system_prompt=context or None,
            batch_size=len(group),
            verbose=False,
        )
        result = []
        for text, done in zip(texts, processed):
            if not done:
                raise RuntimeError("a piece was not decoded")
            if language is None:
                _lang, text = self.model.extract_language(text)
            result.append(detect_and_fix_repetitions((text or "").strip()))
        return result

    # ------------------------------------------------------------------ memory

    def reset_peak_memory(self):
        f = getattr(self.mx, "reset_peak_memory", None) or getattr(self.mx.metal, "reset_peak_memory", None)
        if f:
            f()

    def peak_memory_gb(self):
        f = getattr(self.mx, "get_peak_memory", None) or getattr(self.mx.metal, "get_peak_memory", None)
        return f() / 1024 ** 3 if f else None

    def close(self):
        import gc

        self.model = None
        gc.collect()
        self.mx.clear_cache()
