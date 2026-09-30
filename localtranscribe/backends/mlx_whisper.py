"""Apple Silicon backend: Whisper through `mlx-audio` (MLX, Metal GPU). Only imported on an Apple Silicon Mac.

Checked against mlx-audio 0.5.7 (mlx_audio/stt/models/whisper/whisper.py):
  * `mlx_audio.stt.load(repo)` returns the Whisper `Model`; its `generate(audio, *, language, task, temperature,
    initial_prompt, return_timestamps, condition_on_previous_text, ...)` takes a numpy array and returns an object
    with `.text`. `language` must be a Whisper CODE ("it"), not a name.
  * A scalar `temperature=0.0` means greedy with no fallback ladder; the three thresholds are switched off too, so
    a piece is never re-decoded or dropped as "silence".
  * The audio is padded to Whisper's 30 s window; our pieces (about 20 s) fit in one window.
  * The tokenizer comes from a Hugging Face WhisperProcessor loaded from the model folder: repos without tokenizer
    files (the older mlx-examples "weights.npz" and the bare "multilingual.tiktoken" ones) fail at generate().
    The `mlx-community/whisper-*-asr-*` repos carry them.
  * `generate` decodes one audio at a time (no batching), so batch_size is always 1.
"""
from typing import List, Optional

import numpy as np

from ..textutil import detect_and_fix_repetitions, float_range_normalize
from .base import Backend
from .mlx_qwen import metal_available


def language_code(language):
    """"Italian" / "it" -> "it"; None stays None (Whisper detects the language per piece)."""
    if language is None:
        return None
    from mlx_audio.stt.models.whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

    key = language.strip().lower()
    code = key if key in LANGUAGES else TO_LANGUAGE_CODE.get(key)
    if code is None:
        raise ValueError(f"Whisper does not know the language {language!r}")
    return code


class MlxWhisperBackend(Backend):
    name = "mlx-whisper"

    def __init__(self, repo, device="mps", batch_size=1):
        import mlx.core as mx
        from mlx_audio.stt import load

        self.mx = mx
        self.repo = repo
        self.batch_size = 1  # ponytail: generate() is single-audio; batching would mean calling decode() on a stacked mel
        self.dtype = next((t for t in ("4bit", "5bit", "6bit", "8bit", "fp16", "q4") if t in repo.lower()), "mlx")
        self.notes = []
        if device != "cpu" and metal_available(mx):
            mx.set_default_device(mx.gpu)
            self.device = "mps"
        else:
            mx.set_default_device(mx.cpu)
            self.device = "cpu"
            if device != "cpu":
                self.notes.append("Metal GPU not available here: MLX runs on the CPU (slow).")
        self.model = load(repo)

    def transcribe(self, pieces: List[np.ndarray], language: Optional[str], context: str = "") -> List[str]:
        code = language_code(language)
        out = []
        for piece in pieces:
            res = self.model.generate(
                float_range_normalize(piece),
                language=code,
                task="transcribe",
                temperature=0.0,
                compression_ratio_threshold=None,
                logprob_threshold=None,
                no_speech_threshold=None,
                condition_on_previous_text=False,
                return_timestamps=False,
                initial_prompt=context or None,
                verbose=None,
            )
            out.append(detect_and_fix_repetitions((res.text or "").strip()))
            self.mx.clear_cache()
        return out

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
