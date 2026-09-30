"""NVIDIA GPU (float16) and CPU (float32) backend: OpenAI Whisper through Hugging Face transformers.

Used for the Whisper vs Qwen3-ASR comparison. Pieces are our ~20 s cuts (< 30 s), so Whisper's normal
single-window mode is used, never its long-form mode. Decoding is greedy, without timestamps, and always
"transcribe" (never "translate").
"""
from typing import List, Optional

import numpy as np

from .. import config
from ..textutil import detect_and_fix_repetitions
from .base import Backend

# Whisper's decoder has 448 positions in total: forced start tokens + optional prompt + generated text.
DECODER_POSITIONS = 448
_START_TOKENS_MARGIN = 5      # <|startoftranscript|><|lang|><|transcribe|><|notimestamps|> + one spare
_MAX_PROMPT_TOKENS = 200      # keep the newest tokens of a long context (Whisper itself allows 223)
_WINDOW_TOLERANCE = 8         # samples allowed above 30 s (rounding in the chunker)


def language_code(language: Optional[str]) -> Optional[str]:
    """English language name ("Italian") -> Whisper code ("it"); None stays None (auto-detect).
    Uses transformers' own tables. Unknown names raise ValueError."""
    if language is None:
        return None
    from transformers.models.whisper.tokenization_whisper import LANGUAGES, TO_LANGUAGE_CODE

    key = str(language).strip().lower()
    if key in TO_LANGUAGE_CODE:
        return TO_LANGUAGE_CODE[key]
    if key in LANGUAGES:  # already a code such as "it"
        return key
    raise ValueError(f"Unknown language {language!r} for Whisper (use an English name such as 'Italian', or None to auto-detect).")


class TorchWhisperBackend(Backend):
    name = "torch-whisper"

    def __init__(self, repo, device="cuda", batch_size=8):
        import torch
        import transformers

        self.repo = repo
        self.device = device
        self.batch_size = batch_size
        self._torch = torch
        # float16, not bfloat16: Turing GPUs have no bf16.
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.dtype = str(dtype).replace("torch.", "")
        self._torch_dtype = dtype
        self._device_str = "cuda:0" if device == "cuda" else "cpu"
        self.processor = transformers.WhisperProcessor.from_pretrained(repo)
        kwargs = dict(dtype=dtype, low_cpu_mem_usage=True)
        try:
            model = transformers.WhisperForConditionalGeneration.from_pretrained(repo, attn_implementation="sdpa", **kwargs)
        except (ValueError, ImportError, NotImplementedError):  # sdpa not supported by this build
            model = transformers.WhisperForConditionalGeneration.from_pretrained(repo, **kwargs)
        self.model = model.to(self._device_str).eval()

    def _prompt_ids(self, context):
        context = (context or "").strip()
        if not context:
            return None
        ids = self.processor.get_prompt_ids(context)  # <|startofprev|> + " " + text tokens
        ids = np.asarray(ids).reshape(-1)
        if len(ids) > _MAX_PROMPT_TOKENS:
            ids = np.concatenate([ids[:1], ids[-(_MAX_PROMPT_TOKENS - 1):]])
        return ids

    @staticmethod
    def _strip_prompt(row, prompt):
        """generate() may return the prompt at the start of each row; remove it so it never reaches the text."""
        row = list(row)
        if prompt is None:
            return row
        p = [int(x) for x in prompt]
        for cand in (p, p[1:]):  # with and without <|startofprev|>
            if cand and row[:len(cand)] == cand:
                return row[len(cand):]
        return row

    def transcribe(self, pieces: List[np.ndarray], language: Optional[str], context: str = "") -> List[str]:
        code = language_code(language)  # validate first, even for empty input
        if not pieces:
            return []
        max_samples = 30 * config.SAMPLE_RATE + _WINDOW_TOLERANCE
        for idx, p in enumerate(pieces):
            n = len(p)
            if n > max_samples:
                raise ValueError(
                    f"Piece {idx} is {n / config.SAMPLE_RATE:.1f} s long ({n} samples), but Whisper's encoder sees at most "
                    f"30 s and would silently drop the rest of the speech. Use --chunk 25 or less.")
        torch = self._torch
        prompt = self._prompt_ids(context)
        n_prompt = 0 if prompt is None else len(prompt)
        max_new = max(1, min(config.MAX_NEW_TOKENS, DECODER_POSITIONS - n_prompt - _START_TOKENS_MARGIN))
        out: List[str] = []
        for i in range(0, len(pieces), max(1, self.batch_size)):
            batch = [np.asarray(p, dtype=np.float32) for p in pieces[i:i + self.batch_size]]
            feats = self.processor.feature_extractor(batch, sampling_rate=config.SAMPLE_RATE,
                                                 return_attention_mask=True, return_tensors="pt")
            gen = dict(
                input_features=feats["input_features"].to(self._device_str, dtype=self._torch_dtype),
                task="transcribe",
                return_timestamps=False,
                num_beams=1,
                do_sample=False,
                max_new_tokens=max_new,
            )
            if "attention_mask" in feats:
                gen["attention_mask"] = feats["attention_mask"].to(self._device_str)
            if code is not None:
                gen["language"] = code
            if prompt is not None:
                gen["prompt_ids"] = torch.as_tensor(prompt, dtype=torch.long, device=self._device_str)
            with torch.inference_mode():
                seqs = self.model.generate(**gen)
            if hasattr(seqs, "sequences"):
                seqs = seqs.sequences
            rows = seqs.tolist() if hasattr(seqs, "tolist") else [list(r) for r in seqs]
            rows = [self._strip_prompt(r, prompt) for r in rows]
            texts = self.processor.batch_decode(rows, skip_special_tokens=True)
            out.extend(detect_and_fix_repetitions((t or "").strip()) for t in texts)
        return out

    def reset_peak_memory(self):
        if self.device == "cuda":
            self._torch.cuda.reset_peak_memory_stats()

    def peak_memory_gb(self):
        if self.device == "cuda":
            return self._torch.cuda.max_memory_allocated() / 1024 ** 3
        return None

    def close(self):
        import gc

        self.model = None
        self.processor = None
        gc.collect()
        if self.device == "cuda":
            self._torch.cuda.empty_cache()
