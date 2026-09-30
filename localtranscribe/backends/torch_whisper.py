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

# Silence prepended to every piece before the feature extractor. Whisper often outputs only "Grazie" / "Grazie a tutti"
# for pieces whose speech starts at sample 0 (VoxPopuli clips are cut mid-speech, ~17% of them). 0.05 s of zeros fixes it:
# turbo, 1169 non-empty voxpopuli_it clips: WER 29.59 -> 16.47% (206 -> 4 clips that are "Grazie" only, i.e. reference
# >= 10 words and hypothesis <= 3 words); FLEURS Italian tuning set: 3.01 -> 2.85. Chaotic in the pad length (same clips:
# 0.03 s: 16.41%, 5 failing; 0.08 s: 17.35%, 20), so do not tune casually.
LEAD_IN_S = 0.05
# Linear fade in/out of the first/last FADE_S of the ORIGINAL piece (0 = off, the default; the code path is kept).
# Tried with 0.05 s and rejected: on VoxPopuli it left one clip fewer failing (3 instead of 4), but it caused a 334-word
# repetition loop on turbo (vp_pauses vpl_06) and made the long-form sets worse (turbo vp_long WER 7.23 vs 5.94,
# vp_pauses 15.33 vs 10.66; large-v3 vp_long 6.41 vs 5.62, vp_pauses 15.05 vs 12.30, each vs lead-in only).
FADE_S = 0.0


def _prepare(piece, lead_in_s, fade_s):
    """float32 copy of one piece: optional linear fade in/out of its own edges, then `lead_in_s` of zeros in front."""
    x = np.array(piece, dtype=np.float32, copy=True).reshape(-1)
    n_fade = min(int(round(fade_s * config.SAMPLE_RATE)), len(x) // 2)
    if n_fade > 0:
        ramp = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
        x[:n_fade] *= ramp
        x[len(x) - n_fade:] *= ramp[::-1]
    n_lead = int(round(lead_in_s * config.SAMPLE_RATE))
    if n_lead > 0:
        x = np.concatenate([np.zeros(n_lead, dtype=np.float32), x])
    return x


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
        lead_in_s, fade_s = LEAD_IN_S, FADE_S  # read at call time, so experiments can set them on the module
        out: List[str] = []
        for i in range(0, len(pieces), max(1, self.batch_size)):
            batch = [_prepare(p, lead_in_s, fade_s) for p in pieces[i:i + self.batch_size]]
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
