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

`_single_recorded` is adapted from mlx-audio 0.5.7 (MIT, Copyright (c) 2024 Prince Canuma) and from the mlx-lm
generate_step it includes (MIT, Copyright (c) 2023 Apple Inc.); the license text is in the NOTICE file.
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


class _Recorder:
    """Greedy sampler (same pick as make_sampler(0.0)) that notes, per step, the chosen token's probability
    and the top-k alternatives. Takes (B, V) logits or log-probs. Only small (B,) / (B, k) arrays are kept and
    evaluated right away, so the vocab-sized graphs are freed at once."""

    def __init__(self, mx, k=config.CONFIDENCE_TOP_K):
        self.mx, self.k, self.steps = mx, k, []

    def __call__(self, x):
        mx = self.mx
        lp = x - mx.logsumexp(x, axis=-1, keepdims=True)
        tok = mx.argmax(lp, axis=-1)
        idx = mx.argpartition(-lp, self.k - 1, axis=-1)[..., :self.k]
        val = mx.take_along_axis(lp, idx, axis=-1)
        order = mx.argsort(-val, axis=-1)
        idx, val = mx.take_along_axis(idx, order, axis=-1), mx.take_along_axis(val, order, axis=-1)
        p = mx.exp(mx.take_along_axis(lp, tok[..., None], axis=-1))[..., 0]
        step = (tok, p, idx, mx.exp(val))
        mx.async_eval(*step)
        self.steps.append(step)
        return tok

    def rows(self, counts):
        """Per row, its first counts[i] steps as (ids, p, alt ids, alt p) Python lists."""
        mx = self.mx
        if not self.steps:
            return [([], [], [], []) for _ in counts]
        tok, p, idx, val = (mx.stack(a).tolist() for a in zip(*self.steps))  # (S, B), (S, B), (S, B, k), (S, B, k)
        return [([tok[j][i] for j in range(n)], [p[j][i] for j in range(n)],
                 [idx[j][i] for j in range(n)], [val[j][i] for j in range(n)]) for i, n in enumerate(counts)]


class MlxQwenBackend(Backend):
    name = "mlx"

    def __init__(self, repo, device="mps", batch_size=1, record_confidence=False):
        import mlx.core as mx
        from mlx_audio.stt import load

        self.mx = mx
        self.repo = repo
        self.batch_size = max(1, int(batch_size))
        self.record_confidence = record_confidence
        self.last_records = []  # with record_confidence: one {"aligned", "tokens"} per text of the last transcribe()
        self._recs = []
        self._kept = None  # KV cache etc. of the last recorded piece, for word_continuations (see _single_recorded)
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
        self.last_records = [None] * len(pieces)
        self._kept = None
        order = list(range(len(pieces)))
        step = self.batch_size  # fixed for this call: a fallback inside _run_group may lower self.batch_size
        if step > 1:
            order.sort(key=lambda i: -len(pieces[i]))  # similar lengths together = less padding
        for g in range(0, len(order), step):
            idx = order[g:g + step]
            group = [float_range_normalize(pieces[i]) for i in idx]
            self._recs = []
            for i, text in zip(idx, self._run_group(group, language, context)):
                out[i] = text
            for i, rec in zip(idx, self._recs):
                self.last_records[i] = self._record(rec, out[i], language)
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
        if self.record_confidence:
            return self._single_recorded(piece, language, context)
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

    @staticmethod
    def _key(piece, language, context):
        return len(piece), hash(piece.tobytes()), language, context or ""

    def _single_recorded(self, piece, language, context):
        """Like generate() for one piece: mlx-audio 0.5.7's stream_generate + generate_step copied step for step
        (prefill in chunks, all but the last prompt position first; greedy pick with one step of lookahead; stop at
        EOS or max_tokens), with the recording sampler. Same tokens as stream_generate. The KV cache stays in
        self._kept so that word_continuations can reuse it instead of encoding the audio and prefilling again."""
        mx, m = self.mx, getattr(self.model, "_model", self.model)
        self._kept = None
        feats, mask, n_audio = m._preprocess_audio(piece)
        ids = m._build_prompt(n_audio, language, context or None)
        audio = m.get_audio_features(feats, mask)
        mx.eval(audio)
        del feats, mask
        mx.clear_cache()
        embeds = m._build_inputs_embeds(ids, audio)
        mx.eval(embeds)
        del audio
        mx.clear_cache()
        embeds = embeds[0]
        n_prompt, cache, eos, rec = len(embeds), m.make_cache(), m._eos_token_ids(), _Recorder(mx)
        for i in range(0, n_prompt - 1, 2048):
            m._forward_with_embeds(embeds[i:min(i + 2048, n_prompt - 1)][None], cache=cache)
            mx.eval([c.state for c in cache])
            mx.clear_cache()

        def step(x):  # embeddings (1, T, H) -> the next token, sampled from the log-probs of the last position
            lp = m._forward_with_embeds(x, cache=cache)[:, -1, :]
            return rec(lp - mx.logsumexp(lp, axis=-1, keepdims=True))

        tok, got = step(embeds[-1:][None]), []
        for count in range(config.MAX_NEW_TOKENS):
            nxt = step(m.model.embed_tokens(tok[None]))
            mx.async_eval(nxt)
            t = int(tok.item())
            if t in eos:
                break
            got.append(t)
            if count % 256 == 0:
                mx.clear_cache()
            tok = nxt
        text = self.model._tokenizer.decode(got, skip_special_tokens=True)
        if language is None:
            _lang, text = self.model.extract_language(text)
        self._recs.append(rec.rows([len(got)])[0])
        self._kept = {"key": self._key(piece, language, context), "cache": cache, "n_prompt": n_prompt, "ids": got}
        return detect_and_fix_repetitions((text or "").strip())

    def _batched(self, group, language, context):
        from mlx_audio.lm.sample_utils import make_sampler

        sampler = make_sampler(0.0)  # greedy, like the NVIDIA path
        if self.record_confidence:
            sampler = _Recorder(self.mx)
        texts, gen, _prompt, processed = self.model._generate_chunks_batched(
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
        if self.record_confidence:
            self._recs.extend(sampler.rows(gen))  # only after success: a failed group leaves no records
        return result

    def decode(self, ids):
        return self.model._tokenizer.decode(ids)

    @property
    def eos_ids(self):
        return getattr(self.model, "_model", self.model)._eos_token_ids()

    def word_continuations(self, piece, language, context, requests, max_steps=8):
        """For ONE audio piece: requests = [(prefix_ids, alt_ids)]. Per request and alt, (ids, ps, stop) = the greedy
        tokens (and their probabilities) the model writes after prompt + prefix + [alt], up to the next word (a token
        starting with whitespace), EOS or max_steps; `stop` is that ending token's id. None when max_steps ran out
        (the word is incomplete).
        If the piece is the one _single_recorded just decoded, its KV cache is reused: the prefix is the first tokens
        it generated, so the requests go from the longest prefix to the shortest, trimming the cache back to
        prompt + prefix each time (no audio encoding, no prefill, no prefix feed). Otherwise the prompt and the
        prefixes are fed to a new cache. After each alt the cache is trimmed back to prompt + prefix."""
        # Measured on 30 FLEURS clips (--confidence 59 s vs 40 s plain): encoding the audio + prefilling the prompt
        # cost 12.8 s, feeding the prefixes 4.0 s, forcing the alts 3.6 s; reusing the cache removes the first two.
        # On 100 clips, paired runs: --confidence 1.5-1.9x the plain time before, 1.1-1.4x after.
        mx, m = self.mx, getattr(self.model, "_model", self.model)
        kept, self._kept = self._kept, None
        reuse = (kept is not None and kept["key"] == self._key(float_range_normalize(piece), language, context)
                 and all(list(p) == kept["ids"][:len(p)] for p, _ in requests))
        if reuse:
            cache, n_prompt = kept["cache"], kept["n_prompt"]
            order = sorted(range(len(requests)), key=lambda r: -len(requests[r][0]))
        else:
            feats, mask, n_audio = m._preprocess_audio(float_range_normalize(piece))
            ids = m._build_prompt(n_audio, language, context or None)
            embeds = m._build_inputs_embeds(ids, m.get_audio_features(feats, mask))[0]
            del feats, mask
            cache = m.make_cache()
            m._forward_with_embeds(embeds[None], cache=cache)
            mx.eval([x for c in cache for x in c.state])
            n_prompt = cache[0].offset
            order = sorted(range(len(requests)), key=lambda r: len(requests[r][0]))
        eos = m._eos_token_ids()

        def feed(tokens):  # -> logits of the last position
            return m._forward_with_embeds(m.model.embed_tokens(mx.array([tokens])), cache=cache)[0, -1]

        fed = []  # new cache: prefix tokens currently in the cache after the prompt
        out = [None] * len(requests)
        for r in order:
            prefix, alts = list(requests[r][0]), requests[r][1]
            if reuse:
                for c in cache:
                    c.trim(c.offset - n_prompt - len(prefix))
            else:
                if prefix[:len(fed)] != fed:  # not a longer prefix of the previous one: start again from the prompt
                    for c in cache:
                        c.trim(c.offset - n_prompt)
                    fed = []
                if len(prefix) > len(fed):
                    feed(prefix[len(fed):])
                    fed = prefix
            base = cache[0].offset
            row = []
            for alt in alts:
                got, stop, logits = [], None, feed([alt])
                for step in range(max_steps):
                    lp = logits - mx.logsumexp(logits)
                    t = int(mx.argmax(lp))
                    if t in eos or self.decode([t])[:1].isspace():
                        stop = t
                        break
                    got.append((t, float(mx.exp(lp[t]))))
                    if step < max_steps - 1:
                        logits = feed([t])
                row.append(([t for t, _ in got], [q for _, q in got], stop) if stop is not None else None)
                for c in cache:
                    c.trim(c.offset - base)
            out[r] = row
        mx.clear_cache()
        return out

    def _record(self, rec, text, language):
        """(ids, p, alt ids, alt p) of one piece -> {"aligned", "tokens"} with the token texts."""
        ids, ps, alt_ids, alt_ps = rec
        dec = self.model._tokenizer.decode
        tokens = []
        for t, p, ai, ap in zip(ids, ps, alt_ids, alt_ps):
            alts = [[a, dec([a]), round(q, 5)] for a, q in zip(ai, ap) if a != t]
            tokens.append({"id": t, "text": dec([t]), "p": round(p, 5), "alts": alts})
        full = dec(ids, skip_special_tokens=True)
        if language is None:
            _lang, full = self.model.extract_language(full)
        return {"aligned": detect_and_fix_repetitions((full or "").strip()) == text, "tokens": tokens}

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

        self.model = self._kept = None
        gc.collect()
        self.mx.clear_cache()
