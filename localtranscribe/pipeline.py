"""Cut audio into pieces, run a backend on them, clean and join the text. Backend-independent."""
from typing import Callable, List, Optional, Tuple

import numpy as np

from . import config
from .textutil import fmt_time, group_words, split_audio_into_chunks, strip_context_echo

SR = config.SAMPLE_RATE


def split_wav(wav: np.ndarray, chunk_seconds: float = config.CHUNK_SECONDS) -> List[Tuple[np.ndarray, float]]:
    """Cut at quiet moments into (piece, start offset in seconds) of about `chunk_seconds` (never more than ~+5 s)."""
    return split_audio_into_chunks(wav, SR, max_chunk_sec=chunk_seconds)


def transcribe_pieces(
    backend,
    pieces: List[np.ndarray],
    language: Optional[str],
    context: str = "",
    progress: Optional[Callable[[int, int], None]] = None,
    records: Optional[list] = None,
) -> List[str]:
    """One cleaned text per piece (context echo removed). Runs in slices of a few batches so that
    `progress(done, total)` can be reported; slicing on batch boundaries keeps the batches, and
    therefore the results, the same as one big call. With `records` (a list) and a backend that recorded
    confidence, one {"aligned", "tokens"} per returned text is appended to it."""
    texts: List[str] = []
    step = max(1, backend.batch_size) * 4
    for i in range(0, len(pieces), step):
        raw = backend.transcribe(pieces[i:i + step], language, context)
        cleaned = [strip_context_echo(t, context) for t in raw]
        texts.extend(cleaned)
        if records is not None:
            records.extend(dict(r, aligned=r["aligned"] and c == t)
                           for r, t, c in zip(backend.last_records, raw, cleaned))
        if progress:
            progress(len(texts), len(pieces))
    return texts


def add_candidates(backend, piece, language, context, para):
    """Give each unsure word of `para` "cands": whole-word alternatives from the speech model. Its weakest piece is
    swapped for each of its alternatives and the model finishes the word. [{"w", "p"}], best first. A candidate is
    kept only if the model, after it, goes on with the original next token (or ends, for the last word): a one-word
    replacement that fits the rest of the sentence."""
    words, toks = para["words"], para["tokens"]
    reqs, meta = [], []
    for k, w in enumerate(words):
        if not w.get("unsure"):
            continue
        a, b = w["i"]
        j = min(range(a, b + 1), key=lambda n: toks[n]["p"])
        alts = [x for x in toks[j]["alts"]
                if x[2] >= config.CONFIDENCE_ALT_MIN and not (x[1].startswith("<|") or "<asr_text>" in x[1])
                and not (j > a and x[1][:1].isspace()) and not (j == a and k > 0 and not x[1][:1].isspace())]
        if alts:
            reqs.append(([t["id"] for t in toks[:j]], [x[0] for x in alts]))
            meta.append((w, "".join(t["text"] for t in toks[a:j]), alts, b))
    if not reqs:
        return
    for (w, head, alts, b), row in zip(meta, backend.word_continuations(piece, language, context, reqs)):
        best = {}
        for (_id, text, p), cont in zip(alts, row):
            if cont is None or (cont[2] != toks[b + 1]["id"] if b + 1 < len(toks) else cont[2] not in backend.eos_ids):
                continue
            word = (head + text + "".join(backend.decode([t]) for t in cont[0])).strip()
            p = round(p * float(np.prod(cont[1])), 5)
            if word and word.split() == [word] and word != w["w"] and p > best.get(word, 0):
                best[word] = p
        if best:
            w["cands"] = [{"w": t, "p": p} for t, p in sorted(best.items(), key=lambda x: -x[1])]


def transcribe_wav(backend, wav, language, context="", chunk_seconds=config.CHUNK_SECONDS, progress=None,
                   paragraphs=None) -> str:
    """Whole recording -> text, one "[hh:mm:ss] ..." paragraph per ~20 s piece. With `paragraphs` (a list), one
    {"text", "offset", "aligned", "tokens", "words"} per paragraph is appended to it (needs a backend with
    record_confidence). "words" is one {"w", "p", "i": [first, last piece], "unsure": True if p is low} per word of
    `text.split()`, or None when the pieces do not line up with the text. Unsure words also get "cands"
    (see add_candidates) when the backend has word_continuations."""
    records = [] if paragraphs is not None else None
    chunks = split_wav(wav, chunk_seconds)
    texts = transcribe_pieces(backend, [p for p, _ in chunks], language, context, progress, records)
    offsets = [o for _, o in chunks]
    if paragraphs is not None:
        for (piece, _), t, o, r in zip(chunks, texts, offsets, records):
            if t:
                words = group_words(r["tokens"], t) if r["aligned"] else None
                for w in words or []:
                    if w["p"] < config.CONFIDENCE_UNSURE:
                        w["unsure"] = True
                para = dict(r, text=t, offset=o, words=words)
                if words and hasattr(backend, "word_continuations"):
                    add_candidates(backend, piece, language, context, para)
                paragraphs.append(para)
    return "\n\n".join(f"[{fmt_time(o)}] {t}" for t, o in zip(texts, offsets) if t)
