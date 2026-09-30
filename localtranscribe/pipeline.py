"""Cut audio into pieces, run a backend on them, clean and join the text. Backend-independent."""
from typing import Callable, List, Optional, Tuple

import numpy as np

from . import config
from .textutil import fmt_time, group_words, is_silent, split_audio_into_chunks, strip_context_echo

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
    partial: Optional[Callable[[List[str]], None]] = None,
    skipped: Optional[list] = None,
) -> List[str]:
    """One cleaned text per piece (context echo removed). Runs in slices of a few batches so that
    `progress(done, total)` can be reported; slicing on batch boundaries keeps the batches the same as one big
    call, except that a skipped piece leaves a smaller batch in its slice. With `records` (a list) and a backend that recorded
    confidence, one {"aligned", "tokens"} per returned text is appended to it. `partial(texts_so_far)` is called after
    each slice. A piece with nothing audible (config.SILENCE_DBFS) never reaches the backend: its text is "", its record
    {"aligned": False, "tokens": []}, and its length in seconds is appended to `skipped` (a list), if given."""
    texts: List[str] = []
    step = max(1, backend.batch_size) * 4
    dbfs = config.SILENCE_DBFS  # read at call time: bench.py --no-silence-gate and the tests set it to None
    for i in range(0, len(pieces), step):
        group = pieces[i:i + step]
        keep = [k for k, p in enumerate(group) if dbfs is None or not is_silent(p, SR, dbfs)]
        raw = [""] * len(group)
        recs = [{"aligned": False, "tokens": []} for _ in group]
        if keep:  # a backend is never called with no pieces
            for k, t in zip(keep, backend.transcribe([group[k] for k in keep], language, context)):
                raw[k] = t
            if records is not None:
                for k, r in zip(keep, backend.last_records):
                    recs[k] = r
        if skipped is not None:
            skipped.extend(len(p) / SR for k, p in enumerate(group) if k not in keep)
        cleaned = [strip_context_echo(t, context) for t in raw]
        texts.extend(cleaned)
        if records is not None:
            records.extend(dict(r, aligned=r["aligned"] and c == t) for r, t, c in zip(recs, raw, cleaned))
        if partial:
            partial(texts)
        if progress:
            progress(len(texts), len(pieces))
    return texts


def say_skipped(skipped):
    """One note for the pieces transcribe_pieces left out as silence."""
    if skipped:
        n = len(skipped)
        print(f"  [note] {sum(skipped):.0f} s had no sound at all ({n} piece{'s' if n > 1 else ''} below "
              f"{config.SILENCE_DBFS} dBFS) and were not transcribed.")


def add_candidates(backend, piece, language, context, para):
    """Give each unsure word of `para` "cands": whole-word alternatives from the speech model. Its weakest piece is
    swapped for each of its alternatives and the model finishes the word. [{"w", "p"}], best first. A candidate is
    kept only if the model, after it, goes on with the original next token (or ends, for the last word): a one-word
    replacement that fits the rest of the sentence. "p" is like the word's own p: the product of all its pieces
    (the unchanged pieces before the swapped one, the alternative, the continuation)."""
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
            pre = float(np.prod([toks[n]["p"] for n in range(a, j)]))
            meta.append((w, [t["id"] for t in toks[a:j]], alts, b, pre))
    if not reqs:
        return
    for (w, head, alts, b, pre), row in zip(meta, backend.word_continuations(piece, language, context, reqs)):
        best = {}
        for (_id, _text, p), cont in zip(alts, row):
            if cont is None or (cont[2] != toks[b + 1]["id"] if b + 1 < len(toks) else cont[2] not in backend.eos_ids):
                continue
            word = backend.decode(head + [_id] + cont[0]).strip()  # together: a letter can span two pieces
            p = round(p * float(np.prod(cont[1])) * pre, 5)
            if word and "\ufffd" not in word and word.split() == [word] and word != w["w"] and p > best.get(word, 0):
                best[word] = p
        if best:
            w["cands"] = [{"w": t, "p": p} for t, p in sorted(best.items(), key=lambda x: -x[1])]


def _join_paragraphs(texts, chunks):
    return "\n\n".join(f"[{fmt_time(o)}] {t}" for t, (_, o) in zip(texts, chunks) if t)


def transcribe_wav(backend, wav, language, context="", chunk_seconds=config.CHUNK_SECONDS, progress=None,
                   paragraphs=None, corrector=None, partial=None) -> str:
    """Whole recording -> text, one "[hh:mm:ss] ..." paragraph per ~20 s piece. With `paragraphs` (a list), one
    {"text", "offset", "aligned", "tokens", "words"} per paragraph is appended to it (needs a backend with
    record_confidence). "words" is one {"w", "p", "i": [first, last piece], "unsure": True if p is low} per word of
    `text.split()`, or None when the pieces do not line up with the text. Unsure words also get "cands"
    (see add_candidates) when the backend has word_continuations. `corrector(para)`, if given, then adds "suggest"
    to words (correct.Corrector); it never changes the text. `partial(text_so_far)` is called after each slice."""
    records = [] if paragraphs is not None else None
    skipped = []
    chunks = split_wav(wav, chunk_seconds)
    cands_ok = True

    def add_paragraph(piece, t, o, r):
        nonlocal cands_ok
        if t:
            words = group_words(r["tokens"], t) if r["aligned"] else None
            for w in words or []:
                if w["p"] < config.CONFIDENCE_UNSURE:
                    w["unsure"] = True
            para = dict(r, text=t, offset=o, words=words)
            if words and hasattr(backend, "word_continuations"):
                if cands_ok:
                    try:
                        add_candidates(backend, piece, language, context, para)
                    except Exception as e:  # the transcript is worth more than the alternatives
                        cands_ok = False
                        print(f"  [note] alternative words unavailable ({type(e).__name__}: {e}); "
                              "continuing without suggestions.")
                if corrector:
                    corrector(para)
            paragraphs.append(para)

    if paragraphs is not None and backend.batch_size == 1 and hasattr(backend, "word_continuations"):
        # One piece at a time: the backend keeps the KV cache of the piece it just decoded, and add_candidates
        # reuses it (a later piece would replace it).
        texts = []
        for piece, o in chunks:
            texts += transcribe_pieces(backend, [piece], language, context, records=records, skipped=skipped)
            if partial:
                partial(_join_paragraphs(texts, chunks))
            add_paragraph(piece, texts[-1], o, records[-1])
            if progress:
                progress(len(texts), len(chunks))
    else:
        texts = transcribe_pieces(backend, [p for p, _ in chunks], language, context, progress, records,
                                  (lambda ts: partial(_join_paragraphs(ts, chunks))) if partial else None,
                                  skipped=skipped)
        if paragraphs is not None:
            for (piece, o), t, r in zip(chunks, texts, records):
                add_paragraph(piece, t, o, r)
    say_skipped(skipped)
    return _join_paragraphs(texts, chunks)
