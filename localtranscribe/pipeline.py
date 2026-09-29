"""Cut audio into pieces, run a backend on them, clean and join the text. Backend-independent."""
from typing import Callable, List, Optional, Tuple

import numpy as np

from . import config
from .textutil import fmt_time, split_audio_into_chunks, strip_context_echo

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
            records.extend(dict(r, aligned=r["aligned"] and c == t) for r, t, c in zip(backend.last_records, raw, cleaned))
        if progress:
            progress(len(texts), len(pieces))
    return texts


def transcribe_wav(backend, wav, language, context="", chunk_seconds=config.CHUNK_SECONDS, progress=None,
                   paragraphs=None) -> str:
    """Whole recording -> text, one "[hh:mm:ss] ..." paragraph per ~20 s piece. With `paragraphs` (a list), one
    {"text", "offset", "aligned", "tokens"} per paragraph is appended to it (needs a backend with record_confidence)."""
    records = [] if paragraphs is not None else None
    chunks = split_wav(wav, chunk_seconds)
    texts = transcribe_pieces(backend, [p for p, _ in chunks], language, context, progress, records)
    offsets = [o for _, o in chunks]
    if paragraphs is not None:
        paragraphs.extend(dict(r, text=t, offset=o) for t, o, r in zip(texts, offsets, records) if t)
    return "\n\n".join(f"[{fmt_time(o)}] {t}" for t, o in zip(texts, offsets) if t)
