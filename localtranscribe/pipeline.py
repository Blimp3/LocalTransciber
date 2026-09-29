"""Cut audio into pieces, run a backend on them, clean and join the text. Backend-independent."""
from typing import Callable, List, Optional

import numpy as np

from . import config
from .textutil import split_audio_into_chunks, strip_context_echo

SR = config.SAMPLE_RATE


def split_wav(wav: np.ndarray, chunk_seconds: float = config.CHUNK_SECONDS) -> List[np.ndarray]:
    """Cut at quiet moments into pieces of about `chunk_seconds` (never more than ~+5 s)."""
    return [p for p, _ in split_audio_into_chunks(wav, SR, max_chunk_sec=chunk_seconds)]


def transcribe_pieces(
    backend,
    pieces: List[np.ndarray],
    language: Optional[str],
    context: str = "",
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[str]:
    """One cleaned text per piece (context echo removed). Runs in slices of a few batches so that
    `progress(done, total)` can be reported; slicing on batch boundaries keeps the batches, and
    therefore the results, the same as one big call."""
    texts: List[str] = []
    step = max(1, backend.batch_size) * 4
    for i in range(0, len(pieces), step):
        raw = backend.transcribe(pieces[i:i + step], language, context)
        texts.extend(strip_context_echo(t, context) for t in raw)
        if progress:
            progress(len(texts), len(pieces))
    return texts


def transcribe_wav(backend, wav, language, context="", chunk_seconds=config.CHUNK_SECONDS, progress=None) -> str:
    """Whole recording -> plain text, one paragraph per ~20 s piece."""
    texts = transcribe_pieces(backend, split_wav(wav, chunk_seconds), language, context, progress)
    return "\n\n".join(t for t in texts if t)
