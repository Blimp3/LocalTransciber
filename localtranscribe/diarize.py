"""Speaker diarization for mono recordings: Silero VAD + WavLM speaker embeddings + clustering.

diarize(wav) returns speech runs [(start_s, end_s, speaker_index)]; transcribe_turns() turns them into
[(start_s, speaker_index, text)] by transcribing each run through the backend interface and merging
consecutive runs of the same speaker.

The torch models (Silero VAD, WavLM) are loaded only when this module is used (i.e. with --speakers)
and released as soon as the embeddings are computed, so they do not compete with the speech model
for memory on small machines.
"""
import gc
import sys

import numpy as np

from . import config
from .pipeline import transcribe_pieces
from .textutil import fmt_time, split_audio_into_chunks  # noqa: F401 (fmt_time is re-exported)

SR = config.SAMPLE_RATE
SPK_MODEL = config.SPEAKER_MODEL
FRAME = 0.1        # label resolution (s)
WIN, HOP = 1.5, 0.5  # embedding window and hop (s)
MIN_RUN = 0.6      # shorter speaker runs are absorbed by their neighbours (s)


def _speech_regions(wav):
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    vad = load_silero_vad()
    ts = get_speech_timestamps(torch.from_numpy(wav), vad, sampling_rate=SR,
                               min_silence_duration_ms=200, min_speech_duration_ms=200, speech_pad_ms=50)
    del vad
    return [(t["start"], t["end"]) for t in ts]


def _windows(regions):
    win, hop = int(WIN * SR), int(HOP * SR)
    out = []
    for a, b in regions:
        if b - a <= win:
            out.append((a, b))
            continue
        out.extend((s, s + win) for s in range(a, b - win + 1, hop))
        if out[-1][1] < b:
            out.append((b - win, b))
    return out


def _free_device_memory(device):
    import torch

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _embed(wav, windows, device):
    """x-vector per window. If the GPU path fails (an op missing on MPS, out of memory) it is
    redone on the CPU rather than crashing."""
    try:
        return _embed_on(wav, windows, device)
    except Exception as e:
        if device == "cpu":
            raise
        print(f"  [note] speaker model failed on {device} ({type(e).__name__}); retrying on the CPU.",
              file=sys.stderr, flush=True)
        _free_device_memory(device)
        return _embed_on(wav, windows, "cpu")


def _embed_on(wav, windows, device):
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector

    fe = AutoFeatureExtractor.from_pretrained(SPK_MODEL)
    model = WavLMForXVector.from_pretrained(SPK_MODEL).to(device).eval()
    embs = []
    try:
        for i in range(0, len(windows), 32):
            batch = [wav[a:b] for a, b in windows[i:i + 32]]
            inputs = fe(batch, sampling_rate=SR, return_tensors="pt", padding=True)
            with torch.no_grad():
                e = model(**{k: v.to(device) for k, v in inputs.items()}).embeddings
            embs.append(torch.nn.functional.normalize(e, dim=-1).cpu())
    finally:
        del model
        _free_device_memory(device)
    return torch.cat(embs).numpy()


def _smooth(labels, min_frames):
    """Absorb speech runs shorter than min_frames into the longer neighbouring run (ignores -1 = silence)."""
    labels = labels.copy()
    changed = True
    while changed:
        changed = False
        idx = np.flatnonzero(labels >= 0)
        if len(idx) == 0:
            break
        runs, start = [], idx[0]  # runs over speech frames only (silence gaps don't break a run)
        for p, q in zip(idx, idx[1:]):
            if labels[q] != labels[p]:
                runs.append((start, p))
                start = q
        runs.append((start, idx[-1]))
        for k, (a, b) in enumerate(runs):
            n = np.count_nonzero(labels[a:b + 1] >= 0)
            if n >= min_frames or len(runs) == 1:
                continue
            left = runs[k - 1] if k > 0 else None
            right = runs[k + 1] if k + 1 < len(runs) else None
            nb = max([r for r in (left, right) if r], key=lambda r: r[1] - r[0])
            seg = labels[a:b + 1]
            seg[seg >= 0] = labels[nb[0]]
            changed = True
            break
    return labels


def diarize(wav, n_speakers=2, device="cpu", reference=None):
    """device: 'cuda', 'mps' or 'cpu' for the speaker-embedding model (WavLM is small; CPU works).

    reference: optional [(start_s, end_s, speaker_index)] of known single-speaker stretches. When given,
    each speaker's voice profile is learned from them and every window is classified against the profiles
    (much more reliable on short calls than unsupervised clustering)."""
    from sklearn.cluster import KMeans

    regions = _speech_regions(wav)
    windows = _windows(regions)
    if not windows:
        return []
    if not reference and len(windows) < n_speakers:
        print("[note] too little speech to tell voices apart: everything is labelled as one speaker.", file=sys.stderr)
        return [(a / SR, b / SR, 0) for a, b in regions]
    X = _embed(wav, windows, device)
    if reference:
        cents = []
        for k in range(n_speakers):
            inside = [i for i, (a, b) in enumerate(windows)
                      if any(ra * SR <= a and b <= rb * SR for ra, rb, rk in reference if rk == k)]
            c = X[inside].mean(0)
            cents.append(c / np.linalg.norm(c))
        win_labels = (X @ np.stack(cents).T).argmax(1)
    else:
        win_labels = KMeans(n_speakers, n_init=10, random_state=0).fit_predict(X)

    step = int(FRAME * SR)
    n_frames = len(wav) // step + 1
    votes = np.zeros((n_frames, n_speakers))
    for (a, b), lab in zip(windows, win_labels):
        votes[a // step:(b - 1) // step + 1, lab] += 1
    speech = np.zeros(n_frames, bool)
    for a, b in regions:
        speech[a // step:(b - 1) // step + 1] = True
    labels = np.where(speech & (votes.sum(1) > 0), votes.argmax(1), -1)
    labels = _smooth(labels, int(MIN_RUN / FRAME))

    runs, cur = [], None
    for f, lab in enumerate(labels):
        if lab < 0:
            if cur:
                runs.append(cur)
                cur = None
        elif cur and cur[2] == lab:
            cur[1] = f
        else:
            if cur:
                runs.append(cur)
            cur = [f, f, lab]
    if cur:
        runs.append(cur)
    return [(a * FRAME, (b + 1) * FRAME, int(lab)) for a, b, lab in runs]


def transcribe_turns(backend, wav, runs, language, context="", max_piece=20.0, pad=0.15, progress=None):
    """Transcribe each run through the backend, then merge consecutive runs of the same speaker into turns.
    `pad` overlaps neighbouring runs on purpose: run edges land 0.1-0.45 s off the true change, and on two-voice
    FLEURS dialogues clamping the pad to the gap midpoint lost words (pooled WER 5.28% -> 5.93%) and removed no
    duplicates."""
    pieces, owner = [], []
    for k, (a, b, _) in enumerate(runs):
        seg = wav[max(0, int((a - pad) * SR)):int((b + pad) * SR)]
        for chunk, _ in split_audio_into_chunks(seg, SR, max_chunk_sec=max_piece):
            pieces.append(chunk)
            owner.append(k)
    results = transcribe_pieces(backend, pieces, language, context, progress) if pieces else []
    texts = [""] * len(runs)
    for k, text in zip(owner, results):
        texts[k] = (texts[k] + " " + text.strip()).strip()

    turns = []
    for (a, _, spk), text in zip(runs, texts):
        if not text:
            continue
        if turns and turns[-1][1] == spk:
            turns[-1][2] += " " + text
        else:
            turns.append([a, spk, text])
    return [tuple(t) for t in turns]


def first_appearance_names(turns, prefix="Parlante"):
    """Map cluster ids to 'Parlante 1', 'Parlante 2', ... in order of first appearance."""
    order = {}
    for _, spk, _ in turns:
        order.setdefault(spk, f"{prefix} {len(order) + 1}")
    return order
