"""Chunking and text post-processing shared by every backend.

`split_audio_into_chunks` (with two documented changes), `float_range_normalize` and
`detect_and_fix_repetitions` are ports of the helpers in the official `qwen-asr` package (Apache-2.0, (c) 2026 The Alibaba Qwen team,
qwen_asr/inference/utils.py). They are copied here so that the NVIDIA, CPU and Apple/MLX
backends cut and clean the audio in exactly the same way, without the Mac needing to install
`qwen-asr` and its heavy dependencies.
"""
import math
import os
import re
import shutil
import time
from typing import List, Optional, Tuple

import numpy as np

MIN_ASR_INPUT_SECONDS = 0.5
_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "


def float_range_normalize(audio: np.ndarray) -> np.ndarray:
    """Same conservative range handling qwen-asr applies to every piece: scale down only if the
    peak exceeds 1.0, and clip to [-1, 1]."""
    audio = audio.astype(np.float32)
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        return audio
    if peak > 1.0:
        audio = audio / peak
    return np.clip(audio, -1.0, 1.0)


def split_audio_into_chunks(
    wav: np.ndarray,
    sr: int,
    max_chunk_sec: float,
    search_expand_sec: float = 5.0,
    min_window_ms: float = 100.0,
    min_tail_sec: float = 3.0,
) -> List[Tuple[np.ndarray, float]]:
    """Split a long audio into pieces close to max_chunk_sec, cutting at a low-energy moment.

    Same algorithm as qwen-asr's helper (each cut is the quietest 100 ms within +-5 s of the target),
    with two changes that avoid a failure seen with 20 s pieces on FLEURS clips of 20-24 s:
      * audio that is at most max_chunk_sec + search_expand_sec long stays in one piece (pieces can
        already reach that length when a cut lands late, and 15-30 s pieces are equally accurate);
      * a cut never leaves a final piece shorter than min_tail_sec. Before, the quietest point was
        often the trailing silence, and the model hallucinates a word (e.g. "si") on a 0.5 s scrap.

    Concatenating all returned pieces reproduces the original audio exactly (no overlap, no gap).
    Returns [(piece, offset_seconds)].
    """
    if not (math.isfinite(max_chunk_sec) and max_chunk_sec >= 1):
        raise ValueError("max_chunk_sec must be a finite number of seconds >= 1")
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1).astype(np.float32)

    total_len = int(wav.shape[0])
    max_len = int(max_chunk_sec * sr)
    expand = int(search_expand_sec * sr)
    if total_len <= max_len + expand:
        return [(wav, 0.0)]

    win = max(4, int((min_window_ms / 1000.0) * sr))
    min_tail = int(min_tail_sec * sr)

    chunks: List[Tuple[np.ndarray, float]] = []
    start = 0
    offset_sec = 0.0

    while (total_len - start) > max_len + expand:
        cut = start + max_len
        left = max(start + max_len // 2, cut - expand)  # at least half a piece of progress
        right = min(total_len - min_tail, cut + expand)

        if right - left <= win:
            boundary = min(cut, right)
        else:
            seg_abs = np.abs(wav[left:right])
            window_sums = np.convolve(seg_abs, np.ones(win, dtype=np.float32), mode="valid")
            min_pos = int(np.argmin(window_sums))
            local = seg_abs[min_pos:min_pos + win]
            boundary = left + min_pos + int(np.argmin(local))

        boundary = int(max(boundary, start + 1))
        boundary = int(min(boundary, total_len))

        chunks.append((wav[start:boundary], offset_sec))
        offset_sec += (boundary - start) / float(sr)
        start = boundary

    chunks.append((wav[start:total_len], offset_sec))

    min_len = int(MIN_ASR_INPUT_SECONDS * sr)  # pad pieces the model would reject as too short
    padded = []
    for c, off in chunks:
        if c.shape[0] < min_len:
            c = np.pad(c, (0, min_len - int(c.shape[0])), mode="constant", constant_values=0.0).astype(np.float32)
        padded.append((c, off))
    return padded


def is_silent(piece, sr, dbfs):
    """True when no 100 ms of `piece` reaches `dbfs` (RMS level in dBFS): nothing a speech model could hear. The
    loudest 100 ms decides, not the average, so one short word in a long pause still counts as sound."""
    n = max(1, sr // 10)
    x = np.asarray(piece, dtype=np.float64)
    x = np.pad(x, (0, -len(x) % n))  # whole 100 ms frames (the zeros only soften a last frame shorter than that)
    rms = np.sqrt(np.mean(np.square(x.reshape(-1, n)), axis=1))  # one level per 100 ms
    return not x.size or float(rms.max()) < 10 ** (dbfs / 20)


def detect_and_fix_repetitions(text, threshold=20):
    """Collapse runaway repetitions ("ahahahah..." x100) that ASR decoders sometimes emit."""

    def fix_char_repeats(s, thresh):
        res = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1
            if count > thresh:
                res.append(s[i])
            else:
                res.append(s[i:i + count])
            i += count
        return "".join(res)

    def fix_pattern_repeats(s, thresh, max_len=20):
        n = len(s)
        min_repeat_chars = thresh * 2
        if n < min_repeat_chars:
            return s

        i = 0
        result = []
        found = False
        while i <= n - min_repeat_chars:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break
                pattern = s[i:i + k]
                valid = True
                for rep in range(1, thresh):
                    start_idx = i + rep * k
                    if s[start_idx:start_idx + k] != pattern:
                        valid = False
                        break
                if valid:
                    end_index = i + thresh * k
                    while end_index + k <= n and s[end_index:end_index + k] == pattern:
                        end_index += k
                    result.append(pattern)
                    result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                    i = n
                    found = True
                    break
            if found:
                break
            result.append(s[i])
            i += 1

        if not found:
            result.append(s[i:])
        return "".join(result)

    text = fix_char_repeats(text, threshold)
    return fix_pattern_repeats(text, threshold)


def parse_asr_output(raw: Optional[str], user_language: Optional[str] = None) -> Tuple[str, str]:
    """Split raw Qwen3-ASR output into (language, text), like qwen_asr.inference.utils.parse_asr_output.

    With a forced language the output is plain text. Without one the model writes
    "language Italian<asr_text>the text".
    """
    if raw is None:
        return "", ""
    s = str(raw).strip()
    if not s:
        return "", ""
    s = detect_and_fix_repetitions(s)
    if user_language:
        return user_language, s
    if _ASR_TEXT_TAG not in s:
        return "", s.strip()
    meta_part, text_part = s.split(_ASR_TEXT_TAG, 1)
    if "language none" in meta_part.lower():
        return "", text_part.strip()
    lang = ""
    for line in meta_part.splitlines():
        line = line.strip()
        if line.lower().startswith(_LANG_PREFIX):
            lang = line[len(_LANG_PREFIX):].strip()
            break
    return lang, text_part.strip()


def strip_context_echo(text, context):
    """On near-silent pieces the model can repeat the --context hint verbatim. Cut it only when the
    END of the text is such an echo: whole words (case and punctuation ignored) equal to the start of
    the context. With a 1-2 word context the whole text must be the echo, so real speech that
    mentions a name is never cut."""
    ctx = re.findall(r"\w+", context.lower())
    words = list(re.finditer(r"\w+", text))
    lows = [m.group().lower() for m in words]
    need = min(len(ctx), 3)
    for k in range(len(words)):
        tail = lows[k:]
        if len(tail) >= need and tail == ctx[:len(tail)] and (need >= 3 or k == 0):
            return text[:words[k].start()].strip()
    return text.strip()


def fmt_time(s):
    """Seconds -> hh:mm:ss."""
    s = int(s)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def sync(f):
    """Push a written file to the disk before a rename relies on it (after a power cut an unsynced file can come back
    empty or full of zero bytes). A file system without fsync is not an error."""
    f.flush()
    try:
        os.fsync(f.fileno())
    except OSError:
        pass


def sync_copy(src, dst):
    """Copy the content, then push the copy to the disk too (best effort: Windows needs a writable handle). Only
    the content: a read-only flag copied along would stop Windows from syncing or removing the backup."""
    shutil.copyfile(src, dst)
    try:
        with open(dst, "rb+") as f:
            os.fsync(f.fileno())
    except OSError:
        pass


def backup_name(md_path, kind="bak"):
    """A free "<stem>.<kind>-YYYYMMDD-HHMMSS[-n].md" next to md_path."""
    stem = os.path.splitext(md_path)[0] + f".{kind}-" + time.strftime("%Y%m%d-%H%M%S")
    bak, n = stem + ".md", 1
    while os.path.exists(bak):
        n += 1
        bak = f"{stem}-{n}.md"
    return bak


def group_words(tokens, text):
    """Word-pieces ({"text", "p", ...}) -> [{"w", "p", "i": [first, last]}], one per word of `text.split()`.
    A piece starting with whitespace starts a word; p is the product of its pieces' p; i indexes `tokens`.
    Skips the "language X<asr_text>" prefix and special tokens. None if the words differ from `text.split()`."""
    words, start = [], 0
    for n, t in enumerate(tokens):
        if "<asr_text>" in t["text"]:
            words, start = [], n + 1
    for n in range(start, len(tokens)):
        s = tokens[n]["text"]
        if s.startswith("<|") and s.endswith("|>"):
            continue
        if not words or s[:1].isspace():
            words.append({"w": "", "p": 1.0, "i": [n, n]})
        w = words[-1]
        w["w"] += s
        w["p"] *= tokens[n]["p"]
        w["i"][1] = n
    for w in words:
        w["w"], w["p"] = w["w"].strip(), round(w["p"], 5)
    return words if [w["w"] for w in words] == text.split() else None
