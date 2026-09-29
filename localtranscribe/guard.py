"""Run the proven Bend checker (guard.bend) on AI suggestions.

`check(paragraphs)` gives every word of a paragraph, after the checker has kept only the suggestions that
are allowed: the word is flagged and the suggestion is one of the speech model's own alternatives. Words are
mapped to numbers per paragraph, so the binary only sees ASCII digits (protocol: see guard_cli.bend).
Any problem returns None, so a missing or broken binary never stops a transcription."""
import os
import platform
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

BINARY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "guard-macos-arm64")
TIMEOUT_S = 30


def available() -> bool:
    """True when the prebuilt checker exists for this machine (macOS on Apple silicon for now)."""
    return sys.platform == "darwin" and platform.machine() == "arm64" and os.access(BINARY, os.X_OK)


def _encode(p: dict):
    """One paragraph as (protocol line, id -> word), or None if its word lists do not line up."""
    words, flagged, cands, proposal = p["words"], p["flagged"], p["cands"], p["proposal"]
    if not len(words) == len(flagged) == len(cands):  # a shorter or longer proposal is the checker's business
        return None
    ids: Dict[str, int] = {}

    def num(w: str) -> str:
        return str(ids.setdefault(w, len(ids) + 1))

    slots = ";".join(f"{num(w)} {int(bool(f))} {','.join(num(c) for c in cs) or '-'}"
                     for w, f, cs in zip(words, flagged, cands))
    line = slots + "|" + " ".join(num(w) for w in proposal)
    return line, {i: w for w, i in ids.items()}


def check(paragraphs: List[dict]) -> Optional[List[List[str]]]:
    """paragraphs = [{"words", "flagged", "cands", "proposal"}], one entry per word in each list (a proposal
    list that is shorter keeps the missing words, a longer one is cut).
    Returns the checked words per paragraph, or None if the checker is unavailable or failed."""
    if not paragraphs:
        return []
    if not available():
        return None
    try:
        encoded = [_encode(p) for p in paragraphs]
        if None in encoded:
            return None
        with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="ascii") as f:
            f.write("".join(line + "\n" for line, _ in encoded))
            f.flush()
            run = subprocess.run([BINARY, f.name], capture_output=True, text=True, timeout=TIMEOUT_S)
        if run.returncode != 0 or not run.stdout.endswith("\n"):
            return None
        lines = run.stdout[:-1].split("\n")
        if len(lines) != len(paragraphs):
            return None
        result = []
        for line, (_, names), p in zip(lines, encoded, paragraphs):
            out = [names[int(i)] for i in line.split()]
            if len(out) != len(p["words"]):
                return None
            result.append(out)
        return result
    except Exception:  # any failure, including a bad id or a timeout: the caller falls back to no suggestions
        return None
