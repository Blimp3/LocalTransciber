"""Run the proven Bend checker (guard.bend) on AI suggestions.

`check(paragraphs)` gives every word of a paragraph, after the checker has kept only the suggestions that
are allowed: the word is flagged and the suggestion is one of the speech model's own alternatives. Words are
mapped to numbers per paragraph, so the binary only sees ASCII digits (protocol: see guard_cli.bend).
The binary's answer is then compared with a second, independent implementation of the same four rules in plain
Python (`_expected`); the proof (LAWS.bend / PROOF.bend) stays the specification, the Python only double-checks
the build. Any disagreement, like any other problem, returns None: a missing, broken or wrong binary never
stops a transcription and never lets a suggestion through."""
import os
import platform
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

BIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")


def _binary_name() -> Optional[str]:
    """The prebuilt checker for this machine, or None when there is none."""
    machine = platform.machine().lower()
    if sys.platform == "darwin" and machine == "arm64":
        return "guard-macos-arm64"
    if sys.platform == "win32" and machine in ("amd64", "x86_64"):
        return "guard-windows-x64.exe"
    return None


_NAME = _binary_name()
BINARY = os.path.join(BIN_DIR, _NAME or "guard-macos-arm64")
TIMEOUT_S = 30


def available() -> bool:
    """True when the prebuilt checker exists for this machine (macOS on Apple silicon, Windows x64)."""
    return _NAME is not None and os.access(BINARY, os.X_OK)


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


def _expected(p: dict) -> List[str]:
    """The rules in plain Python, independent of the binary: a proposal is applied only on a flagged word and
    only if it is one of that word's candidates. A missing proposal keeps the word, extra proposals are ignored."""
    prop = p["proposal"]
    return [prop[i] if i < len(prop) and f and prop[i] in cs else w
            for i, (w, f, cs) in enumerate(zip(p["words"], p["flagged"], p["cands"]))]


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
        # mkstemp and close before running: on Windows the child cannot open a NamedTemporaryFile still open here.
        fd, path = tempfile.mkstemp(suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="ascii", newline="\n") as f:
                f.write("".join(line + "\n" for line, _ in encoded))
            run = subprocess.run([BINARY, path], capture_output=True, text=True, timeout=TIMEOUT_S)
        finally:
            os.unlink(path)
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
        # Second opinion: the binary must return exactly what the Python rules give (same length, same words).
        if any(out != _expected(p) for out, p in zip(result, paragraphs)):
            print("[note] the checker's answer failed the Python re-check; no suggestions", file=sys.stderr)
            return None
        return result
    except Exception:  # any failure, including a bad id or a timeout: the caller falls back to no suggestions
        return None
