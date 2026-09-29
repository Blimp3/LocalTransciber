"""Text normalisation and word error rate, shared by the benchmark scripts."""
import csv
import os
import re
import sys
import unicodedata

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def normalize(text):
    """Lowercase, drop punctuation (apostrophes become spaces: l'uomo -> l uomo), collapse spaces."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("’", "'")
    text = re.sub(r"[^\w\s]", " ", text).replace("_", " ")
    return " ".join(text.split())


def wer(refs, hyps):
    """Corpus word error rate (0..1) of normalised text. refs/hyps are strings or lists of strings."""
    import jiwer

    if isinstance(refs, str):
        refs, hyps = [refs], [hyps]
    return jiwer.wer([normalize(r) for r in refs], [normalize(h) for h in hyps])


def cer(refs, hyps):
    import jiwer

    if isinstance(refs, str):
        refs, hyps = [refs], [hyps]
    return jiwer.cer([normalize(r) for r in refs], [normalize(h) for h in hyps])


def load_references(data_dir, limit=None):
    """Rows of <data_dir>/references.csv as dicts with at least 'file' and 'reference'."""
    with open(os.path.join(data_dir, "references.csv"), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows
