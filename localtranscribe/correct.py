"""Suggest fixes for unsure words: a small local language model rescores the speech model's own alternatives.
Nothing here decides anything: every proposal goes through the proven Bend checker (guard.check) and the user
confirms each accepted suggestion in the review page."""
import math
import re

from . import config

_LM = None


def load_lm(repo=config.CORRECT_MODEL):
    """(model, tokenizer), loaded once."""
    global _LM
    if _LM is None:
        from mlx_lm import load

        _LM = load(repo)
    return _LM


def lm_scorer(repo=config.CORRECT_MODEL):
    """score(texts) -> total log-probability of each text under the small model (one padded forward pass)."""
    import mlx.core as mx

    model, tok = load_lm(repo)

    def score(texts):
        start = tok.pad_token_id  # <|endoftext|>; the model has no BOS: the text is scored after this token
        ids = [[start] + tok.encode(t) for t in texts]
        n = max(map(len, ids))
        x = mx.array([r + [start] * (n - len(r)) for r in ids])
        logp = model(x[:, :-1]).astype(mx.float32)
        logp = logp - mx.logsumexp(logp, axis=-1, keepdims=True)
        got = mx.take_along_axis(logp, x[:, 1:, None], axis=-1)[..., 0]
        mask = mx.array([[1.0] * (len(r) - 1) + [0.0] * (n - len(r)) for r in ids])
        return (got * mask).sum(axis=1).tolist()

    return score


def _key(word):
    """The word without case and punctuation: a candidate that only changes those is not a correction."""
    return re.sub(r"[\W_]+", "", word.lower())


def _useful(cand, word):
    """A real correction: not only case/punctuation, and not a bare symbol ("e" -> "&" broke a held-out clip)."""
    return _key(cand) not in ("", _key(word))


def options(words, score):
    """{word index: [(option, LM log-probability of the paragraph with it, speech-model p)]} for each unsure word
    that has candidates; the original comes first."""
    todo = [(k, [(w["w"], w["p"])] + [(c["w"], c["p"]) for c in w["cands"] if _useful(c["w"], w["w"])])
            for k, w in enumerate(words) if w.get("unsure") and w.get("cands")]
    todo = [(k, opts) for k, opts in todo if len(opts) > 1]
    texts = [" ".join(x["w"] if j != k else o for j, x in enumerate(words)) for k, opts in todo for o, _ in opts]
    lm = iter(score(texts) if texts else [])
    return {k: [(o, next(lm), p) for o, p in opts] for k, opts in todo}


def choose(opts, alpha, margin):
    """The best candidate if it beats the original by `margin` nats (LM + alpha * log speech p), else None."""
    s = [lm + alpha * math.log(max(p, 1e-9)) for _, lm, p in opts]
    best = max(range(1, len(opts)), key=s.__getitem__)
    return opts[best][0] if s[best] - s[0] >= margin else None


def propose(words, opts, alpha, margin):
    """One proposed word per word: the chosen swap, or the original."""
    swaps = {k: choose(o, alpha, margin) for k, o in opts.items()}
    return [swaps.get(k) or w["w"] for k, w in enumerate(words)]


class Corrector:
    """Callable for pipeline.transcribe_wav: adds "suggest" to words of a paragraph that the checker accepts."""

    def __init__(self, score=None, alpha=config.CORRECT_ALPHA, margin=config.CORRECT_MARGIN, check=None):
        self.score, self.alpha, self.margin = score, alpha, margin
        self.check, self.real_check = check, check is None
        self.off = False

    def __call__(self, para):
        """Never raises: on any failure it says so once and stops suggesting, so the transcription goes on."""
        if self.off:
            return
        try:
            self.suggest(para)
        except Exception as e:
            self.stop(f"--correct stopped: {e}; no more suggestions for this recording.")

    def stop(self, note):
        print(f"  [note] {note}")
        self.off = True

    def suggest(self, para):
        words = para["words"]
        if self.check is None or self.real_check:
            from . import guard  # imported only when used

            if not guard.available():
                return self.stop("the Bend checker is not available; no suggestions are made.")
            self.check = guard.check
        if self.score is None:
            self.score = lm_scorer()
        opts = options(words, self.score)
        if not opts:
            return
        prop = propose(words, opts, self.alpha, self.margin)
        got = self.check([{"words": [w["w"] for w in words], "flagged": [bool(w.get("unsure")) for w in words],
                           "cands": [[c["w"] for c in w.get("cands", [])] for w in words], "proposal": prop}])
        if got is None:
            return self.stop("the Bend checker is not available; no suggestions are made.")
        for w, new in zip(words, got[0]):
            if new != w["w"]:
                w["suggest"] = new
