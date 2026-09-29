"""--confidence: the recording sampler and how the MLX backend and the pipeline keep its records aligned.

mlx is replaced by numpy (same function names for what the recorder uses) and the model by a scripted fake, so this
runs anywhere. The real model is checked by comparing transcripts with and without recording (see the session notes)."""
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_mlx_backend import FakeQwenModel, install_fake_mlx, restore  # noqa: E402

SR = 16000
V, EOS = 8, 7


def logits_for(tok):
    row = np.zeros(V, dtype=np.float64)
    row[tok], row[(tok + 1) % V], row[(tok + 2) % V] = 4.0, 2.0, 1.0
    return row


class _Tok:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(97 + i) for i in ids if not (skip_special_tokens and i == EOS))


class ScriptedModel(FakeQwenModel):
    """Token script per piece length; calls the sampler like mlx-audio does (plus one step of lookahead)."""

    def __init__(self, scripts, fail_after_sampling=False):
        super().__init__()
        self.scripts, self._tokenizer, self.fail_after_sampling = scripts, _Tok(), fail_after_sampling

    def stream_generate(self, audio, *, max_tokens, sampler, language=None, system_prompt=None, **kw):
        script = self.scripts[len(audio)] + [EOS]
        for j in range(len(script) + 1):
            tok = sampler(logits_for(script[min(j, len(script) - 1)])[None, :])
            if j < len(script):
                if script[j] == EOS:
                    return
                yield int(tok[0]), None

    def _generate_chunks_batched(self, chunks, *, sampler, **kw):
        self.batched_calls.append([len(c[0]) for c in chunks])
        scripts = [self.scripts[len(c[0])] + [EOS] for c in chunks]
        for j in range(max(map(len, scripts)) + 1):
            sampler(np.stack([logits_for(s[j] if j < len(s) else 0) for s in scripts]))
        if self.fail_after_sampling:
            raise RuntimeError("boom")
        gen = [len(s) - 1 for s in scripts]
        texts = [self._tokenizer.decode(s[:-1]) for s in scripts]
        return texts, gen, gen, [True] * len(chunks)


def patch_numpy_as_mx(core):
    for name in ("argmax", "argpartition", "argsort", "take_along_axis", "exp", "stack"):
        setattr(core, name, getattr(np, name))
    core.async_eval = lambda *a: None
    core.logsumexp = lambda x, axis, keepdims=False: np.log(np.exp(x).sum(axis=axis, keepdims=keepdims))


class ConfidenceTests(unittest.TestCase):
    def make(self, batch, scripts, record=True, fail=False):
        self.model = ScriptedModel(scripts, fail_after_sampling=fail)
        saved = install_fake_mlx(self.model)
        self.addCleanup(restore, saved)
        patch_numpy_as_mx(sys.modules["mlx.core"])
        from localtranscribe.backends.mlx_qwen import MlxQwenBackend

        return MlxQwenBackend("mlx-community/Qwen3-ASR-1.7B-4bit", batch_size=batch, record_confidence=record)

    def pieces(self, *lengths):
        return [np.full(n, 0.1, dtype=np.float32) for n in lengths]

    def test_recorder_returns_argmax_and_records_p_and_topk(self):
        be = self.make(1, {})
        from localtranscribe.backends.mlx_qwen import _Recorder

        rec = _Recorder(be.mx, k=3)
        x = np.stack([logits_for(2), logits_for(5)])
        self.assertEqual(rec(x).tolist(), [2, 5])
        self.assertEqual(rec(np.log(np.exp(x) / np.exp(x).sum(axis=1, keepdims=True))).tolist(), [2, 5])  # log-probs too
        self.assertEqual([r[0] for r in rec.rows([2, 1])], [[2, 2], [5]])  # each row keeps its own first n steps
        first = rec.rows([1, 1])
        self.assertEqual(first[0][0], [2])
        self.assertEqual(first[1][0], [5])
        self.assertEqual(first[0][2][0], [2, 3, 4])  # chosen, then the runners-up in order
        soft = np.exp(logits_for(2)) / np.exp(logits_for(2)).sum()
        self.assertAlmostEqual(first[0][1][0], soft[2], places=6)
        self.assertAlmostEqual(first[0][3][0][1], soft[3], places=6)

    def test_single_records_tokens_and_skips_generate(self):
        be = self.make(1, {5: [0, 1, 2]})
        out = be.transcribe(self.pieces(5), "Italian")
        self.assertEqual(out, ["abc"])
        rec = be.last_records[0]
        self.assertEqual([t["id"] for t in rec["tokens"]], [0, 1, 2])
        self.assertEqual([t["text"] for t in rec["tokens"]], ["a", "b", "c"])
        self.assertEqual(rec["tokens"][0]["alts"][0][:2], [1, "b"])
        self.assertEqual(len(rec["tokens"][0]["alts"]), 4)  # top 5 without the chosen one
        self.assertTrue(rec["aligned"])
        self.assertEqual(self.model.generate_calls, [])

    def test_batched_cuts_each_row_and_keeps_original_order(self):
        scripts = {10: [0], 24: [1, 2, 3, 4], 12: [2, 2], 23: [5, 6]}
        be = self.make(2, scripts)
        lengths = (10, 24, 12, 23)
        out = be.transcribe(self.pieces(*lengths), "Italian")
        self.assertEqual(out, ["".join(chr(97 + i) for i in scripts[n]) for n in lengths])
        self.assertEqual([[t["id"] for t in r["tokens"]] for r in be.last_records], [scripts[n] for n in lengths])
        self.assertTrue(all(r["aligned"] for r in be.last_records))
        self.assertEqual(self.model.batched_calls, [[24, 23], [12, 10]])  # sorted by length, records still in piece order

    def test_failed_batch_leaves_no_records_of_its_own(self):
        scripts = {10: [0], 11: [1, 1]}
        be = self.make(2, scripts, fail=True)
        out = be.transcribe(self.pieces(10, 11), "Italian")
        self.assertEqual(out, ["a", "bb"])  # fell back to single mode
        self.assertEqual([[t["id"] for t in r["tokens"]] for r in be.last_records], [[0], [1, 1]])

    def test_off_by_default_records_nothing(self):
        be = self.make(1, {}, record=False)
        be.transcribe(self.pieces(5), "Italian")
        self.assertEqual(be.last_records, [None])
        self.assertEqual(len(self.model.generate_calls), 1)

    def test_sidecar_paragraphs_match_the_md_paragraphs_when_a_piece_is_empty(self):
        from localtranscribe import pipeline

        class Stub:
            batch_size = 1
            last_records = []

            def transcribe(self, pieces, language, context=""):
                self.last_records = [{"aligned": True, "tokens": [len(p)]} for p in pieces]
                return ["uno", "", "tre"][:len(pieces)]

        paragraphs = []
        with mock.patch.object(pipeline, "split_wav", lambda wav, secs: [(np.zeros(1), 0.0), (np.zeros(2), 20.0), (np.zeros(3), 40.0)]):
            text = pipeline.transcribe_wav(Stub(), None, "Italian", paragraphs=paragraphs)
        self.assertEqual(text.split("\n\n"), ["[00:00:00] uno", "[00:00:40] tre"])
        self.assertEqual([(p["text"], p["offset"], p["tokens"]) for p in paragraphs],
                         [("uno", 0.0, [1]), ("tre", 40.0, [3])])

    def test_context_echo_clears_aligned(self):
        from localtranscribe import pipeline

        class Stub:
            batch_size = 1
            last_records = [{"aligned": True, "tokens": []}]

            def transcribe(self, pieces, language, context=""):
                return ["Mario Rossi ciao"]

        records = []
        with mock.patch.object(pipeline, "strip_context_echo", lambda t, c: "ciao"):
            pipeline.transcribe_pieces(Stub(), [np.zeros(1)], "Italian", "Mario Rossi", records=records)
        self.assertFalse(records[0]["aligned"])


if __name__ == "__main__":
    unittest.main()
