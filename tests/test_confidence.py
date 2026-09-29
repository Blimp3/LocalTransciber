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
                self.last_records = [{"aligned": False, "tokens": [len(p)]} for p in pieces]
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


def tk(text, p=1.0):
    return {"text": text, "p": p, "alts": []}


class GroupWordsTests(unittest.TestCase):
    def test_groups_on_leading_space_with_product_p_and_spans(self):
        from localtranscribe.textutil import group_words

        toks = [tk(" accom", 0.5), tk("un", 0.5), tk("ano", 1.0), tk(" dino", 0.9), tk(".", 0.5)]
        self.assertEqual(group_words(toks, "accomunano dino."),
                         [{"w": "accomunano", "p": 0.25, "i": [0, 2]}, {"w": "dino.", "p": 0.45, "i": [3, 4]}])

    def test_skips_language_prefix_and_special_tokens(self):
        from localtranscribe.textutil import group_words

        toks = [tk("language"), tk(" Italian"), tk("<asr_text>"), tk("ciao"), tk(" mondo"), tk("<|im_end|>")]
        self.assertEqual([w["i"] for w in group_words(toks, "ciao mondo")], [[3, 3], [4, 4]])

    def test_mismatch_gives_none(self):
        from localtranscribe.textutil import group_words

        self.assertIsNone(group_words([tk("ciao"), tk(" mondo")], "ciao mondi"))


class WordsInSidecarTests(unittest.TestCase):
    def run_wav(self, aligned):
        from localtranscribe import pipeline

        class Stub:
            batch_size = 1

            def transcribe(self, pieces, language, context=""):
                self.last_records = [{"aligned": aligned, "tokens": [tk("ciao", 0.99), tk(" mondo", 0.3)]}]
                return ["ciao mondo"]

        paragraphs = []
        with mock.patch.object(pipeline, "split_wav", lambda wav, secs: [(np.zeros(1), 0.0)]):
            pipeline.transcribe_wav(Stub(), None, "Italian", paragraphs=paragraphs)
        return paragraphs[0]["words"]

    def test_unsure_word_is_flagged(self):
        words = self.run_wav(True)
        self.assertNotIn("unsure", words[0])
        self.assertTrue(words[1]["unsure"])

    def test_unaligned_record_gives_none(self):
        self.assertIsNone(self.run_wav(False))


class CandidatesTests(unittest.TestCase):
    """Words "accomunano" (pieces 0-2, weakest 1, unsure), "dino" (piece 3, unsure), "ok" (sure)."""

    ALTS_MID = [[30, "mi", 0.3], [31, " split", 0.2], [32, "<|im_end|>", 0.1], [33, "ni", 0.1], [34, "mu", 0.05],
                [35, "zz", 0.04], [36, "qq", 0.03], [37, "mi", 0.02], [38, "mo", 0.019]]
    ALTS_LAST = [[40, " dino", 0.3], [41, " ", 0.25], [42, "dine", 0.2], [43, " dine", 0.1]]
    # (ids, ps, stop token): the next word of the sentence is " dino" (13) after word 0 and " ok" (14) after word 1
    CONTS = {30: ([20], [0.5], 13), 33: ([21], [0.5], 13), 34: ([], [], 13), 35: None, 36: ([22], [1.0], 13),
             37: ([20], [0.9], 13), 40: ([], [], 14), 41: ([], [], 14), 43: ([], [], 14),
             50: ([], [], 7), 51: ([], [], 14), 52: ([], [], 7)}

    def run_wav(self, has_method=True, sure=False, last=False):
        from localtranscribe import pipeline

        p = 0.99 if sure else 0.4
        toks = [{"id": 10, "text": " acco", "p": 0.99, "alts": []},
                {"id": 11, "text": "mu", "p": p, "alts": self.ALTS_MID},
                {"id": 12, "text": "nano", "p": 0.99, "alts": []},
                {"id": 13, "text": " dino", "p": 0.99 if sure else 0.5, "alts": self.ALTS_LAST},
                {"id": 14, "text": " ok", "p": 0.3 if last else 0.99,
                 "alts": [[50, " ak", 0.3], [51, " ek", 0.2], [52, " ik", 0.1]] if last else []}]
        conts, calls = self.CONTS, []

        class Stub:
            batch_size = 1
            eos_ids = {7}

            def transcribe(self, pieces, language, context=""):
                self.last_records = [{"aligned": True, "tokens": toks} for _ in pieces]
                return ["accomunano dino ok"]

            def decode(self, ids):
                return {20: "mi", 21: "ni", 22: " x"}[ids[0]]

        if has_method:
            def wc(self, piece, language, context, requests):
                calls.append(requests)
                return [[conts[a] for a in alts] for _, alts in requests]

            Stub.word_continuations = wc
        paragraphs = []
        with mock.patch.object(pipeline, "split_wav", lambda wav, secs: [(np.zeros(1), 0.0)]):
            pipeline.transcribe_wav(Stub(), None, "Italian", paragraphs=paragraphs)
        return paragraphs[0]["words"], calls

    def test_alts_sent_to_the_backend_are_filtered(self):
        _, calls = self.run_wav()
        self.assertEqual(len(calls), 1)  # one call per piece
        # the whitespace alt would split the word, the special token is skipped; for the later word
        # (first piece) the alt without whitespace would glue onto the previous word
        self.assertEqual(calls[0], [([10], [30, 33, 34, 35, 36, 37]), ([10, 11, 12], [40, 41, 43])])

    def test_alt_below_the_floor_is_not_tried(self):
        _, calls = self.run_wav()
        self.assertIn(37, calls[0][0][1])  # exactly at CONFIDENCE_ALT_MIN: kept
        self.assertNotIn(38, calls[0][0][1])  # just below: dropped (CONTS has no entry for it)

    def test_candidates_composed_deduped_sorted(self):
        words, _ = self.run_wav()
        # accomimi: p .3*.5, duplicate (.02*.9) dropped; acconini .1*.5; accomu = alt p only;
        # incomplete (None) and whitespace ("qq x") dropped
        self.assertEqual(words[0]["cands"], [{"w": "accomimi", "p": 0.15}, {"w": "acconini", "p": 0.05},
                                             {"w": "accomu", "p": 0.05}])
        # " dino" is unchanged and " " is empty: both dropped
        self.assertEqual(words[1]["cands"], [{"w": "dine", "p": 0.1}])
        self.assertNotIn("cands", words[2])

    def test_stop_token_must_be_the_original_next_token(self):
        self.CONTS = {**self.CONTS, 33: ([21], [0.5], 99)}  # after "acconini" the model wants another word
        words, _ = self.run_wav()
        self.assertEqual([c["w"] for c in words[0]["cands"]], ["accomimi", "accomu"])

    def test_last_word_candidate_must_end_the_text(self):
        words, _ = self.run_wav(last=True)
        # " ak" and " ik" end with EOS: kept; " ek" wants to go on with " ok": dropped
        self.assertEqual([c["w"] for c in words[2]["cands"]], ["ak", "ik"])

    def test_no_unsure_words_no_call(self):
        words, calls = self.run_wav(sure=True)
        self.assertEqual(calls, [])
        self.assertTrue(all("cands" not in w for w in words))

    def test_backend_without_word_continuations_adds_nothing(self):
        words, _ = self.run_wav(has_method=False)
        self.assertTrue(words[0]["unsure"])
        self.assertTrue(all("cands" not in w for w in words))


if __name__ == "__main__":
    unittest.main()
