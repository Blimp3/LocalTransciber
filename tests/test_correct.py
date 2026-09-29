"""--correct: rescoring choice logic, the guard gate and the pipeline hook, with a stub language model and stub checker."""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from localtranscribe import correct, pipeline  # noqa: E402


def words():
    return [{"w": "il", "p": 0.99}, {"w": "cane", "p": 0.6, "unsure": True, "cands": [{"w": "pane", "p": 0.3}]},
            {"w": "mangia", "p": 0.5, "unsure": True}, {"w": "ora", "p": 0.95}]


def stub_score(table):
    """score(texts): log-probability looked up by the text, default -10."""
    return lambda texts: [table.get(t, -10.0) for t in texts]


def accept_all(paras):
    return [p["proposal"] for p in paras]


class ChoiceTests(unittest.TestCase):
    def test_options_only_unsure_with_cands(self):
        o = correct.options(words(), stub_score({}))
        self.assertEqual(list(o), [1])  # "mangia" is unsure but has no candidates
        self.assertEqual([x[0] for x in o[1]], ["cane", "pane"])

    def test_case_punctuation_and_symbol_cands_are_not_options(self):
        w = [{"w": "e", "p": 0.5, "unsure": True, "cands": [{"w": "&", "p": 0.4}, {"w": "E.", "p": 0.3}]}]
        self.assertEqual(correct.options(w, stub_score({})), {})

    def test_margin(self):
        w = words()
        o = correct.options(w, stub_score({"il pane mangia ora": -5.0, "il cane mangia ora": -8.0}))
        self.assertEqual(correct.propose(w, o, 0, 3.0), ["il", "pane", "mangia", "ora"])  # gain 3 >= 3
        self.assertEqual(correct.propose(w, o, 0, 3.5), ["il", "cane", "mangia", "ora"])  # not enough

    def test_alpha_weighs_the_speech_model(self):
        w = words()
        o = correct.options(w, stub_score({"il pane mangia ora": -5.0, "il cane mangia ora": -8.0}))
        # log(0.3) - log(0.6) = -0.69: alpha 1 lowers the gain from 3 to 2.31
        self.assertEqual(correct.propose(w, o, 1, 3.0)[1], "cane")
        self.assertEqual(correct.propose(w, o, 1, 2.0)[1], "pane")

    def test_no_options_no_lm_call(self):
        called = []
        o = correct.options([{"w": "a", "p": 0.4, "unsure": True}], lambda t: called.append(t) or [])
        self.assertEqual((o, called), ({}, []))


class CorrectorTests(unittest.TestCase):
    def make(self, check):
        table = {"il pane mangia ora": -1.0, "il cane mangia ora": -9.0}
        return correct.Corrector(score=stub_score(table), alpha=0, margin=2, check=check)

    def test_guard_accepted_words_get_suggest(self):
        para = {"words": words()}
        self.make(accept_all)(para)
        self.assertEqual([w.get("suggest") for w in para["words"]], [None, "pane", None, None])

    def test_guard_rejection_leaves_nothing(self):
        para = {"words": words()}
        self.make(lambda ps: [p["words"] for p in ps])(para)
        self.assertFalse(any("suggest" in w for w in para["words"]))

    def test_guard_unavailable_no_suggestions_one_note(self):
        c = self.make(lambda ps: None)
        out = io.StringIO()
        with redirect_stdout(out):
            for _ in range(2):
                para = {"words": words()}
                c(para)
                self.assertFalse(any("suggest" in w for w in para["words"]))
        self.assertEqual(out.getvalue().count("[note]"), 1)

    def test_scorer_failure_notes_once_and_stops(self):
        def boom(texts):
            raise RuntimeError("out of memory")

        c = correct.Corrector(score=boom, alpha=0, margin=2, check=accept_all)
        out = io.StringIO()
        with redirect_stdout(out):
            for _ in range(2):
                para = {"words": words()}
                c(para)
                self.assertEqual(para, {"words": words()})
        self.assertEqual(out.getvalue().count("[note]"), 1)
        self.assertIn("out of memory", out.getvalue())
        self.assertTrue(c.off)

    def test_checker_unavailable_never_loads_the_model(self):
        from unittest import mock

        from localtranscribe import guard

        c = correct.Corrector(score=lambda t: self.fail("scorer called"))
        out = io.StringIO()
        with mock.patch.object(guard, "available", return_value=False), redirect_stdout(out):
            c({"words": words()})
            c({"words": words()})
        self.assertEqual(out.getvalue().count("[note]"), 1)

    def test_guard_gets_flags_cands_and_proposal(self):
        seen = []
        self.make(lambda ps: seen.extend(ps) or accept_all(ps))({"words": words()})
        self.assertEqual(seen[0]["flagged"], [False, True, True, False])
        self.assertEqual(seen[0]["cands"], [[], ["pane"], [], []])
        self.assertEqual(seen[0]["proposal"], ["il", "pane", "mangia", "ora"])

    def test_paragraph_without_options_skips_guard(self):
        c = self.make(lambda ps: self.fail("guard called"))
        c({"words": [{"w": "a", "p": 0.99}]})


class FakeBackend:
    batch_size, eos_ids = 1, {0}
    last_records = []
    word_continuations = None  # only its presence matters: add_candidates is stubbed below

    def transcribe(self, pieces, language, context):
        self.last_records = [{"aligned": True, "tokens": []} for _ in pieces]
        return ["ciao mondo"] * len(pieces)


class HookTests(unittest.TestCase):
    def run_it(self, corrector):
        orig, orig_add = pipeline.group_words, pipeline.add_candidates
        pipeline.add_candidates = lambda *a: None
        pipeline.group_words = lambda toks, t: [{"w": "ciao", "p": 0.4, "unsure": True}, {"w": "mondo", "p": 0.99}]
        try:
            paras = []
            text = pipeline.transcribe_wav(FakeBackend(), np.zeros(16000 * 3, dtype=np.float32), "it",
                                           paragraphs=paras, corrector=corrector)
        finally:
            pipeline.group_words, pipeline.add_candidates = orig, orig_add
        return text, paras

    def test_hook_is_optional_and_never_touches_text(self):
        t0, p0 = self.run_it(None)
        t1, p1 = self.run_it(lambda para: para["words"][0].update(suggest="x"))
        self.assertEqual(t0, t1)
        self.assertNotIn("suggest", p0[0]["words"][0])
        self.assertEqual(p1[0]["words"][0]["suggest"], "x")


if __name__ == "__main__":
    unittest.main()
