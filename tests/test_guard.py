"""The Bend checker binary and its Python wrapper (localtranscribe/guard.py).

Binary tests need bin/guard-macos-arm64 (build it with build_guard.sh) and are skipped without it."""
import os
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from localtranscribe import guard  # noqa: E402


def para(words, flagged, cands, proposal):
    return {"words": words, "flagged": flagged, "cands": cands, "proposal": proposal}


def expected(p):
    """The rules, in plain Python: a proposal is applied only on a flagged word and only if it is a candidate."""
    return [q if f and q in cs else w
            for w, f, cs, q in zip(p["words"], p["flagged"], p["cands"], p["proposal"])]


@unittest.skipUnless(guard.available(), "bin/guard-macos-arm64 is not built (run build_guard.sh)")
class TestGuardBinary(unittest.TestCase):
    def run_cli(self, text):
        with tempfile.NamedTemporaryFile("w", suffix=".txt") as f:
            f.write(text)
            f.flush()
            return subprocess.run([guard.BINARY, f.name], capture_output=True, text=True, timeout=30)

    def test_protocol_example(self):
        r = self.run_cli("1 0 -;2 1 7,8;3 1 9|5 8 4\n")
        self.assertEqual((r.returncode, r.stdout), (0, "1 8 3\n"))

    def test_protocol_edge_cases(self):
        self.assertEqual(self.run_cli("").stdout, "")
        self.assertEqual(self.run_cli("|\n1 0 -|\n").stdout, "\n1\n")
        self.assertEqual(self.run_cli("1 0 -|5").stdout, "1\n")  # no final newline
        self.assertEqual(self.run_cli("1 1 2;3 1 4|2\n").stdout, "2 3\n")  # missing proposal keeps the word

    def test_malformed_input_is_an_error(self):
        for bad in ["x\n", "1 0|5\n", "1 2 -|5\n", "1 0 a|5\n", "1 1 2,|5\n", "1 0 -|5|6\n", "1 0 -;|5\n",
                    "1 0 -|5\n\n", "1 0 -|5 x\n"]:
            r = self.run_cli(bad)
            self.assertNotEqual(r.returncode, 0, bad)
            self.assertEqual(r.stdout, "", bad)
            self.assertIn("malformed", r.stderr)

    def test_missing_file_and_no_args(self):
        self.assertNotEqual(subprocess.run([guard.BINARY, "/nonexistent/x"], capture_output=True).returncode, 0)
        self.assertNotEqual(subprocess.run([guard.BINARY], capture_output=True).returncode, 0)

    def test_unflagged_proposal_rejected(self):
        p = para(["a", "b"], [False, False], [["x"], ["y"]], ["x", "y"])
        self.assertEqual(guard.check([p]), [["a", "b"]])

    def test_free_text_rejected(self):
        p = para(["a", "b"], [True, True], [["x"], []], ["zzz", "anything"])
        self.assertEqual(guard.check([p]), [["a", "b"]])

    def test_allowed_candidate_applied(self):
        p = para(["a", "b", "c"], [True, False, True], [["x", "y"], ["z"], ["q"]], ["y", "z", "q"])
        self.assertEqual(guard.check([p]), [["y", "b", "q"]])

    def test_shorter_and_longer_proposals(self):
        w, f, c = ["a", "b", "c"], [True] * 3, [["x"], ["y"], ["z"]]
        self.assertEqual(guard.check([para(w, f, c, ["x"])]), [["x", "b", "c"]])
        self.assertEqual(guard.check([para(w, f, c, ["x", "y", "z", "extra", "more"])]), [["x", "y", "z"]])
        self.assertEqual(guard.check([para(w, f, c, [])]), [["a", "b", "c"]])

    def test_special_characters_and_repeats(self):
        w = ["è", "l'anno,", "è", "«ciao»;", "a|b", "1 2", "-", "", "è"]
        f = [True, True, False, True, True, True, True, True, True]
        c = [["e"], ["l'anno", "anno."], ["e"], ["ciao|", "x y"], ["a;b"], ["1,2"], ["--"], ["vuoto"], ["é", "è"]]
        q = ["e", "anno.", "e", "x y", "a;b", "3", "--", "vuoto", "é"]
        p = para(w, f, c, q)
        self.assertEqual(guard.check([p]), [expected(p)])
        self.assertEqual(expected(p), ["e", "anno.", "è", "x y", "a;b", "1 2", "--", "vuoto", "é"])

    def test_empty_inputs(self):
        self.assertEqual(guard.check([]), [])
        self.assertEqual(guard.check([para([], [], [], [])]), [[]])
        self.assertEqual(guard.check([para([], [], [], []), para(["a"], [True], [["b"]], ["b"]), para([], [], [], [])]),
                         [[], ["b"], []])

    def test_mismatched_lists_give_none(self):
        self.assertIsNone(guard.check([para(["a"], [True, False], [["b"]], ["b"])]))

    def test_200_paragraphs(self):
        ps = [para([f"w{i}", "x"], [True, False], [[f"c{i}"], []], [f"c{i}", "y"]) for i in range(200)]
        self.assertEqual(guard.check(ps), [[f"c{i}", "x"] for i in range(200)])

    def test_random_paragraphs_obey_the_four_rules(self):
        rng = random.Random(4)
        vocab = ["è", "e", "l'anno", "casa", "Casa", "a,", "«x»", "1", "10", "-", "uno due", "ciao;", "p|q"]
        ps = []
        for _ in range(300):
            n = rng.randint(0, 25)
            words = [rng.choice(vocab) for _ in range(n)]
            cands = [rng.sample(vocab, rng.randint(0, 4)) if rng.random() < 0.6 else [] for _ in range(n)]
            proposal = [rng.choice(cs) if cs and rng.random() < 0.5 else rng.choice(vocab + ["free text"])
                        for cs in cands]
            if rng.random() < 0.2:
                proposal = proposal[:rng.randint(0, n)]
            ps.append(para(words, [rng.random() < 0.5 for _ in range(n)], cands, proposal))
        out = guard.check(ps)
        self.assertEqual(len(out), 300)
        for p, o in zip(ps, out):
            self.assertEqual(len(o), len(p["words"]))  # same word count
            for i, w in enumerate(o):
                if not p["flagged"][i]:
                    self.assertEqual(w, p["words"][i])  # sure words are never touched
                self.assertTrue(w == p["words"][i] or w in p["cands"][i])  # only the ASR's own alternatives
            padded = p["proposal"] + [None] * (len(p["words"]) - len(p["proposal"]))
            self.assertEqual(o, expected(dict(p, proposal=[q if q is not None else "\0" for q in padded])))


class TestGuardFallback(unittest.TestCase):
    def test_none_when_the_binary_is_unavailable(self):
        p = para(["a"], [True], [["b"]], ["b"])
        with mock.patch.object(guard, "available", return_value=False):
            self.assertIsNone(guard.check([p]))
            self.assertEqual(guard.check([]), [])

    def test_none_when_the_binary_fails(self):
        p = para(["a"], [True], [["b"]], ["b"])
        with mock.patch.object(guard, "available", return_value=True):
            for binary in ["/nonexistent/guard", "/usr/bin/false", "/bin/echo"]:
                with mock.patch.object(guard, "BINARY", binary):
                    self.assertIsNone(guard.check([p]), binary)


if __name__ == "__main__":
    unittest.main()
