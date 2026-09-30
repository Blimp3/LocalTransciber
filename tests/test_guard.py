"""The Bend checker binary and its Python wrapper (localtranscribe/guard.py).

Binary tests need the checker for this platform (bin/guard-macos-arm64 from build_guard.sh, or
bin/guard-windows-x64.exe from build_guard_windows.sh) and are skipped without it."""
import os
import contextlib
import io
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


@unittest.skipUnless(guard.available(), "no checker binary for this platform (run build_guard.sh or build_guard_windows.sh)")
class TestGuardBinary(unittest.TestCase):
    def run_cli(self, text):
        fd, path = tempfile.mkstemp(suffix=".txt")  # closed before the run: Windows cannot reopen an open temp file
        try:
            with os.fdopen(fd, "w", newline="\n") as f:
                f.write(text)
            return subprocess.run([guard.BINARY, path], capture_output=True, text=True, timeout=30)
        finally:
            os.unlink(path)

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


# A stand-in for the checker: a Python script run through sys.executable and a tiny .cmd/.sh launcher (the
# checker is called as `BINARY <file>`), so it works on Windows and macOS. It prints whatever FAKE_OUT holds and
# exits with FAKE_EXIT, so each test controls exactly what "the binary" answers.
FAKE = "import os, sys\nsys.stdout.write(os.environ['FAKE_OUT'])\nsys.exit(int(os.environ.get('FAKE_EXIT', '0')))\n"


class TestPythonRecheck(unittest.TestCase):
    def check_with_fake(self, out, ps, code=0):
        """Run guard.check() against a fake binary that prints `out` and exits with `code`; returns (result, stderr text)."""
        with tempfile.TemporaryDirectory() as d:
            script = os.path.join(d, "fake_guard.py")
            with open(script, "w") as f:
                f.write(FAKE)
            if sys.platform == "win32":
                launcher, text = os.path.join(d, "fake_guard.cmd"), f'@"{sys.executable}" "{script}"\r\n'
            else:
                launcher, text = os.path.join(d, "fake_guard.sh"), f'#!/bin/sh\nexec "{sys.executable}" "{script}"\n'
            with open(launcher, "w", newline="") as f:
                f.write(text)
            os.chmod(launcher, 0o755)
            err = io.StringIO()
            with mock.patch.object(guard, "available", return_value=True), \
                    mock.patch.object(guard, "BINARY", launcher), \
                    mock.patch.dict(os.environ, {"FAKE_OUT": out, "FAKE_EXIT": str(code)}), \
                    contextlib.redirect_stderr(err):
                return guard.check(ps), err.getvalue()

    def test_none_when_the_binary_fails(self):
        p = para(["a"], [True], [["b"]], ["b"])  # ids: a=1 b=2, so the correct answer is "2\n"
        missing = os.path.join(tempfile.gettempdir(), "no_such_dir_for_guard", "guard")
        with mock.patch.object(guard, "available", return_value=True), mock.patch.object(guard, "BINARY", missing):
            self.assertIsNone(guard.check([p]))  # binary does not exist
        self.assertIsNone(self.check_with_fake("2\n", [p], code=3)[0])  # non-zero exit
        self.assertIsNone(self.check_with_fake("2 ", [p])[0])  # no trailing newline (a valid answer once cut)
        self.assertIsNone(self.check_with_fake("2\n2\n", [p])[0])  # wrong number of lines

    # Ids are numbered per paragraph in order of first appearance (see guard._encode).
    def test_wrong_answer_on_an_unflagged_slot_fails_closed(self):
        p = para(["a", "b"], [True, False], [["x"], ["y"]], ["x", "y"])
        # ids: a=1 x=2 b=3 y=4. Correct answer is "2 3"; the fake applies the proposal y on the unflagged slot.
        res, err = self.check_with_fake("2 4\n", [p])
        self.assertIsNone(res)
        self.assertEqual(err, "[note] the checker's answer failed the Python re-check; no suggestions\n")

    def test_missing_a_legal_suggestion_is_also_a_disagreement(self):
        p = para(["a"], [True], [["x"]], ["x"])
        res, err = self.check_with_fake("1\n", [p])
        self.assertIsNone(res)
        self.assertEqual(err.count("\n"), 1)  # exactly one line

    def test_correct_answer_passes(self):
        p = para(["a", "b"], [True, False], [["x"], ["y"]], ["x", "y"])
        res, err = self.check_with_fake("2 3\n", [p])
        self.assertEqual(res, [["x", "b"]])
        self.assertEqual(err, "")

    def test_expected_rules(self):
        p = para(["a", "b", "c"], [True, True, False], [["x"], ["y"], ["z"]], ["x", "zzz", "z", "extra"])
        self.assertEqual(guard._expected(p), ["x", "b", "c"])
        self.assertEqual(guard._expected(dict(p, proposal=["x"])), ["x", "b", "c"])  # missing proposal keeps the word


if __name__ == "__main__":
    unittest.main()
