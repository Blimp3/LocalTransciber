import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from localtranscribe import audio, cli, pipeline
from localtranscribe.devices import SetupError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Backend:
    name = "cpu"
    record_confidence = False
    notes = []

    def describe(self):
        return "stub"


def run(argv, texts, load=None):
    """cli.main with a stub backend, silent audio and the given transcripts (an Exception item is raised)."""
    it = iter(texts)

    def fake(*a, **k):
        t = next(it)
        if isinstance(t, BaseException):
            raise t
        return t

    loader = load or (lambda *a: Backend())
    with mock.patch.object(cli, "_load_backend", loader), \
            mock.patch.object(audio, "load_audio", lambda p: np.zeros(16000, "float32")), \
            mock.patch.object(pipeline, "transcribe_wav", fake), \
            contextlib.redirect_stdout(io.StringIO()):
        return cli.main(argv + ["--device", "cpu", "--model", "light", "--batch-size", "1"])


class CliTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.d = tmp.name

    def p(self, name):
        path = os.path.join(self.d, name)
        if name.endswith((".wav", ".mp3")):
            open(path, "wb").write(b"x")
        return path

    def test_rerun_keeps_backup_of_different_text(self):
        wav, md = self.p("a.wav"), self.p("a.md")
        open(md, "w").write("HAND REVIEWED\n")
        self.assertEqual(run([wav], ["fresh"]), 0)
        self.assertEqual(open(md).read(), "fresh\n")
        baks = [f for f in os.listdir(self.d) if ".bak-" in f]
        self.assertEqual(len(baks), 1)
        self.assertTrue(baks[0].endswith(".md"))
        self.assertEqual(open(os.path.join(self.d, baks[0])).read(), "HAND REVIEWED\n")
        self.assertFalse([f for f in os.listdir(self.d) if f.endswith(".tmp")])

    def test_backup_names_do_not_collide(self):
        wav, md = self.p("a.wav"), self.p("a.md")
        with mock.patch.object(cli.time, "strftime", lambda f: "20260930-101010"):
            for old, new in (("one", "two"), ("two", "three")):
                open(md, "w").write(old + "\n")
                run([wav], [new])
        self.assertEqual(len([f for f in os.listdir(self.d) if ".bak-" in f]), 2)

    def test_identical_rerun_makes_no_backup(self):
        wav, md = self.p("a.wav"), self.p("a.md")
        open(md, "w").write("same\n")
        run([wav], ["same"])
        self.assertEqual(sorted(os.listdir(self.d)), ["a.md", "a.wav"])

    def test_duplicate_output_refused_before_backend_loads(self):
        loaded = []
        with self.assertRaises(SetupError) as cm:
            with mock.patch.object(cli, "_load_backend", lambda *a: loaded.append(1)):
                cli.run(cli.build_parser().parse_args([self.p("a.wav"), self.p("a.mp3"), "--device", "cpu",
                                                       "--model", "light", "--batch-size", "1"]))
        self.assertIn("a.wav", str(cm.exception))
        self.assertIn("a.mp3", str(cm.exception))
        self.assertEqual(loaded, [])
        self.assertEqual(run([self.p("a.wav"), self.p("a.mp3")], ["x", "y"]), 2)

    def test_stale_sidecar_removed_without_confidence(self):
        wav, side = self.p("a.wav"), self.p("a.review.json")
        json.dump({"version": 1, "paragraphs": []}, open(side, "w"))
        run([wav], ["text"])
        self.assertFalse(os.path.exists(side))

    def test_sidecar_written_with_confidence(self):
        wav, side = self.p("a.wav"), self.p("a.review.json")

        class Rec(Backend):
            name = "mlx"
            record_confidence = True

        def fake(b, wav_, lang, ctx, chunk, prog, paragraphs, corrector):
            paragraphs.append({"offset": 0.0, "text": "t"})
            return "t"

        with mock.patch.object(cli, "_load_backend", lambda *a: Rec()), \
                mock.patch.object(audio, "load_audio", lambda p: np.zeros(16000, "float32")), \
                mock.patch.object(pipeline, "transcribe_wav", fake), \
                contextlib.redirect_stdout(io.StringIO()):
            cli.main([wav, "--device", "cpu", "--model", "light", "--batch-size", "1", "--confidence"])
        self.assertEqual(json.load(open(side))["paragraphs"][0]["text"], "t")

    def test_failing_file_does_not_stop_the_next(self):
        a, b = self.p("a.wav"), self.p("b.wav")
        self.assertEqual(run([a, b], [ValueError("boom"), "ok"]), 1)
        self.assertFalse(os.path.exists(os.path.join(self.d, "a.md")))
        self.assertEqual(open(os.path.join(self.d, "b.md")).read(), "ok\n")

    def test_setup_error_still_propagates(self):
        self.assertEqual(run([self.p("a.wav")], [SetupError("no model")]), 2)

    def test_keyboard_interrupt_still_escapes(self):
        self.assertEqual(run([self.p("a.wav")], [KeyboardInterrupt()]), 130)

    def test_chunk_rejected(self):
        for bad in ("0", "-5", "nan", "inf", "4.9", "121", "abc"):
            with self.subTest(bad), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.build_parser().parse_args(["x.wav", "--chunk", bad])
        self.assertEqual(cli.build_parser().parse_args(["x.wav", "--chunk", "30"]).chunk, 30)

    def _offline(self, env_value):
        env = {k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"}
        if env_value is not None:
            env["HF_HUB_OFFLINE"] = env_value
        code = ("import os,sys,types\n"
                "m=types.ModuleType('localtranscribe.cli'); m.main=lambda: print(os.environ.get('HF_HUB_OFFLINE'))\n"
                "import localtranscribe; sys.modules['localtranscribe.cli']=m\n"
                "import runpy; runpy.run_module('localtranscribe.__main__', run_name='__main__')\n")
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
        return r.stdout.strip()

    def test_offline_by_default_but_overridable(self):
        self.assertEqual(self._offline(None), "1")
        self.assertEqual(self._offline("0"), "0")


if __name__ == "__main__":
    unittest.main()
