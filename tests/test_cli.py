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

    def test_duplicate_output_ignores_case(self):
        with self.assertRaises(SetupError):  # a.md and A.md are one file on macOS and Windows disks
            cli._check_distinct_outputs([self.p("a.wav"), self.p("A.mp3")], None)

    def test_stale_sidecar_removed_without_confidence(self):
        wav, side = self.p("a.wav"), self.p("a.review.json")
        json.dump({"version": 1, "paragraphs": []}, open(side, "w"))
        run([wav], ["text"])
        self.assertFalse(os.path.exists(side))

    def _conf_run(self, wav, extra_patches=()):
        class Rec(Backend):
            name = "mlx"
            record_confidence = True

        def fake(b, wav_, lang, ctx, chunk, prog, paragraphs, corrector, partial=None):
            paragraphs.append({"offset": 0.0, "text": "t"})
            return "t"

        with contextlib.ExitStack() as st:
            for p in extra_patches:
                st.enter_context(p)
            st.enter_context(mock.patch.object(cli, "_load_backend", lambda *a: Rec()))
            st.enter_context(mock.patch.object(audio, "load_audio", lambda p: np.zeros(16000, "float32")))
            st.enter_context(mock.patch.object(pipeline, "transcribe_wav", fake))
            out = st.enter_context(contextlib.redirect_stdout(io.StringIO()))
            code = cli.main([wav, "--device", "cpu", "--model", "light", "--batch-size", "1", "--confidence"])
        return code, out.getvalue()

    def test_sidecar_written_with_confidence(self):
        wav, side = self.p("a.wav"), self.p("a.review.json")
        self._conf_run(wav)
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

    def _cli_run(self, argv):
        return cli.run(cli.build_parser().parse_args(argv + ["--device", "cpu", "--model", "light", "--batch-size", "1"]))

    def test_missing_input_refused_before_backend_loads(self):
        loaded = []
        with mock.patch.object(cli, "_load_backend", lambda *a: loaded.append(1)):
            with self.assertRaises(SetupError) as cm:
                self._cli_run([self.p("here.wav"), os.path.join(self.d, "gone.wav")])
        self.assertIn("File not found", str(cm.exception))
        self.assertIn("gone.wav", str(cm.exception))
        self.assertNotIn("here.wav", str(cm.exception))
        self.assertEqual(loaded, [])

    def test_unwritable_folder_refused_before_backend_loads(self):
        loaded = []
        wav = self.p("a.wav")
        with mock.patch.object(cli, "_load_backend", lambda *a: loaded.append(1)), \
                mock.patch.object(cli.os, "open", side_effect=[PermissionError("no"), AssertionError("retried")]) as probe:
            with self.assertRaises(SetupError) as cm:
                self._cli_run([wav])
        self.assertEqual(probe.call_count, 1)  # one attempt: a retry loop hangs on a folder that denies writing
        self.assertIn(self.d, str(cm.exception))
        self.assertIn("--out-dir", str(cm.exception))
        self.assertEqual(loaded, [])

    def test_empty_text_keeps_existing_transcript(self):
        wav, md, side = self.p("a.wav"), self.p("a.md"), self.p("a.review.json")
        open(md, "wb").write(b"HAND REVIEWED\n")
        open(side, "w").write("{}")
        self.assertEqual(run([wav], ["  \n"]), 1)
        self.assertEqual(open(md, "rb").read(), b"HAND REVIEWED\n")
        self.assertEqual(open(side).read(), "{}")
        self.assertEqual(sorted(os.listdir(self.d)), ["a.md", "a.review.json", "a.wav"])

    def test_replace_retried_then_succeeds(self):
        wav, md = self.p("a.wav"), self.p("a.md")
        open(md, "w").write("old\n")
        real, calls = os.replace, []

        def flaky(src, dst):
            calls.append(1)
            if len(calls) <= 2:
                raise PermissionError("in use")
            return real(src, dst)

        with mock.patch.object(cli, "SAVE_RETRY_SECONDS", 0), mock.patch.object(cli.os, "replace", flaky):
            self.assertEqual(run([wav], ["new"]), 0)
        self.assertEqual(open(md).read(), "new\n")
        self.assertEqual(len([f for f in os.listdir(self.d) if ".bak-" in f]), 1)
        self.assertFalse([f for f in os.listdir(self.d) if ".new-" in f or f.endswith(".tmp")])

    def _locked(self, *locked):
        """os.replace that fails whenever it touches one of the `locked` paths."""
        real = os.replace

        def replace(src, dst):
            if any(os.path.abspath(x) in (os.path.abspath(src), os.path.abspath(dst)) for x in locked):
                raise PermissionError("in use by another program")
            return real(src, dst)

        return mock.patch.object(cli.os, "replace", replace)

    def test_locked_target_saves_next_to_it_and_keeps_the_old_one(self):
        wav, md = self.p("a.wav"), self.p("a.md")
        open(md, "w").write("HAND REVIEWED\n")
        with mock.patch.object(cli, "SAVE_RETRY_SECONDS", 0), self._locked(md):
            self.assertEqual(run([wav], ["fresh"]), 1)
        self.assertEqual(open(md).read(), "HAND REVIEWED\n")
        news = [f for f in os.listdir(self.d) if ".new-" in f]
        self.assertEqual(len(news), 1)
        self.assertTrue(news[0].startswith("a.new-") and news[0].endswith(".md"))
        self.assertEqual(open(os.path.join(self.d, news[0])).read(), "fresh\n")
        self.assertFalse([f for f in os.listdir(self.d) if f.endswith(".tmp") or ".bak-" in f])

    def test_locked_target_and_folder_falls_back_to_home(self):
        wav, md = self.p("a.wav"), self.p("a.md")
        open(md, "w").write("old\n")
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        real = cli._write_new

        def write(path, text):
            if os.path.dirname(path) == self.d:
                raise PermissionError("read-only")
            return real(path, text)

        with mock.patch.object(cli, "SAVE_RETRY_SECONDS", 0), self._locked(md), \
                mock.patch.object(cli, "_write_new", write), \
                mock.patch.object(cli.os.path, "expanduser", lambda p: home.name):
            self.assertEqual(run([wav], ["fresh"]), 1)
        self.assertEqual(open(md).read(), "old\n")
        (name,) = os.listdir(home.name)
        self.assertIn(".new-", name)
        self.assertEqual(open(os.path.join(home.name, name)).read(), "fresh\n")

    def _partial_run(self, wav, fake):
        with mock.patch.object(cli, "_load_backend", lambda *a: Backend()), \
                mock.patch.object(audio, "load_audio", lambda p: np.zeros(16000, "float32")), \
                mock.patch.object(pipeline, "transcribe_wav", fake), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            code = cli.main([wav, "--device", "cpu", "--model", "light", "--batch-size", "1"])
        return code, out.getvalue()

    def test_partial_file_holds_the_first_pieces_and_is_kept_on_failure(self):
        wav = self.p("a.wav")
        partial_md = os.path.join(self.d, "a.partial.md")

        def fake(*a, partial=None, **k):
            partial("[00:00:00] uno")
            partial("[00:00:00] uno\n\n[00:00:20] due")
            raise ValueError("boom")

        code, out = self._partial_run(wav, fake)
        self.assertEqual(code, 1)
        self.assertEqual(open(partial_md).read(), "[00:00:00] uno\n\n[00:00:20] due\n")
        self.assertIn(partial_md, out)
        self.assertFalse(os.path.exists(os.path.join(self.d, "a.md")))

    def test_partial_file_removed_after_success_silently(self):
        wav = self.p("a.wav")
        open(os.path.join(self.d, "a.partial.md"), "w").write("stale from a crashed run\n")

        def fake(*a, partial=None, **k):
            partial("[00:00:00] uno")
            return "[00:00:00] uno"

        code, out = self._partial_run(wav, fake)
        self.assertEqual(code, 0)
        self.assertEqual(sorted(os.listdir(self.d)), ["a.md", "a.wav"])
        self.assertNotIn("partial", out)

    def test_failing_partial_write_does_not_fail_the_run(self):
        wav = self.p("a.wav")

        def fake(*a, partial=None, **k):
            partial("[00:00:00] uno")
            partial("[00:00:00] uno due")
            return "[00:00:00] uno due"

        real = cli._write_atomic

        def write(path, text):
            if path.endswith(".partial.md"):
                raise OSError("disk full")
            return real(path, text)

        with mock.patch.object(cli, "_write_atomic", write):
            code, out = self._partial_run(wav, fake)
        self.assertEqual(code, 0)
        self.assertEqual(open(os.path.join(self.d, "a.md")).read(), "[00:00:00] uno due\n")
        self.assertEqual(out.count("[note] cannot keep the part"), 1)

    def test_interrupt_keeps_the_partial_and_exits_130(self):
        wav = self.p("a.wav")

        def fake(*a, partial=None, **k):
            partial("[00:00:00] uno")
            raise KeyboardInterrupt()

        code, out = self._partial_run(wav, fake)
        self.assertEqual(code, 130)
        self.assertEqual(open(os.path.join(self.d, "a.partial.md")).read(), "[00:00:00] uno\n")
        self.assertIn("a.partial.md", out)

    def test_transcript_written_with_unix_line_ends(self):
        wav, seen = self.p("a.wav"), []

        def spy(path, mode="r", *a, **k):
            if "w" in mode:
                seen.append(k.get("newline"))  # newline="\n" is what keeps Windows from writing CRLF
            return open(path, mode, *a, **k)

        with mock.patch.object(cli, "open", spy, create=True):
            run([wav], ["[00:00:00] uno\n\n[00:00:20] due"])
        self.assertEqual(seen, ["\n"])
        self.assertEqual(open(os.path.join(self.d, "a.md"), "rb").read(), b"[00:00:00] uno\n\n[00:00:20] due\n")

    def test_turns_partial_grows(self):
        from localtranscribe import diarize

        class B:
            batch_size = 1
            texts = iter(["uno", "due", "tre", "quattro", "cinque", "sei"])

            def transcribe(self, pieces, language, context):
                return [next(self.texts) for _ in pieces]

        runs = [(0.0, 2.0, 0), (3.0, 5.0, 1), (6.0, 8.0, 1), (9.0, 11.0, 0), (12.0, 14.0, 0), (15.0, 17.0, 1)]
        seen = []
        turns = diarize.transcribe_turns(B(), np.zeros(16000 * 20, np.float32), runs, "Italian",
                                         partial=lambda t: seen.append(t))
        self.assertEqual(seen, [[(0.0, 0, "uno"), (3.0, 1, "due tre"), (9.0, 0, "quattro")], turns])
        self.assertEqual(turns, [(0.0, 0, "uno"), (3.0, 1, "due tre"), (9.0, 0, "quattro cinque"), (15.0, 1, "sei")])
        self.assertEqual(cli._format_turns(turns[:2]), "[00:00:00] Parlante 1: uno\n\n[00:00:03] Parlante 2: due tre")

    def test_wav_partial_is_the_prefix_of_the_final_text(self):
        class B:
            batch_size = 1

            def transcribe(self, pieces, language, context):
                return [f"t{len(p)}" for p in pieces]

        chunks = [(np.zeros(n), n * 20.0) for n in range(1, 7)]
        seen = []
        with mock.patch.object(pipeline, "split_wav", lambda *a: chunks):
            text = pipeline.transcribe_wav(B(), None, "Italian", partial=seen.append)
        self.assertEqual(len(seen), 2)
        self.assertTrue(text.startswith(seen[0]) and 0 < len(seen[0]) < len(text))
        self.assertEqual(seen[-1], text)

    def _home(self):
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        return home.name, mock.patch.object(cli.os.path, "expanduser", lambda p: home.name)

    def test_old_transcript_is_never_moved(self):
        wav, md = self.p("a.wav"), self.p("a.md")
        open(md, "w").write("HAND REVIEWED\n")
        real, moved = os.replace, []
        with mock.patch.object(cli.os, "replace", lambda a, b: (moved.append(a), real(a, b))):
            self.assertEqual(run([wav], ["fresh"]), 0)
        self.assertNotIn(md, moved)
        self.assertEqual(open(md).read(), "fresh\n")
        (bak,) = [f for f in os.listdir(self.d) if ".bak-" in f]
        self.assertEqual(open(os.path.join(self.d, bak)).read(), "HAND REVIEWED\n")

    def test_interrupt_while_swapping_keeps_the_old_transcript(self):
        wav, md = self.p("a.wav"), self.p("a.md")
        open(md, "w").write("HAND REVIEWED\n")
        real = os.replace

        def replace(src, dst):
            if dst == md:
                raise KeyboardInterrupt()
            return real(src, dst)

        def fake(*a, partial=None, **k):
            partial("fresh")
            return "fresh"

        with mock.patch.object(cli.os, "replace", replace):
            code, out = self._partial_run(wav, fake)
        self.assertEqual(code, 130)
        self.assertEqual(open(md).read(), "HAND REVIEWED\n")
        self.assertEqual(open(os.path.join(self.d, "a.partial.md")).read(), "fresh\n")
        self.assertIn("a.partial.md", out)
        for f in os.listdir(self.d):
            self.assertFalse(f.endswith(".tmp"))
            if ".bak-" in f:
                self.assertEqual(open(os.path.join(self.d, f)).read(), "HAND REVIEWED\n")

    def test_interrupt_just_after_the_swap_keeps_the_old_text_as_a_backup(self):
        wav, md = self.p("a.wav"), self.p("a.md")
        open(md, "w").write("HAND REVIEWED\n")
        real = os.replace

        def replace(src, dst):
            real(src, dst)
            if dst == md:
                raise KeyboardInterrupt()

        with mock.patch.object(cli.os, "replace", replace):
            self.assertEqual(run([wav], ["fresh"]), 130)
        self.assertEqual(open(md).read(), "fresh\n")
        (bak,) = [f for f in os.listdir(self.d) if ".bak-" in f]
        self.assertEqual(open(os.path.join(self.d, bak)).read(), "HAND REVIEWED\n")

    def test_fallback_needs_no_rename(self):
        wav, md = self.p("a.wav"), self.p("a.md")
        open(md, "w").write("old\n")
        home, patch_home = self._home()
        with mock.patch.object(cli, "SAVE_RETRY_SECONDS", 0), patch_home, \
                mock.patch.object(cli.os, "replace", side_effect=PermissionError("held by the antivirus")):
            self.assertEqual(run([wav], ["fresh"]), 1)
        self.assertEqual(open(md).read(), "old\n")
        (new,) = [f for f in os.listdir(self.d) if ".new-" in f]
        self.assertEqual(open(os.path.join(self.d, new)).read(), "fresh\n")
        self.assertFalse([f for f in os.listdir(self.d) if f.endswith(".tmp") or ".bak-" in f])
        self.assertEqual(os.listdir(home), [])

    def test_fallback_names_do_not_collide(self):
        wav, md = self.p("a.wav"), self.p("a.md")
        open(md, "w").write("old\n")
        home, patch_home = self._home()
        with mock.patch.object(cli, "SAVE_RETRY_SECONDS", 0), self._locked(md), patch_home, \
                mock.patch.object(cli.time, "strftime", lambda f: "20260930-101010"):
            run([wav], ["one"])
            run([wav], ["two"])
        news = sorted(open(os.path.join(self.d, f)).read() for f in os.listdir(self.d) if ".new-" in f)
        self.assertEqual(news, ["one\n", "two\n"])

    def test_every_place_failing_is_reported_and_keeps_the_partial(self):
        wav, md = self.p("a.wav"), self.p("a.md")
        open(md, "w").write("old\n")
        home, patch_home = self._home()

        def fake(*a, partial=None, **k):
            partial("fresh")
            return "fresh"

        with mock.patch.object(cli, "SAVE_RETRY_SECONDS", 0), self._locked(md), patch_home, \
                mock.patch.object(cli, "_write_new", side_effect=PermissionError("read-only")):
            code, out = self._partial_run(wav, fake)
        self.assertEqual(code, 1)
        self.assertEqual(open(md).read(), "old\n")
        self.assertEqual(open(os.path.join(self.d, "a.partial.md")).read(), "fresh\n")
        self.assertIn("[fail] a.wav", out)
        self.assertIn("the part transcribed so far is in " + os.path.join(self.d, "a.partial.md"), out)
        self.assertNotIn("done in", out)
        self.assertIn("1 of 1 files had problems (see the notes above)", out)
        self.assertEqual(os.listdir(home), [])

    def test_fallback_note_and_summary(self):
        wav, md = self.p("a.wav"), self.p("a.md")
        open(md, "w").write("old\n")
        home, patch_home = self._home()

        def fake(*a, partial=None, **k):
            return "fresh"

        with mock.patch.object(cli, "SAVE_RETRY_SECONDS", 0), self._locked(md), patch_home:
            code, out = self._partial_run(wav, fake)
        (new,) = [f for f in os.listdir(self.d) if ".new-" in f]
        self.assertIn(f"[note] could not replace {md} (in use by another program)", out)
        self.assertIn("saved as " + os.path.join(self.d, new), out)
        self.assertIn("1 of 1 files had problems (see the notes above)", out)

    def test_fallback_keeps_the_old_sidecar_without_confidence(self):
        wav, md, side = self.p("a.wav"), self.p("a.md"), self.p("a.review.json")
        open(md, "w").write("old\n")
        open(side, "w").write("{}")
        home, patch_home = self._home()
        with mock.patch.object(cli, "SAVE_RETRY_SECONDS", 0), self._locked(md), patch_home:
            self.assertEqual(run([wav], ["fresh"]), 1)
        self.assertEqual(open(side).read(), "{}")

    def test_fallback_with_confidence_puts_the_sidecar_next_to_the_new_file(self):
        wav, md, side = self.p("a.wav"), self.p("a.md"), self.p("a.review.json")
        open(md, "w").write("old\n")
        open(side, "w").write("{}")
        home, patch_home = self._home()
        code, out = self._conf_run(wav, [mock.patch.object(cli, "SAVE_RETRY_SECONDS", 0), self._locked(md), patch_home])
        self.assertEqual(code, 1)
        (new,) = [f for f in os.listdir(self.d) if f.startswith("a.new-") and f.endswith(".md")]
        newside = os.path.join(self.d, new[:-3] + ".review.json")
        self.assertEqual(json.load(open(newside))["paragraphs"][0]["text"], "t")
        self.assertEqual(open(side).read(), "{}")

    def test_sidecar_failure_after_a_save_is_a_note(self):
        wav, side = self.p("a.wav"), self.p("a.review.json")
        open(side, "w").write("{}")
        real = cli._write_atomic

        def write(path, text):
            if path.endswith(".review.json"):
                raise OSError("disk full")
            return real(path, text)

        code, out = self._conf_run(wav, [mock.patch.object(cli, "_write_atomic", write)])
        self.assertEqual(code, 1)
        self.assertEqual(open(os.path.join(self.d, "a.md")).read(), "t\n")
        self.assertFalse(os.path.exists(side))
        self.assertIn("[note] could not write " + side, out)
        self.assertNotIn("[fail]", out)
        self.assertIn("1 of 1 files had problems", out)

    def test_sidecar_failure_after_a_fallback_counts_once(self):
        wav, md = self.p("a.wav"), self.p("a.md")
        open(md, "w").write("old\n")
        home, patch_home = self._home()
        real = cli._write_atomic

        def write(path, text):
            if path.endswith(".review.json"):
                raise OSError("disk full")
            return real(path, text)

        code, out = self._conf_run(wav, [mock.patch.object(cli, "SAVE_RETRY_SECONDS", 0), self._locked(md), patch_home,
                                         mock.patch.object(cli, "_write_atomic", write)])
        self.assertEqual(code, 1)
        self.assertIn("1 of 1 files had problems", out)

    def test_probe_that_cannot_be_removed_does_not_block(self):
        wav = self.p("a.wav")
        with mock.patch.object(cli.os, "remove", side_effect=PermissionError("held by the antivirus")):
            cli._check_inputs_and_folders([wav], None)  # the folder is writable: no SetupError
        if hasattr(os, "O_TEMPORARY"):  # Windows deletes the probe itself on close
            self.assertEqual(os.listdir(self.d), ["a.wav"])

    def test_new_out_dir_created_before_the_model_loads(self):
        wav, out = self.p("a.wav"), os.path.join(self.d, "new", "sub")
        cli._check_inputs_and_folders([wav], out)
        self.assertEqual(os.listdir(out), [])

    def test_partial_write_tried_again_after_a_failure(self):
        wav, calls = self.p("a.wav"), []
        real = cli._write_atomic

        def write(path, text):
            calls.append(path)
            if len(calls) == 1:
                raise PermissionError("held by the antivirus")
            return real(path, text)

        def fake(*a, partial=None, **k):
            partial("uno")
            partial("uno due")
            raise ValueError("boom")

        with mock.patch.object(cli, "_write_atomic", write):
            code, out = self._partial_run(wav, fake)
        self.assertEqual(open(os.path.join(self.d, "a.partial.md")).read(), "uno due\n")
        self.assertEqual(out.count("[note] cannot keep the part"), 1)

    def test_speakers_partial_is_written(self):
        from localtranscribe import diarize

        wav = self.p("a.wav")

        def turns(b, wav_, runs, lang, context="", progress=None, partial=None):
            partial([(0.0, 0, "uno")])
            partial([(0.0, 0, "uno"), (3.0, 1, "due")])
            raise ValueError("boom")

        with mock.patch.object(diarize, "diarize", lambda *a, **k: [(0.0, 2.0, 0)]), \
                mock.patch.object(diarize, "transcribe_turns", turns), \
                mock.patch.object(cli, "_load_backend", lambda *a: Backend()), \
                mock.patch.object(audio, "load_audio", lambda p: np.zeros(16000, "float32")), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            code = cli.main([wav, "--device", "cpu", "--model", "light", "--batch-size", "1", "--speakers", "2"])
        self.assertEqual(code, 1)
        self.assertEqual(open(os.path.join(self.d, "a.partial.md")).read(),
                         "[00:00:00] Parlante 1: uno\n\n[00:00:03] Parlante 2: due\n")
        self.assertIn("a.partial.md", out.getvalue())

    def test_wav_partial_one_piece_at_a_time(self):
        class B:
            batch_size = 1
            record_confidence = True

            def transcribe(self, pieces, language, context):
                self.last_records = [{"aligned": False, "tokens": []} for _ in pieces]
                return [f"t{len(p)}" for p in pieces]

            def word_continuations(self, *a):
                raise AssertionError("no words, no candidates")

        chunks = [(np.zeros(n), n * 20.0) for n in range(1, 4)]
        seen, paragraphs = [], []
        with mock.patch.object(pipeline, "split_wav", lambda *a: chunks):
            text = pipeline.transcribe_wav(B(), None, "Italian", paragraphs=paragraphs, partial=seen.append)
        self.assertEqual(seen, ["[00:00:20] t1", "[00:00:20] t1\n\n[00:00:40] t2", text])
        self.assertEqual(len(paragraphs), 3)

    def test_half_written_backup_copy_removed(self):
        wav, md = self.p("a.wav"), self.p("a.md")
        open(md, "w").write("old\n")
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)

        def half(src, dst):
            with open(dst, "w") as f:
                f.write("ol")
            raise OSError("disk full")

        with mock.patch.object(cli, "SAVE_RETRY_SECONDS", 0), mock.patch("shutil.copyfile", half), \
                mock.patch.object(cli.os.path, "expanduser", lambda p: home.name):
            self.assertEqual(run([wav], ["fresh"]), 1)
        self.assertEqual(open(md).read(), "old\n")
        self.assertFalse([f for f in os.listdir(self.d) if ".bak-" in f or f.endswith(".tmp")])
        (new,) = [f for f in os.listdir(self.d) if ".new-" in f]
        self.assertEqual(open(os.path.join(self.d, new)).read(), "fresh\n")

    def test_new_text_and_backup_on_disk_before_the_swap(self):
        wav, md = self.p("a.wav"), self.p("a.md")
        open(md, "w").write("HAND REVIEWED\n")
        events, real_fsync, real_replace = [], os.fsync, os.replace

        def fsync(fd):
            events.append(("synced", os.fstat(fd).st_ino))
            real_fsync(fd)

        def replace(a, b):
            events.append(("replaced", os.stat(a).st_ino, b))
            real_replace(a, b)

        with mock.patch.object(cli.os, "fsync", fsync), mock.patch.object(cli.os, "replace", replace):
            self.assertEqual(run([wav], ["fresh"]), 0)
        (bak,) = [f for f in os.listdir(self.d) if ".bak-" in f]
        (swap,) = [i for i, e in enumerate(events) if e[0] == "replaced" and e[2] == md]
        self.assertIn(("synced", events[swap][1]), events[:swap])
        self.assertIn(("synced", os.stat(os.path.join(self.d, bak)).st_ino), events[:swap])

    def test_file_system_without_fsync_still_saves(self):
        wav, md = self.p("a.wav"), self.p("a.md")
        open(md, "w").write("old\n")
        with mock.patch.object(cli.os, "fsync", side_effect=OSError(22, "Invalid argument")):
            self.assertEqual(run([wav], ["fresh"]), 0)
        self.assertEqual(open(md).read(), "fresh\n")
        self.assertEqual(len([f for f in os.listdir(self.d) if ".bak-" in f]), 1)

    def test_undecodable_old_transcript_is_kept(self):
        wav, md = self.p("a.wav"), self.p("a.md")
        open(md, "wb").write(b"caff\xe8 rivisto\r\n")  # saved as ANSI by a Windows editor
        self.assertEqual(run([wav], ["fresh"]), 0)
        (bak,) = [f for f in os.listdir(self.d) if ".bak-" in f]
        self.assertEqual(open(os.path.join(self.d, bak), "rb").read(), b"caff\xe8 rivisto\r\n")

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
