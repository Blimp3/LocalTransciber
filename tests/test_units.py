"""Unit tests that need no model, no GPU and no network.   python -m unittest discover -s tests -v"""
import os
import stat
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from localtranscribe import config, devices, pipeline, textutil  # noqa: E402

SR = config.SAMPLE_RATE
SAMPLE = os.path.join(ROOT, "tests", "fleurs_it_sample.wav")


def noisy_speechlike(seconds, seed=0, gap_every=7.0):
    """Noise bursts separated by short silences, so the chunker has quiet places to cut."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    wav = (rng.standard_normal(n) * 0.1).astype(np.float32)
    step = int(gap_every * SR)
    for s in range(step, n, step):
        wav[s:s + int(0.3 * SR)] *= 0.001
    return wav


class FakeBackend:
    """Returns the piece length in seconds as its 'transcript', or a canned text."""

    name = "fake"
    batch_size = 2

    def __init__(self, texts=None):
        self.texts = texts
        self.calls = []

    def transcribe(self, pieces, language, context=""):
        self.calls.append((len(pieces), language, context))
        if self.texts is not None:
            return [self.texts.pop(0) for _ in pieces]
        return [f"{len(p) / SR:.1f}s" for p in pieces]


class ChunkerTests(unittest.TestCase):
    def test_short_audio_is_one_piece(self):
        for seconds in (0.3, 5, 20, 24.9):
            wav = noisy_speechlike(seconds)
            pieces = textutil.split_audio_into_chunks(wav, SR, 20)
            self.assertEqual(len(pieces), 1, seconds)

    def test_pieces_reproduce_audio_and_respect_limits(self):
        for seconds in (26, 47.3, 61, 125.5, 300):
            wav = noisy_speechlike(seconds, seed=int(seconds))
            pieces = textutil.split_audio_into_chunks(wav, SR, 20)
            self.assertTrue(np.array_equal(np.concatenate([p for p, _ in pieces]), wav), seconds)
            lengths = [len(p) / SR for p, _ in pieces]
            self.assertLessEqual(max(lengths), 25.01, seconds)
            self.assertGreaterEqual(lengths[-1], 3.0 - 1e-6, seconds)  # no scrap at the end
            offsets = [o for _, o in pieces]
            self.assertAlmostEqual(offsets[1], lengths[0], places=3)

    def test_matches_qwen_asr_helper_before_the_tail(self):
        """The cuts equal the official helper's (the only change is at the end of the audio)."""
        try:
            from qwen_asr.inference.utils import split_audio_into_chunks as official
        except Exception:
            self.skipTest("qwen-asr is not installed here (Mac install)")
        wav = noisy_speechlike(400, seed=3)
        ours = [len(p) for p, _ in textutil.split_audio_into_chunks(wav, SR, 20)]
        theirs = [len(p) for p, _ in official(wav, SR, max_chunk_sec=20)]
        common = min(len(ours), len(theirs)) - 2
        self.assertGreater(common, 10)
        self.assertEqual(ours[:common], theirs[:common])

    def test_repetition_fix_matches_qwen_asr(self):
        samples = ["ciao " * 60, "a" * 200 + " fine", "normale frase di prova", "ah" * 80, "no no no " * 30 + "basta", ""]
        try:
            from qwen_asr.inference.utils import detect_and_fix_repetitions as official
        except Exception:
            official = None
        for s in samples:
            fixed = textutil.detect_and_fix_repetitions(s)
            if official:
                self.assertEqual(fixed, official(s), s[:30])
        self.assertEqual(textutil.detect_and_fix_repetitions("a" * 200), "a")
        self.assertEqual(textutil.detect_and_fix_repetitions("normale frase"), "normale frase")


class TextTests(unittest.TestCase):
    def test_strip_context_echo(self):
        ctx = "Mario Rossi Politecnico di Milano LoRaWAN"
        self.assertEqual(textutil.strip_context_echo("Buongiorno a tutti. Mario Rossi Politecnico di Milano", ctx), "Buongiorno a tutti.")
        self.assertEqual(textutil.strip_context_echo("  nessun eco  ", ctx), "nessun eco")
        self.assertEqual(textutil.strip_context_echo(" testo ", ""), "testo")

    def test_strip_context_echo_keeps_real_speech(self):
        sce = textutil.strip_context_echo
        name = "Mario Rossi"
        for t in ("Mario Rossi apre la riunione.", "Oggi Mario Rossi presenta il progetto.", "Ha parlato con Mario Rossi"):
            self.assertEqual(sce(t, name), t)
        self.assertEqual(sce("Questo aroma cambia tutto.", "Roma"), "Questo aroma cambia tutto.")
        self.assertEqual(sce("Roma", "Roma"), "")
        self.assertEqual(sce("Mario Rossi", name), "")
        long_ctx = "Mario Rossi, Politecnico di Milano, LoRaWAN"
        self.assertEqual(sce("Sono Mario Rossi, Politecnico di Milano, e parlo di LoRaWAN.", long_ctx),
                         "Sono Mario Rossi, Politecnico di Milano, e parlo di LoRaWAN.")
        self.assertEqual(sce(long_ctx, long_ctx), "")

    def test_parse_asr_output(self):
        self.assertEqual(textutil.parse_asr_output("language Italian<asr_text>ciao a tutti"), ("Italian", "ciao a tutti"))
        self.assertEqual(textutil.parse_asr_output("language None<asr_text>"), ("", ""))
        self.assertEqual(textutil.parse_asr_output("  solo testo ", "Italian"), ("Italian", "solo testo"))

    def test_float_range_normalize(self):
        x = np.array([0.5, -2.0, 1.0], dtype=np.float32)
        y = textutil.float_range_normalize(x)
        self.assertAlmostEqual(float(np.abs(y).max()), 1.0)
        z = np.array([0.5, -0.25], dtype=np.float32)
        self.assertTrue(np.array_equal(textutil.float_range_normalize(z), z))


class PipelineTests(unittest.TestCase):
    def test_transcribe_wav_joins_pieces_and_strips_echo(self):
        wav = noisy_speechlike(60)
        chunks = pipeline.split_wav(wav, 20)
        fake = FakeBackend(texts=["uno", "due Mario Rossi Politecnico di Milano", ""] + ["x"] * 20)
        text = pipeline.transcribe_wav(fake, wav, "Italian", "Mario Rossi Politecnico di Milano", 20)
        self.assertEqual(text.split("\n\n")[:2], ["[00:00:00] uno", f"[{textutil.fmt_time(chunks[1][1])}] due"])
        self.assertGreaterEqual(chunks[1][1], 1)
        self.assertNotIn("\n" * 3, text)
        self.assertNotIn("Mario", text)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertEqual(fake.calls[0][2], "Mario Rossi Politecnico di Milano")

    def test_progress_and_slicing(self):
        wav = noisy_speechlike(400)
        seen = []
        fake = FakeBackend()
        pipeline.transcribe_wav(fake, wav, "Italian", progress=lambda d, t: seen.append((d, t)))
        self.assertEqual(seen[-1][0], seen[-1][1])
        self.assertTrue(all(n <= fake.batch_size * 4 for n, _, _ in fake.calls))


class SilenceGateTests(unittest.TestCase):
    def test_is_silent_levels(self):
        def square(db, n=SR * 20):  # RMS exactly `db` dBFS
            return (np.where(np.arange(n) % 2, 1.0, -1.0) * 10 ** (db / 20)).astype(np.float32)

        self.assertTrue(textutil.is_silent(np.zeros(SR * 20, np.float32), SR, -60))
        self.assertTrue(textutil.is_silent(np.zeros(0, np.float32), SR, -60))
        self.assertTrue(textutil.is_silent(square(-61), SR, -60))
        self.assertFalse(textutil.is_silent(square(-59), SR, -60))
        piece = np.zeros(SR * 20, np.float32)
        piece[12345:12345 + SR // 10] = square(-50, SR // 10)  # one quiet word of 100 ms, not on a frame boundary
        self.assertFalse(textutil.is_silent(piece, SR, -60))

    def test_silent_pieces_never_reach_the_backend(self):
        pieces = [noisy_speechlike(3), np.zeros(SR * 20, np.float32), noisy_speechlike(4),
                  np.zeros(SR * 10, np.float32)]
        fake, skipped, seen, parts = FakeBackend(), [], [], []
        texts = pipeline.transcribe_pieces(fake, pieces, "Italian", progress=lambda d, t: seen.append((d, t)),
                                           partial=parts.append, skipped=skipped)
        self.assertEqual(texts, ["3.0s", "", "4.0s", ""])
        self.assertEqual(fake.calls, [(2, "Italian", "")])
        self.assertEqual(skipped, [20.0, 10.0])
        self.assertEqual(seen[-1], (4, 4))
        self.assertEqual(parts[-1], texts)
        fake = FakeBackend()
        self.assertEqual(pipeline.transcribe_pieces(fake, pieces[1::2], "Italian"), ["", ""])
        self.assertEqual(fake.calls, [])  # nothing to hear, no call at all
        fake, skipped = FakeBackend(), []
        with mock.patch.object(config, "SILENCE_DBFS", None):  # read when called, so it can be switched off
            texts = pipeline.transcribe_pieces(fake, pieces, "Italian", skipped=skipped)
        self.assertEqual(texts, ["3.0s", "20.0s", "4.0s", "10.0s"])
        self.assertEqual(skipped, [])

    def test_transcribe_wav_keeps_offsets_and_says_what_it_skipped(self):
        import contextlib
        import io

        chunks = [(noisy_speechlike(20), 0.0), (np.zeros(SR * 20, np.float32), 20.0), (noisy_speechlike(20), 40.0)]
        with mock.patch.object(pipeline, "split_wav", lambda *a: chunks), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            text = pipeline.transcribe_wav(FakeBackend(), None, "Italian")
        self.assertEqual(text, "[00:00:00] 20.0s\n\n[00:00:40] 20.0s")
        self.assertEqual(out.getvalue(), "  [note] 20 s had no sound at all (1 piece below -60 dBFS) and were not "
                                         "transcribed.\n")

    def test_turns_skip_a_silent_run(self):
        import contextlib
        import io

        from localtranscribe import diarize

        wav = np.concatenate([noisy_speechlike(5), np.zeros(SR * 10, np.float32), noisy_speechlike(5, seed=1)])
        runs = [(0.0, 5.0, 0), (6.0, 14.0, 1), (15.0, 20.0, 0)]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            turns = diarize.transcribe_turns(FakeBackend(texts=["uno", "tre"]), wav, runs, "Italian")
        self.assertEqual(turns, [(0.0, 0, "uno tre")])
        self.assertEqual(out.getvalue().count("[note]"), 1)


class WerNormalizeTests(unittest.TestCase):
    def test_normalize_strips_timestamps(self):
        from benchmark import wer

        self.assertEqual(wer.normalize("[00:01:23] Ciao, mondo.\n\n[01:00:00] Sì"), "ciao mondo sì")


class DeviceChoiceTests(unittest.TestCase):
    def test_nvidia_model_choice(self):
        with mock.patch.object(devices, "is_apple_silicon", return_value=False):
            self.assertEqual(devices.choose_model("auto", "cuda", 24), config.TORCH_MODELS["best"])
            # the automatic rule is the precheck's: best from 6 GB of video memory (it was 12 GB before the precheck)
            for vram, want in ((12, "best"), (8, "best"), (6, "best"), (4, "light"), (3, "light"), (2, "light")):
                self.assertEqual(devices.choose_model("auto", "cuda", vram), config.TORCH_MODELS[want], vram)
            self.assertEqual(devices.choose_model("auto", "cpu", None), config.TORCH_MODELS["light"])
            self.assertEqual(devices.choose_model("best", "cpu", None), config.TORCH_MODELS["best"])
            self.assertEqual(devices.choose_model("acme/other-model", "cuda", 24), "acme/other-model")

    def test_mac_model_choice(self):
        with mock.patch.object(devices, "is_apple_silicon", return_value=True):
            self.assertEqual(devices.choose_model("auto", "mps", 16.0), config.MAC_MODELS["best"])
            self.assertEqual(devices.choose_model("auto", "mps", 24.0), config.MAC_MODELS["best"])
            self.assertEqual(devices.choose_model("auto", "mps", 8.0), config.MAC_MODELS["best"])
            self.assertEqual(devices.choose_model("best", "mps", 8.0), config.MAC_MODELS["best"])
            self.assertEqual(devices.choose_model("Qwen/Qwen3-ASR-0.6B", "mps", 8.0), config.MAC_MODELS["light"])
            self.assertEqual(devices.choose_model("mlx-community/Qwen3-ASR-1.7B-4bit", "mps", 8.0),
                             "mlx-community/Qwen3-ASR-1.7B-4bit")
            with self.assertRaises(devices.SetupError):
                devices.choose_model("someone/not-mlx", "mps", 8.0)

    def test_mac_batch_and_device(self):
        with mock.patch.object(devices, "is_apple_silicon", return_value=True):
            with mock.patch.object(devices, "total_ram_gb", return_value=8.0):
                self.assertEqual(devices.choose_batch_size(0, "mps", "x"), config.MAC_BATCH_SIZE)
            with mock.patch.object(devices, "total_ram_gb", return_value=24.0):
                self.assertEqual(devices.choose_batch_size(0, "mps", "x"), config.MAC_BATCH_SIZE)
            self.assertEqual(devices.choose_batch_size(5, "mps", "x"), 5)
            self.assertEqual(devices.resolve_device("auto"), "mps")
            self.assertEqual(devices.resolve_device("cpu"), "cpu")
            with self.assertRaises(devices.SetupError):
                devices.resolve_device("cuda")

    def test_non_mac_rejects_mps_and_scales_batch(self):
        with mock.patch.object(devices, "is_apple_silicon", return_value=False):
            with self.assertRaises(devices.SetupError):
                devices.resolve_device("mps")
            with mock.patch.object(devices, "cuda_memory_gb", return_value=(24.0, 22.0)):
                self.assertEqual(devices.choose_batch_size(0, "cuda", "Qwen/Qwen3-ASR-1.7B"), 8)
            with mock.patch.object(devices, "cuda_memory_gb", return_value=(6.0, 5.5)):
                self.assertLess(devices.choose_batch_size(0, "cuda", "Qwen/Qwen3-ASR-1.7B"), 8)
            with mock.patch.object(devices, "cuda_memory_gb", return_value=(4.0, 3.5)):
                self.assertEqual(devices.choose_batch_size(0, "cuda", "Qwen/Qwen3-ASR-0.6B"), 4)
            with mock.patch.object(devices, "cuda_memory_gb", return_value=(4.0, 3.5)):
                self.assertEqual(devices.choose_batch_size(0, "cuda", "Qwen/Qwen3-ASR-1.7B"), 1)  # tight: one piece at a time
            with mock.patch.object(devices, "cuda_memory_gb", return_value=(8.0, 7.5)):
                self.assertEqual(devices.choose_batch_size(0, "cuda", "Qwen/Qwen3-ASR-1.7B"), 8)
            with mock.patch.object(devices, "cuda_memory_gb", return_value=(6.0, 5.5)):
                self.assertEqual(devices.choose_batch_size(0, "cuda", "Qwen/Qwen3-ASR-1.7B"), 2)

    def test_intel_mac_is_refused(self):
        with mock.patch.object(sys, "platform", "darwin"), \
                mock.patch("platform.machine", return_value="x86_64"), \
                mock.patch.object(devices, "_sysctl", return_value="0"):
            with self.assertRaises(devices.SetupError) as cm:
                devices.check_platform()
            self.assertIn("Intel", str(cm.exception))


class ImportHygieneTests(unittest.TestCase):
    def test_mlx_and_torch_are_not_imported_by_the_light_modules(self):
        import subprocess

        code = ("import sys; sys.path.insert(0, %r);"
                "import localtranscribe.cli, localtranscribe.devices, localtranscribe.pipeline, localtranscribe.backends,"
                "localtranscribe.precheck, localtranscribe.setup_models;"
                "bad=[m for m in ('mlx','mlx_audio','torch','qwen_asr','transformers') if m in sys.modules];"
                "print(bad)" % ROOT)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip()
        self.assertEqual(out, "[]")

    def test_windows_backend_never_imports_mlx(self):
        import subprocess

        code = ("import sys; sys.path.insert(0, %r);"
                "from localtranscribe import devices;"
                "print(devices.is_apple_silicon(), 'mlx' in sys.modules, 'mlx_audio' in sys.modules)" % ROOT)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.split()
        if out and out[0] == "False":
            self.assertEqual(out[1:], ["False", "False"])


class AudioTests(unittest.TestCase):
    def test_decode_sample(self):
        from localtranscribe.audio import load_audio

        wav = load_audio(SAMPLE)
        self.assertEqual(wav.dtype, np.float32)
        self.assertEqual(wav.ndim, 1)
        self.assertAlmostEqual(len(wav) / SR, 6.54, delta=0.05)
        self.assertLessEqual(float(np.abs(wav).max()), 1.0)

    def test_decode_resamples_and_downmixes_various_formats(self):
        import av

        from localtranscribe.audio import load_audio

        base = load_audio(SAMPLE)
        stereo_48k = np.repeat(np.interp(np.arange(0, len(base), 1 / 3), np.arange(len(base)), base)[None, :], 2, axis=0)
        made = 0
        with tempfile.TemporaryDirectory() as d:
            for ext, codec, fmt in (("flac", "flac", "s16"), ("wav", "pcm_s16le", "s16"), ("ogg", "libvorbis", "fltp"),
                                    ("mp3", "libmp3lame", "fltp"), ("m4a", "aac", "fltp"), ("opus", "libopus", "flt")):
                path = os.path.join(d, "t." + ext)
                try:
                    _encode(av, path, stereo_48k.astype(np.float32), 48000, codec, fmt)
                except Exception:
                    continue  # this PyAV build has no such encoder: nothing to test
                wav = load_audio(path)
                made += 1
                self.assertAlmostEqual(len(wav) / SR, len(base) / SR, delta=0.25, msg=ext)
                self.assertEqual(wav.dtype, np.float32)
        self.assertGreaterEqual(made, 2)

    def test_missing_audio_track_is_reported(self):
        from localtranscribe.audio import load_audio

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.txt")
            with open(path, "w") as f:
                f.write("not audio")
            with self.assertRaises(Exception):
                load_audio(path)


def _encode(av, path, samples, rate, codec, fmt):
    layout = "stereo"
    with av.open(path, "w") as out:
        stream = out.add_stream(codec, rate=rate)
        stream.layout = layout
        stream.format = fmt
        step = 1024
        pts = 0
        for i in range(0, samples.shape[1], step):
            block = samples[:, i:i + step]
            if fmt in ("fltp",):
                data = np.ascontiguousarray(block)
            elif fmt == "flt":
                data = np.ascontiguousarray(block.T.reshape(1, -1))
            else:
                data = np.ascontiguousarray((block.T.reshape(1, -1) * 32767).astype(np.int16))
            frame = av.AudioFrame.from_ndarray(data, format=fmt, layout=layout)
            frame.sample_rate = rate
            frame.pts = pts
            pts += block.shape[1]
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)


class ChunkerGuardTests(unittest.TestCase):
    def _run(self, fn, secs=10):
        import signal

        def boom(*a):
            raise TimeoutError("chunker did not finish")
        if not hasattr(signal, "SIGALRM"):  # Windows: no alarm, run without the timeout guard
            return fn()
        old = signal.signal(signal.SIGALRM, boom)
        signal.alarm(secs)
        try:
            return fn()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    def test_noise_then_long_zeros_makes_bounded_progress(self):
        rng = np.random.default_rng(0)
        wav = np.concatenate([rng.standard_normal(3 * SR) * 0.1, np.zeros(30 * SR)]).astype(np.float32)
        pieces = self._run(lambda: textutil.split_audio_into_chunks(wav, SR, 5))
        self.assertLessEqual(len(pieces), 33 * 2 // 5 + 3)
        self.assertTrue(np.array_equal(np.concatenate([p for p, _ in pieces]), wav))

    def test_bad_chunk_seconds_raise(self):
        for bad in (float("nan"), float("inf"), 0, -5, 0.5):
            with self.assertRaises(ValueError):
                textutil.split_audio_into_chunks(np.zeros(SR * 30, np.float32), SR, bad)

    def test_default_20s_cuts_are_unchanged(self):
        wav = noisy_speechlike(60, seed=1)
        pieces = textutil.split_audio_into_chunks(wav, SR, 20)
        # the old rule (left = max(start, cut - expand)) picks the same cut whenever cut - expand >= start + max_len // 2
        # i.e. for expand 5 s and max 20 s always: 15 s >= 10 s
        self.assertEqual([round(o, 3) for _, o in pieces], self._old_offsets(wav, 20))

    @staticmethod
    def _old_offsets(wav, max_chunk_sec):
        import inspect
        src = inspect.getsource(textutil.split_audio_into_chunks).replace(
            "left = max(start + max_len // 2, cut - expand)", "left = max(start, cut - expand)")
        ns = {"np": np, "List": list, "Tuple": tuple, "math": __import__("math"),
              "MIN_ASR_INPUT_SECONDS": textutil.MIN_ASR_INPUT_SECONDS}
        exec(src, ns)
        return [round(o, 3) for _, o in ns["split_audio_into_chunks"](wav, SR, max_chunk_sec)]


class BackupNameTests(unittest.TestCase):
    def test_free_name_next_to_the_transcript(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d, mock.patch.object(textutil.time, "strftime", lambda f: "20260930-101010"):
            md = os.path.join(d, "a.md")
            first = textutil.backup_name(md)
            self.assertEqual(first, os.path.join(d, "a.bak-20260930-101010.md"))
            open(first, "w").close()
            self.assertEqual(textutil.backup_name(md), os.path.join(d, "a.bak-20260930-101010-2.md"))
            self.assertEqual(textutil.backup_name(md, "new"), os.path.join(d, "a.new-20260930-101010.md"))

    def test_backup_of_a_read_only_transcript_is_writable(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "a.md"), os.path.join(d, "a.bak.md")
            open(src, "w").write("old\n")
            os.chmod(src, stat.S_IREAD)
            try:
                textutil.sync_copy(src, dst)
                self.assertTrue(os.stat(dst).st_mode & stat.S_IWRITE)
                os.remove(dst)  # Windows refuses to remove a read-only file
            finally:
                os.chmod(src, stat.S_IREAD | stat.S_IWRITE)


class DiarizeHelpersTests(unittest.TestCase):
    def test_smooth_absorbs_short_runs(self):
        from localtranscribe import diarize

        labels = np.array([0] * 20 + [1] * 2 + [0] * 20 + [-1] * 5 + [1] * 20)
        out = diarize._smooth(labels, 6)
        self.assertEqual(set(out[20:22]), {0})
        self.assertEqual(set(out[-20:]), {1})

    def test_names_time_and_turn_merging(self):
        from localtranscribe import diarize

        self.assertEqual(diarize.fmt_time(3725), "01:02:05")
        self.assertEqual([textutil.fmt_time(s) for s in (0, 83.9, 3725)], ["00:00:00", "00:01:23", "01:02:05"])
        self.assertEqual(diarize.first_appearance_names([(0, 7, "a"), (1, 3, "b"), (2, 7, "c")]), {7: "Parlante 1", 3: "Parlante 2"})
        wav = np.ones(SR * 30, dtype=np.float32)
        runs = [(0.0, 5.0, 0), (5.0, 8.0, 0), (9.0, 12.0, 1), (13.0, 14.0, 0)]
        fake = FakeBackend(texts=["uno", "due", "tre", "quattro"])
        turns = diarize.transcribe_turns(fake, wav, runs, "Italian")
        self.assertEqual(turns, [(0.0, 0, "uno due"), (9.0, 1, "tre"), (13.0, 0, "quattro")])


class DiarizeFewWindowsTests(unittest.TestCase):
    def setUp(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("sklearn not installed")

    def test_fewer_windows_than_speakers_is_one_speaker_without_loading_a_model(self):
        from localtranscribe import diarize

        with mock.patch.object(diarize, "_speech_regions", return_value=[(SR, int(2.5 * SR))]), \
                mock.patch.object(diarize, "_embed", side_effect=AssertionError("model loaded")):
            out = diarize.diarize(np.zeros(SR * 5, np.float32), n_speakers=2)
        self.assertEqual(out, [(1.0, 2.5, 0)])

    def test_two_windows_two_speakers_still_clusters(self):
        from localtranscribe import diarize

        regions = [(SR, int(2.5 * SR)), (4 * SR, int(5.5 * SR))]
        emb = lambda wav, windows, dev: np.array([[1.0, 0.0], [0.0, 1.0]])
        with mock.patch.object(diarize, "_speech_regions", return_value=regions), \
                mock.patch.object(diarize, "_embed", side_effect=emb):
            out = diarize.diarize(np.zeros(SR * 6, np.float32), n_speakers=2)
        self.assertEqual({lab for _, _, lab in out}, {0, 1})


class SetupModelsTests(unittest.TestCase):
    def test_cache_check_uses_the_patterns_mlx_audio_downloads_with(self):
        import types

        from localtranscribe import setup_models

        pats = ["*.json", "*.safetensors"]
        fake_hf = types.SimpleNamespace(snapshot_download=mock.Mock())
        fake_utils = types.SimpleNamespace(DEFAULT_ALLOW_PATTERNS=pats)
        mods = {"huggingface_hub": fake_hf, "mlx_audio": types.ModuleType("mlx_audio"), "mlx_audio.utils": fake_utils}
        with mock.patch.dict(sys.modules, mods), mock.patch.object(devices, "is_apple_silicon", return_value=True):
            self.assertTrue(setup_models.is_cached("mlx-community/Qwen3-ASR-0.6B-8bit"))
            self.assertEqual(fake_hf.snapshot_download.call_args.kwargs["allow_patterns"], pats)
            self.assertTrue(setup_models.is_cached(config.CORRECT_MODEL))
            self.assertEqual(fake_hf.snapshot_download.call_args.kwargs["allow_patterns"], pats)
            self.assertTrue(setup_models.is_cached("microsoft/wavlm-base-plus-sv"))
            self.assertIsNone(fake_hf.snapshot_download.call_args.kwargs["allow_patterns"])


class CliTests(unittest.TestCase):
    def test_parser_defaults_match_the_old_tool(self):
        from localtranscribe.cli import build_parser

        args = build_parser().parse_args(["a.wav"])
        self.assertEqual((args.language, args.chunk, args.speakers, args.context, args.out_dir),
                         ("Italian", 20, 0, "", None))

    def test_output_name_rule(self):
        # <input name without extension>.md, next to the input unless --out-dir is given
        self.assertEqual(os.path.splitext(os.path.basename("dir/My call.m4a"))[0] + ".md", "My call.md")


if __name__ == "__main__":
    unittest.main()
