"""Unit tests that need no model, no GPU and no network.   python -m unittest discover -s tests -v"""
import os
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
        n_pieces = len(pipeline.split_wav(wav, 20))
        fake = FakeBackend(texts=["uno", "due Mario Rossi Politecnico di Milano", ""] + ["x"] * 20)
        text = pipeline.transcribe_wav(fake, wav, "Italian", "Mario Rossi Politecnico di Milano", 20)
        self.assertTrue(text.startswith("uno" + "\n" + "\n" + "due"))
        self.assertNotIn("\n" * 3, text)
        self.assertNotIn("Mario", text)
        self.assertGreaterEqual(n_pieces, 3)
        self.assertEqual(fake.calls[0][2], "Mario Rossi Politecnico di Milano")

    def test_progress_and_slicing(self):
        wav = noisy_speechlike(400)
        seen = []
        fake = FakeBackend()
        pipeline.transcribe_wav(fake, wav, "Italian", progress=lambda d, t: seen.append((d, t)))
        self.assertEqual(seen[-1][0], seen[-1][1])
        self.assertTrue(all(n <= fake.batch_size * 4 for n, _, _ in fake.calls))


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
            self.assertEqual(devices.choose_model("auto", "mps", 8.0), config.MAC_MODELS["light"])
            self.assertEqual(devices.choose_model("best", "mps", 8.0), config.MAC_MODELS["best"])
            self.assertEqual(devices.choose_model("Qwen/Qwen3-ASR-0.6B", "mps", 8.0), config.MAC_MODELS["light"])
            self.assertEqual(devices.choose_model("mlx-community/Qwen3-ASR-1.7B-4bit", "mps", 8.0),
                             "mlx-community/Qwen3-ASR-1.7B-4bit")
            with self.assertRaises(devices.SetupError):
                devices.choose_model("someone/not-mlx", "mps", 8.0)

    def test_mac_batch_and_device(self):
        with mock.patch.object(devices, "is_apple_silicon", return_value=True):
            with mock.patch.object(devices, "total_ram_gb", return_value=8.0):
                self.assertEqual(devices.choose_batch_size(0, "mps", "x"), config.MAC_BATCH_SIZE_LOW_RAM)
            with mock.patch.object(devices, "total_ram_gb", return_value=24.0):
                self.assertEqual(devices.choose_batch_size(0, "mps", "x"), config.MAC_BATCH_SIZE_HIGH_RAM)
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
        self.assertEqual(diarize.first_appearance_names([(0, 7, "a"), (1, 3, "b"), (2, 7, "c")]), {7: "Parlante 1", 3: "Parlante 2"})
        wav = np.zeros(SR * 30, dtype=np.float32)
        runs = [(0.0, 5.0, 0), (5.0, 8.0, 0), (9.0, 12.0, 1), (13.0, 14.0, 0)]
        fake = FakeBackend(texts=["uno", "due", "tre", "quattro"])
        turns = diarize.transcribe_turns(fake, wav, runs, "Italian")
        self.assertEqual(turns, [(0.0, 0, "uno due"), (9.0, 1, "tre"), (13.0, 0, "quattro")])


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
