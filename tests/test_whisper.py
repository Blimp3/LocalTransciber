"""Checks of the MLX Whisper backend against a stand-in for mlx-audio's Whisper (same generate() surface, 0.5.7)."""
import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from test_mlx_backend import install_fake_mlx, restore  # noqa: E402

SR = 16000
TOK = "mlx_audio.stt.models.whisper.tokenizer"


class FakeWhisper:
    def __init__(self, texts=None):
        self.calls, self.texts = [], texts

    def generate(self, audio, **kw):
        self.calls.append(dict(kw, n=len(audio)))
        text = self.texts[len(self.calls) - 1] if self.texts else f" piece:{len(audio)} "
        return types.SimpleNamespace(text=text)


def install(model):
    saved = install_fake_mlx(model)
    tok = types.ModuleType(TOK)
    tok.LANGUAGES = {"it": "italian", "en": "english"}
    tok.TO_LANGUAGE_CODE = {"italian": "it", "english": "en"}
    for name in ("mlx_audio.stt.models", "mlx_audio.stt.models.whisper", TOK):
        saved[name] = sys.modules.get(name)
    sys.modules["mlx_audio.stt.models"] = types.ModuleType("mlx_audio.stt.models")
    sys.modules["mlx_audio.stt.models.whisper"] = types.ModuleType("mlx_audio.stt.models.whisper")
    sys.modules[TOK] = tok
    return saved


def restore_all(saved):
    restore(saved)
    sys.modules.pop("localtranscribe.backends.mlx_whisper", None)


class MlxWhisperTests(unittest.TestCase):
    def make(self, texts=None, repo="mlx-community/whisper-large-v3-turbo-asr-4bit"):
        self.model = FakeWhisper(texts)
        self.addCleanup(restore_all, install(self.model))
        from localtranscribe.backends.mlx_whisper import MlxWhisperBackend

        return MlxWhisperBackend(repo, device="mps", batch_size=4)

    def pieces(self, *seconds):
        return [np.full(int(s * SR), 0.1, dtype=np.float32) for s in seconds]

    def test_dispatch_by_repo_id(self):
        self.model = FakeWhisper()
        self.addCleanup(restore_all, install(self.model))
        from localtranscribe import backends

        with mock.patch.object(backends, "is_apple_silicon", return_value=True):
            b = backends.create_backend("mlx-community/Whisper-Large-V3-turbo-asr-4bit", "mps", 1)
            self.assertEqual((b.name, b.batch_size), ("mlx-whisper", 1))
            self.assertEqual(backends.create_backend("mlx-community/Qwen3-ASR-1.7B-4bit", "mps", 1).name, "mlx")
        fake = types.ModuleType("localtranscribe.backends.torch_whisper")
        fake.TorchWhisperBackend = lambda repo, device, batch_size: ("torch", repo)
        with mock.patch.object(backends, "is_apple_silicon", return_value=False), \
                mock.patch.dict(sys.modules, {"localtranscribe.backends.torch_whisper": fake}):
            self.assertEqual(backends.create_backend("openai/whisper-large-v3", "cuda", 2), ("torch", "openai/whisper-large-v3"))

    def test_language_mapping(self):
        for given, code in (("Italian", "it"), ("italian", "it"), ("it", "it"), (None, None)):
            be = self.make()
            be.transcribe(self.pieces(1.0), given)
            self.assertEqual(self.model.calls[-1]["language"], code)
        with self.assertRaises(ValueError):
            self.make().transcribe(self.pieces(1.0), "Klingon")

    def test_prompt_task_and_greedy_options(self):
        be = self.make()
        be.transcribe(self.pieces(2.0), "Italian", "Mario Rossi")
        c = self.model.calls[0]
        self.assertEqual(c["initial_prompt"], "Mario Rossi")
        self.assertEqual(c["task"], "transcribe")
        self.assertEqual(c["temperature"], 0.0)  # a scalar: no fallback ladder
        self.assertFalse(c["return_timestamps"])
        self.assertFalse(c["condition_on_previous_text"])
        self.assertIsNone(c["logprob_threshold"])
        be.transcribe(self.pieces(2.0), "Italian", "")
        self.assertIsNone(self.model.calls[1]["initial_prompt"])

    def test_one_string_per_piece_in_order(self):
        be = self.make()
        out = be.transcribe(self.pieces(19.0, 5.0, 24.0), "Italian")
        self.assertEqual(out, [f"piece:{19 * SR}", f"piece:{5 * SR}", f"piece:{24 * SR}"])

    def test_repetition_cleanup(self):
        be = self.make([" ciao " * 60])
        out = be.transcribe(self.pieces(3.0), "Italian")[0]
        self.assertTrue(out.startswith("ciao"))
        self.assertLess(out.count("ciao"), 20)

    def test_memory_and_describe(self):
        be = self.make()
        self.assertEqual(be.peak_memory_gb(), 3.0)
        self.assertIn("4bit", be.describe())
        be.close()


if __name__ == "__main__":
    unittest.main()
