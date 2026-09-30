"""Checks of the PyTorch Whisper backend that need neither a model nor a GPU: the transformers classes are mocked."""
import os
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import torch
    import transformers
    from localtranscribe.backends import torch_whisper
    from localtranscribe import config
    HAVE = True
except ImportError:  # torch / transformers missing
    HAVE = False

PROMPT = [50361, 11, 22, 33]  # <|startofprev|> + 3 text tokens


class FakeFeatureExtractor:
    def __init__(self, with_mask=False):
        self.with_mask = with_mask
        self.calls = []
        self.batches = []  # the arrays received, per call

    def __call__(self, batch, sampling_rate=None, return_attention_mask=None, return_tensors=None):
        self.calls.append((len(batch), sampling_rate, return_attention_mask))
        self.batches.append([np.array(b) for b in batch])
        # the largest sample of each piece encodes its id (the first one is the lead-in zero), so the fake model can echo it back
        feats = torch.tensor([[[float(np.max(b))]] for b in batch])
        out = {"input_features": feats}
        if self.with_mask:
            out["attention_mask"] = torch.ones(len(batch), 3)
        return out


class FakeProcessor:
    with_mask = False
    texts = None  # optional override: id -> raw decoded text

    def __init__(self):
        self.feature_extractor = FakeFeatureExtractor(FakeProcessor.with_mask)

    @classmethod
    def from_pretrained(cls, repo):
        return cls()

    def get_prompt_ids(self, text):
        return np.array(PROMPT)

    def batch_decode(self, rows, skip_special_tokens=True):
        # a row is [prompt?] + [id]; a real decoder drops special tokens, the prompt text tokens stay unless stripped
        assert skip_special_tokens is True
        out = []
        for r in rows:
            t = "PROMPTTEXT " if r[:3] == PROMPT[1:] else ""
            i = r[-1]
            t += (FakeProcessor.texts or {}).get(i, f"text{i}")
            out.append(f"  {t} ")
        return out


class FakeModel:
    def __init__(self):
        self.generate_calls = []
        self.echo_prompt = "strip"  # "strip": rows begin with the prompt as real generate() does; "none": they do not

    @classmethod
    def from_pretrained(cls, repo, **kw):
        m = cls()
        m.load_kwargs = kw
        m.repo = repo
        return m

    def to(self, dev):
        self.dev = dev
        return self

    def eval(self):
        return self

    def generate(self, **kw):
        self.generate_calls.append(kw)
        ids = [int(x) for x in kw["input_features"][:, 0, 0].tolist()]
        rows = []
        for i in ids:
            row = list(PROMPT) if "prompt_ids" in kw and self.echo_prompt == "strip" else []
            rows.append(row + [i])
        return torch.tensor(rows)


def tf():
    """The module object `import transformers` returns right now. Reading a lazy attribute can make transformers
    swap its sys.modules entry, so touch the attributes first and look the module up afterwards."""
    import transformers as m
    m.WhisperProcessor, m.WhisperForConditionalGeneration
    return sys.modules["transformers"]


def piece(i):
    return np.full(16000, float(i), dtype=np.float32)


def make(device="cpu", batch_size=2, mask=False):
    FakeProcessor.with_mask = mask
    FakeProcessor.texts = None
    with mock.patch.object(tf(), "WhisperProcessor", FakeProcessor), \
            mock.patch.object(tf(), "WhisperForConditionalGeneration", FakeModel):
        return torch_whisper.TorchWhisperBackend("openai/fake", device=device, batch_size=batch_size)


@unittest.skipUnless(HAVE, "torch / transformers not installed")
class LanguageTests(unittest.TestCase):
    def test_names(self):
        self.assertEqual(torch_whisper.language_code("Italian"), "it")
        self.assertEqual(torch_whisper.language_code("italian"), "it")
        self.assertEqual(torch_whisper.language_code(" English "), "en")
        self.assertEqual(torch_whisper.language_code("German"), "de")

    def test_none_is_auto(self):
        self.assertIsNone(torch_whisper.language_code(None))

    def test_unknown(self):
        with self.assertRaises(ValueError):
            torch_whisper.language_code("Klingonese")

    def test_unknown_raises_at_call_time_even_for_empty_input(self):
        b = make()
        with self.assertRaises(ValueError):
            b.transcribe([], "Klingonese")
        with self.assertRaises(ValueError):
            b.transcribe([piece(1)], "Klingonese")


@unittest.skipUnless(HAVE, "torch / transformers not installed")
class BackendTests(unittest.TestCase):
    def test_empty_input(self):
        b = make()
        self.assertEqual(b.transcribe([], "Italian"), [])
        self.assertEqual(b.model.generate_calls, [])

    def test_batching_order_across_batches(self):
        b = make(batch_size=2)
        out = b.transcribe([piece(i) for i in range(1, 6)], "Italian")
        self.assertEqual(out, [f"text{i}" for i in range(1, 6)])
        self.assertEqual([c["input_features"].shape[0] for c in b.model.generate_calls], [2, 2, 1])

    def test_language_passed_and_auto(self):
        b = make()
        b.transcribe([piece(1)], "Italian")
        b.transcribe([piece(1)], None)
        self.assertEqual(b.model.generate_calls[0]["language"], "it")
        self.assertNotIn("language", b.model.generate_calls[1])

    def test_task_and_greedy_args(self):
        b = make()
        b.transcribe([piece(1)], "Italian")
        kw = b.model.generate_calls[0]
        self.assertEqual(kw["task"], "transcribe")
        self.assertEqual(kw["num_beams"], 1)
        self.assertIs(kw["do_sample"], False)
        self.assertIs(kw["return_timestamps"], False)
        self.assertEqual(kw["max_new_tokens"], min(config.MAX_NEW_TOKENS, 443))
        self.assertNotIn("prompt_ids", kw)

    def test_context_becomes_prompt_and_is_stripped(self):
        b = make()
        out = b.transcribe([piece(7)], "Italian", "uccelli dinosauri")
        kw = b.model.generate_calls[0]
        self.assertEqual(kw["prompt_ids"].tolist(), PROMPT)
        self.assertEqual(out, ["text7"])
        self.assertNotIn("PROMPTTEXT", out[0])

    def test_generate_that_does_not_echo_the_prompt(self):
        b = make()
        b.model.echo_prompt = "none"
        self.assertEqual(b.transcribe([piece(7)], "Italian", "x y"), ["text7"])

    def test_empty_or_blank_context_means_no_prompt(self):
        b = make()
        b.transcribe([piece(1)], "Italian", "")
        b.transcribe([piece(1)], "Italian", "   ")
        self.assertTrue(all("prompt_ids" not in c for c in b.model.generate_calls))

    def test_long_context_is_clamped(self):
        b = make()
        b.processor.get_prompt_ids = lambda text: np.arange(1000)
        b.transcribe([piece(1)], "Italian", "parole " * 500)
        kw = b.model.generate_calls[0]
        n = len(kw["prompt_ids"])
        self.assertLessEqual(n, 200)
        self.assertLessEqual(n + 4 + kw["max_new_tokens"], 448)
        self.assertEqual(kw["prompt_ids"][0].item(), 0)  # first token (<|startofprev|>) kept in front

    def test_repetition_fix_applied_once(self):
        b = make()
        FakeProcessor.texts = {1: "ah" * 100}
        with mock.patch.object(torch_whisper, "detect_and_fix_repetitions", side_effect=lambda t: t) as m:
            out = b.transcribe([piece(1), piece(2)], "Italian")
        self.assertEqual(m.call_count, 2)  # once per piece
        self.assertEqual(out[0], "ah" * 100)
        # and the real function does collapse it
        self.assertLess(len(b.transcribe([piece(1)], "Italian")[0]), 20)

    def test_mask_requested_and_forwarded(self):
        b = make(mask=True)
        b.transcribe([piece(1)], "Italian")
        self.assertTrue(b.processor.feature_extractor.calls[0][2])
        self.assertIn("attention_mask", b.model.generate_calls[0])

    def test_piece_longer_than_30s_refused(self):
        b = make()
        sr = config.SAMPLE_RATE
        ok = np.ones(int(29.9 * sr), dtype=np.float32)
        self.assertEqual(b.transcribe([ok], "Italian"), ["text1"])
        self.assertEqual(b.transcribe([np.ones(30 * sr + 3, dtype=np.float32)], "Italian"), ["text1"])  # tolerance
        n_calls = len(b.model.generate_calls)
        bad = np.ones(int(30.5 * sr), dtype=np.float32)
        with self.assertRaises(ValueError) as cm:
            b.transcribe([piece(1), ok, bad], "Italian")
        msg = str(cm.exception)
        self.assertIn("Piece 2", msg)
        self.assertIn("30.5 s", msg)
        self.assertIn("--chunk 25", msg)
        self.assertEqual(len(b.model.generate_calls), n_calls)  # refused before any decoding

    def test_lead_in_prepended_before_feature_extractor(self):
        b = make(batch_size=2)
        pieces = [piece(1), piece(2), np.full(8000, 3.0, dtype=np.float32)]
        out = b.transcribe(pieces, "Italian")  # code defaults: lead-in only
        self.assertEqual(out, ["text1", "text2", "text3"])  # order and content preserved
        n_lead = round(torch_whisper.LEAD_IN_S * config.SAMPLE_RATE)
        self.assertEqual(torch_whisper.LEAD_IN_S, 0.05)
        self.assertEqual(n_lead, 800)
        got = [x for batch in b.processor.feature_extractor.batches for x in batch]
        self.assertEqual([len(x) for x in got], [16000 + n_lead, 16000 + n_lead, 8000 + n_lead])
        for x, p in zip(got, pieces):
            self.assertEqual(x.dtype, np.float32)
            self.assertTrue(np.all(x[:n_lead] == 0))
            np.testing.assert_array_equal(x[n_lead:], p)  # fade off: the piece itself is untouched
        self.assertEqual(len(pieces[0]), 16000)  # caller's arrays are not modified

    def test_lead_in_and_fade_read_at_call_time(self):
        b = make()
        with mock.patch.object(torch_whisper, "LEAD_IN_S", 0.1):
            b.transcribe([piece(1)], "Italian")
        with mock.patch.object(torch_whisper, "LEAD_IN_S", 0.0):
            b.transcribe([piece(1)], "Italian")
        first, second = b.processor.feature_extractor.batches
        self.assertEqual(len(first[0]), 16000 + 1600)
        self.assertEqual(len(second[0]), 16000)  # LEAD_IN_S = 0 disables it

    def test_fade_off_by_default(self):
        self.assertEqual(torch_whisper.LEAD_IN_S, 0.05)
        self.assertEqual(torch_whisper.FADE_S, 0.0)
        b = make()
        b.transcribe([np.full(16000, 2.0, dtype=np.float32)], "Italian")
        got = b.processor.feature_extractor.batches[0][0]
        self.assertEqual(len(got), 16000 + 800)
        self.assertTrue(np.all(got[:800] == 0))
        self.assertTrue(np.all(got[800:] == 2.0))  # the piece itself is untouched, edges included

    def test_fade_shape_when_on(self):
        b = make()
        sr = config.SAMPLE_RATE
        x = np.full(2 * sr, 2.0, dtype=np.float32)
        with mock.patch.object(torch_whisper, "FADE_S", 0.05):
            b.transcribe([x], "Italian")
        got = b.processor.feature_extractor.batches[0][0]
        n_lead, n_fade = 800, 800
        self.assertEqual(len(got), len(x) + n_lead)
        self.assertTrue(np.all(got[:n_lead] == 0))                     # lead-in is silence, not faded audio
        body = got[n_lead:]
        self.assertEqual(body[0], 0.0)                                 # ramp starts at zero ...
        self.assertAlmostEqual(float(body[n_fade - 1]), 2.0, places=5)  # ... reaches full level after FADE_S
        self.assertTrue(np.all(np.diff(body[:n_fade]) > 0))            # linear, increasing
        np.testing.assert_allclose(np.diff(body[:n_fade]), 2.0 / (n_fade - 1), rtol=1e-3)
        self.assertTrue(np.all(body[n_fade:-n_fade] == 2.0))           # middle untouched
        np.testing.assert_allclose(body[-n_fade:], body[:n_fade][::-1])  # fade-out mirrors fade-in
        self.assertEqual(body[-1], 0.0)
        self.assertTrue(np.all(x == 2.0))                              # caller's array not modified

    def test_fade_longer_than_half_the_piece_is_clamped(self):
        b = make()
        x = np.ones(1000, dtype=np.float32)
        with mock.patch.object(torch_whisper, "FADE_S", 0.05):  # 800 samples each side > 500
            b.transcribe([x], "Italian")
        body = b.processor.feature_extractor.batches[0][0][800:]
        self.assertEqual(len(body), 1000)
        self.assertTrue(np.all(np.isfinite(body)) and body.max() <= 1.0 and body.min() >= 0.0)

    def test_refusal_uses_original_length_not_padded(self):
        b = make()
        sr = config.SAMPLE_RATE
        # 29.99 s fits; with the lead-in it is 30.04 s, which must NOT be refused
        near = np.ones(int(29.99 * sr), dtype=np.float32)
        self.assertEqual(b.transcribe([near], "Italian"), ["text1"])
        self.assertEqual(len(b.processor.feature_extractor.batches[-1][0]), len(near) + 800)
        # 30.5 s is refused although 30.5 + 0.05 is longer still, and the message reports the original 30.5 s
        n_calls = len(b.model.generate_calls)
        with self.assertRaises(ValueError) as cm:
            b.transcribe([np.ones(int(30.5 * sr), dtype=np.float32)], "Italian")
        self.assertIn("30.5 s", str(cm.exception))
        self.assertNotIn("30.6 s", str(cm.exception))
        self.assertEqual(len(b.model.generate_calls), n_calls)

    def test_attention_mask_only_when_provided(self):
        b = make(mask=False)
        b.transcribe([piece(1)], "Italian")
        self.assertNotIn("attention_mask", b.model.generate_calls[0])
        b = make(mask=True)
        b.transcribe([piece(1)], "Italian")
        self.assertIn("attention_mask", b.model.generate_calls[0])

    def test_cpu_dtype_and_metadata(self):
        b = make("cpu", batch_size=3)
        self.assertEqual((b.name, b.repo, b.device, b.dtype, b.batch_size),
                         ("torch-whisper", "openai/fake", "cpu", "float32", 3))
        self.assertEqual(b.model.load_kwargs["dtype"], torch.float32)
        self.assertNotIn("torch_dtype", b.model.load_kwargs)
        self.assertEqual(b.model.load_kwargs["attn_implementation"], "sdpa")
        self.assertIn("torch-whisper", b.describe())
        self.assertIsNone(b.peak_memory_gb())
        b.reset_peak_memory()
        b.close()
        self.assertIsNone(b.model)

    def test_cuda_dtype_and_device(self):
        b = make("cuda")
        self.assertEqual(b.dtype, "float16")
        self.assertEqual(b.model.load_kwargs["dtype"], torch.float16)
        self.assertEqual(b._device_str, "cuda:0")
        self.assertEqual(b.model.dev, "cuda:0")

    def test_sdpa_fallback(self):
        calls = []

        def fake_from_pretrained(repo, **kw):
            calls.append(kw)
            if "attn_implementation" in kw:
                raise ValueError("sdpa not supported")
            return FakeModel()

        FakeProcessor.with_mask = False
        with mock.patch.object(tf(), "WhisperProcessor", FakeProcessor), \
                mock.patch.object(tf(), "WhisperForConditionalGeneration") as M:
            M.from_pretrained.side_effect = fake_from_pretrained
            torch_whisper.TorchWhisperBackend("x", device="cpu")
        self.assertEqual(len(calls), 2)
        self.assertNotIn("attn_implementation", calls[1])


if __name__ == "__main__":
    unittest.main()
