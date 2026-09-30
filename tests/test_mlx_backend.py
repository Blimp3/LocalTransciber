"""Checks of the Apple/MLX backend that run anywhere (no Mac needed).

1. The backend logic is exercised against a stand-in for mlx and mlx-audio that has the same call signatures
   as the pinned mlx-audio 0.5.7 (ordering, batching, fallback, how --context and the piece length are passed).
2. `test_api_contract` downloads the pinned mlx-audio wheel from PyPI (skipped when offline) and checks with
   the AST that every function and argument the backend uses really exists in that exact version.

What can NOT be checked here: running the real model on Metal. That is what mac_selftest.sh is for.
"""
import ast
import io
import json
import os
import re
import sys
import types
import unittest
import urllib.request
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SR = 16000


class _Result:
    def __init__(self, text):
        self.text = text


class FakeQwenModel:
    """Same public/private call surface as mlx_audio's Qwen3ASRModel (0.5.7)."""

    def __init__(self, fail_batched=False):
        self.generate_calls = []
        self.batched_calls = []
        self.fail_batched = fail_batched

    def generate(self, audio, *, max_tokens=8192, batch_size=1, language=None, chunk_duration=1200.0,
                 system_prompt=None, verbose=False, **kw):
        self.generate_calls.append(dict(n=len(audio), language=language, system_prompt=system_prompt,
                                        chunk_duration=chunk_duration, max_tokens=max_tokens, batch_size=batch_size))
        return _Result(f"single:{len(audio)}")

    def _generate_chunks_batched(self, chunks, *, max_tokens, sampler, logits_processors, language,
                                 system_prompt, batch_size, verbose):
        self.batched_calls.append(dict(sizes=[len(c[0]) for c in chunks], system_prompt=system_prompt,
                                       batch_size=batch_size, max_tokens=max_tokens))
        if self.fail_batched:
            raise RuntimeError("boom")
        texts = [f"batched:{len(c[0])}" for c in chunks]
        return texts, [1] * len(chunks), [1] * len(chunks), [True] * len(chunks)

    def extract_language(self, text):
        return "Italian", text


def install_fake_mlx(model):
    core = types.ModuleType("mlx.core")
    core.cpu, core.gpu = "cpu", "gpu"
    core.metal = types.SimpleNamespace(is_available=lambda: True)
    core.default = None
    core.set_default_device = lambda d: setattr(core, "default", d)
    core.clear_cache = lambda: None
    core.get_peak_memory = lambda: 3 * 1024 ** 3
    core.reset_peak_memory = lambda: None
    mlx = types.ModuleType("mlx")
    mlx.core = core
    stt = types.ModuleType("mlx_audio.stt")
    stt.load = lambda repo: model
    sample_utils = types.ModuleType("mlx_audio.lm.sample_utils")
    sample_utils.make_sampler = lambda temp=0.0, *a, **k: "sampler"
    pkg = types.ModuleType("mlx_audio")
    lm = types.ModuleType("mlx_audio.lm")
    mods = {"mlx": mlx, "mlx.core": core, "mlx_audio": pkg, "mlx_audio.stt": stt,
            "mlx_audio.lm": lm, "mlx_audio.lm.sample_utils": sample_utils}
    saved = {k: sys.modules.get(k) for k in mods}
    sys.modules.update(mods)
    return saved


def restore(saved):
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v
    sys.modules.pop("localtranscribe.backends.mlx_qwen", None)


class MlxBackendLogicTests(unittest.TestCase):
    def make(self, batch, fail=False, device="mps"):
        self.model = FakeQwenModel(fail_batched=fail)
        saved = install_fake_mlx(self.model)
        self.addCleanup(restore, saved)
        from localtranscribe.backends.mlx_qwen import MlxQwenBackend

        return MlxQwenBackend("mlx-community/Qwen3-ASR-1.7B-8bit", device=device, batch_size=batch)

    def pieces(self, *seconds):
        return [np.full(int(s * SR), 0.1, dtype=np.float32) for s in seconds]

    def test_single_mode_passes_context_and_never_lets_mlx_resplit(self):
        be = self.make(1)
        out = be.transcribe(self.pieces(19.0, 24.0), "Italian", "Mario Rossi")
        self.assertEqual(out, [f"single:{19 * SR}", f"single:{24 * SR}"])
        for call, n in zip(self.model.generate_calls, (19 * SR, 24 * SR)):
            self.assertEqual(call["system_prompt"], "Mario Rossi")
            self.assertEqual(call["language"], "Italian")
            self.assertEqual(call["batch_size"], 1)
            self.assertGreater(call["chunk_duration"], n / SR)  # longer than the piece: mlx-audio keeps it whole
        self.assertEqual(self.model.batched_calls, [])

    def test_empty_context_is_none(self):
        be = self.make(1)
        be.transcribe(self.pieces(3.0), "Italian", "")
        self.assertIsNone(self.model.generate_calls[0]["system_prompt"])

    def test_batched_mode_keeps_order_and_groups_similar_lengths(self):
        be = self.make(2)
        secs = (10.0, 24.0, 12.0, 23.0, 5.0)
        out = be.transcribe(self.pieces(*secs), "Italian", "")
        self.assertEqual(out, [f"batched:{int(s * SR)}" if s != 5.0 else f"single:{5 * SR}" for s in secs])
        sizes = [c["sizes"] for c in self.model.batched_calls]
        self.assertEqual(sizes, [[24 * SR, 23 * SR], [12 * SR, 10 * SR]])  # longest first, neighbours together
        self.assertEqual(len(self.model.generate_calls), 1)  # the odd piece out runs alone

    def test_batched_failure_falls_back_to_single_and_stays_there(self):
        be = self.make(2, fail=True)
        out = be.transcribe(self.pieces(10.0, 11.0, 12.0, 13.0), "Italian", "")
        self.assertEqual(out, [f"single:{int(s * SR)}" for s in (10.0, 11.0, 12.0, 13.0)])
        self.assertEqual(be.batch_size, 1)
        self.assertEqual(len(self.model.batched_calls), 1)  # tried once, then never again
        be.transcribe(self.pieces(10.0, 11.0), "Italian", "")
        self.assertEqual(len(self.model.batched_calls), 1)

    def test_cpu_device_and_memory_api(self):
        be = self.make(1, device="cpu")
        self.assertEqual(be.device, "cpu")
        self.assertAlmostEqual(be.peak_memory_gb(), 3.0)
        self.assertEqual(be.dtype, "8bit")

    def test_no_metal_falls_back_to_cpu_with_note(self):
        self.make(1)
        sys.modules["mlx.core"].metal.is_available = lambda: False
        from localtranscribe.backends.mlx_qwen import MlxQwenBackend

        be2 = MlxQwenBackend("mlx-community/Qwen3-ASR-0.6B-8bit", device="mps", batch_size=1)
        self.assertEqual(be2.device, "cpu")
        self.assertTrue(be2.notes)


def _pinned(name):
    with open(os.path.join(ROOT, "requirements-mac.txt"), encoding="utf-8") as f:
        m = re.search(rf"^{re.escape(name)}(?:\[[^\]]*\])?==([\w.]+)", f.read(), re.M)
    return m.group(1) if m else None


def _fetch_wheel(name, version):
    meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30))
    url = next(f["url"] for f in meta["urls"] if f["filename"].endswith(".whl"))
    return zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=120).read()))


def _funcs(tree):
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def _params(fn):
    a = fn.args
    return [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs] + ([a.kwarg.arg] if a.kwarg else [])


class MlxApiContractTests(unittest.TestCase):
    """The pinned mlx-audio really has everything mlx_qwen.py relies on."""

    @classmethod
    def setUpClass(cls):
        version = _pinned("mlx-audio")
        if not version:
            raise unittest.SkipTest("mlx-audio is not pinned in requirements-mac.txt")
        try:
            cls.zf = _fetch_wheel("mlx-audio", version)
        except Exception as e:  # offline, PyPI down...
            raise unittest.SkipTest(f"cannot download mlx-audio {version}: {e}") from e
        cls.version = version

    def parse(self, path):
        return ast.parse(self.zf.read(path).decode("utf-8"))

    def test_generate_signature(self):
        funcs = _funcs(self.parse("mlx_audio/stt/models/qwen3_asr/qwen3_asr.py"))
        gen = _params(funcs["generate"])
        for arg in ("audio", "max_tokens", "batch_size", "language", "chunk_duration", "system_prompt", "verbose"):
            self.assertIn(arg, gen, f"generate() lacks {arg} in mlx-audio {self.version}")
        batched = _params(funcs["_generate_chunks_batched"])
        for arg in ("chunks", "max_tokens", "sampler", "logits_processors", "language", "system_prompt",
                    "batch_size", "verbose"):
            self.assertIn(arg, batched, f"_generate_chunks_batched() lacks {arg}")
        self.assertIn("extract_language", funcs)
        self.assertIn("_build_prompt", funcs)

    def test_private_decoding_api_used_by_the_recorded_path(self):
        """_single_recorded and word_continuations drive the model by hand (a copy of stream_generate's loop)."""
        funcs = _funcs(self.parse("mlx_audio/stt/models/qwen3_asr/qwen3_asr.py"))
        for name in ("_preprocess_audio", "get_audio_features", "_build_inputs_embeds", "_forward_with_embeds",
                     "make_cache", "_eos_token_ids", "stream_generate"):
            self.assertIn(name, funcs, f"{name}() is gone in mlx-audio {self.version}")
        self.assertIn("cache", _params(funcs["_forward_with_embeds"]))
        self.assertEqual(_params(funcs["_build_prompt"])[1:4], ["num_audio_tokens", "language", "system_prompt"])
        step = self.parse("mlx_audio/lm/generate.py")
        self.assertIn("prefill_step_size", _params(_funcs(step)["generate_step"]))
        src = self.zf.read("mlx_audio/lm/generate.py").decode()
        self.assertIn("count = min(prefill_step_size, total - processed - 1)", src)  # the prefill split we copy

    def test_load_and_helpers_exist(self):
        stt_init = self.zf.read("mlx_audio/stt/__init__.py").decode()
        self.assertIn("load", stt_init)
        utils = _funcs(self.parse("mlx_audio/utils.py"))
        self.assertIn("get_model_path", utils)
        sampler = _funcs(self.parse("mlx_audio/lm/sample_utils.py"))["make_sampler"]
        self.assertEqual(_params(sampler)[0], "temp")

    def test_system_prompt_semantics(self):
        src = self.zf.read("mlx_audio/stt/models/qwen3_asr/qwen3_asr.py").decode()
        # the context becomes the system message, followed by one newline (documented small difference)
        self.assertIn('system_content = f"{system_prompt}\\n" if system_prompt else ""', src)
        self.assertIn("<|im_start|>system\\n{system_content}<|im_end|>", src)


if __name__ == "__main__":
    unittest.main()
