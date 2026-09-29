"""NVIDIA GPU (float16) and CPU (float32) backend: PyTorch + the official `qwen-asr` package."""
from typing import List, Optional

import numpy as np

from .. import config
from .base import Backend


class TorchQwenBackend(Backend):
    name = "torch"

    def __init__(self, repo, device="cuda", batch_size=8):
        import torch
        from qwen_asr import Qwen3ASRModel

        self.repo = repo
        self.device = device
        self.batch_size = batch_size
        self._torch = torch
        if device == "cuda":
            # float16, not bfloat16: Turing GPUs have no bf16 and fp16 measured identical accuracy to fp32.
            dtype, device_map = torch.float16, "cuda:0"
        else:
            dtype, device_map = torch.float32, "cpu"
        self.dtype = str(dtype).replace("torch.", "")
        self.model = Qwen3ASRModel.from_pretrained(
            repo,
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=batch_size,
            max_new_tokens=config.MAX_NEW_TOKENS,
        )

    def transcribe(self, pieces: List[np.ndarray], language: Optional[str], context: str = "") -> List[str]:
        if not pieces:
            return []
        results = self.model.transcribe(
            audio=[(p, config.SAMPLE_RATE) for p in pieces], language=language, context=context
        )
        return [r.text for r in results]

    def reset_peak_memory(self):
        if self.device == "cuda":
            self._torch.cuda.reset_peak_memory_stats()

    def peak_memory_gb(self):
        if self.device == "cuda":
            return self._torch.cuda.max_memory_allocated() / 1024 ** 3
        return None

    def close(self):
        import gc

        self.model = None
        gc.collect()
        if self.device == "cuda":
            self._torch.cuda.empty_cache()
