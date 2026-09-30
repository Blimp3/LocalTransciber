"""Backend factory. Heavy imports (torch, qwen-asr, mlx) happen only inside the chosen backend."""
import os

from ..devices import is_apple_silicon
from .base import Backend


def create_backend(repo, device, batch_size) -> Backend:
    if "whisper" in os.path.basename(repo.rstrip("/\\")).lower():  # the repo name, not a folder above it
        if is_apple_silicon():
            from .mlx_whisper import MlxWhisperBackend

            return MlxWhisperBackend(repo, device=device, batch_size=batch_size)
        from .torch_whisper import TorchWhisperBackend

        return TorchWhisperBackend(repo, device=device, batch_size=batch_size)
    if is_apple_silicon():
        from .mlx_qwen import MlxQwenBackend

        return MlxQwenBackend(repo, device=device, batch_size=batch_size)
    from .torch_qwen import TorchQwenBackend

    return TorchQwenBackend(repo, device=device, batch_size=batch_size)


__all__ = ["Backend", "create_backend"]
