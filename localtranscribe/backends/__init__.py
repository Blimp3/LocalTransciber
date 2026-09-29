"""Backend factory. Heavy imports (torch, qwen-asr, mlx) happen only inside the chosen backend."""
from ..devices import is_apple_silicon
from .base import Backend


def create_backend(repo, device, batch_size) -> Backend:
    if is_apple_silicon():
        from .mlx_qwen import MlxQwenBackend

        return MlxQwenBackend(repo, device=device, batch_size=batch_size)
    from .torch_qwen import TorchQwenBackend

    return TorchQwenBackend(repo, device=device, batch_size=batch_size)


__all__ = ["Backend", "create_backend"]
