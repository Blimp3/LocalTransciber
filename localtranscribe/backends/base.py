"""The one interface every speech-to-text backend implements."""
from typing import List, Optional

import numpy as np


class Backend:
    """A loaded speech-to-text model.

    `transcribe` takes already-cut pieces (mono float32, 16 kHz, each at most ~25 s) and returns
    one cleaned string per piece, in the same order. Chunking, joining and file output are done
    by the callers, so they are identical on every platform.
    """

    name = "base"
    repo = ""
    device = "cpu"
    batch_size = 1
    dtype = ""

    def describe(self) -> str:
        return f"{self.name}, {self.repo}, device {self.device}, {self.dtype}, batch {self.batch_size}"

    def transcribe(self, pieces: List[np.ndarray], language: Optional[str], context: str = "") -> List[str]:
        raise NotImplementedError

    def reset_peak_memory(self) -> None:
        pass

    def peak_memory_gb(self) -> Optional[float]:
        """Peak accelerator memory since the last reset (None if the backend cannot tell)."""
        return None

    def close(self) -> None:
        pass
