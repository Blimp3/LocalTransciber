"""Audio decoding with PyAV (ffmpeg is bundled in the `av` wheel, nothing to install)."""
import numpy as np

from .config import SAMPLE_RATE


def load_audio(path, sr=SAMPLE_RATE):
    """Decode any audio/video file (wav, mp3, m4a, ogg, opus, flac, mp4, mkv, ...) to a
    mono float32 numpy array at `sr` Hz. Video tracks are ignored; the first audio track is used."""
    import av

    frames = []
    with av.open(str(path)) as container:
        streams = container.streams.audio
        if not streams:
            raise ValueError("no audio track found")
        stream = streams[0]
        stream.thread_type = "AUTO"
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)
        for frame in container.decode(stream):
            for out in _as_list(resampler.resample(frame)):
                frames.append(out.to_ndarray().reshape(-1))
        for out in _as_list(resampler.resample(None)):  # flush what the resampler still holds
            frames.append(out.to_ndarray().reshape(-1))
    if not frames:
        raise ValueError("the audio track is empty")
    return np.ascontiguousarray(np.concatenate(frames), dtype=np.float32)


def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, (list, tuple)) else [x]
