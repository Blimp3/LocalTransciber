"""Model presets, thresholds and defaults.

Everything a maintainer may want to tune lives in this one file.
"""

SAMPLE_RATE = 16000

DEFAULT_LANGUAGE = "Italian"

# Pieces of ~20 s are the most accurate for Qwen3-ASR (measured on FLEURS Italian:
# 15-30 s pieces give 2.4-2.9 % WER, 60 s 3.5 %, 120 s 5.2 %, 300 s 11.7 %). Do not raise.
CHUNK_SECONDS = 20

# Tokens the model may generate for one piece (a 20-25 s piece needs ~100-150).
MAX_NEW_TOKENS = 1024

# --------------------------------------------------------------------------------------
# NVIDIA GPU (Windows/Linux) and CPU fallback: PyTorch + the official qwen-asr package
# --------------------------------------------------------------------------------------
TORCH_MODELS = {
    "best": "Qwen/Qwen3-ASR-1.7B",
    "light": "Qwen/Qwen3-ASR-0.6B",
}
# --model auto: "best" when the GPU has at least this much memory, otherwise "light".
# On CPU "auto" always means "light".
NVIDIA_BEST_MIN_VRAM_GB = 12

# --------------------------------------------------------------------------------------
# Apple Silicon: MLX (mlx-audio) with 8-bit quantised weights
# --------------------------------------------------------------------------------------
MAC_MODELS = {
    "best": "mlx-community/Qwen3-ASR-1.7B-8bit",
    "light": "mlx-community/Qwen3-ASR-0.6B-8bit",
}
# --model auto: "best" when the Mac has at least this much TOTAL unified memory (GB), otherwise
# "light". PROVISIONAL: to be confirmed with the report of `mac_selftest.sh` on real hardware.
MAC_BEST_MIN_RAM_GB = 16

# Extra Mac presets that mac_selftest.sh also measures (name -> repo). The 4-bit 1.7B model is
# smaller than the 8-bit one but has not been measured yet.
MAC_SELFTEST_MODELS = {
    "1.7B-8bit": "mlx-community/Qwen3-ASR-1.7B-8bit",
    "1.7B-4bit": "mlx-community/Qwen3-ASR-1.7B-4bit",
    "0.6B-8bit": "mlx-community/Qwen3-ASR-0.6B-8bit",
}

# Mac batch size (pieces decoded together). Small on purpose: every extra piece in a batch adds
# activation memory and a fanless MacBook Air throttles under sustained load anyway.
MAC_BATCH_SIZE_LOW_RAM = 1     # total RAM below MAC_BEST_MIN_RAM_GB
MAC_BATCH_SIZE_HIGH_RAM = 2

# Upstream Hugging Face ids the user may type on a Mac -> the MLX repo that replaces them.
MAC_MODEL_ALIASES = {
    "Qwen/Qwen3-ASR-1.7B": MAC_MODELS["best"],
    "Qwen/Qwen3-ASR-0.6B": MAC_MODELS["light"],
}

# --------------------------------------------------------------------------------------
# Speaker separation (--speakers N)
# --------------------------------------------------------------------------------------
SPEAKER_MODEL = "microsoft/wavlm-base-plus-sv"

# Approximate download sizes in GB, only used to tell the user what is about to be fetched
# (the real sizes are queried from Hugging Face when possible).
APPROX_DOWNLOAD_GB = {
    "Qwen/Qwen3-ASR-1.7B": 4.7,
    "Qwen/Qwen3-ASR-0.6B": 1.9,
    "mlx-community/Qwen3-ASR-1.7B-8bit": 2.5,
    "mlx-community/Qwen3-ASR-1.7B-4bit": 1.6,
    "mlx-community/Qwen3-ASR-0.6B-8bit": 1.0,
    "microsoft/wavlm-base-plus-sv": 0.4,
}
