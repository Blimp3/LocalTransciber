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
# --model auto and the hardware precheck use the same rule (precheck.nvidia_preset_for): "best" when the
# GPU has at least NVIDIA_BEST_MIN_VRAM_GB of memory, "light" from NVIDIA_LIGHT_MIN_VRAM_GB, and below
# that the CPU. On CPU "auto" always means "light".
# Was 12 GB before the precheck existed; best measured 4.0 GB at batch 1 and 5.2 GB at batch 8 on an
# RTX 6000 (fp16, 20 s pieces), so 6 GB cards run it comfortably at a small batch.
NVIDIA_BEST_MIN_VRAM_GB = 6
NVIDIA_LIGHT_MIN_VRAM_GB = 3   # light: 1.7 GB at batch 1 + reserve below

# --------------------------------------------------------------------------------------
# Apple Silicon: MLX (mlx-audio) with quantised weights (best 4-bit, light 8-bit)
# --------------------------------------------------------------------------------------
MAC_MODELS = {
    "best": "mlx-community/Qwen3-ASR-1.7B-4bit",
    "light": "mlx-community/Qwen3-ASR-0.6B-8bit",
}
# --model auto: "best" when the Mac has at least this much TOTAL unified memory (GB), otherwise
# "light". Measured (M2 8 GB, 30 FLEURS clips): peak GPU memory light 1.8, best 2.5, 1.7B-8bit 3.3 GB, so all
# fit on 8 GB, and best (4-bit) halves light's WER (2.8% vs 5.6%) at 9.5x real time (light 17x).
MAC_BEST_MIN_RAM_GB = 8
# Measured Mac numbers for the plain-words descriptions (M2 8 GB, 30 FLEURS Italian clips).
MAC_WER_PERCENT = {"best": 2.8, "light": 5.6}
MAC_REALTIME_X = {"best": 9.5, "light": 17}
MAC_BEST_PEAK_GB = 2.5

# Extra Mac presets that mac_selftest.sh also measures (name -> repo). The 4-bit 1.7B model is
# smaller than the 8-bit one; it is now the "best" preset (measured), 8-bit stays for comparison.
MAC_SELFTEST_MODELS = {
    "1.7B-8bit": "mlx-community/Qwen3-ASR-1.7B-8bit",
    "1.7B-4bit": "mlx-community/Qwen3-ASR-1.7B-4bit",
    "0.6B-8bit": "mlx-community/Qwen3-ASR-0.6B-8bit",
}

# Mac batch size (pieces decoded together). Small on purpose: every extra piece in a batch adds
# activation memory and a fanless MacBook Air throttles under sustained load anyway.
# M2 8 GB, best 4-bit, 30 clips: batch 1 / 2 / 4 all about 11.5x real time, batching only adds process memory.
MAC_BATCH_SIZE_LOW_RAM = 1     # total RAM below MAC_BATCH_MIN_RAM_GB
MAC_BATCH_SIZE_HIGH_RAM = 2
MAC_BATCH_MIN_RAM_GB = 16      # not measured on a 16 GB Mac yet

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

# --------------------------------------------------------------------------------------
# Hardware precheck (localtranscribe/precheck.py, standard library only)
# --------------------------------------------------------------------------------------
# Small file in the repo folder that remembers the choice made by the check (git-ignored).
SETTINGS_FILE = "localtranscribe_settings.json"

# A "16 GB" computer may report 15.6 GB, a "6 GB" graphics card 5.9 GB: memory thresholds allow this much less.
MEMORY_ROUNDING_TOLERANCE_GB = 0.5

# --- NVIDIA (PyTorch cu128 wheel, requirements-windows.txt) ---
# Peak video memory at batch 1, measured on a Quadro RTX 6000 (fp16, 20 s pieces): best 4.0 GB, light 1.7 GB;
# every extra piece decoded together costs about 0.16 GB (best: 5.2 GB at batch 8, light: 2.8 GB).
NVIDIA_PEAK_VRAM_GB = {"best": 4.0, "light": 1.7}
NVIDIA_VRAM_PER_EXTRA_PIECE_GB = 0.16
# Kept free on top of the peak: CUDA context and allocator slack (measured 0.75 GB for best and 0.45 GB for light
# as whole-process usage minus torch's peak allocation) plus a safety margin.
NVIDIA_VRAM_RESERVE_GB = 1.25
NVIDIA_MAX_BATCH_SIZE = 8

# Oldest GPU the CUDA 12.8 build of PyTorch has kernels for. `torch.cuda.get_arch_list()` of torch 2.11.0+cu128
# (the version pinned in requirements-windows.txt) returns
#     ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
# so Turing (GeForce GTX 16 / RTX 20 series, Quadro RTX, 2018) is the oldest supported generation; Pascal (sm_61,
# GTX 10 series) and Volta (sm_70) are not. Re-check this list whenever the torch pin changes.
NVIDIA_MIN_COMPUTE_CAPABILITY = (7, 5)

# Oldest NVIDIA driver that supports CUDA 12.8 (the CUDA the torch wheel is built with). Source: NVIDIA CUDA
# Toolkit Release Notes, Table 3 "CUDA Toolkit and Corresponding Driver Versions", row "CUDA 12.8 GA":
#     Linux x86_64 >=570.26, Windows x86_64 >=570.65
# https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html
# (Drivers back to 525.60.13 / 528.33 can run CUDA 12.x programs through "minor version compatibility", but NVIDIA
# does not guarantee every feature of the wheel there, so the check uses the toolkit minimum.)
NVIDIA_MIN_DRIVER = {"windows": "570.65", "linux": "570.26"}

# --- Apple Silicon ---
# The MLX wheels need macOS 14 (Sonoma) or newer.
MAC_MIN_MACOS = 14
# On this much memory or less the check adds "close memory-heavy apps".
MAC_LOW_RAM_GB = 8

# --- CPU only ---
# Light on the CPU (float32), measured on a 6-core/12-thread Xeon W-3235: about 6 GB process RAM at the peak,
# 2.6x real time (1 hour of audio in about 23 minutes).
CPU_LIGHT_PEAK_RAM_GB = 6.0
CPU_LIGHT_REALTIME_X = 2.6
CPU_LOW_RAM_WARN_GB = 8
# Best on the CPU has not been measured: float32 weights alone are about 7 GB. ESTIMATE; below this it is not offered.
CPU_BEST_MIN_RAM_GB = 16
CPU_BATCH_SIZE_HIGH_RAM = 2    # pieces decoded together on the CPU when RAM is at least CPU_BATCH_MIN_RAM_GB
CPU_BATCH_SIZE_LOW_RAM = 1
CPU_BATCH_MIN_RAM_GB = 12

# --- Accuracy and speed, only used to describe the presets in plain words ---
# Word error rate on FLEURS Italian (100 clips, two sets and a 24-minute file), NVIDIA float16.
WER_PERCENT = {"best": (2.4, 2.7), "light": (4.4, 5.8)}
NVIDIA_REALTIME_X = 26         # RTX 6000: best 25-26x, light 27x

# --- Disk space ---
# Virtual environment size by install type (measured: NVIDIA about 5 GB, CPU about 1.2 GB, Mac about 1 GB).
VENV_SIZE_GB = {"nvidia": 5.0, "cpu": 1.2, "mac": 1.0}
# Free space that must remain after the install (uv download cache, temporary files, the transcripts).
DISK_MARGIN_GB = 1.0
# requirements file per install type
REQUIREMENTS_FILES = {"nvidia": "requirements-windows.txt", "cpu": "requirements-cpu.txt", "mac": "requirements-mac.txt"}
