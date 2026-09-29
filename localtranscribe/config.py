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

# --confidence: alternatives saved per decoded word-piece (the chosen one is not counted twice).
CONFIDENCE_TOP_K = 5
# --confidence: a word whose probability (product of its pieces') is below this is marked "unsure".
# 0.9 flags 9.4% of the words and catches 70% of the wrong ones (FLEURS Italian, 100 clips, best preset).
CONFIDENCE_UNSURE = 0.9
# --confidence: an alternative piece below this probability is not tried for candidates. All 20 hits (reference word
# among the candidates) came from alts with p >= 0.027; 0.02 drops 433 of the 734 alts tried (FLEURS Italian, 100 clips).
CONFIDENCE_ALT_MIN = 0.02

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
# "light". Measured (M2 8 GB, 100 FLEURS clips, 25.3 min): peak GPU memory light 1.8, best 2.5, 1.7B-8bit 3.3 GB,
# so all fit on 8 GB, and best (4-bit) makes about 40% fewer mistakes than light (WER 3.5% vs 5.7%) at 6.4x real
# time (light 18.7x). Speed varies with memory pressure (6-12x measured for best).
MAC_BEST_MIN_RAM_GB = 8
# Measured Mac numbers for the plain-words descriptions (M2 8 GB, 100 FLEURS Italian clips).
MAC_WER_PERCENT = {"best": 3.5, "light": 5.7}
MAC_REALTIME_X = {"best": 6.4, "light": 18.7}
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
# M2 8 GB, best 4-bit, first 30 clips: batch 1 5.4x, batch 2 3.3x, batch 4 6.8x real time. Batch 2 was slower than
# batch 1 in 3 of 4 measurements and never faster, and batching changes some transcripts. No 16 GB Mac was
# available to test, so batch 1 everywhere. --batch-size still overrides.
MAC_BATCH_SIZE = 1

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
    "mlx-community/Qwen3-0.6B-4bit": 0.35,
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

# --- AI correction (--correct, localtranscribe/correct.py) ---
# A small local model rescores the speech model's own alternatives at unsure words; the Bend checker vets each choice.
# Score of an option = LM log-probability of the paragraph + ALPHA * log(speech model p); a swap is proposed when it
# beats the original by MARGIN nats. Sweep on the 100 FLEURS clips (91 unsure words with alternatives, 32 of them wrong,
# 15 with the right answer among the alternatives): ALPHA 4, MARGIN 1 gave 13 suggestions = 5 fixes, 1 break, 7 neutral
# (WER if all accepted 3.35% vs 3.51%). A weak model is best used as a tie-breaker, hence the high ALPHA.
CORRECT_MODEL = "mlx-community/Qwen3-0.6B-4bit"
CORRECT_ALPHA = 4.0
CORRECT_MARGIN = 1.0
