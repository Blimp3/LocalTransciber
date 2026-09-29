"""Hardware detection and the automatic choice of device, model and batch size.

Nothing here imports torch or mlx at module level, so it is cheap to import on any platform.
"""
import os
import platform
import subprocess
import sys

from . import config, precheck

GIB = 1024 ** 3


class SetupError(RuntimeError):
    """A problem the user can fix; the message is shown as-is without a traceback."""


# ----------------------------------------------------------------------------- platform

def is_apple_silicon():
    """True on an Apple Silicon Mac running a native (arm64) Python."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def check_platform():
    """Refuse clearly on Macs that cannot run the MLX backend."""
    if sys.platform != "darwin":
        return
    if platform.machine() == "arm64":
        return
    hw_arm = _sysctl("hw.optional.arm64")
    if hw_arm == "1":
        raise SetupError(
            "This Python runs under Rosetta (Intel emulation) on an Apple Silicon Mac. "
            "Delete the .venv folder and run setup_mac.sh again from a normal Terminal window."
        )
    raise SetupError(
        "This tool needs an Apple Silicon Mac (M1 or newer). Intel Macs are not supported."
    )


def _sysctl(name):
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


# ----------------------------------------------------------------------------- memory

def total_ram_gb():
    """Total physical RAM in GiB (on a Mac this is the unified memory the GPU shares)."""
    try:
        if sys.platform == "darwin":
            v = _sysctl("hw.memsize")
            if v:
                return int(v) / GIB
        elif sys.platform.startswith("win"):
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullTotalPhys / GIB
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024 / GIB
    except Exception:
        pass
    return None


def cuda_memory_gb():
    """(total_gb, free_gb) of GPU 0, or None without a usable CUDA GPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info(0)
        return total / GIB, free / GIB
    except Exception:
        return None


def peak_rss_gb():
    """Peak resident memory of this process in GiB (None if unknown)."""
    try:
        if sys.platform.startswith("win"):
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            ctypes.windll.psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
            return pmc.PeakWorkingSetSize / GIB
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (rss if sys.platform == "darwin" else rss * 1024) / GIB  # bytes on macOS, KiB on Linux
    except Exception:
        return None


# ----------------------------------------------------------------------------- device

def resolve_device(requested="auto"):
    """Return 'cuda', 'mps' or 'cpu'.

    On an Apple Silicon Mac 'mps' means the Apple GPU (transcription runs through MLX/Metal;
    diarization through PyTorch/MPS) and 'cpu' means MLX on the CPU cores.
    """
    requested = (requested or "auto").lower()
    if requested not in ("auto", "cuda", "mps", "cpu"):
        raise SetupError(f"Unknown --device {requested!r}; use auto, cuda, mps or cpu.")
    if is_apple_silicon():
        if requested == "cuda":
            raise SetupError("CUDA is not available on a Mac. Use --device auto, mps or cpu.")
        return "cpu" if requested == "cpu" else "mps"
    if requested == "mps":
        raise SetupError("--device mps only exists on Apple Silicon Macs.")
    if requested == "cpu":
        return "cpu"
    has_cuda = cuda_memory_gb() is not None
    if requested == "cuda":
        if not has_cuda:
            raise SetupError(
                "--device cuda was requested but PyTorch cannot see an NVIDIA GPU. "
                "Check the NVIDIA driver, or use --device cpu."
            )
        return "cuda"
    return "cuda" if has_cuda else "cpu"


def accelerator_memory_gb(device):
    """Memory the chosen accelerator offers: VRAM on NVIDIA, total unified RAM on a Mac, else None."""
    if device == "cuda":
        m = cuda_memory_gb()
        return m[0] if m else None
    if device == "mps":
        return total_ram_gb()
    return None


def diarization_device(device):
    """Device for the speaker-embedding model: cuda -> mps -> cpu, never above the requested one."""
    if device == "cpu":
        return "cpu"
    try:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            return "cuda"
        if device == "mps" and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ----------------------------------------------------------------------------- model / batch

def choose_model(requested, device, mem_gb):
    """Turn --model best|light|auto|<repo id> into the Hugging Face repo id for this platform."""
    requested = (requested or "auto").strip()
    mac = is_apple_silicon()
    table = config.MAC_MODELS if mac else config.TORCH_MODELS
    key = requested.lower()
    if key == "auto":  # the same rule as the hardware precheck
        if mac:
            big = precheck.mac_preset_for(mem_gb) == "best"
        else:
            big = device == "cuda" and precheck.nvidia_preset_for(mem_gb) == "best"
        return table["best" if big else "light"]
    if key in table:
        return table[key]
    if mac:
        repo = config.MAC_MODEL_ALIASES.get(requested, requested)
        if "/" in repo and not repo.startswith("mlx-community/") and not os.path.isdir(repo):
            raise SetupError(
                f"{requested!r} is not an MLX model. On a Mac use --model best, --model light or an "
                "mlx-community/... repo id."
            )
        return repo
    return requested


def choose_batch_size(requested, device, repo, mem_gb=None):
    """Pieces decoded together. Scales down on small accelerators; an explicit --batch-size wins."""
    if requested:
        return max(1, int(requested))
    if is_apple_silicon():
        return precheck.mac_batch_size()
    if device == "cuda":
        m = cuda_memory_gb()
        free = m[1] if m else (mem_gb or 8)
        # From the free video memory: peak at batch 1 plus the measured cost of every extra piece (see config.py).
        return precheck.nvidia_batch_size(precheck.preset_of_repo(repo), free)
    return precheck.cpu_batch_size(total_ram_gb())


def resolve_run_config(device_arg="auto", model_arg="auto", batch_arg=0, saved=None, notes=None):
    """The one place that turns --device/--model/--batch-size into concrete choices.
    Returns (device, repo, batch_size, accelerator_memory_gb).

    `saved` is the choice remembered by the hardware check (precheck.load_settings). It fills in only what the
    caller leaves as None (device_arg, model_arg) or 0 (batch_arg); explicit values always win. It is used only when
    it was made for the device that is actually in use, and the saved batch size is a ceiling: it is lowered when
    the video memory that is free right now cannot hold it. Reasons for ignoring it are appended to `notes`."""
    check_platform()
    implicit_device = device_arg is None
    if saved and implicit_device and saved["device"] == "cpu":
        device_arg = "cpu"  # the check chose CPU mode (e.g. unsupported GPU or driver)
    device = resolve_device(device_arg or "auto")
    mem_gb = accelerator_memory_gb(device)
    use_saved = bool(saved) and saved["device"] == device
    if saved and not use_saved and implicit_device and notes is not None:
        notes.append(f"The saved hardware choice was made for the {saved['device']} but this run uses the {device}; "
                     "using the automatic choice instead (run check_hardware to update it).")
    if model_arg is None:
        model_arg = saved["preset"] if use_saved else "auto"
    repo = choose_model(model_arg, device, mem_gb)
    batch = choose_batch_size(batch_arg, device, repo, mem_gb)
    if use_saved and not batch_arg and repo == choose_model(saved["preset"], device, mem_gb):
        batch = min(batch, saved["batch_size"])
    return device, repo, batch, mem_gb
