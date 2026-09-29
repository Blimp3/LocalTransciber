"""Hardware precheck: look at this computer, recommend the model that should run on it, remember the choice.

Standard library only, so it runs before any package is installed (the setup scripts start it with the bare
uv-managed Python), and it never imports torch or mlx.

    python -m localtranscribe.precheck [auto|best|light|both|cpu] [--yes] [--json] [--no-save]
    python -m localtranscribe.precheck --print preset|device|models|requirements|batch    (read the saved choice)

  best / light / both   choose the model(s) without asking (both = download both, use the recommended one)
  cpu                   force the CPU-only install even if a usable NVIDIA GPU exists
  --yes                 accept the recommendation without asking (also the case when stdin is not a terminal)
  --json                machine-readable result on stdout; never asks; saves only together with --yes
  --no-save             do not write localtranscribe_settings.json

Exit code: 0 = fine, 1 = a blocking problem (unsupported computer, not enough disk space), 2 = bad arguments.
"""
import argparse
import csv
import datetime
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap

try:
    from . import config
except ImportError:  # started as a plain script: python localtranscribe/precheck.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from localtranscribe import config

GIB = 1024 ** 3
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repo folder
PRESETS = ("best", "light")
DEVICES = ("cuda", "mps", "cpu")
SETTINGS_VERSION = 1
NVIDIA_QUERY = "name,memory.total,memory.free,driver_version,compute_cap"


# ----------------------------------------------------------------------------- running programs

def _exec(cmd, timeout=10):
    """(returncode, stdout, stderr); returncode is None when the program could not run or timed out."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout)
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        return None, "", f"no answer within {timeout} seconds"
    except Exception as e:  # not installed, not allowed, ...
        return None, "", f"{type(e).__name__}: {e}"


def _run(cmd, timeout=10):
    """stdout of a program that succeeded, else None."""
    rc, out, _err = _exec(cmd, timeout)
    return out.strip() if rc == 0 else None


def _sysctl(name):
    return _run(["sysctl", "-n", name], timeout=5) or None


def _int(text):
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------- pure rules (also used by devices.py)

def nvidia_preset_for(total_gb):
    """'best', 'light' or None (use the CPU) for a GPU with this much memory."""
    if total_gb is None:
        return None
    tol = config.MEMORY_ROUNDING_TOLERANCE_GB
    if total_gb >= config.NVIDIA_BEST_MIN_VRAM_GB - tol:
        return "best"
    if total_gb >= config.NVIDIA_LIGHT_MIN_VRAM_GB - tol:
        return "light"
    return None


def mac_preset_for(total_gb):
    """'best' with enough unified memory, else 'light'. Measured rule (see config.MAC_BEST_MIN_RAM_GB)."""
    if total_gb is not None and total_gb >= config.MAC_BEST_MIN_RAM_GB - config.MEMORY_ROUNDING_TOLERANCE_GB:
        return "best"
    return "light"


def nvidia_batch_size(preset, free_gb):
    """Pieces decoded together, from the free video memory: peak at batch 1 plus the measured cost per extra piece."""
    spare = free_gb - config.NVIDIA_VRAM_RESERVE_GB - config.NVIDIA_PEAK_VRAM_GB[preset]
    extra = int(spare // config.NVIDIA_VRAM_PER_EXTRA_PIECE_GB) if spare > 0 else 0
    return max(1, min(config.NVIDIA_MAX_BATCH_SIZE, 1 + extra))


def mac_batch_size(total_gb):
    high = total_gb is not None and total_gb >= config.MAC_BATCH_MIN_RAM_GB - config.MEMORY_ROUNDING_TOLERANCE_GB
    return config.MAC_BATCH_SIZE_HIGH_RAM if high else config.MAC_BATCH_SIZE_LOW_RAM


def cpu_batch_size(ram_gb):
    ram = ram_gb or 8
    return config.CPU_BATCH_SIZE_HIGH_RAM if ram >= config.CPU_BATCH_MIN_RAM_GB else config.CPU_BATCH_SIZE_LOW_RAM


def preset_of_repo(repo):
    return "light" if "0.6B" in repo else "best"


def _version_tuple(text):
    parts = re.findall(r"\d+", text or "")
    return tuple(int(p) for p in parts[:3]) or None


def driver_ok(driver, system):
    """True/False for an NVIDIA driver version against the CUDA 12.8 minimum; None if unknown."""
    have = _version_tuple(driver)
    need = _version_tuple(config.NVIDIA_MIN_DRIVER.get(system))
    if have is None or need is None:
        return None
    return have >= need


def requirements_for(device):
    kind = {"cuda": "nvidia", "mps": "mac"}.get(device, "cpu")
    return config.REQUIREMENTS_FILES[kind]


# ----------------------------------------------------------------------------- formatting

def fmt_gb(value):
    if value is None:
        return "unknown"
    if value >= 100:
        return f"{value:.0f}"
    text = f"{value:.1f}"
    return text[:-2] if text.endswith(".0") else text


def _wer_text(preset):
    lo, hi = config.WER_PERCENT[preset]
    return f"{lo:g}-{hi:g}"


# ----------------------------------------------------------------------------- detection: operating system

def detect_os():
    system = platform.system()
    info = {"system": system, "name": system, "version": platform.version(), "machine": platform.machine(),
            "mac_version": None, "linux_name": None}
    if system == "Windows":
        parts = platform.version().split(".")
        major, build = _int(parts[0]), _int(parts[2]) if len(parts) > 2 else None
        if major == 10 and build is not None:
            info["name"] = "Windows 11" if build >= 22000 else "Windows 10"
            info["name"] += f" (build {build})"
        else:
            info["name"] = f"Windows {platform.release()} ({platform.version()})"
    elif system == "Darwin":
        ver = _run(["sw_vers", "-productVersion"], timeout=5) or platform.mac_ver()[0] or None
        info["mac_version"] = ver
        info["name"] = f"macOS {ver}" if ver else "macOS"
    elif system == "Linux":
        name = None
        try:
            with open("/etc/os-release", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        name = line.split("=", 1)[1].strip().strip('"')
        except OSError:
            pass
        info["linux_name"] = name
        info["name"] = f"{name} (Linux)" if name else "Linux"
    return info


# ----------------------------------------------------------------------------- detection: CPU and memory

def _windows_physical_cores():
    """Physical core count through GetLogicalProcessorInformationEx (RelationProcessorCore = 0)."""
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    fn = k32.GetLogicalProcessorInformationEx
    fn.argtypes = [wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    fn.restype = wintypes.BOOL
    size = wintypes.DWORD(0)
    fn(0, None, ctypes.byref(size))
    if not size.value:
        return None
    buf = ctypes.create_string_buffer(size.value)
    if not fn(0, buf, ctypes.byref(size)):
        return None
    raw, offset, count = buf.raw, 0, 0
    while offset + 8 <= size.value:
        relationship = int.from_bytes(raw[offset:offset + 4], "little")
        length = int.from_bytes(raw[offset + 4:offset + 8], "little")
        if length == 0:
            break
        count += relationship == 0
        offset += length
    return count or None


def detect_cpu(system):
    cpu = {"name": None, "cores": None, "threads": os.cpu_count(), "performance_cores": None, "efficiency_cores": None}
    try:
        if system == "Windows":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                cpu["name"] = " ".join(str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).split()) or None
            cpu["cores"] = _windows_physical_cores()
        elif system == "Darwin":
            cpu["name"] = _sysctl("machdep.cpu.brand_string")
            cpu["cores"] = _int(_sysctl("hw.physicalcpu"))
            cpu["threads"] = _int(_sysctl("hw.logicalcpu")) or cpu["threads"]
            cpu["performance_cores"] = _int(_sysctl("hw.perflevel0.physicalcpu"))
            cpu["efficiency_cores"] = _int(_sysctl("hw.perflevel1.physicalcpu"))
        elif system == "Linux":
            pairs = set()
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                for block in f.read().split("\n\n"):
                    fields = dict((a.strip(), b.strip()) for a, _, b in (ln.partition(":") for ln in block.splitlines()))
                    cpu["name"] = cpu["name"] or fields.get("model name") or fields.get("Hardware")
                    if "core id" in fields:
                        pairs.add((fields.get("physical id"), fields["core id"]))
            cpu["cores"] = len(pairs) or None
    except Exception:
        pass
    return cpu


def _windows_memory():
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
    return st.ullTotalPhys / GIB, st.ullAvailPhys / GIB


def _mac_available_gb():
    """Free + inactive + speculative + purgeable pages from vm_stat (what macOS can hand out without swapping)."""
    text = _run(["vm_stat"], timeout=5)
    if not text:
        return None
    m = re.search(r"page size of (\d+) bytes", text)
    page = int(m.group(1)) if m else 16384
    pages = 0
    for label in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        m = re.search(label + r":\s+(\d+)", text)
        pages += int(m.group(1)) if m else 0
    return pages * page / GIB


def detect_memory(system):
    total = avail = None
    try:
        if system == "Windows":
            total, avail = _windows_memory()
        elif system == "Darwin":
            size = _int(_sysctl("hw.memsize"))
            total = size / GIB if size else None
            avail = _mac_available_gb()
        elif system == "Linux":
            with open("/proc/meminfo", encoding="utf-8", errors="replace") as f:
                info = {ln.split(":")[0]: _int(ln.split()[1]) for ln in f if ":" in ln and len(ln.split()) > 1}
            total = info["MemTotal"] * 1024 / GIB if info.get("MemTotal") else None
            free = info.get("MemAvailable", info.get("MemFree"))
            avail = free * 1024 / GIB if free else None
    except Exception:
        pass
    return {"total_gb": total, "available_gb": avail}


# ----------------------------------------------------------------------------- detection: disk and caches

def hf_hub_dir(env=None):
    """Folder of the Hugging Face model cache: HF_HUB_CACHE, else HF_HOME/hub, else ~/.cache/huggingface/hub."""
    env = os.environ if env is None else env
    if env.get("HF_HUB_CACHE"):
        return env["HF_HUB_CACHE"]
    if env.get("HF_HOME"):
        return os.path.join(env["HF_HOME"], "hub")
    if env.get("XDG_CACHE_HOME"):
        return os.path.join(env["XDG_CACHE_HOME"], "huggingface", "hub")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def _existing_parent(path):
    path = os.path.abspath(path)
    while path and not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def _same_volume(a, b):
    if os.path.splitdrive(a)[0].upper() != os.path.splitdrive(b)[0].upper():
        return False
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return True


def _free_gb(path):
    try:
        return shutil.disk_usage(path).free / GIB
    except Exception:
        return None


def detect_disk(root=None):
    root = os.path.abspath(root or ROOT)
    hf = hf_hub_dir()
    hf_at, root_at = _existing_parent(hf), _existing_parent(root)
    return {"repo_path": root, "repo_free_gb": _free_gb(root_at), "hf_path": hf, "hf_free_gb": _free_gb(hf_at),
            "same_volume": _same_volume(root_at, hf_at)}


def model_cached(repo, hub=None):
    """True when the Hugging Face cache holds weight files for this repo (what an offline run needs)."""
    snaps = os.path.join(hub or hf_hub_dir(), "models--" + repo.replace("/", "--"), "snapshots")
    for ext in ("safetensors", "bin", "npz", "gguf"):
        if glob.glob(os.path.join(snaps, "*", "**", "*." + ext), recursive=True):
            return True
    return False


def installed_torch_version(root=None):
    """Version of torch inside <repo>/.venv (from its dist-info folder name), or None."""
    venv = os.path.join(root or ROOT, ".venv")
    patterns = [os.path.join(venv, "Lib", "site-packages", "torch-*.dist-info"),
                os.path.join(venv, "lib", "python3*", "site-packages", "torch-*.dist-info")]
    for pattern in patterns:
        for path in glob.glob(pattern):
            return os.path.basename(path)[len("torch-"):-len(".dist-info")]
    return None


def venv_ready(root, kind):
    """The virtual environment already holds the right PyTorch build, so the install costs (almost) no space."""
    version = installed_torch_version(root)
    if version is None:
        return False
    if kind == "nvidia":
        return "+cu" in version
    if kind == "cpu":
        return "+cpu" in version
    return "+" not in version


# ----------------------------------------------------------------------------- detection: NVIDIA

def _find_nvidia_smi(system):
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if system == "Windows":
        for base in (os.environ.get("SystemRoot", r"C:\Windows") + r"\System32",
                     os.environ.get("ProgramFiles", r"C:\Program Files") + r"\NVIDIA Corporation\NVSMI"):
            path = os.path.join(base, "nvidia-smi.exe")
            if os.path.exists(path):
                return path
    return None


def _mib_to_gb(text):
    m = re.search(r"\d+", text or "")
    return int(m.group()) / 1024 if m else None


def _parse_capability(text):
    m = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", text or "")
    return [int(m.group(1)), int(m.group(2))] if m else None


def parse_nvidia_smi(text):
    """Rows of `nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version[,compute_cap] --format=csv,noheader`."""
    gpus = []
    for row in csv.reader([ln for ln in (text or "").splitlines() if ln.strip()], skipinitialspace=True):
        if len(row) < 4 or _mib_to_gb(row[1]) is None:
            continue
        gpus.append({"index": len(gpus), "name": row[0].strip(), "total_gb": _mib_to_gb(row[1]),
                     "free_gb": _mib_to_gb(row[2]), "driver": row[3].strip(),
                     "compute_capability": _parse_capability(row[4]) if len(row) > 4 else None})
    return gpus


def detect_nvidia(system):
    """{'smi': 'found'|'missing'|'failed', 'message', 'driver', 'gpus': [...]}; never raises."""
    result = {"smi": "missing", "message": None, "driver": None, "gpus": []}
    exe = _find_nvidia_smi(system) if system in ("Windows", "Linux") else None
    if not exe:
        return result
    result["smi"] = "found"
    cmd = [exe, f"--query-gpu={NVIDIA_QUERY}", "--format=csv,noheader"]
    rc, out, err = _exec(cmd, timeout=15)
    if rc not in (None, 0) and "compute_cap" in (out + err):  # drivers before ~510 do not know compute_cap
        cmd[1] = "--query-gpu=" + NVIDIA_QUERY.replace(",compute_cap", "")
        rc, out, err = _exec(cmd, timeout=15)
    gpus = parse_nvidia_smi(out) if rc == 0 else []
    if not gpus:
        result["smi"] = "failed"
        lines = [ln.strip() for ln in (err.strip() or out.strip() or "no answer").splitlines() if ln.strip()]
        result["message"] = (lines[0] if lines else "no answer")[:200]
        return result
    result["gpus"] = gpus
    result["driver"] = gpus[0]["driver"]
    return result


# ----------------------------------------------------------------------------- detection: Apple Silicon

def parse_system_profiler_hardware(text):
    """(model name, model identifier, chip) from `system_profiler -json SPHardwareDataType` (text output as fallback)."""
    try:
        item = json.loads(text)["SPHardwareDataType"][0]
        return item.get("machine_name"), item.get("machine_model"), item.get("chip_type")
    except Exception:
        pass

    def field(*labels):
        m = re.search(r"^\s*(?:%s):\s*(.+?)\s*$" % "|".join(labels), text or "", re.M)
        return m.group(1) if m else None

    return field("Model Name"), field("Model Identifier"), field("Chip")


def parse_system_profiler_gpu_cores(text):
    """GPU core count from `system_profiler -json SPDisplaysDataType` (text output as fallback)."""
    try:
        for item in json.loads(text)["SPDisplaysDataType"]:
            cores = _int(item.get("sppci_cores"))
            if cores:
                return cores
    except Exception:
        pass
    m = re.search(r"Total Number of Cores:\s*(\d+)", text or "")
    return int(m.group(1)) if m else None


def detect_apple():
    """Chip, memory, model, GPU cores and Rosetta state of a Mac, or None on other systems."""
    if sys.platform != "darwin":
        return None
    arch = platform.machine()
    hw_arm = _sysctl("hw.optional.arm64")
    translated = _sysctl("sysctl.proc_translated")
    size = _int(_sysctl("hw.memsize"))
    apple = {
        "apple_silicon": hw_arm == "1" or arch == "arm64",
        "python_arch": arch,
        "rosetta": translated == "1" or (hw_arm == "1" and arch != "arm64"),
        "chip": _sysctl("machdep.cpu.brand_string"),
        "unified_gb": size / GIB if size else None,
        "model_name": None,
        "model_id": _sysctl("hw.model"),
        "gpu_cores": None,
    }
    text = _run(["system_profiler", "-json", "SPHardwareDataType"], timeout=15)
    if text:
        name, model_id, chip = parse_system_profiler_hardware(text)
        apple["model_name"] = name
        apple["model_id"] = model_id or apple["model_id"]
        apple["chip"] = apple["chip"] or chip
    text = _run(["system_profiler", "-json", "SPDisplaysDataType"], timeout=10)  # can be slow: optional
    if text:
        apple["gpu_cores"] = parse_system_profiler_gpu_cores(text)
    name = f"{apple['model_name'] or ''} {apple['model_id'] or ''}"
    apple["fanless"] = "macbook air" in name.lower() or "macbookair" in name.lower()
    return apple


# ----------------------------------------------------------------------------- detection: everything

def detect(root=None):
    """The hardware profile: a plain, JSON-serialisable dict. Every part is optional and failures are swallowed."""
    os_info = detect_os()
    system = os_info["system"]
    return {
        "os": os_info,
        "cpu": detect_cpu(system),
        "memory": detect_memory(system),
        "disk": detect_disk(root),
        "nvidia": detect_nvidia(system),
        "apple": detect_apple() if system == "Darwin" else None,
        "python": platform.python_version(),
    }


# ----------------------------------------------------------------------------- recommendation

def _option(kind, preset, device, batch, fit, tagline, summary, warnings=()):
    table = config.MAC_MODELS if kind == "mac" else config.TORCH_MODELS
    return {"key": preset, "preset": preset, "device": device, "kind": kind, "model": table[preset],
            "name": "Qwen3-ASR " + ("1.7B" if preset == "best" else "0.6B"), "batch_size": batch,
            "fit": fit,  # good = recommended range, tight = possible but not comfortable, poor = will probably fail
            "offered": fit in ("good", "tight"), "tagline": tagline, "summary": summary, "warnings": list(warnings),
            "requirements": config.REQUIREMENTS_FILES[kind]}


def _assess_mac(profile, rep):
    apple = profile["apple"] or {}
    ver = profile["os"].get("mac_version")
    if not apple.get("apple_silicon"):
        rep["blocking"].append("This Mac has an Intel processor. LocalTranscribe needs an Apple Silicon Mac "
                               "(M1 or newer); Intel Macs are not supported.")
        return
    if apple.get("rosetta"):
        rep["blocking"].append("This Python runs under Rosetta (Intel emulation) on an Apple Silicon Mac. Delete the .venv "
                               "folder and run setup_mac.sh again from a normal Terminal window.")
        return
    major = _int(ver.split(".")[0]) if ver else None
    if major is not None and major < config.MAC_MIN_MACOS:
        rep["blocking"].append(f"macOS {ver} is too old. LocalTranscribe needs macOS {config.MAC_MIN_MACOS} (Sonoma) or "
                               "newer: Apple menu > System Settings > General > Software Update.")
        return
    if major is None:
        rep["warnings"].append("Could not read the macOS version; macOS %d or newer is required." % config.MAC_MIN_MACOS)
    ram = apple.get("unified_gb") or profile["memory"]["total_gb"]
    if ram is None:
        rep["warnings"].append("Could not read the memory size of this Mac; assuming the light model.")
    preset = mac_preset_for(ram)
    batch = mac_batch_size(ram)
    rep["platform"] = "mac"
    ram_txt = fmt_gb(ram) + " GB"
    best_summary = (f"about {config.MAC_WER_PERCENT['best']:g} mistakes per 100 words and about "
                    f"{config.MAC_REALTIME_X['best']:g}x real time on an M2 Mac; needs about {config.MAC_BEST_PEAK_GB:g} GB of memory")
    light_summary = (f"less accurate ({config.MAC_WER_PERCENT['light']:g} mistakes per 100 words on an M2 Mac), "
                     f"but lighter on memory and faster ({config.MAC_REALTIME_X['light']:g}x real time)")
    best_warn = [] if preset == "best" else [
        f"best is tight on {ram_txt} of memory: it may be slow or run out of memory (not tested on this kind of Mac). "
        "light is the safer choice."]
    if preset == "best":
        best_fit, light_fit = "good", "good"
    else:
        best_fit, light_fit = "tight", "good"
    rep["options"] = {"best": _option("mac", "best", "mps", batch, best_fit, "most accurate", best_summary, best_warn),
                      "light": _option("mac", "light", "mps", batch, light_fit, "smaller and faster", light_summary)}
    rep["recommended"] = preset
    rep["order"] = [preset] + [p for p in PRESETS if p != preset]
    if apple.get("fanless"):
        rep["notes"].append("This Mac has no fan: keep it plugged in for long files.")
    if ram is not None and ram <= config.MAC_LOW_RAM_GB + config.MEMORY_ROUNDING_TOLERANCE_GB:
        rep["notes"].append(f"With {ram_txt} of memory, close memory-heavy apps (browser tabs, video calls) while transcribing.")


def _pick_gpu(profile, rep):
    """The NVIDIA GPU this tool can use (most memory among the supported ones), or None; explains why in `rep`."""
    nv = profile["nvidia"]
    system = profile["os"]["system"].lower()
    gpus = nv.get("gpus") or []
    if not gpus:
        if nv.get("smi") == "failed":
            rep["warnings"].append(f"The NVIDIA driver did not answer ({nv.get('message')}). If this PC has an NVIDIA card, "
                                   "repair or reinstall its driver (nvidia.com/drivers) and run the check again; "
                                   "the processor will be used for now.")
        else:
            rep["notes"].append("No NVIDIA graphics card driver was found, so the processor will be used. If this PC has an "
                                "NVIDIA card, install its driver (nvidia.com/drivers) and run the check again.")
        return None
    minimum = tuple(config.NVIDIA_MIN_COMPUTE_CAPABILITY)
    supported, too_old = [], []
    for g in gpus:
        (too_old if g["compute_capability"] and tuple(g["compute_capability"]) < minimum else supported).append(g)
    if not supported:
        g = too_old[0]
        cc = ".".join(str(n) for n in g["compute_capability"])
        rep["warnings"].append(
            f"The NVIDIA GPU ({g['name']}, compute capability {cc}) is too old for the GPU version of PyTorch that "
            f"this tool installs (it needs compute capability {minimum[0]}.{minimum[1]} or newer: GeForce GTX 16 / RTX 20 "
            "series or newer, 2018). The processor will be used instead: it works, but is much slower.")
        return None
    if driver_ok(nv.get("driver"), system) is False:
        need = config.NVIDIA_MIN_DRIVER[system]
        best = max(supported, key=lambda g: g["total_gb"] or 0)
        later = nvidia_preset_for(best["total_gb"])
        hint = f' (this graphics card would then run the "{later}" model)' if later else ""
        rep["warnings"].append(
            f"The NVIDIA driver {nv['driver']} is too old for CUDA 12.8 (needs {need} or newer). Update it "
            f"(nvidia.com/drivers) and run the check again to use the graphics card{hint}; until then the processor is used.")
        return None
    if driver_ok(nv.get("driver"), system) is None:
        rep["notes"].append("Could not read the NVIDIA driver version; the GPU is used if PyTorch can see it.")
    best = max(supported, key=lambda g: ((g["total_gb"] or 0), -g["index"]))
    if nvidia_preset_for(best["total_gb"]) is None:
        rep["warnings"].append(
            f"The NVIDIA GPU ({best['name']}) has only {fmt_gb(best['total_gb'])} GB of video memory; at least "
            f"{fmt_gb(config.NVIDIA_LIGHT_MIN_VRAM_GB)} GB are needed. The processor will be used instead.")
        return None
    if len(gpus) > 1:
        rep["notes"].append(f"{len(gpus)} NVIDIA GPUs found; using GPU {best['index']} ({best['name']}, "
                            f"{fmt_gb(best['total_gb'])} GB), the supported one with the most memory.")
    for g in too_old:
        rep["notes"].append(f"GPU {g['index']} ({g['name']}) is too old for this tool and is not used.")
    return best


def _cpu_options(profile, rep, kind_note=None):
    ram = profile["memory"]["total_gb"]
    batch = cpu_batch_size(ram)
    minutes = 60 / config.CPU_LIGHT_REALTIME_X
    light_warn = []
    if ram is not None and ram < config.CPU_LOW_RAM_WARN_GB:
        light_warn.append(f"Only {fmt_gb(ram)} GB of memory: the light model needs about {fmt_gb(config.CPU_LIGHT_PEAK_RAM_GB)} GB "
                          "while it loads, so it may fail or be very slow. Close other programs.")
    best_fit = "good" if ram is not None and ram >= config.CPU_BEST_MIN_RAM_GB else "poor"
    best_warn = [] if best_fit == "good" else [
        f"best on the processor needs roughly {fmt_gb(config.CPU_BEST_MIN_RAM_GB - 4)} GB of free memory (estimate) "
        f"and this computer has {fmt_gb(ram)} GB: it will probably fail or crawl. Use light."]
    light_summary = (f"{_wer_text('light')} mistakes per 100 words; slow without a graphics card: about "
                     f"{config.CPU_LIGHT_REALTIME_X:g}x real time, so 1 hour of audio takes about {minutes:.0f} minutes "
                     "(measured on a 6-core Xeon; yours may be slower or faster)")
    best_summary = (f"{_wer_text('best')} mistakes per 100 words, but several times slower than light and needs much more "
                    "memory (estimate: not measured on the processor)")
    rep["platform"] = "cpu"
    rep["options"] = {"best": _option("cpu", "best", "cpu", batch, best_fit, "most accurate", best_summary, best_warn),
                      "light": _option("cpu", "light", "cpu", batch, "good", "the practical choice without a graphics card",
                                       light_summary, light_warn)}
    rep["recommended"] = "light"
    rep["order"] = ["light", "best"]


def _assess_pc(profile, rep, force_cpu):
    system = profile["os"]["system"]
    if system == "Windows":
        ver = _version_tuple(profile["os"]["version"])
        if ver and ver[0] < 10:
            rep["warnings"].append("Windows 10 or 11 is required; this version has not been tested.")
    else:
        rep["warnings"].append("Linux is not officially supported: this check is best effort.")
    gpu = None
    if force_cpu:
        rep["notes"].append("CPU-only mode was requested: an NVIDIA graphics card, if there is one, will not be used.")
    else:
        gpu = _pick_gpu(profile, rep)
    if gpu is None:
        _cpu_options(profile, rep)
        return
    preset = nvidia_preset_for(gpu["total_gb"])
    total, free = gpu["total_gb"], gpu["free_gb"] if gpu["free_gb"] is not None else gpu["total_gb"]
    rep["platform"] = "nvidia"
    rep["gpu_index"] = gpu["index"]
    rep["gpu_count"] = len(profile["nvidia"]["gpus"])
    speed = f"about {config.NVIDIA_REALTIME_X}x real time on an RTX 6000 (1 hour of audio in a few minutes)"

    def need(p):
        return config.NVIDIA_PEAK_VRAM_GB[p] + config.NVIDIA_VRAM_RESERVE_GB

    def option(p):
        fit = "good" if (p == "light" or preset == "best") else "poor"
        warns = []
        if fit == "poor":
            warns.append(f"best needs about {fmt_gb(config.NVIDIA_BEST_MIN_VRAM_GB)} GB of video memory and this graphics card "
                         f"has {fmt_gb(total)} GB: it will probably run out of memory. light is the safer choice.")
        elif free < need(p):
            warns.append(f"Only {fmt_gb(free)} GB of the video memory is free right now (other programs are using the "
                         f"graphics card); {p} needs about {fmt_gb(need(p))} GB. Close them before transcribing.")
        if p == "best":
            tagline = "most accurate"
            summary = (f"about {_wer_text('best')} mistakes per 100 words; {speed}; "
                       f"needs about {fmt_gb(config.NVIDIA_BEST_MIN_VRAM_GB)} GB of video memory")
        else:
            tagline = "smaller and lighter"
            summary = (f"about twice as many mistakes ({_wer_text('light')} per 100 words), needs less video memory "
                       f"(about {fmt_gb(config.NVIDIA_LIGHT_MIN_VRAM_GB)} GB), similar speed")
        return _option("nvidia", p, "cuda", nvidia_batch_size(p, free), fit, tagline, summary, warns)

    rep["options"] = {p: option(p) for p in PRESETS}
    rep["recommended"] = preset
    rep["order"] = [preset] + [p for p in PRESETS if p != preset]


def assess(profile, force_cpu=False):
    """What can run here: blocking problems, warnings, notes, the options (best/light) and the recommended one."""
    rep = {"platform": None, "blocking": [], "warnings": [], "notes": [], "options": {}, "order": [], "recommended": None,
           "gpu_index": None, "gpu_count": 0}
    system = profile["os"]["system"]
    if system == "Darwin":
        _assess_mac(profile, rep)
    elif system in ("Windows", "Linux"):
        _assess_pc(profile, rep, force_cpu)
    else:
        rep["blocking"].append(f"{system} is not supported. LocalTranscribe runs on Windows, macOS (Apple Silicon) and Linux.")
    return rep


# ----------------------------------------------------------------------------- disk plan

def disk_plan(profile, option, presets, root=None):
    """Space the install needs for this option, what is free, and a problem text if it does not fit.
    A virtual environment that already holds the right PyTorch build and models already in the cache cost nothing."""
    root = root or ROOT
    kind = option["kind"]
    table = config.MAC_MODELS if kind == "mac" else config.TORCH_MODELS
    hub = hf_hub_dir()
    repos = [table[p] for p in presets] + [config.SPEAKER_MODEL]
    venv_gb = 0.0 if venv_ready(root, kind) else config.VENV_SIZE_GB[kind]
    models_gb = sum(config.APPROX_DOWNLOAD_GB.get(r, 0.0) for r in repos if not model_cached(r, hub))
    margin = config.DISK_MARGIN_GB
    d = profile["disk"]
    plan = {"venv_gb": venv_gb, "models_gb": models_gb, "margin_gb": margin, "needs_gb": venv_gb + models_gb + margin,
            "already_installed": venv_gb + models_gb == 0, "free_gb": d["repo_free_gb"], "hf_free_gb": d["hf_free_gb"],
            "same_volume": d["same_volume"], "problem": None}
    if d["same_volume"]:
        short = d["repo_free_gb"] is not None and d["repo_free_gb"] < plan["needs_gb"]
        where = f"on the drive that holds {d['repo_path']}"
        text = f"{fmt_gb(d['repo_free_gb'])} GB free {where}, about {fmt_gb(plan['needs_gb'])} GB needed"
    else:
        need_repo, need_hf = venv_gb + margin, models_gb + margin
        short_repo = d["repo_free_gb"] is not None and d["repo_free_gb"] < need_repo
        short_hf = d["hf_free_gb"] is not None and d["hf_free_gb"] < need_hf
        short = short_repo or short_hf
        text = (f"{fmt_gb(d['repo_free_gb'])} GB free where the program lives ({d['repo_path']}), about {fmt_gb(need_repo)} GB "
                f"needed; {fmt_gb(d['hf_free_gb'])} GB free where the models are stored ({d['hf_path']}), "
                f"about {fmt_gb(need_hf)} GB needed")
    if short:
        plan["problem"] = (f"Not enough free disk space: {text}. Free some space (or set HF_HOME to another drive for the "
                           "models) and run this again.")
    return plan


# ----------------------------------------------------------------------------- summary text

def describe_computer(profile):
    """One line: 'MacBook Air (Apple M2, 8-core GPU), 16 GB memory, macOS 15.6' / 'Windows 11 (build ...), CPU, ...'."""
    os_info, apple, cpu, mem = profile["os"], profile.get("apple"), profile["cpu"], profile["memory"]
    if apple and apple.get("apple_silicon") and os_info["system"] == "Darwin":
        inner = [apple.get("chip") or cpu.get("name") or "Apple Silicon"]
        if apple.get("gpu_cores"):
            inner.append(f"{apple['gpu_cores']}-core GPU")
        name = apple.get("model_name") or "Mac"
        ram = apple.get("unified_gb") or mem["total_gb"]
        return f"{name} ({', '.join(inner)}), {fmt_gb(ram)} GB memory, {os_info['name']}"
    parts = [os_info["name"]]
    if cpu.get("name"):
        cores = f"{cpu['cores']} cores, " if cpu.get("cores") else ""
        parts.append(f"{cpu['name']} ({cores}{cpu.get('threads') or '?'} threads)")
    if mem["total_gb"]:
        free = f" ({fmt_gb(mem['available_gb'])} GB free)" if mem["available_gb"] else ""
        parts.append(f"{fmt_gb(mem['total_gb'])} GB memory{free}")
    return ", ".join(parts)


def describe_graphics(profile):
    nv = profile["nvidia"]
    if not nv.get("gpus"):
        return None
    names = [g["name"] if g["name"].upper().startswith("NVIDIA") else "NVIDIA " + g["name"] for g in nv["gpus"]]
    return "; ".join(f"{n}, {fmt_gb(g['total_gb'])} GB video memory" for n, g in zip(names, nv["gpus"])) + \
        (f", driver {nv['driver']}" if nv.get("driver") else "")


def hardware_summary(profile):
    """Short text stored in the settings file."""
    parts = [describe_computer(profile)]
    g = describe_graphics(profile)
    if g:
        parts.append(g)
    return "; ".join(parts)


def _disk_line(plan):
    if plan is None:
        return None
    free = fmt_gb(plan["free_gb"]) + " GB free" if plan["free_gb"] is not None else "free space unknown"
    if plan["already_installed"]:
        return f"{free} (everything is already installed and downloaded)"
    parts = []
    if plan["venv_gb"]:
        parts.append(f"program {fmt_gb(plan['venv_gb'])} GB")
    if plan["models_gb"]:
        parts.append(f"models {fmt_gb(plan['models_gb'])} GB")
    detail = ": " + ", ".join(parts) + f", plus {fmt_gb(plan['margin_gb'])} GB spare" if parts else ""
    line = f"{free} (needs about {plan['needs_gb']:.0f} GB{detail})"
    if not plan["same_volume"] and plan["hf_free_gb"] is not None:
        line += f"; models go to {plan['hf_free_gb']:.0f} GB free elsewhere"
    return line


def summary_lines(profile, rep, plan=None):
    """The plain-English summary shown to the user."""
    lines = ["Hardware check", f"  Computer : {describe_computer(profile)}"]
    graphics = describe_graphics(profile)
    if graphics:
        lines.append(f"  Graphics : {graphics}")
    disk = _disk_line(plan)
    if disk:
        lines.append(f"  Disk     : {disk}")
    for text in rep["blocking"]:
        lines.append(f"Problem: {text}")
    if rep["recommended"]:
        opt = rep["options"][rep["recommended"]]
        where = {"cuda": "on the NVIDIA GPU", "mps": "on the Apple GPU", "cpu": "on the processor"}[opt["device"]]
        lines.append(f"Recommended: {opt['key']} - {opt['name']} {where} ({opt['tagline']})")
        lines.append(f"    {opt['summary']}")
        others = [rep["options"][k] for k in rep["order"] if k != rep["recommended"] and rep["options"][k]["offered"]]
        if others:
            lines.append("Also possible:")
            for o in others:
                lines.append(f"  {o['key']} - {o['name']} ({o['tagline']}): {o['summary']}")
        for text in opt["warnings"]:
            lines.append(f"Warning: {text}")
    for text in rep["warnings"]:
        lines.append(f"Warning: {text}")
    for text in rep["notes"]:
        lines.append(f"Note: {text}")
    return lines


# ----------------------------------------------------------------------------- settings file

def settings_path(root=None):
    return os.path.join(root or ROOT, config.SETTINGS_FILE)


def read_settings(path=None):
    """(settings, problem): settings is a validated dict or None; problem is None when the file is simply missing."""
    path = path or settings_path()
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        return None, f"{os.path.basename(path)} could not be read ({type(e).__name__}: {e})"
    if not isinstance(data, dict):
        return None, f"{os.path.basename(path)} does not hold a settings object"
    batch = data.get("batch_size")
    if data.get("preset") not in PRESETS or data.get("device") not in DEVICES \
            or not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
        return None, f"{os.path.basename(path)} has missing or invalid values (preset, device or batch_size)"
    out = {"preset": data["preset"], "device": data["device"], "batch_size": batch, "model": data.get("model"),
           "download": data.get("download") if data.get("download") in ("best", "light", "both") else data["preset"]}
    gpu_index, gpu_count = data.get("gpu_index"), data.get("gpu_count")
    out["gpu_index"] = gpu_index if isinstance(gpu_index, int) and not isinstance(gpu_index, bool) and gpu_index >= 0 else None
    out["gpu_count"] = gpu_count if isinstance(gpu_count, int) and not isinstance(gpu_count, bool) and gpu_count >= 1 else 1
    return out, None


def load_settings(path=None):
    return read_settings(path)[0]


def settings_dict(profile, rep, key, download, chosen_by):
    opt = rep["options"][key]
    return {
        "version": SETTINGS_VERSION,
        "preset": opt["preset"],
        "device": opt["device"],
        "model": opt["model"],
        "batch_size": opt["batch_size"],
        "download": download,
        "gpu_index": rep["gpu_index"] if opt["device"] == "cuda" else None,
        "gpu_count": rep["gpu_count"] if opt["device"] == "cuda" else 0,
        "recommended": rep["recommended"],
        "chosen_by": chosen_by,
        "hardware": hardware_summary(profile),
        "date": datetime.date.today().isoformat(),
    }


def save_settings(settings, path=None):
    """Write the settings file (atomically). Returns None on success or a text explaining the failure."""
    path = path or settings_path()
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(settings, f, indent=2, ensure_ascii=True)
            f.write("\n")
        os.replace(tmp, path)
        return None
    except OSError as e:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return f"{type(e).__name__}: {e}"


# ----------------------------------------------------------------------------- the check itself

def _say(text=""):
    """Print one message, wrapped for a console; never fails on characters the console cannot show."""
    if text.strip():
        label = re.match(r"\s+\w+\s*: ", text)  # "  Computer : ..." continues under the text after the label
        lead = len(text) - len(text.lstrip())
        indent = " " * (len(label.group()) if label else lead + (2 if lead else 4))
        text = textwrap.fill(text, width=110, subsequent_indent=indent, break_long_words=False, break_on_hyphens=False)
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"), flush=True)


def stdin_is_terminal():
    """True when a person can answer a question. (On Windows os.isatty() also says True for NUL, so ask the console.)"""
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return False
        if os.name == "nt":
            import ctypes

            mode = ctypes.c_ulong()
            handle = ctypes.windll.kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
            return bool(ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)))
        return True
    except Exception:
        return False


def _ask_choice(rep, default, ask, say):
    others = [k for k in rep["order"] if k != default and rep["options"][k]["offered"]]
    hint = ", or type " + " / ".join(f'"{k}"' for k in others) if others else ""
    prompt = f'Press Enter to use "{default}"{hint}: '
    for _ in range(3):
        try:
            answer = ask(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return default
        if not answer:
            return default
        if answer in rep["options"]:
            return answer
        say(f'Please press Enter or type one of: {", ".join(k for k in rep["order"] if k in rep["options"])}.')
    return default


def parse_words(words):
    """(preset word, force_cpu) from the positional arguments; raises ValueError for anything else."""
    presets, cpu = [], False
    for w in words:
        w = w.lower()
        if w == "cpu":
            cpu = True
        elif w in ("auto", "best", "light", "both"):
            presets.append(w)
        else:
            raise ValueError(f"unknown choice {w!r}; use auto, best, light, both or cpu")
    if len(presets) > 1:
        raise ValueError("give only one of auto, best, light, both")
    return (presets[0] if presets else "auto"), cpu


def run(words=(), yes=False, as_json=False, save=True, interactive=None, ask=None, profile=None, root=None,
        settings_file=None, check_disk=True, out=None):
    """Detect, recommend, let the user choose, save. Returns (exit code, settings dict or None)."""
    say = out or _say
    ask = ask or input
    wanted, force_cpu = parse_words(words)
    explicit = wanted != "auto" or force_cpu
    if interactive is None:
        interactive = stdin_is_terminal()
    prompting = interactive and not yes and not as_json and not explicit

    profile = profile or detect(root)
    rep = assess(profile, force_cpu)
    rec = rep["recommended"]
    explicit_preset = bool(rec) and wanted in PRESETS
    key = wanted if explicit_preset else rec
    chosen_by = "setup argument" if explicit_preset else "recommended"

    def plan_for(k, both):
        if not (check_disk and k):
            return None
        return disk_plan(profile, rep["options"][k], list(PRESETS) if both else [k], root)

    if not as_json:
        for line in summary_lines(profile, rep, plan_for(rec, wanted == "both")):
            say(line)
    if key and prompting:
        key = _ask_choice(rep, key, ask, say)
        chosen_by = "recommended" if key == rec else "user"
    if key and not as_json:
        if key != rec:
            say(f'Using "{key}" as requested.' if chosen_by == "setup argument" else f'Using "{key}" (your choice).')
            for text in rep["options"][key]["warnings"]:
                say(f"Warning: {text}")
        elif not prompting and not explicit:
            say(f'Using the recommended choice: "{key}".')
    download = "both" if (key and wanted == "both") else key
    plan = plan_for(key, wanted == "both")
    problems = list(rep["blocking"])
    if plan and plan["problem"]:
        problems.append(plan["problem"])
        if not as_json:
            say(f"Problem: {plan['problem']}")
            for k in (k for k in PRESETS if k != key and k in rep["options"]):
                alt = plan_for(k, False)
                if alt and not alt["problem"] and alt["needs_gb"] < plan["needs_gb"]:
                    say(f'  "{k}" needs only about {alt["needs_gb"]:.0f} GB: run this again and choose {k}.')
    settings = settings_dict(profile, rep, key, download, chosen_by) if key and not problems else None
    saved = False
    if settings and save and (not as_json or yes):
        problem = save_settings(settings, settings_file)
        saved = problem is None
        if not as_json:
            if saved:
                dev = {"cuda": "NVIDIA GPU", "mps": "Apple GPU", "cpu": "processor"}[settings["device"]]
                say(f'Saved: {settings["preset"]} on the {dev}, batch size {settings["batch_size"]} '
                    f"({config.SETTINGS_FILE}).")
            else:
                say(f"Warning: could not save the choice ({problem}); the automatic choice is used next time.")
    if as_json:
        result = {"hardware": profile, "assessment": rep, "selected": key, "download": download, "disk": plan,
                  "problems": problems, "settings": settings, "saved": saved,
                  "settings_file": (settings_file or settings_path(root)) if saved else None}
        (out or print)(json.dumps(result, indent=2, default=str))  # raw: the console wrapping of `say` would break it
    return (1 if problems else 0), settings


def first_run_check(root=None, settings_file=None, out=None, profile=None):
    """For the command line tool when no settings exist (setup skipped, or the file is damaged): check once,
    without asking, print the recommendation, save it. Prefers a model that is already downloaded.
    Returns the settings dict, or None when nothing could be recommended (the caller carries on automatically)."""
    say = out or _say
    profile = profile or detect(root)
    rep = assess(profile)
    key = rec = rep["recommended"]
    if not rec:
        return None
    for line in summary_lines(profile, rep):
        say(line)
    hub = hf_hub_dir()
    if not model_cached(rep["options"][rec]["model"], hub):
        for k in rep["order"]:
            if k != rec and model_cached(rep["options"][k]["model"], hub):
                say(f'Using "{k}": it is already downloaded ("{rec}" is recommended for this computer; '
                    f"run the setup script again to download it).")
                return settings_dict(profile, rep, k, k, "downloaded model")  # not saved: the next run checks again
    settings = settings_dict(profile, rep, key, key, "recommended")
    problem = save_settings(settings, settings_file or settings_path(root))
    if problem:
        say(f"Warning: could not save the choice ({problem}).")
    else:
        say(f'Saved this choice ({config.SETTINGS_FILE}); run check_hardware to change it.')
    return settings


def main(argv=None):
    ap = argparse.ArgumentParser(prog="check_hardware", description="Check this computer and choose the speech model.",
                                 epilog="Choices: auto (ask; default), best, light, both, cpu. Explicit choices are not asked.")
    ap.add_argument("choice", nargs="*", metavar="choice", help="auto, best, light, both and/or cpu")
    ap.add_argument("--yes", "-y", action="store_true", help="accept the recommendation without asking")
    ap.add_argument("--json", action="store_true", help="print the result as JSON (saves only together with --yes)")
    ap.add_argument("--no-save", action="store_true", help="do not write the settings file")
    ap.add_argument("--print", dest="print_key", choices=("preset", "device", "models", "requirements", "batch"),
                    help="print one value of the saved choice and exit (1 when nothing is saved)")
    args = ap.parse_args(argv)
    if args.print_key:
        settings, _problem = read_settings()
        if not settings:
            return 1
        value = {"preset": settings["preset"], "device": settings["device"], "models": settings["download"],
                 "requirements": requirements_for(settings["device"]), "batch": settings["batch_size"]}[args.print_key]
        print(value)
        return 0
    try:
        code, _settings = run(args.choice, yes=args.yes, as_json=args.json, save=not args.no_save)
    except ValueError as e:
        ap.error(str(e))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return code


if __name__ == "__main__":
    sys.exit(main())
