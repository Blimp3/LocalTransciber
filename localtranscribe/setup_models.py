"""Download the models once, so that every later run works fully offline.

    python -m localtranscribe.setup_models [--model auto|best|light|both|<repo id> ...] [--no-diarization]
"""
import argparse
import os
import sys

from . import config, devices
from .devices import SetupError


def resolve_repos(names, device, mem_gb):
    repos = []
    for name in names:
        key = name.strip().lower()
        if key == "both":
            wanted = [devices.choose_model("best", device, mem_gb), devices.choose_model("light", device, mem_gb)]
        else:
            wanted = [devices.choose_model(name, device, mem_gb)]
        for r in wanted:
            if r not in repos:
                repos.append(r)
    return repos


def repo_size_gb(repo):
    """Download size from Hugging Face (None if it cannot be queried)."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, files_metadata=True)
        total = sum((s.size or 0) for s in (info.siblings or []))
        return total / 1e9 if total else None
    except Exception:
        return None


def is_cached(repo):
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo, local_files_only=True)
        return True
    except Exception:
        return False


def describe_downloads(repos):
    """Print what is about to be downloaded (models already on this computer are marked); returns the GB still to fetch."""
    todo = 0.0
    print("Models needed (each is downloaded once from Hugging Face):")
    for r in repos:
        size = repo_size_gb(r)
        approx = size is None
        size = size if size is not None else config.APPROX_DOWNLOAD_GB.get(r, 0.0)
        cached = is_cached(r)
        if not cached:
            todo += size
        note = "already on this computer" if cached else "to download"
        print(f"  {r:<40} {'~' if approx else ''}{size:5.2f} GB  ({note})")
    print(f"  {'still to download':<40} {'~' if todo else ' '}{todo:5.2f} GB", flush=True)
    return todo


def download_repo(repo):
    if devices.is_apple_silicon() and repo.startswith("mlx-community/"):
        from mlx_audio.utils import get_model_path  # same file patterns mlx-audio uses when loading

        return get_model_path(repo)
    from huggingface_hub import snapshot_download

    return snapshot_download(repo)


def verify_offline(repo):
    """Confirm the files are in the local cache (what a run with HF_HUB_OFFLINE=1 will need)."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo, local_files_only=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download the speech models for offline use.")
    ap.add_argument("--model", nargs="+", default=["auto"],
                    help="auto (default), best, light, both, or Hugging Face repo ids")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--no-diarization", action="store_true", help="skip the speaker model used by --speakers")
    ap.add_argument("--dry-run", action="store_true", help="only list what would be downloaded")
    args = ap.parse_args(argv)

    os.environ.pop("HF_HUB_OFFLINE", None)  # downloading needs the network
    try:
        devices.check_platform()
        device = devices.resolve_device(args.device)
        mem_gb = devices.accelerator_memory_gb(device)
        repos = resolve_repos(args.model, device, mem_gb)
        if not args.no_diarization:
            repos.append(config.SPEAKER_MODEL)
        label = {"cuda": "NVIDIA GPU", "mps": "Apple Silicon", "cpu": "CPU"}[device]
        mem = f", {mem_gb:.0f} GB {'VRAM' if device == 'cuda' else 'memory'}" if mem_gb else ""
        print(f"Detected: {label}{mem}")
        describe_downloads(repos)
        if args.dry_run:
            return 0
    except SetupError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    failed = []
    for repo in repos:
        print(f"\nDownloading {repo} ...", flush=True)
        try:
            path = download_repo(repo)
            verify_offline(repo)
            print(f"  OK -> {path}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(repo)
    if failed:
        print("\nSome downloads failed: " + ", ".join(failed) + "\nCheck the internet connection and run the setup again.",
              file=sys.stderr)
        return 1
    print("\nAll models are downloaded. Transcription now works without internet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
