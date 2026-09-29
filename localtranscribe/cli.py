"""Command-line interface: transcribe audio/video files to .md (Markdown), locally.

    transcribe.bat recording.m4a [video.mp4 ...] [--context "names, jargon"] [--speakers 2]   (Windows)
    ./transcribe.sh recording.m4a [...]                                                        (Mac)
"""
import argparse
import json
import os
import sys
import time

from . import config, devices, precheck
from .devices import SetupError


def build_parser():
    ap = argparse.ArgumentParser(
        prog="transcribe",
        description="Local speech-to-text with Qwen3-ASR (NVIDIA GPU, Apple Silicon or CPU).",
    )
    ap.add_argument("files", nargs="+", help="audio or video files")
    ap.add_argument("--language", default=config.DEFAULT_LANGUAGE,
                    help='spoken language (default Italian; "auto" to detect)')
    ap.add_argument("--context", default="", help="optional names/terms that appear in the audio, to improve spelling")
    ap.add_argument("--chunk", type=float, default=config.CHUNK_SECONDS,
                    help=f"max seconds per piece (default {config.CHUNK_SECONDS}; longer hurts accuracy)")
    ap.add_argument("--out-dir", help="write transcripts here instead of next to the input files")
    ap.add_argument("--speakers", type=int, default=0,
                    help="split the transcript by speaker (e.g. 2 for a phone call); works best on calls longer than a few minutes")
    ap.add_argument("--model", default=None,
                    help="best, light, auto or a Hugging Face repo id (default: the choice saved by the hardware check, "
                         "see check_hardware; without one, auto = the best model the memory allows, light on CPU)")
    ap.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"],
                    help="auto picks the NVIDIA GPU, the Apple GPU, or the CPU (default: the saved hardware choice, else auto)")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="pieces decoded together (default: the saved hardware choice, lowered if the free memory is "
                         "tight; else chosen from the available memory)")
    ap.add_argument("--confidence", action="store_true",
                    help="also save each word-piece's confidence and the model's alternatives to <recording>.review.json "
                         "(for the review tool; Apple Silicon only for now)")
    ap.add_argument("--stats", action="store_true", help="print device, model and peak memory at the end")
    return ap


def _load_backend(repo, device, batch):
    from .backends import create_backend

    try:
        return create_backend(repo, device, batch)
    except SetupError:
        raise
    except Exception as e:
        raise SetupError(_explain_load_error(e, repo)) from e


def _explain_load_error(e, repo):
    text = f"{type(e).__name__}: {e}"
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    missing = any(s in text for s in ("LocalEntryNotFound", "offline mode", "local_files_only", "not find the requested files",
                                      "couldn't connect", "Cannot find an appropriate cached snapshot"))
    if offline and missing:
        return (f"The model {repo} has not been downloaded yet.\n"
                "Run the setup script again (setup_windows.bat / setup_mac.sh) while connected to the internet,\n"
                "or download it once with:  python -m localtranscribe.setup_models --model " + repo)
    if "out of memory" in text.lower():
        return (f"Not enough memory to load {repo}. Try --model light, --batch-size 1, or close other programs.\n" + text)
    return f"Could not load {repo}: {text}"


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except SetupError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


def _saved_choice(args):
    """The choice remembered by the hardware check (setup or check_hardware), or None.

    When there is no usable settings file (someone skipped the setup, or the file is damaged) the check runs once
    now, without asking, prints its recommendation and saves it; the run then continues with it."""
    if args.model and args.device and args.batch_size:
        return None  # everything was given on the command line: nothing to look up
    path = precheck.settings_path()
    saved, problem = precheck.read_settings(path)
    if saved:
        return saved
    if problem:
        print(f"Note: {problem}; checking this computer again.")
    else:
        print("No saved hardware check yet (setup and check_hardware make one); checking this computer now.")
    try:
        return precheck.first_run_check(settings_file=path)
    except Exception as e:  # the check must never stop a transcription
        print(f"Note: the hardware check failed ({type(e).__name__}: {e}); using the automatic choice.")
        return None


def _pin_gpu(saved):
    """With several NVIDIA GPUs use the one the check chose (nvidia-smi order = PCI bus order). Must run before torch
    is imported, and never overrides a CUDA_VISIBLE_DEVICES set by the user."""
    if (saved and saved["device"] == "cuda" and saved.get("gpu_count", 1) > 1 and saved.get("gpu_index") is not None
            and "CUDA_VISIBLE_DEVICES" not in os.environ):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(saved["gpu_index"])


def run(args):
    saved = _saved_choice(args)
    _pin_gpu(saved)
    notes = []
    device, repo, batch, _mem_gb = devices.resolve_run_config(args.device, args.model, args.batch_size,
                                                              saved=saved, notes=notes)
    for note in notes:
        print(f"Note: {note}")
    language = None if args.language.lower() == "auto" else args.language
    if args.chunk > 60:
        print(f"Warning: --chunk {args.chunk:g} is much longer than the tested {config.CHUNK_SECONDS} s; expect more errors.")

    from .audio import load_audio
    from .pipeline import transcribe_wav

    label = {"cuda": "NVIDIA GPU", "mps": "Apple GPU", "cpu": "CPU"}[device]
    backend = None

    def get_backend():
        nonlocal backend
        if backend is None:
            print(f"Loading {repo} on the {label} (batch {batch})...", flush=True)
            t0 = time.perf_counter()
            backend = _load_backend(repo, device, batch)
            if args.confidence:
                if backend.name == "mlx" and not args.speakers:
                    backend.record_confidence = True
                else:
                    print("  [note] --confidence is not supported with --speakers or on this backend yet; "
                          "continuing without it.")
            for note in getattr(backend, "notes", []):
                print(f"  [note] {note}")
            print(f"  loaded in {time.perf_counter() - t0:.0f}s ({backend.describe()})", flush=True)
        return backend

    if not args.speakers:
        get_backend()  # fail early if the model is missing, before touching any file

    failed = 0
    audio_total = 0.0
    time_total = 0.0
    for path in args.files:
        name = os.path.basename(path)
        try:
            wav = load_audio(path)
        except Exception as e:
            print(f"\n[skip] {name}: could not read audio ({e})")
            failed += 1
            continue
        duration = len(wav) / config.SAMPLE_RATE
        print(f"\n{name}: {duration / 60:.1f} min of audio, transcribing...", flush=True)

        start = time.perf_counter()
        paragraphs = [] if getattr(backend, "record_confidence", False) else None
        if args.speakers:
            from .diarize import diarize, first_appearance_names, fmt_time, transcribe_turns

            runs = diarize(wav, n_speakers=args.speakers, device=devices.diarization_device(device))
            b = get_backend()  # loaded after the speaker model has been released (matters on 8 GB Macs)
            turns = transcribe_turns(b, wav, runs, language, context=args.context, progress=_progress())
            names = first_appearance_names(turns)
            text = "\n\n".join(f"[{fmt_time(a)}] {names[s]}: {t}" for a, s, t in turns if t)
        else:
            text = transcribe_wav(get_backend(), wav, language, args.context, args.chunk, _progress(), paragraphs)
        elapsed = time.perf_counter() - start
        audio_total += duration
        time_total += elapsed

        out_dir = args.out_dir or os.path.dirname(os.path.abspath(path))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        if paragraphs is not None:
            side = os.path.splitext(out_path)[0] + ".review.json"
            with open(side, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "model": repo, "top_k": config.CONFIDENCE_TOP_K, "paragraphs": paragraphs},
                          f, ensure_ascii=False)
            print(f"  confidence -> {side}")
        print(f"  done in {elapsed:.0f}s ({duration / max(elapsed, 1e-6):.0f}x realtime) -> {out_path}")

    if args.stats and backend is not None:
        peak = backend.peak_memory_gb()
        rss = devices.peak_rss_gb()
        print("\nStats: " + backend.describe())
        if audio_total:
            print(f"  audio {audio_total / 60:.1f} min in {time_total:.0f}s = {audio_total / max(time_total, 1e-6):.1f}x realtime")
        if peak is not None:
            print(f"  peak accelerator memory {peak:.2f} GB")
        if rss is not None:
            print(f"  peak process memory {rss:.2f} GB")
    return 1 if failed else 0


def _progress():
    state = {"last": -1}

    def report(done, total):
        pct = int(100 * done / max(total, 1))
        if total >= 8 and pct // 25 > state["last"] // 25 and done < total:
            print(f"  ... {pct}% ({done}/{total} pieces)", flush=True)
        state["last"] = pct

    return report


if __name__ == "__main__":
    sys.exit(main())
