"""Command-line interface: transcribe audio/video files to .md (Markdown), locally.

    transcribe.bat recording.m4a [video.mp4 ...] [--context "names, jargon"] [--speakers 2]   (Windows)
    ./transcribe.sh recording.m4a [...]                                                        (Mac)
"""
import argparse
import json
import math
import os
import sys
import tempfile
import time

from . import config, devices, precheck
from .devices import SetupError
from .textutil import backup_name, fmt_time, sync, sync_copy

SAVE_RETRY_SECONDS = 0.5  # between the 3 attempts to put the finished transcript in place (Windows: file in use)


def _chunk_seconds(text):
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid number: {text!r}")
    if not (math.isfinite(v) and 5 <= v <= 120):
        raise argparse.ArgumentTypeError(f"--chunk must be a number from 5 to 120 seconds, got {text}")
    return v


def _out_path(path, out_dir):
    out_dir = out_dir or os.path.dirname(os.path.abspath(path))
    return os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + ".md")


def _check_distinct_outputs(files, out_dir):
    """Two inputs that would write the same .md (a.wav + a.mp3) must not silently overwrite each other."""
    seen = {}
    for f in files:
        out = os.path.abspath(_out_path(f, out_dir)).casefold()  # macOS and Windows disks ignore case
        if out in seen:
            raise SetupError(f"{seen[out]} and {f} would both write {out}. Rename one of them or run them separately.")
        seen[out] = f


def _check_inputs_and_folders(files, out_dir):
    """Before the model loads: every input exists and every output folder can really be written."""
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        raise SetupError("File not found: " + ", ".join(missing))
    for folder in dict.fromkeys(os.path.dirname(_out_path(f, out_dir)) for f in files):
        try:
            os.makedirs(folder, exist_ok=True)
            with tempfile.TemporaryFile(dir=folder):  # deleted by the system on close: no remove to fail on Windows
                pass
        except OSError as e:
            raise SetupError(f"Cannot write in the folder {folder} ({e}). Use --out-dir with a folder that can be "
                             "written, or copy the recording to such a folder.") from e


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _write_tmp(path, text):
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            sync(f)
    except BaseException:
        _remove(tmp)
        raise
    return tmp


def _write_atomic(path, text):
    tmp = _write_tmp(path, text)
    try:
        os.replace(tmp, path)
    except BaseException:
        _remove(tmp)
        raise


def _write_new(path, text):
    """Write a file that does not exist yet, directly: no rename for another program to block."""
    f = open(path, "x", encoding="utf-8", newline="\n")
    try:
        with f:
            f.write(text)
            sync(f)
    except BaseException:
        _remove(path)
        raise


def _differs(path, text):
    """True if `path` exists and holds something other than `text`."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read() != text
    except FileNotFoundError:
        return False
    except UnicodeDecodeError:
        return True


def _put_in_place(tmp, out_path, text):
    """`tmp` holds the new text: copy a different previous transcript to its backup name, then put `tmp` in place
    (3 attempts). out_path is never moved away, so it always holds the old or the new text. Returns the backup path
    or None; if every attempt fails the backup copy is removed again."""
    bak = None
    for attempt in range(3):
        try:
            if bak is None and _differs(out_path, text):
                b = backup_name(out_path)
                try:
                    sync_copy(out_path, b)
                except BaseException:
                    _remove(b)
                    raise
                bak = b
            os.replace(tmp, out_path)
            return bak
        except OSError:
            if attempt == 2:
                if bak:
                    _remove(bak)  # out_path still holds the previous transcript
                raise
            time.sleep(SAVE_RETRY_SECONDS)


def _save_transcript(out_path, text):
    """Save `text` as out_path; the previous transcript keeps its name until the new one replaces it. If out_path
    cannot be replaced, save a free <stem>.new-<time>.md next to it, else in the home folder. Returns the path
    written; raises if every place failed."""
    try:
        tmp = _write_tmp(out_path, text)
        try:
            bak = _put_in_place(tmp, out_path, text)
        except BaseException:
            _remove(tmp)
            raise
    except OSError as e:
        err = why = e
    else:
        if bak:
            print(f"  previous transcript kept as {bak}")
        return out_path
    for folder in (os.path.dirname(out_path), os.path.expanduser("~")):
        alt = backup_name(os.path.join(folder, os.path.basename(out_path)), "new")
        try:
            _write_new(alt, text)
        except OSError as e:
            why = e
            continue
        print(f"  [note] could not replace {out_path} ({err}); is it open in another program? "
              f"The transcript was saved as {alt} instead.")
        return alt
    raise why


def _partial_writer(out_path):
    """(path, write): write(text) keeps the text so far in <stem>.partial.md. A failure prints one note; the next
    write tries again (on Windows another program can hold the file for a moment)."""
    path = os.path.splitext(out_path)[0] + ".partial.md"
    state = {"noted": False}

    def write(text):
        if text.strip():
            try:
                _write_atomic(path, text + "\n")
            except Exception as e:  # a partial file is a courtesy, never a reason to stop
                if not state["noted"]:
                    state["noted"] = True
                    print(f"  [note] cannot keep the part transcribed so far in {path} ({e}); "
                          "trying again with the next pieces.")

    return path, write


def _say_partial(path):
    if os.path.exists(path):
        print(f"  the part transcribed so far is in {path}")


def _format_turns(turns):
    from .diarize import first_appearance_names

    names = first_appearance_names(turns)
    return "\n\n".join(f"[{fmt_time(a)}] {names[s]}: {t}" for a, s, t in turns if t)


def build_parser():
    ap = argparse.ArgumentParser(
        prog="transcribe",
        description="Local speech-to-text with Qwen3-ASR (NVIDIA GPU, Apple Silicon or CPU).",
    )
    ap.add_argument("files", nargs="+", help="audio or video files")
    ap.add_argument("--language", default=config.DEFAULT_LANGUAGE,
                    help='spoken language (default Italian; "auto" to detect)')
    ap.add_argument("--context", default="", help="optional names/terms that appear in the audio, to improve spelling")
    ap.add_argument("--chunk", type=_chunk_seconds, default=config.CHUNK_SECONDS,
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
    ap.add_argument("--correct", action="store_true",
                    help="suggest fixes for unsure words with a small local model, checked by the Bend rule checker "
                         "(implies --confidence; the suggestions go to <recording>.review.json, the .md is unchanged)")
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
    for stream in (sys.stdout, sys.stderr):  # a console that cannot show a character must not fail a print mid-save
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
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
    _check_distinct_outputs(args.files, args.out_dir)
    _check_inputs_and_folders(args.files, args.out_dir)
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
            if args.confidence or args.correct:
                if backend.name == "mlx" and not args.speakers:
                    backend.record_confidence = True
                else:
                    print("  [note] --confidence and --correct are not supported with --speakers or on this backend "
                          "yet; continuing without them.")
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
        out_path = _out_path(path, args.out_dir)
        partial_path, write_partial = _partial_writer(out_path)
        try:
            duration = len(wav) / config.SAMPLE_RATE
            print(f"\n{name}: {duration / 60:.1f} min of audio, transcribing...", flush=True)

            start = time.perf_counter()
            paragraphs = [] if getattr(backend, "record_confidence", False) else None
            corrector = None
            if args.correct and paragraphs is not None:
                from .correct import Corrector

                corrector = Corrector()
            if args.speakers:
                from .diarize import diarize, transcribe_turns

                runs = diarize(wav, n_speakers=args.speakers, device=devices.diarization_device(device))
                b = get_backend()  # first file: loaded after the speaker model is released (8 GB Macs)
                turns = transcribe_turns(b, wav, runs, language, context=args.context, progress=_progress(),
                                         partial=lambda ts: write_partial(_format_turns(ts)))
                text = _format_turns(turns)
            else:
                text = transcribe_wav(get_backend(), wav, language, args.context, args.chunk, _progress(), paragraphs,
                                      corrector, partial=write_partial)
            elapsed = time.perf_counter() - start
            if not text.strip():
                print(f"\n[warn] {name}: no speech was recognised (silent recording?); nothing was written")
                failed += 1
                continue
            audio_total += duration
            time_total += elapsed

            written = _save_transcript(out_path, text + "\n")
            _remove(partial_path)
            problem = written != out_path
            side = os.path.splitext(written)[0] + ".review.json"
            if paragraphs is not None:
                try:
                    _write_atomic(side, json.dumps({"version": 1, "model": repo, "top_k": config.CONFIDENCE_TOP_K,
                                                    "paragraphs": paragraphs}, ensure_ascii=False))
                    print(f"  confidence -> {side}")
                except Exception as e:  # the transcript is saved; only its review data is missing
                    _remove(side)  # an older one would not match the new transcript
                    print(f"  [note] could not write {side} ({e}); the transcript is saved without it.")
                    problem = True
            else:
                _remove(side)  # a sidecar of an older run would not match this transcript
            failed += problem
            rt = duration / max(elapsed, 1e-6)
            print(f"  done in {elapsed:.0f}s ({rt:.{0 if rt >= 10 else 1}f}x realtime) -> {written}")
        except (SetupError, KeyboardInterrupt):
            _say_partial(partial_path)
            raise
        except Exception as e:
            print(f"\n[fail] {name}: {type(e).__name__}: {e}")
            _say_partial(partial_path)
            failed += 1
            continue

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
    if failed:
        print(f"\n{failed} of {len(args.files)} files had problems (see the notes above)")
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
