"""Accuracy and speed benchmark on a folder of clips with references.csv (FLEURS format).

    python benchmark/bench.py --data benchmark/data/fleurs_it [--model best|light|<repo>] [--device auto]
                              [--n 100] [--batch-size 0] [--chunk 20] [--json out.json] [--hyp-csv hyp.csv]

Uses the same pipeline as the command line tool: PyAV decoding, the shared 20-second chunker, the
selected backend. All pieces of all clips go through the backend together (like a long recording
would), so batching behaves as in real use. Prints one RESULT line.
"""
import argparse
import csv
import json
import os
import platform
import sys
import time

import wer as werlib  # also puts the repo root on sys.path

from localtranscribe import config, devices
from localtranscribe.audio import load_audio
from localtranscribe.pipeline import split_wav, transcribe_pieces


def run(args):
    device, repo, batch, mem_gb = devices.resolve_run_config(args.device, args.model, args.batch_size)
    rows = werlib.load_references(args.data, args.n)
    t0 = time.perf_counter()
    wavs = [load_audio(os.path.join(args.data, r["file"])) for r in rows]
    decode_s = time.perf_counter() - t0
    audio_s = sum(len(w) for w in wavs) / config.SAMPLE_RATE

    from localtranscribe.backends import create_backend

    t0 = time.perf_counter()
    backend = create_backend(repo, device, batch)
    load_s = time.perf_counter() - t0

    language = None if args.language.lower() == "auto" else args.language
    pieces, owner = [], []
    for k, w in enumerate(wavs):
        for p, _ in split_wav(w, args.chunk):
            pieces.append(p)
            owner.append(k)

    transcribe_pieces(backend, pieces[:1], language)  # warm-up (kernel compilation, caches); not timed
    backend.reset_peak_memory()
    t0 = time.perf_counter()
    texts = transcribe_pieces(backend, pieces, language)
    run_s = time.perf_counter() - t0

    hyps = [""] * len(rows)
    for k, t in zip(owner, texts):
        hyps[k] = (hyps[k] + " " + t).strip()
    refs = [r["reference"] for r in rows]
    w_err = werlib.wer(refs, hyps)
    c_err = werlib.cer(refs, hyps)
    peak = backend.peak_memory_gb()
    res = {
        "backend": backend.name, "model": repo, "device": backend.device, "dtype": backend.dtype,
        "batch_size": backend.batch_size, "clips": len(rows), "pieces": len(pieces),
        "audio_seconds": round(audio_s, 1), "seconds": round(run_s, 2),
        "xrealtime": round(audio_s / run_s, 2), "rtf": round(run_s / audio_s, 4),
        "wer_percent": round(100 * w_err, 3), "cer_percent": round(100 * c_err, 3),
        "load_seconds": round(load_s, 2), "decode_seconds": round(decode_s, 2),
        "peak_accelerator_gb": None if peak is None else round(peak, 3),
        "peak_process_gb": None if devices.peak_rss_gb() is None else round(devices.peak_rss_gb(), 3),
        "system": f"{platform.system()} {platform.machine()}, python {platform.python_version()}",
    }
    print(f"RESULT | {repo} | {backend.device} {backend.dtype} batch {backend.batch_size} | WER {res['wer_percent']:.2f}% "
          f"| CER {res['cer_percent']:.2f}% | {len(rows)} clips, {audio_s / 60:.1f} min audio in {run_s:.1f}s "
          f"({res['xrealtime']:.1f}x realtime, RTF {res['rtf']:.3f}) | load {load_s:.1f}s | "
          f"peak accelerator {'n/a' if peak is None else f'{peak:.2f} GB'}"
          f" | peak process {res['peak_process_gb']} GB", flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    if args.hyp_csv:
        with open(args.hyp_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "reference", "hypothesis"])
            for r, h in zip(rows, hyps):
                w.writerow([r["file"], r["reference"], h])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with clips and references.csv (see download_fleurs.py)")
    ap.add_argument("--model", default="auto", help="best, light, auto or a Hugging Face repo id")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--n", type=int, default=0, help="use only the first N clips (default: all)")
    ap.add_argument("--chunk", type=float, default=config.CHUNK_SECONDS)
    ap.add_argument("--language", default=config.DEFAULT_LANGUAGE)
    ap.add_argument("--json", help="write the result as JSON to this file")
    ap.add_argument("--hyp-csv", help="write hypotheses next to references to this CSV")
    args = ap.parse_args(argv)
    try:
        run(args)
    except devices.SetupError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
