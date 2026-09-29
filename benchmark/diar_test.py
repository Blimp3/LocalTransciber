"""End-to-end check of --speakers on a built two-speaker recording.

    python benchmark/diar_test.py --data benchmark/data/fleurs_it [--turns 6] [--model auto] [--device auto]

Builds a recording that alternates female and male FLEURS clips (different speakers for sure, gender
comes from references.csv), separated by short silences, then runs diarization and the transcription of
each turn through the normal backend. Reports how many speakers were found, how many of the clips were
attributed consistently, the WER of the joined transcript, the torch device used for the speaker model
and the peak memory. Exit code 0 when both speakers are separated correctly.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import wer as werlib
from localtranscribe import config, devices
from localtranscribe.audio import load_audio

SR = config.SAMPLE_RATE


def build(rows, data_dir, turns):
    fem = [r for r in rows if r.get("gender", "").upper() == "FEMALE"][: (turns + 1) // 2]
    mal = [r for r in rows if r.get("gender", "").upper() == "MALE"][: (turns + 1) // 2]
    order = []
    for f, m in zip(fem, mal):
        order += [f, m]
    order = order[:turns]
    silence = np.zeros(int(0.5 * SR), dtype=np.float32)
    parts, spans, t = [], [], 0
    for i, r in enumerate(order):
        w = load_audio(os.path.join(data_dir, r["file"]))
        parts += [w, silence]
        spans.append((t / SR, (t + len(w)) / SR, i % 2))  # expected speaker: 0 = female, 1 = male
        t += len(w) + len(silence)
    return np.concatenate(parts), spans, [r["reference"] for r in order]


def attribute(runs, spans):
    """For each clip, the diarized speaker that covers most of it."""
    result = []
    for a, b, _ in spans:
        cover = {}
        for ra, rb, spk in runs:
            ov = min(b, rb) - max(a, ra)
            if ov > 0:
                cover[spk] = cover.get(spk, 0) + ov
        result.append(max(cover, key=cover.get) if cover else -1)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder from download_fleurs.py (needs the gender column)")
    ap.add_argument("--turns", type=int, default=6, help="number of alternating clips (default 6, ~90 s)")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--json", help="write the result as JSON")
    args = ap.parse_args(argv)

    device, repo, batch, _ = devices.resolve_run_config(args.device, args.model, 0)
    rows = werlib.load_references(args.data)
    if not any(r.get("gender") for r in rows):
        print("references.csv has no gender column; download the clips with download_fleurs.py", file=sys.stderr)
        return 2
    wav, spans, refs = build(rows, args.data, args.turns)

    from localtranscribe.diarize import diarize, first_appearance_names, transcribe_turns

    ddev = devices.diarization_device(device)
    t0 = time.perf_counter()
    runs = diarize(wav, n_speakers=2, device=ddev)
    diar_s = time.perf_counter() - t0
    rss_after_diar = devices.peak_rss_gb()

    from localtranscribe.backends import create_backend

    backend = create_backend(repo, device, batch)
    t0 = time.perf_counter()
    turns = transcribe_turns(backend, wav, runs, config.DEFAULT_LANGUAGE)
    asr_s = time.perf_counter() - t0
    names = first_appearance_names(turns)

    who = attribute(runs, spans)
    expected = [s for _, _, s in spans]
    same = sum(1 for w, e in zip(who, expected) if w == e)
    flipped = sum(1 for w, e in zip(who, expected) if w == 1 - e)
    correct = max(same, flipped) if -1 not in who else 0
    text = " ".join(t for _, _, t in turns)
    result = {
        "speaker_model_device": ddev, "asr_device": backend.device, "model": repo,
        "clips": len(spans), "speakers_found": len({s for _, s, _ in turns}),
        "clips_attributed_consistently": correct, "turns": len(turns),
        "wer_percent": round(100 * werlib.wer(" ".join(refs), text), 2),
        "diarization_seconds": round(diar_s, 1), "transcription_seconds": round(asr_s, 1),
        "audio_seconds": round(len(wav) / SR, 1),
        "peak_process_gb_after_diarization": None if rss_after_diar is None else round(rss_after_diar, 2),
        "peak_process_gb_final": None if devices.peak_rss_gb() is None else round(devices.peak_rss_gb(), 2),
        "peak_accelerator_gb": None if backend.peak_memory_gb() is None else round(backend.peak_memory_gb(), 2),
        "labels": [names.get(s, "?") for _, s, _ in turns],
    }
    ok = result["speakers_found"] == 2 and correct == len(spans)
    print(f"DIARIZATION | speaker model on {ddev}, ASR on {backend.device} | {len(spans)} alternating clips "
          f"({result['audio_seconds']:.0f}s) | speakers found {result['speakers_found']} | "
          f"consistent {correct}/{len(spans)} | turns {len(turns)} | WER {result['wer_percent']:.2f}% | "
          f"diarize {diar_s:.1f}s, transcribe {asr_s:.1f}s | peak process {result['peak_process_gb_final']} GB "
          f"| {'PASS' if ok else 'FAIL'}", flush=True)
    result["pass"] = ok
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
