"""Build one long recording (a 16 kHz mono WAV) by concatenating the benchmark clips in order.

    python benchmark/make_longform.py --data <fleurs folder> --out long_it.wav [--n 100]

Feed the result to the normal command line tool and score it with score_long.py.
"""
import argparse
import os
import sys
import wave

import numpy as np

import wer as werlib
from localtranscribe.audio import load_audio


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="output .wav path")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    wavs = [load_audio(os.path.join(args.data, r["file"])) for r in werlib.load_references(args.data, args.n)]
    pcm = (np.clip(np.concatenate(wavs), -1, 1) * 32767).astype("<i2")
    with wave.open(args.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    print(f"Wrote {args.out}: {len(pcm) / 16000 / 60:.1f} min from {len(wavs)} clips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
