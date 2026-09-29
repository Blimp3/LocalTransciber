"""Score a long-form transcript against the FLEURS references joined in order.

    python benchmark/score_long.py --data <fleurs folder> --hyp long_it.txt [--n 100]

The long recording is the FLEURS clips concatenated in references.csv order (see make_longform.py),
so the reference text is simply all references joined. Prints the word error rate.
"""
import argparse
import sys

import wer as werlib


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with references.csv")
    ap.add_argument("--hyp", required=True, nargs="+", help="transcript .txt file(s)")
    ap.add_argument("--n", type=int, default=0, help="number of clips the long file was built from (default: all)")
    args = ap.parse_args()
    reference = " ".join(r["reference"] for r in werlib.load_references(args.data, args.n))
    for path in args.hyp:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        ref_words, hyp_words = len(werlib.normalize(reference).split()), len(werlib.normalize(text).split())
        print(f"{path}: WER {100 * werlib.wer(reference, text):.2f}%  (words: reference {ref_words}, transcript {hyp_words})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
