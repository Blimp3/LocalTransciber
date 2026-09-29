"""Download FLEURS Italian test clips for benchmarking (about 0.75 MB per clip).

    python benchmark/download_fleurs.py [--n 100] [--out benchmark/data/fleurs_it]

FLEURS (Conneau et al., 2022, Google) is licensed CC-BY-4.0: https://huggingface.co/datasets/google/fleurs
The clips are the first N recordings of the Italian test archive, so the set is the same on every
machine and a smaller --n is always a prefix of a bigger one. Only that many megabytes are
transferred (the download stops once N clips are collected). Nothing is written to the repo folder
outside benchmark/data/, which is git-ignored.

Output: <out>/it_000.wav ... plus references.csv with columns file, reference, source, gender.
"""
import argparse
import csv
import io
import os
import sys
import tarfile
import time
import urllib.request

BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/it_it"
TSV_URL = f"{BASE}/test.tsv"
TAR_URL = f"{BASE}/audio/test.tar.gz"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "data", "fleurs_it")


def _open(url, tries=4):
    last = None
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "local-transcribe"}), timeout=60)
        except Exception as e:  # network hiccup: retry with a short back-off
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not download {url}: {last}")


def read_tsv():
    """file_name -> (raw transcription, gender) for the whole test split."""
    with _open(TSV_URL) as r:
        text = r.read().decode("utf-8")
    table = {}
    for row in csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE):
        if len(row) >= 7:
            table[row[1]] = (row[2].strip(), row[6].strip())
    return table


def existing(out):
    path = os.path.join(out, "references.csv")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if os.path.exists(os.path.join(out, r["file"]))]


def download(n=100, out=DEFAULT_OUT, force=False, quiet=False):
    """Make sure `out` holds the first n clips; returns the list of rows (dicts)."""
    os.makedirs(out, exist_ok=True)
    have = [] if force else existing(out)
    if len(have) >= n:
        return have[:n]
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    table = read_tsv()
    say(f"FLEURS Italian test split: {len(table)} recordings; fetching the first {n} (CC-BY-4.0).")
    rows = []
    for attempt in range(3):
        rows = []
        try:
            with _open(TAR_URL) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
                for member in tar:
                    name = os.path.basename(member.name)
                    if not member.isfile() or name not in table:
                        continue
                    data = tar.extractfile(member).read()
                    fname = f"it_{len(rows):03d}.wav"
                    with open(os.path.join(out, fname), "wb") as f:
                        f.write(data)
                    ref, gender = table[name]
                    rows.append({"file": fname, "reference": ref, "source": name, "gender": gender})
                    if len(rows) % 20 == 0:
                        say(f"  {len(rows)}/{n} clips")
                    if len(rows) >= n:
                        break
            break
        except Exception as e:
            say(f"  download interrupted ({type(e).__name__}: {e}); retrying ({attempt + 1}/3)")
            time.sleep(3)
    if len(rows) < n:
        raise RuntimeError(f"only {len(rows)} clips could be downloaded")
    with open(os.path.join(out, "references.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "reference", "source", "gender"])
        w.writeheader()
        w.writerows(rows)
    say(f"Saved {len(rows)} clips to {out}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="number of clips (default 100)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="destination folder")
    ap.add_argument("--force", action="store_true", help="download again even if the folder is complete")
    args = ap.parse_args()
    try:
        download(args.n, args.out, args.force)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
