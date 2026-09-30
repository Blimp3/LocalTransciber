"""Compare two bench.py --hyp-csv files of the same set: silence gate off vs on. No model, no audio.

    python ab_compare.py <repo root> <off.csv> <on.csv>

Prints: corpus WER off / on (FLEURS it300: also tuning it_000-it_099 and held-out it_100-it_299), per clip whether the
ON hypothesis is identical, is the OFF one with only its end removed (what the gate should do on vp_pauses), or changed
elsewhere, and the most frequent inserted word runs (>= 2 words, aligned to no reference word) in each file: invented
phrases such as "grazie a tutti" or "non è vero" show up there, wherever they are in a clip.
"""
import collections
import csv
import sys

sys.path.insert(0, sys.argv[1] + "/benchmark")
import wer as werlib  # noqa: E402  (also puts the repo root on sys.path)
import jiwer  # noqa: E402


def load(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {r["file"]: r for r in csv.DictReader(f)}


def inserted_runs(refs, hyps):
    out = collections.Counter()
    o = jiwer.process_words([werlib.normalize(r) for r in refs], [werlib.normalize(h) for h in hyps])
    for words, chunks in zip(o.hypotheses, o.alignments):
        for c in chunks:
            if c.type == "insert" and c.hyp_end_idx - c.hyp_start_idx >= 2:
                out[" ".join(words[c.hyp_start_idx:c.hyp_end_idx])] += 1
    return out


def main(root, off_path, on_path):
    off, on = load(off_path), load(on_path)
    assert off.keys() == on.keys(), "the two files cover different clips"
    files = sorted(off)
    subsets = {"all": files}
    if all(f.startswith("it_") for f in files) and len(files) == 300:
        subsets["tuning it_000-099"] = [f for f in files if int(f[3:6]) < 100]
        subsets["held-out it_100-299"] = [f for f in files if int(f[3:6]) >= 100]
    for name, fs in subsets.items():
        refs = [off[f]["reference"] for f in fs]
        w_off = werlib.wer(refs, [off[f]["hypothesis"] for f in fs])
        w_on = werlib.wer(refs, [on[f]["hypothesis"] for f in fs])
        print(f"WER {name}: off {100 * w_off:.2f}%  on {100 * w_on:.2f}%  ({len(fs)} clips)")
    kinds = collections.Counter()
    for f in files:
        a, b = werlib.normalize(off[f]["hypothesis"]).split(), werlib.normalize(on[f]["hypothesis"]).split()
        if a == b:
            kinds["identical"] += 1
        elif a[:len(b)] == b:
            kinds["end removed"] += 1
            print(f"  {f}: end removed: {' '.join(a[len(b):])!r}")
        else:
            kinds["changed elsewhere"] += 1
            i = next(k for k in range(min(len(a), len(b)) + 1) if k == min(len(a), len(b)) or a[k] != b[k])
            print(f"  {f}: CHANGED ELSEWHERE at word {i}: off {' '.join(a[i:i + 8])!r} / on {' '.join(b[i:i + 8])!r}")
    print("clips:", dict(kinds))
    refs = [off[f]["reference"] for f in files]
    for tag, d in (("off", off), ("on", on)):
        runs = inserted_runs(refs, [d[f]["hypothesis"] for f in files])
        print(f"inserted runs >= 2 words ({tag}): {sum(runs.values())}; most frequent:", runs.most_common(8))


if __name__ == "__main__":
    main(*sys.argv[1:4])
