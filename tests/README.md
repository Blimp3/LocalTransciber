# Test data

`fleurs_it_sample.wav` is one unmodified clip of the **FLEURS** speech dataset, Italian test split
(`data/it_it/audio/test.tar.gz`, file `10210638754804006465.wav`, 6.5 s, 16 kHz mono), and
`fleurs_it_sample.txt` is its reference transcription (the dataset's `raw_transcription` field):

> Eppure gli uccelli hanno molte caratteristiche che li accomunano ai dinosauri.

- Dataset: *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*,
  Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa,
  Clara Rivera, Ankur Bapna (Google), 2022. https://arxiv.org/abs/2205.12446
- Source: https://huggingface.co/datasets/google/fleurs
- Licence: **Creative Commons Attribution 4.0 International (CC-BY-4.0)**,
  https://creativecommons.org/licenses/by/4.0/ - the clip is redistributed here unchanged, with this attribution.

It is the only audio file in the repository. It is used by the unit tests and by `mac_selftest.sh` as a
quick smoke test. The larger benchmark clips are downloaded on demand by `benchmark/download_fleurs.py` into
`benchmark/data/` (git-ignored) and are never committed.

Run the unit tests (they need no model and no GPU):

    python -m unittest discover -s tests -v
