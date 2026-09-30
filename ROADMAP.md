# Roadmap

What LocalTranscribe does today and what comes next. The transcriber is complete on its own; the later steps add
review and correction on top without changing how it is used.

## Done

- Local Italian transcription with Qwen3-ASR on Apple Silicon (MLX), NVIDIA GPUs (PyTorch) and CPU, with the same
  audio chunking and text cleanup on every platform.
- Hardware check that recommends a model, explains why, and remembers the choice; double-click launchers for Mac
  (`Trascrivi.command`) and Windows (`transcribe.bat`).
- Two-speaker separation (`--speakers 2`) with `[hh:mm:ss]` timestamps per turn.
- Transcripts written as Markdown (`<recording>.md`), every paragraph starting with its `[hh:mm:ss]` timestamp (the
  benchmark scripts strip the timestamps before scoring).
- Measured presets: NVIDIA (RTX 6000, float16), CPU, and Apple Silicon (100 FLEURS clips per preset, batch sizes,
  speaker separation), plus a fresh-clone install test on a Mac.
- `--confidence` (Mac): each word's probability is saved to `<recording>.review.json` without changing the
  transcript. Words below 0.9 are flagged unsure (9.4 % of words, 70 % of the mistakes on 100 FLEURS clips) and get
  whole-word candidates from the speech model itself: the weakest piece of the word is swapped for each likely
  alternative and the model finishes the word, kept only if it then continues with the original next word. The
  candidates reuse the transcription's own cache, so this costs about 10 % extra time.
- Review page (`Rivedi.command`, `python -m localtranscribe.review`): editable transcript beside the audio on
  `127.0.0.1`, click a timestamp to jump, save writes the `.md` back.
- AI suggestions with guardrails (`--correct`, Mac). Four parts, each a safety net for the previous:
  1. **Flag the unsure words** from the speech model's own confidence.
  2. **A small local model (Qwen3-0.6B) chooses only there**, among the speech model's own candidates, while the
     transcription runs.
  3. **A rule checker accepts or rejects each choice**: only flagged words change, only to one of their candidates,
     no word is added or removed, a rejected choice leaves the word as it was. The rules (`LAWS.bend`) are proven in
     [Bend](https://github.com/HigherOrderCO/Bend) (`PROOF.bend`) and the checker runs as a native binary.
  4. **You have the final say**: the review page highlights each suggestion; nothing is applied until you accept it
     and save.
  Tuned on 100 FLEURS clips (12 suggestions, 5 fixes, 1 new mistake) and checked unchanged on 200 other clips: 25
  suggestions, 11 fixes, 3 new mistakes; accepting all lowers the word error rate from 3.77 % to 3.57 %.
  `Trascrivi.command` offers the suggestions for single-speaker transcripts.
- Reliability fixes after an external review: a new run never overwrites a transcript you edited (the old one is kept
  as `<name>.bak-<date>.md`), one failing file no longer stops the others, `--context` never cuts real speech, a very
  short voice note no longer crashes `--speakers`, and the review page saves in order.

## Next: better suggestions

- Measure on real calls and lectures, not only FLEURS read speech.
- Reach more mistakes: the right word is among the candidates for only about 38 % of flagged mistakes; many of the
  rest are numbers written out ("dieci" for "10") or two neighbouring words wrong together.
- Fewer false alarms: a bit less than half of the suggestions are real fixes (a higher weight on the speech model,
  ALPHA 8, gave fewer new mistakes on the held-out clips, but not on the tuning clips).
- Let the review page remember rejected suggestions.

## Under evaluation

- **Whisper as an alternative backend**, A/B-tested against Qwen3-ASR on real phone calls and lecture recordings
  rather than only on FLEURS read speech. The backend interface already allows it.
- **Voice-activity trimming before chunking**, to cut hallucinations on long silences and speed things up.
- **Optional speech enhancement (denoising) before transcription**, enabled only where measurements show it helps
  (very noisy recordings); on clean audio it tends to hurt.

## Later

- Windows: the rule checker as a prebuilt binary, and confidence recording on the PyTorch path.
- Confidence recording together with `--speakers`.
- Idea: word-level timestamps from the Qwen3 forced aligner (mlx-audio already includes it), so the review page can
  jump to, or play, exactly one word. It needs a second model download and more memory on 8 GB Macs. Not for
  generating candidates: a short clip loses the sentence the speech model needs to choose the right word.

## Not planned

- A cloud or API-based corrector. Everything stays on your computer.
