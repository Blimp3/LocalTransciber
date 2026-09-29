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
- `--confidence` (Mac): each word-piece's probability and the model's top alternatives are saved to
  `<recording>.review.json`, one entry per paragraph, without changing the transcript.
- Review page (`Rivedi.command`, `python -m localtranscribe.review`): editable transcript beside the audio on
  `127.0.0.1`, click a timestamp to jump, save writes the `.md` back.

## Next: local AI correction, with guardrails

The goal is fewer mistakes without inventing text. Four parts, each one a safety net for the previous:

1. **Flag the unsure words** from the ASR model's own confidence (already recorded by `--confidence`).
2. **A small local language model suggests fixes only there**, choosing among the ASR model's own alternatives rather
   than writing freely. It runs on the same machine, alongside the transcriber, on paragraphs as they finish.
3. **A rule checker accepts or rejects each suggestion**: changes only at flagged positions, each replacement one of
   the ASR candidates, no words inserted or deleted, a rejected suggestion leaves the text untouched. The rules are
   written and proved in [Bend](https://github.com/HigherOrderCO/Bend) and compiled to a native binary.
4. **You have the final say**: the review tool highlights accepted suggestions and nothing is applied until confirmed.

The candidates are the speech model's own whole-word guesses: each unsure word's weakest piece is swapped for its
alternatives and the model finishes the word (`"cands"` in the sidecar).

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
