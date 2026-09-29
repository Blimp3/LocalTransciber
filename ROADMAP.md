# Roadmap

What LocalTranscribe does today and what comes next. The transcriber is complete on its own; the later steps add
review and correction on top without changing how it is used.

## Done

- Local Italian transcription with Qwen3-ASR on Apple Silicon (MLX), NVIDIA GPUs (PyTorch) and CPU, with the same
  audio chunking and text cleanup on every platform.
- Hardware check that recommends a model, explains why, and remembers the choice; double-click launchers for Mac
  (`Trascrivi.command`) and Windows (`transcribe.bat`).
- Two-speaker separation (`--speakers 2`) with `[hh:mm:ss]` timestamps per turn.
- Transcripts written as Markdown (`<recording>.md`).
- Measured presets: NVIDIA (RTX 6000, float16), CPU, and Apple Silicon (100 FLEURS clips per preset, batch sizes,
  speaker separation), plus a fresh-clone install test on a Mac.
- `--confidence` (Mac): each word-piece's probability and the model's top alternatives are saved to
  `<recording>.review.json`, one entry per paragraph, without changing the transcript.

## Next: a timestamp on every paragraph

- Prefix each paragraph with `[hh:mm:ss]`, the same format `--speakers` already uses, so a reader or the review tool
  can jump to it. The chunker already knows each piece's offset.
- The benchmark scripts that read `.md` files strip the timestamps before scoring.

## Then: a review tool

- A local page served by Python's built-in `http.server` on `127.0.0.1` and opened in the default browser. No new
  dependencies.
- The editable transcript on the left, the audio on the right. Clicking a paragraph's timestamp jumps the audio there;
  saving writes the `.md` back.
- The audio is served as the decoded 16 kHz WAV, so every input format plays in every browser.
- Entry point `python -m localtranscribe.review <recording>` plus a double-click launcher.

## Then: local AI correction, with guardrails

The goal is fewer mistakes without inventing text. Four parts, each one a safety net for the previous:

1. **Flag the unsure words** from the ASR model's own confidence (already recorded by `--confidence`).
2. **A small local language model suggests fixes only there**, choosing among the ASR model's own alternatives rather
   than writing freely. It runs on the same machine, alongside the transcriber, on paragraphs as they finish.
3. **A rule checker accepts or rejects each suggestion**: changes only at flagged positions, each replacement one of
   the ASR candidates, no words inserted or deleted, a rejected suggestion leaves the text untouched. The rules are
   written and proved in [Bend](https://github.com/HigherOrderCO/Bend) and compiled to a native binary.
4. **You have the final say**: the review tool highlights accepted suggestions and nothing is applied until confirmed.

## Under evaluation

- **Whisper as an alternative backend**, A/B-tested against Qwen3-ASR on real phone calls and lecture recordings
  rather than only on FLEURS read speech. The backend interface already allows it.
- **Voice-activity trimming before chunking**, to cut hallucinations on long silences and speed things up.
- **Optional speech enhancement (denoising) before transcription**, enabled only where measurements show it helps
  (very noisy recordings); on clean audio it tends to hurt.

## Later

- Windows: the rule checker as a prebuilt binary, and confidence recording on the PyTorch path.
- Confidence recording together with `--speakers`.

## Not planned

- A cloud or API-based corrector. Everything stays on your computer.
