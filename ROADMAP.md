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
     [Bend](https://github.com/bendlang/bend) (`PROOF.bend`) and the checker runs as a native binary, prebuilt
     for Apple Silicon and for Windows x64 (on Windows it is used once confidence recording exists there; until then
     `--correct` needs a Mac). Python re-checks the binary's answer with an independent implementation of the same
     rules; any disagreement means no suggestions.
  4. **You have the final say**: the review page highlights each suggestion; nothing is applied until you accept it
     and save.
  Tuned on 100 FLEURS clips (12 suggestions, 5 fixes, 1 new mistake) and checked unchanged on 200 other clips: 25
  suggestions, 11 fixes, 3 new mistakes; accepting all lowers the word error rate from 3.77 % to 3.57 %.
  `Trascrivi.command` offers the suggestions for single-speaker transcripts.
- Reliability fixes after an external review: a new run never overwrites a transcript you edited (the old one is kept
  as `<name>.bak-<date>.md`), one failing file no longer stops the others, `--context` never cuts real speech, a very
  short voice note no longer crashes `--speakers`, and the review page saves in order.
- A finished transcript is never lost: the output folder is checked before the model loads, the text done so far is
  kept in `<name>.partial.md` while a long recording runs, a `.md` that cannot be replaced (open in another program)
  is saved as `<name>.new-<date>.md`, an empty result writes nothing, and the review page keeps a `.bak` copy when
  the file changed on disk.
- Silent pieces (no tenth of a second louder than -60 dBFS, such as digital silence) are no longer sent to the speech
  model: on such pieces every model tested invented a phrase ("Non è vero.", "Grazie a tutti."). The program says how
  many seconds it left out as silence. Checked without a model on 1,930 speech pieces (FLEURS, VoxPopuli, also after
  phone and Opus codecs): the quietest is 30 dB above the limit. Measured on 12 speeches with 2-15 s pauses inserted:
  the gate removes exactly the invented words of the silent pieces and changes nothing else, for every model tested
  (Qwen3-ASR 1.7B: word error rate 7.2 % to 6.7 %); every other test set is unchanged. Also measured and not
  adopted: 0.5 s or 1 s of silence in front of every piece, meant to save the first words of a piece that starts
  right on speech. On 1,177 VoxPopuli clips it recovers about as many first words as it loses and shifts the
  decoding of the rest (word error rate 17.1 % to 17.3 %); on FLEURS it changes nothing.

## Next: better suggestions

- Measure on real calls and lectures, not only FLEURS read speech. First real lecture (95 minutes of university
  mathematics, NVIDIA, Qwen3-ASR 1.7B): 31x real time, 285 paragraphs, no invented text, no loops, no stock phrases
  at pauses; about 1 % of the cuts between pieces may repeat the word at the cut (to be checked by ear).
- Reach more mistakes: the right word is among the candidates for only about 38 % of flagged mistakes; many of the
  rest are numbers written out ("dieci" for "10") or two neighbouring words wrong together.
- Fewer false alarms: a bit less than half of the suggestions are real fixes (a higher weight on the speech model,
  ALPHA 8, gave fewer new mistakes on the held-out clips, but not on the tuning clips).
- Let the review page remember rejected suggestions.

## Under evaluation

- **Whisper as an alternative backend**, A/B-tested against Qwen3-ASR on real phone calls and lecture recordings
  rather than only on FLEURS read speech. The backend interface already allows it.
- **Trimming long silences inside pieces.** On the 12 speeches with inserted pauses, 94 % of the pause time lies
  inside pieces that also hold speech. After the silence gate, Qwen3-ASR reads through those pauses without
  inventing words (what looked like invented text at file starts turned out to be real speech that the reference
  transcripts leave out), so no trimming is planned for it. Whisper loses the speech that follows 3 s or more of
  digital silence in a piece, so a trim belongs to the Whisper evaluation. A real room is louder than the gate's
  limit (VoxPopuli's background is around -42 dBFS): pauses with noise need a voice activity detector, not a level.
- **Parakeet as a small, fast candidate.** NVIDIA's Parakeet TDT 0.6B v3 (CC-BY-4.0) covers 25 European languages
  including Italian (NVIDIA reports 3.0 % WER on FLEURS Italian), and the MLX library this program already uses on
  the Mac can run it, also in a much smaller ternary version. To be tested on the Mac against Qwen3-ASR and Whisper
  on the same clips.
- **Optional speech enhancement (denoising) before transcription**, enabled only where measurements show it helps
  (very noisy recordings); on clean audio it tends to hurt.

## Later

- Windows: confidence recording on the PyTorch path.
- Confidence recording together with `--speakers`.
- Idea: word-level timestamps from the Qwen3 forced aligner (mlx-audio already includes it), so the review page can
  jump to, or play, exactly one word. It needs a second model download and more memory on 8 GB Macs. Not for
  generating candidates: a short clip loses the sentence the speech model needs to choose the right word.

## Not planned

- A cloud or API-based corrector. Everything stays on your computer.
