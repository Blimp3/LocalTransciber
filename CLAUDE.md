# LocalTranscribe: working notes for Claude sessions

Local Italian speech-to-text (Qwen3-ASR on MLX / CUDA / CPU). The README explains the product; this file holds the
decisions and the roadmap. **Update the Status section at the end of every session.**

## How we work
- Opus plans and reviews. Execution (setup, tests, implementation) goes to the `executor` subagent
  (`.claude/agents/executor.md`: Sonnet 5.5, medium effort). `.claude/settings.local.json` makes Sonnet 5.5 at medium
  effort the default for every other subagent too.
- Bend rules (from the user's global CLAUDE.md): read `bend guide` first, keep the rules in `LAWS.bend` (written by the
  human, the AI does not edit it), prove them in `PROOF.bend`, run `bend PROOF.bend` before every commit (it must print
  ALL PROOFS CHECK), and parallelize wherever possible.
- Stdlib first, no new dependencies without asking. Commit only when the user asks.
- Transcripts are private: `*.md` and `*.txt` are gitignored (except README.md, CLAUDE.md and `.claude/`).

## Decisions (2026-09-29)
1. Transcripts are written as `<recording>.md` (it was `.txt`), next to the recording or in `--out-dir`.
2. Every paragraph gets a `[hh:mm:ss]` timestamp (the same format as `--speakers`), so the GUI can jump to it.
3. Review GUI: very simple. The **transcript on the left** (editable), the **audio on the right**, and clicking a
   paragraph's timestamp jumps the audio there.
4. AI correction: first a **very small local model** that runs **at the same time as the transcriber**. An API model
   is only an idea: **do not implement it**.
5. Anti-hallucination design (approved):
   1. **Flag the unsure words**, using the ASR model's own per-token confidence.
   2. **The AI suggests fixes only there**, ideally by choosing from the ASR model's own 2nd/3rd guesses rather than
      writing freely.
   3. **A Bend checker accepts or rejects** each suggestion: changes only at flagged words, each change small,
      everything else rejected. The rules live in `LAWS.bend`, the proofs in `PROOF.bend`.
   4. **The user has the final say**: the GUI highlights accepted suggestions and nothing is applied until confirmed.
6. Now: Mac only. Later: Windows (compile the Bend checker to C with `bend guard.bend -o guard.c` and ship a prebuilt
   `.exe`; get confidence scores from the torch/qwen-asr path).

## Roadmap
### Phase 1: macOS testing on this Mac (NEXT)
The machine: Apple M2, 8 GB memory, macOS 27, about 53 GB free, `uv` not installed, no `.venv` yet. On 8 GB the
hardware check should recommend **light**.
1. `bash setup_mac.sh`. It installs `uv` with the official astral.sh installer (curl | sh). **Ask the user before
   running it.** It runs the hardware check, creates `.venv`, installs about 1 GB of packages and downloads the model.
   Confirm it recommends light, and record what it prints.
2. Unit tests: `.venv/bin/python -m unittest discover -s tests -v`.
3. Smoke test: `./transcribe.sh tests/fleurs_it_sample.wav --out-dir <scratch> --stats` → a `.md` file, with the text
   close to `tests/fleurs_it_sample.txt`.
4. `bash mac_selftest.sh --n 30` first (quicker), then the full run if the user wants it (about 1 hour, plugged in).
   It downloads FLEURS clips plus all three Mac presets, and a 1.7B preset may run out of memory on 8 GB; the report
   records that and carries on. The report is `mac_selftest_report.txt`.
5. From the report, adjust `MAC_MODELS`, `MAC_BEST_MIN_RAM_GB` and the batch sizes in `localtranscribe/config.py`,
   and remove the "provisional / estimate" wording in the README where it is now measured.
6. After setup, look into these for the later phases (read only, nothing to build yet):
   - Does `.venv` already contain `mlx_lm`, pulled in by mlx-audio? If yes, the small local LLM needs no new dependency.
   - In mlx-audio 0.5.7's `qwen3_asr.py`, does `_generate_chunks_batched(..., logits_processors=...)` let us record
     per-token probabilities and the top-k alternatives? That would be the confidence source for Phase 4.

### Phase 2: paragraph timestamps
- `textutil.split_audio_into_chunks` already returns `(piece, offset_seconds)`; `pipeline.split_wav` drops the offset.
  Keep it, and prefix each paragraph with `[{diarize.fmt_time(offset)}] `.
- `benchmark/bench.py` uses `split_wav` + `transcribe_pieces` directly and is unaffected. But
  `benchmark/mac_selftest.py` (step_smoke) and `benchmark/score_long.py` read the `.md` file, so they must strip the
  timestamps before computing the word error rate (WER). Add a unit test for the stripping.

### Phase 3: review GUI
- A local page served by Python's `http.server` on 127.0.0.1 and opened in the default browser. No new dependencies.
- Left: editable paragraphs. Right: an `<audio>` element. Clicking a timestamp sets `currentTime`. Save writes back to
  the `.md` file.
- Serve the audio decoded with `audio.load_audio` as a 16 kHz WAV, so every input format plays in every browser.
- Entry point: `python -m localtranscribe.review <recording>`, plus a double-click launcher like `Trascrivi.command`.

### Phase 4: local AI correction with the Bend checker
- **Confidence:** record per-token probability and the top-k alternatives during ASR decoding (MLX path first). Save
  them in a sidecar file (e.g. `<recording>.review.json`: per paragraph the offset, words, flags, alternatives and
  suggestions), so the `.md` stays clean.
- **Concurrent small LLM:** a very small MLX model (0.5-1.7B, 4-bit) consumes finished paragraphs from a queue while
  ASR continues (`pipeline.transcribe_pieces` already produces results slice by slice). The 8 GB M2 is the memory
  budget: measure ASR light + LLM together. Use a separate process if two MLX models do not work well in one.
- **The LLM only chooses** among the ASR alternatives at flagged positions, or leaves the word as it is.
- **Bend checker** (`guard.bend`, compiled to a native binary, called from Python through a simple stdin/stdout
  protocol, and parallel per paragraph). A first draft of the laws, for the user to own in `LAWS.bend`:
  - the output differs from the input only at flagged positions;
  - each replacement is one of the ASR candidates for that position;
  - no words are inserted or deleted (same word count);
  - a rejected suggestion leaves the text identical.
- **GUI:** highlight each accepted suggestion; the user accepts or rejects each one; nothing is applied automatically.

### Later (not now)
- Windows: the Bend checker as a prebuilt `.exe` (from `-o guard.c`), and confidence from the torch path.
- API corrector: an idea only.

## Status
- 2026-09-29: output switched to `.md`; Sonnet executor configured; roadmap agreed. Unit tests pass (137) with the
  system python3. Next: Phase 1.
