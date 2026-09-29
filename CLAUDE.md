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
### Phase 1: macOS testing on this Mac (DONE 2026-09-29, see Status)
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

### Phase 2: paragraph timestamps (NEXT)
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
  budget: measure ASR best (1.7B-4bit, 2.5 GB peak) + LLM together. Use a separate process if two MLX models do not work well in one.
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
- 2026-09-29, Phase 1 done on the M2 Air (8 GB, macOS 27). Committed on `md-output-and-roadmap`.
  - Setup: uv 0.12.20 (official installer, `~/.local/bin`), `.venv` Python 3.11.15, mlx 0.32.2, mlx-audio 0.5.7,
    huggingface-hub 1.33.0. `mlx_lm` is NOT installed.
  - Bug fixed: `setup_models` checked the cache without mlx-audio's file patterns, so hub 1.33 raised
    IncompleteSnapshotError (no README.md / .gitattributes) and setup failed after a good download. The download and
    both checks now share `allow_patterns()`, and there is a regression test for it.
  - Tests: 139 pass with `.venv` and with system python3 (1 skip: qwen-asr is not installed on a Mac). Smoke test
    WER 0 on `fleurs_it_sample`.
  - `mac_selftest.sh --n 30`: 43.7 min (23 of them downloads), no out-of-memory, no thermal warnings, speakers PASS.
    0.6B-8bit 5.62% WER, 17.2x real time, 1.8 GB. 1.7B-4bit 2.81%, 9.5x, 2.5 GB. 1.7B-8bit 2.14%, 3.5x, 3.3 GB.
  - Decision (user): Mac *best* = 1.7B-4bit, recommended from 8 GB (`MAC_BEST_MIN_RAM_GB = 8`). The batch rule now
    has its own `MAC_BATCH_MIN_RAM_GB = 16`, so 8 GB stays at batch 1. The precheck text and README use the measured
    numbers. This Mac's saved choice is now best.
  - Batch re-run on 4-bit: batches 1, 2 and 4 all ran at about 11.5x; batching only adds process memory. The
    self-test's batch 2 at 0.6x (on 8-bit) did not reproduce. 8-bit was not re-run.
  - Phase 4 findings: a local LLM needs `mlx-lm` as a new dependency (ask first) or its own loop over mlx-audio's
    copies in `mlx_audio/lm/`. Confidence needs no mlx-audio patch: pass a recording greedy sampler through the
    existing `sampler=` argument (chosen-token probability + top-k per step, both paths). The single-piece
    `stream_generate` also yields full-vocab logprobs.
  - Open: the full `--n 100` self-test, batch 2 on a 16 GB Mac. Next: Phase 2.
- 2026-09-29, later: full self-test, mlx-lm and word-piece confidence (not committed yet).
  - Full `mac_selftest.sh` (100 clips, 25.3 min of audio, 22.8 min run, no errors, speakers PASS on best):
    0.6B-8bit 5.74% WER, 18.7x, 1.8 GB; 1.7B-4bit (best) 3.51%, 6.4x, 2.5 GB; 1.7B-8bit 2.89%, 2.3x, 3.3 GB.
    NVIDIA fp16 on the same set: 2.74% / 5.82%. Best on the Mac is about 40% fewer mistakes than light, not half.
    Speed swings with memory pressure (best measured 6.4-11.6x; this run started with ~2 GB free and swap in use).
  - Batch 2 was slower than batch 1 in 3 of 4 runs and never faster, and there's no 16 GB Mac to test on:
    `MAC_BATCH_SIZE = 1` on every Mac (the 16 GB batch-2 rule was removed). The README and precheck text now use
    the 100-clip numbers.
  - `mlx-lm==0.31.3` is installed and pinned in `requirements-mac.txt`; it added only protobuf, with mlx /
    mlx-audio / transformers unchanged. Smoke test with `Qwen3-0.6B-4bit`: 0.44 GB peak, ~15 tok/s. Asked to fix
    "e o comprato", it also dropped "ho", which shows why the LLM must only choose among the ASR alternatives.
  - `--confidence` (MLX only, opt-in): a greedy recording sampler (`_Recorder` in `backends/mlx_qwen.py`) keeps
    each chosen word-piece's probability and the top 5 alternatives, on both the single-piece (`stream_generate`)
    and the batched path. It writes `<recording>.review.json` (git-ignored) with one entry per `.md` paragraph.
    The `.md` is unchanged, transcripts are identical with and without recording (30 clips, batch 1 and 2),
    +2.4% time, same peak memory. 6% of word-pieces fall below p 0.9 and 0.7% below p 0.5. Not supported with
    `--speakers` or on the torch path yet. Tests: 146 pass (`tests/test_confidence.py`, 7 new).
  - Open: the correction model choice (Qwen3-0.6B/1.7B 4-bit?), and ASR + LLM measured together on 8 GB. Grouping
    word-pieces into words and flagging unsure ones. Next: commit this, then Phase 2 (paragraph timestamps; the
    sidecar can take each paragraph's offset then).
