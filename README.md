<h1 align="center">LocalTranscribe</h1>

<p align="center">
  Italian speech-to-text that runs only on your own computer.<br>
  It writes a Markdown transcript next to each recording. It uploads nothing.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
  <img alt="Platforms" src="https://img.shields.io/badge/platform-macOS%20Apple%20Silicon%20%7C%20Windows%20NVIDIA%20%7C%20CPU-lightgrey">
  <img alt="Python 3.11" src="https://img.shields.io/badge/python-3.11-blue">
  <img alt="Model: Qwen3-ASR" src="https://img.shields.io/badge/model-Qwen3--ASR-orange">
  <img alt="Works offline" src="https://img.shields.io/badge/works-offline-brightgreen">
</p>

> **Italiano:** la [Guida rapida](#guida-rapida-italiano) è in fondo alla pagina.

LocalTranscribe makes text transcripts of Italian audio and video recordings, for example interviews, phone calls,
meetings and lectures. It uses the open speech model [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR). The setup
downloads the models one time. After that, all work runs offline on your Apple Silicon Mac, your NVIDIA GPU or a
plain CPU.

## Features

- **Local and offline.** After the setup, the launchers switch the model library to offline mode. There is no
  account, no cloud service and no telemetry. Transcripts are plain Markdown files that you control.
- **Any audio or video file.** The program reads wav, mp3, m4a, ogg, opus, flac, mp4, mkv, webm and more. FFmpeg is
  included, so you do not need to install anything.
- **Uses the hardware you have.** The program runs on Apple Silicon (MLX on the Apple GPU), NVIDIA GPUs (PyTorch +
  CUDA) or the CPU only.
- **Selects the model for your computer.** A [hardware check](#hardware-check) recommends *best* or *light* and tells
  you why. It saves the choice.
- **Speaker separation.** With `--speakers 2`, a call between two people becomes lines such as
  `[00:01:23] Parlante 1: ...`.
- **Context hints.** `--context "Mario Rossi, LoRaWAN"` helps the model spell names and technical words correctly.
- **Word confidence** (Mac). `--confidence` saves the probability of each word and marks the unsure words. For each
  unsure word, it also saves whole-word alternatives from the speech model itself.
- **AI suggestions, checked and confirmed** (Mac). `--correct` lets a small local model choose among those
  alternatives at unsure words. A rule checker rejects all other changes. A proof written in
  [Bend](https://github.com/bendlang/bend) shows that the checker is correct. You accept or reject each suggestion
  on the review page. Nothing changes without your decision.
- **Review page.** A local page shows the transcript next to the audio. You can edit the text. Click a timestamp to
  move the audio to that point.
- **Never loses a transcript.** A new run keeps your edited transcript as a backup. After a crash or a closed laptop,
  the text done so far stays on disk. If the program cannot replace a file, it saves the new file under another name.
  See [Your transcripts are safe](#your-transcripts-are-safe).
- **Double-click launchers** for people who never open a terminal. On a Mac, use `Trascrivi.command` and
  `Rivedi.command`. On Windows, drag and drop files onto `transcribe.bat`.

## Requirements

| | Apple Silicon Mac | Windows + NVIDIA GPU | Any PC, CPU only |
|---|---|---|---|
| System | macOS 14 (Sonoma) or newer, M1 or newer. Intel Macs are not supported. | Windows 10/11, NVIDIA driver 570.65 or newer, GeForce GTX 16 / RTX 20 series (2018) or newer | Windows 10/11 |
| GPU memory | 8 GB unified memory or more for *best* | 6 GB for *best*, 3 GB for *light* | - |
| Free disk space | about 5 GB (6 GB with both models) | about 12 GB (10 GB with one model) | about 4 GB |
| Internet | setup only | setup only | setup only |

Linux works with a manual installation (see [Development](#development)). Linux is not officially supported.

## Quick start

Get the code in one of these two ways:

- [Download the ZIP](https://github.com/Blimp3/LocalTransciber/archive/refs/heads/main.zip) and unpack it in a
  permanent location, for example your Documents folder.
- Clone the repository:

```bash
git clone https://github.com/Blimp3/LocalTransciber.git
cd LocalTransciber
```

### macOS (Apple Silicon)

```bash
bash setup_mac.sh            # 10-20 minutes, once
./transcribe.sh recording.m4a
```

`setup_mac.sh` does these steps:

1. It checks that the Mac is supported.
2. If the small helper tool [uv](https://docs.astral.sh/uv/) is missing, it installs uv. It uses the official
   installer and tells you before it starts.
3. It runs the hardware check. Press Enter to accept the recommended model.
4. It makes a Python environment in `.venv`.
5. It installs about 1 GB of packages.
6. It downloads about 2.4 GB of models.

To skip the model question, run `bash setup_mac.sh light` (or `best`, or `both`).

If you do not want to use Terminal, use the launcher:

1. In Finder, **double-click `Trascrivi.command`**.
2. Select one or more recordings.

The `.md` transcripts appear next to the recordings. For a single-speaker transcript, the launcher also offers the
AI suggestions (`--correct`). Accept or reject them later with `Rivedi.command`.

If macOS refuses to open a downloaded script the first time, do these steps (only once):

1. Right-click the script.
2. Choose **Open**.
3. Choose **Open** again.

### Windows (NVIDIA GPU or CPU)

1. Double-click **`setup_windows.bat`**.
   The script installs uv if uv is missing, runs the hardware check and makes `.venv`. Then it installs the
   packages (1 - 5 GB) and downloads the models. The NVIDIA build is the largest. If the script finds no usable
   NVIDIA card, it installs the CPU-only packages.
2. **Drag audio or video files onto `transcribe.bat`.**

From a command prompt, use `setup_windows.bat light|both|cpu [--yes]` and `transcribe.bat recording.m4a [options]`.

If the setup stops because of a network problem, run it again. It continues from the point where it stopped.

## Usage

```bash
./transcribe.sh interview.m4a --context "Mario Rossi, Politecnico di Milano, LoRaWAN"   # Mac
transcribe.bat  call.m4a --speakers 2                                                  # Windows
```

The program writes the transcript `<recording>.md` next to the recording, or into `--out-dir`. It cuts long
recordings at quiet moments into pieces of about 20 seconds. Each piece becomes a paragraph that starts with its
`[hh:mm:ss]` timestamp. With `--speakers`, each turn starts with the timestamp and the speaker label.

| Option | What it does |
|---|---|
| `--context "names, terms"` | Names and technical words that occur in the recording. They help the model spell these words correctly. |
| `--speakers 2` | Separates the speakers: `[00:01:23] Parlante 1: ...`. Reliable on calls of a few minutes or more. It can merge very short replies such as "sì" into the other speaker. |
| `--out-dir folder` | Writes the transcripts to this folder instead of next to the recordings. |
| `--language auto` | Detects the language instead of assuming Italian. You can also give a language, for example `English`, `French` or `Spanish`. |
| `--model best`, `light`, or a Hugging Face repo id | Selects the model. Default: the choice that the hardware check saved. |
| `--device auto`, `cuda`, `mps`, `cpu` | Selects the hardware. Default: the saved choice. On a Mac, `mps` is the Apple GPU. |
| `--batch-size N` | Number of pieces that the model processes together. Default: the saved choice. The program lowers it if free memory is low. A lower value uses less memory. |
| `--confidence` | Also saves these data to `<recording>.review.json`: the confidence of each word-piece and the model's alternatives, plus per-word `"words"` with an `"unsure"` flag and whole-word `"cands"` for the unsure words. Apple Silicon only for now. |
| `--correct` | Suggests fixes for unsure words. A small local model (Qwen3-0.6B, downloaded at setup) chooses among the speech model's own alternatives. The Bend checker accepts or rejects each choice. The accepted choices go to `<recording>.review.json` for the review page. The program does not change the `.md`. About 10% slower. Implies `--confidence`. Apple Silicon only for now. |
| `--stats` | Prints the device, model, speed and peak memory at the end. |
| `--chunk 20` | Seconds per piece. Keep the value 20: longer pieces are measurably less accurate. |

### Your transcripts are safe

A transcript can take an hour to make and more time to correct. So the program never discards a transcript:

- **If you transcribe the same recording again**, your edits stay. If the `.md` changed, the program keeps the old
  file as `<recording>.bak-<date>.md`.
- **Before the model loads**, the program checks that every recording exists and that it can write to the output
  folder. So it reports a locked folder in seconds, not after the work is done.
- **During a transcription**, the text done so far is in `<recording>.partial.md`. This file appears when the first
  pieces are done, usually within a minute. It grows every few pieces. If the computer crashes, the battery dies or
  you press Ctrl+C, the file stays. The program removes it when the transcript is complete.
- **If the program cannot replace the `.md`**, it tries three times. Then it saves the new transcript as
  `<recording>.new-<date>.md` next to the old one and tells you. If that also fails, it saves the file in your home
  folder. The program does not touch the old transcript. On Windows, the cause can be one of these:
  - The file is open in Word or in an editor that keeps the file locked.
  - The file is marked read-only.
  - An antivirus or sync program holds the file.

  Tip: close the transcript in your editor before you transcribe the same recording again or save from the review
  page.
- **If the model returns no text at all**, the program writes nothing. It tells you that no speech was recognised.
  It does not change an existing transcript.
- **If a piece of the recording is silent**, the program does not send it to the speech model. On a silent piece,
  the model would only invent a phrase ("Grazie a tutti."). A piece is about 20 s long. It is silent if no tenth of
  a second in it is louder than -60 dBFS (decibels relative to full scale). Digital silence is an example. The
  program tells you how many seconds it left out as silence.
- **A pause with background noise** still goes to the model. In the measurements, Qwen3-ASR read through 2-15 s
  pauses and did not invent words.
- **The review page** checks the file on disk before it saves your version. If the file changed after you opened
  the page, the page first keeps a `.bak` copy of it. A new transcription or a second browser tab can cause such a
  change. If the save fails, your edits stay on the page and the page tells you why.

### Review a transcript

To open the review page, do one of these:

- On a Mac, double-click `Rivedi.command` and select the recording.
- Run `.venv/bin/python -m localtranscribe.review recording.m4a`.
- On Windows, run `.venv\Scripts\python.exe -m localtranscribe.review recording.m4a`.

The transcript is on the left and the audio is on the right. Use the page as follows:

1. Click a timestamp to move the audio to that point.
2. Edit the text.
3. Press *Salva* to write the `.md` back.

The page runs only on this computer (127.0.0.1).

If you transcribed the recording with `--correct`, the page highlights each suggested word:

- *Accetta* puts the suggested word into the text. The program writes it only when you press *Salva*.
- *Rifiuta* hides the suggestion.

The page offers a suggestion only while the word is still the original word.

**Quality of the suggestions.** The settings were tuned on 100 FLEURS clips. Then they were checked, without
changes, on 200 other clips. On those 200 clips, `--correct` made 25 suggestions:

- 11 fixed a mistake.
- 3 would have added a mistake.
- 11 changed a wrong word into another wrong word.

If you accept all of them, the word error rate goes down from 3.77 % to 3.57 %. On the tuning clips, there were
12 suggestions, 5 fixes and 1 new mistake. If you accept all of them, the rate goes from 3.51 % to
3.35 %.

The small model alone does worse than the speech model. So it only breaks ties between the speech model's own
guesses. A bit less than half of the suggestions are real fixes. That is why every suggestion waits for you.

## Models

LocalTranscribe has two presets of Qwen3-ASR. The hardware check selects one. `--model` overrides this choice.

| Preset | Model | Download | Memory (measured) | Word error rate, Italian | Speed |
|---|---|---|---|---|---|
| **best** | Qwen3-ASR-1.7B | NVIDIA 4.7 GB, Mac 1.6 GB (4-bit) | NVIDIA 4.0 GB (batch 1) to 5.2 GB (batch 8)<br>Mac 2.5 GB | NVIDIA **2.4 - 2.7 %**<br>Mac 4-bit **3.5 %** | RTX 6000 **26x** real time (1 hour in 2.3 min)<br>M2 Mac **6x** |
| **light** | Qwen3-ASR-0.6B | NVIDIA / CPU 1.9 GB, Mac 1.0 GB (8-bit) | NVIDIA 1.7 - 2.8 GB<br>CPU about 6 GB RAM<br>Mac 1.8 GB | NVIDIA **4.4 - 5.8 %**<br>Mac **5.7 %** | NVIDIA 27x<br>6-core CPU **2.6x** (1 hour in 23 min)<br>M2 Mac **19x** |

The hardware check and `--model auto` use the same rules to select the default:

- **NVIDIA:** *best* with 6 GB of video memory or more. *light* from about 3 GB. The program uses the CPU in these
  cases, and the check tells you why:
  - The card has less than about 3 GB of video memory.
  - The card is older than the GTX 16 / RTX 20 series.
  - The driver is older than 570.65 (Windows) / 570.26 (Linux).
- **Mac:** *best* (1.7B, 4-bit) with 8 GB of memory or more. Otherwise *light* (0.6B, 8-bit). On an 8 GB M2 Air,
  *best* makes about 40 % fewer mistakes than *light*. It still runs at about 6x real time. To use the 8-bit 1.7B
  model (2.9 % word error rate, 2.3x, 3.3 GB), give `--model mlx-community/Qwen3-ASR-1.7B-8bit`.
- **CPU only:** always *light*.

## Accuracy and speed

The table shows the word error rate (WER) on the FLEURS Italian test set. A lower WER is better. The WER ignores
case and punctuation. The test data is 100 read sentences (about 25 minutes) and a 24-minute recording made from
them.

| Model | 100 clips, set A | 100 clips, set B | 24-minute file | Speed | Peak GPU memory |
|---|---|---|---|---|---|
| best (1.7B), NVIDIA RTX 6000, float16 | **2.57 %** | 2.74 % | **2.44 %** | 25 - 26x | 5.2 GB (batch 8) |
| light (0.6B), NVIDIA RTX 6000, float16 | 4.38 % | 5.82 % | 5.18 % | 27x | 2.8 GB (batch 8) |
| light (0.6B), CPU, float32 | - | 4.68 % (first 10 clips) | - | 2.6x | about 6 GB RAM |
| best 4-bit (1.7B), M2 MacBook Air 8 GB, MLX | - | 3.51 % | - | 6.4x | 2.5 GB |
| 1.7B 8-bit, M2 MacBook Air 8 GB, MLX | - | 2.89 % | - | 2.3x | 3.3 GB |
| light 8-bit (0.6B), M2 MacBook Air 8 GB, MLX | - | 5.74 % | - | 18.7x | 1.8 GB |

`benchmark/download_fleurs.py` downloads set B, and the Mac self-test uses set B. So you can compare the Mac rows
directly with the NVIDIA rows in the set B column. Mac speed changes with memory pressure (6 - 12x measured for
*best*). On a real 6.6-minute phone call, the *best* model gave the same text as the previous version of the program.

## Hardware check

The setup scripts run the hardware check first, before they download large files. So you see a full disk or an
unsupported graphics card early, when it is still cheap to stop. The check uses only the Python standard library.
You can run it again at any time:

```
check_hardware.bat                 (Windows)
bash check_hardware.sh             (Mac; Linux best effort)
check_hardware.bat --json          machine-readable; add --yes to save the recommendation without asking
```

<details>
<summary>What the check reads and what it recommends</summary>

The check reads:

- the operating system and the processor
- the total and the available memory
- the free disk space where the program and the Hugging Face model cache are (`HF_HOME`, else
  `~/.cache/huggingface`)
- the NVIDIA cards, through `nvidia-smi` (memory, driver, architecture), or the Apple chip (model, memory, GPU cores,
  Rosetta)

| Your computer | Recommended | Notes |
|---|---|---|
| NVIDIA card with 6 GB or more | **best** on the GPU | batch size 1 - 8 from the free video memory |
| NVIDIA card with about 3 - 6 GB | **light** on the GPU | |
| NVIDIA card below 3 GB, older than GTX 16 / RTX 20 (compute capability < 7.5), or driver older than 570.65 / 570.26 | **light** on the CPU | the check says which reason applies |
| No NVIDIA card or driver | **light** on the CPU | warns below 8 GB of memory |
| Apple Silicon Mac, 8 GB or more | **best** (1.7B, 4-bit) | batch size 1<br>"close memory-heavy apps" on 8 GB<br>"keep it plugged in" on a MacBook Air |
| Intel Mac, macOS older than 14, or Python under Rosetta | nothing | the setup stops and explains what to do |
| Not enough free disk space | nothing | the setup stops and says which choice would fit |

The check saves the choice in `localtranscribe_settings.json` next to the program. Git ignores this file. The file
holds the preset, device, batch size, a hardware summary and the date. The launchers use it as their default.
`--model`, `--device` and `--batch-size` still override it.

If the file is missing or damaged, the program runs the check one time, prints the recommendation and continues. To
get the question again, delete the file.

</details>

## How it works

```
recording -> PyAV decoder (16 kHz mono) -> cut at quiet moments into ~20 s pieces -> Qwen3-ASR -> Markdown
                                           (same code on every platform)             NVIDIA: PyTorch + qwen-asr, float16
                                                                                     CPU:    PyTorch + qwen-asr, float32
                                                                                     Mac:    MLX (mlx-audio), 4-bit / 8-bit
--speakers N: Silero VAD -> WavLM voice embeddings -> clustering -> per-speaker turns -> the same speech model
```

- `localtranscribe/backends/` holds the MLX and PyTorch backends behind one interface. The program imports `mlx`
  only on a Mac.
- `localtranscribe/textutil.py` cuts the audio and cleans the text in the same way on every platform. The chunker
  is adapted from `qwen-asr`, with two changes:
  - Audio up to 25 s stays in one piece.
  - No cut leaves a final piece shorter than 3 s. Before this change, the model invented a word on half-second
    scraps. This raised the WER from 2.6 % to 3.1 %.
- `pipeline.transcribe_pieces` skips a piece if no tenth of a second in it is louder than -60 dBFS. The threshold
  is `SILENCE_DBFS` in `config.py`, and `None` turns it off. This silence gate applies to every backend, to
  `--speakers` and to `benchmark/bench.py`. For A/B runs, `benchmark/bench.py` has `--no-silence-gate`.
  Measurements without a model:
  - The quietest of 1,930 speech pieces reaches -30 dBFS. The pieces come from FLEURS, VoxPopuli and 12 speeches,
    also after G.711 and Opus 12 kbit/s.
  - Digital silence after those codecs stays at -81 dBFS or lower.
- `localtranscribe/precheck.py` is the hardware check. `devices.py` uses the same rules, so `--model auto` and the
  check always agree. `config.py` holds every model name and threshold.
- `--context` becomes the system prompt of the model on both backends.

## Development

```bash
python -m unittest discover -s tests -v                                    # unit tests, no model needed
python benchmark/download_fleurs.py --n 100                                # FLEURS clips into benchmark/data/ (~90 MB)
python benchmark/bench.py --data benchmark/data/fleurs_it --model best     # WER, speed, peak memory
python benchmark/make_longform.py --data benchmark/data/fleurs_it --out long.wav
./transcribe.sh long.wav --out-dir out && python benchmark/score_long.py --data benchmark/data/fleurs_it --hyp out/long.md
python benchmark/diar_test.py --data benchmark/data/fleurs_it              # two-speaker check of --speakers
bash mac_selftest.sh                                                       # Mac: all presets and speakers, about 25 min; writes mac_selftest_report.txt
```

On Windows, use `.venv\Scripts\python.exe` and `transcribe.bat`.

The self-test report contains no personal data. The self-test removes the home folder, user name and host name from
the report. The Mac rows in the tables above come from this report.

To install manually on Linux:

1. Run `uv venv --python 3.11 .venv`.
2. Install the packages. `--torch-backend` needs uv 0.7.14 or newer.
   - NVIDIA: `uv pip install --torch-backend cu128 -r requirements-windows.txt`
   - CPU: `uv pip install --torch-backend cpu -r requirements-cpu.txt`
3. Run `PYTHONPATH=. .venv/bin/python -m localtranscribe file.m4a`.

## Roadmap

Done:

- the review page
- the AI suggestions, measured on clips that they were not tuned on, and offered from the double-click launcher
- the silence gate, which skips silent pieces

Next:

- measure on real calls and lectures
- cover more mistakes with fewer false alarms

Details are in [ROADMAP.md](ROADMAP.md).

## Contributing

Issues and pull requests are welcome. Before you open one:

1. Run the unit tests.
2. Use the standard library where possible.
3. If you want a new dependency, discuss it first. Do not add it by default.
4. If you change the chunker, the prompts or the model presets, include `benchmark/bench.py` numbers from before and
   after the change.

A rule checker written in [Bend](https://github.com/bendlang/bend) guards the AI suggestions. It has four files:

| File | What it is |
|---|---|
| `LAWS.bend` | The rules. This file is the maintainer's specification. Propose changes to it in an issue first. |
| `guard.bend` | The checker. |
| `PROOF.bend` | The proof that the checker obeys every rule. |
| `guard_cli.bend` | The small command-line wrapper that Python calls. |

If you change one of these files, do these steps:

1. Run `bend PROOF.bend`. It must print `ALL PROOFS CHECK`.
2. Run `bash build_guard.sh`. It rebuilds `bin/guard-macos-arm64` only when the proofs pass.

At run time, an independent Python implementation of the same four rules (`localtranscribe/guard.py`) checks the
answer of the checker again. If the two disagree, there are no suggestions. The proof stays the specification.

**Windows checker (maintainers only).** The repository includes `bin/guard-windows-x64.exe`, like the Mac binary.
Bend 2 does not support Windows, so the Windows build uses an unofficial bridge. `build_guard_windows.sh` takes the
compiler out of the official Linux release and runs it on Bun. It adds a small path preload and POSIX shims from
`win/`. The build never edits the proofs or the generated C.

To build the Windows checker:

1. Get the pinned tools with `BEND_TOOLCHAIN=<folder> bash win/fetch_toolchain.sh`. You get Bun 1.4.2, zig 0.16.0
   and Bend 2.0.32, all sha256-checked.
2. Make sure that a Python 3 is on `PATH`, or set `PYTHON=...`.
3. From Git Bash, run `BEND_TOOLCHAIN=<folder> bash build_guard_windows.sh`.

`win/compat/abi_fix.h` works around a clang `musttail` bug on Windows x64. `win/compat/musttail_repro.c` reproduces
the bug.

The exe sha256 is `913dcf5362f71805db9faf02885aaad1d3d34ba924714cae5bf404515b3bc219`. To verify it, rebuild with
`build_guard_windows.sh` (tools: `win/fetch_toolchain.sh`) and compare.

## Troubleshooting

- **"has not been downloaded yet"**: Run the setup script again while you are online.
- **Windows: "Could not create the Python environment"**: There is no internet, or a proxy blocks `github.com`. uv
  downloads Python from there. Install Python 3.11 first. Then run the setup again.
- **Out of memory**: Use `--model light` or `--batch-size 1`, or close other programs.
- **"Not enough free disk space"** during setup: Do one of these:
  - Free some space. The message tells you how much.
  - Select the smaller model (`setup_windows.bat light`).
  - Move the model cache with the `HF_HOME` environment variable.
- **The program does not use the NVIDIA card**: Run `check_hardware.bat`. It tells you if the card is too old for
  the GPU build of PyTorch, or if you must update the driver from nvidia.com/drivers. Then run the setup again.
- **"no speech was recognised"**: The recording is silent, or the model heard no speech. The program writes nothing
  and does not change an existing transcript. A "[note] ... left out as silence" line means that pieces of about
  20 s were quieter than -60 dBFS. These pieces did not go to the model. If you can hear speech there, make the file
  louder and run it again.
- **"could not read audio (no audio track found)"**: The file has no sound track, for example a silent video.
- **"could not replace ... The transcript was saved as ..."**: The `.md` is open in another program. On Windows,
  Word locks it. The new transcript is in `<recording>.new-<date>.md`. Close the other program and keep the file
  that you want.
- **The run stopped half-way** (crash, empty battery, Ctrl+C): The text done so far is in `<recording>.partial.md`.
  A new run replaces the partial file. If you corrected the partial file by hand, copy it first. Then run the
  transcription again to get the whole text.
- **Mac: "cannot be opened because the developer cannot be verified"**: Right-click the file and choose **Open**.

## Updating and uninstalling

**Update.**

- If you cloned the repository, run `git pull`. Then run the setup script again.
- If you use the ZIP, do these steps:
  1. Download the new ZIP and unpack it.
  2. Run the setup script in the new folder.
  3. Delete the old folder.

The setup does not download the models again, because they are outside the program folder (see below). Your
transcripts are next to your recordings, not in the program folder.

**Uninstall.** The setup installs nothing system-wide except the small tool uv. Delete these items:

- The program folder. It contains `.venv`: about 1 GB on a Mac and up to 5 GB with the NVIDIA packages.
- The models (2 - 7 GB) in the Hugging Face cache. The cache is `~/.cache/huggingface/hub` on Mac and Linux, and
  `%USERPROFILE%\.cache\huggingface\hub` on Windows. Delete the folders named `models--mlx-community--Qwen3-...`,
  `models--Qwen--Qwen3-ASR-...` and `models--microsoft--wavlm-base-plus-sv`.
- Optional, only if the setup installed uv for you (you did not use uv before): uv and its files. Remove the uv
  download cache and the Python that uv downloaded, with `uv cache clean` and `uv python uninstall 3.11`. Then
  delete `uv` itself (`~/.local/bin/uv` on a Mac).

## Privacy

All work runs on your computer. The setup downloads the models from Hugging Face and the Python packages from PyPI.
After the setup, the launchers set `HF_HUB_OFFLINE=1`. So the audio and the transcripts never leave the computer.

A transcript can have related files next to it:

- `<recording>.review.json`: word confidence and suggestions. This file repeats the whole text.
- `<recording>.bak-<date>.md`: earlier versions.
- `.partial.md` or `.new-<date>.md`: after an interrupted or blocked save.

When you delete or share a transcript, remember these files too.

## License

LocalTranscribe is released under the [Apache License 2.0](LICENSE). This repository also contains code from other
projects. The notices are in [`NOTICE`](NOTICE):

- The audio chunker and repetition filter in `localtranscribe/textutil.py` are adapted from
  [`qwen-asr`](https://github.com/QwenLM/Qwen3-ASR) (Apache-2.0, copyright The Alibaba Qwen team).
  [How it works](#how-it-works) describes the changes.
- The decoding loop that records word confidence (`_single_recorded` in `localtranscribe/backends/mlx_qwen.py`) is
  adapted from [`mlx-audio`](https://github.com/Blaizzy/mlx-audio) (MIT, copyright Prince Canuma). It is also
  adapted from the [`mlx-lm`](https://github.com/ml-explore/mlx-lm) generation loop that `mlx-audio` includes (MIT,
  copyright Apple Inc.).
- `bin/guard-macos-arm64` is compiled from `guard_cli.bend` with [Bend](https://github.com/bendlang/bend)
  (Apache-2.0, copyright HigherOrderCO) and contains Bend's runtime code.
- `bin/guard-windows-x64.exe` is compiled from the same sources and contains the same Bend runtime code. It also
  contains code that [Zig](https://ziglang.org) links in statically: Zig's runtime (MIT), the mingw-w64 C runtime
  and winpthreads. The details are in [`NOTICE`](NOTICE).

Components that the program downloads and uses:

| Component | License |
|---|---|
| Qwen3-ASR models (`Qwen/Qwen3-ASR-1.7B`, `-0.6B`) and the `qwen-asr` package | Apache-2.0 |
| MLX conversions (`mlx-community/Qwen3-ASR-...`) | Apache-2.0 (as the originals) |
| Correction model `mlx-community/Qwen3-0.6B-4bit` (used by `--correct`, a conversion of `Qwen/Qwen3-0.6B`) | Apache-2.0 |
| MLX, `mlx-audio`, `mlx-lm` | MIT |
| Silero VAD | MIT |
| WavLM speaker model `microsoft/wavlm-base-plus-sv` (used by `--speakers`) | The model card points to the license of Microsoft's UniSpeech repository (CC BY-SA 3.0). The WavLM code (microsoft/unilm) is MIT. The weights are not included here. The setup downloads them from Hugging Face. |
| PyTorch, transformers, scikit-learn, PyAV | BSD / Apache-2.0 (PyAV bundles FFmpeg libraries under their own LGPL/GPL terms) |
| uv (installed by the setup scripts) | MIT or Apache-2.0 |
| Bend (only to rebuild the rule checker, not needed to use the program) | Apache-2.0 |
| FLEURS test clips (tests and benchmark) | CC-BY-4.0, Google. Attribution in [`tests/README.md`](tests/README.md) |

## Acknowledgements

Thanks to:

- the Qwen team for Qwen3-ASR and the `qwen-asr` reference implementation
- the MLX and `mlx-audio` maintainers and the `mlx-community` for the Apple Silicon conversions
- Silero for the VAD (voice activity detection) model
- Microsoft for WavLM
- HigherOrderCO for Bend
- Google for FLEURS

---

## Guida rapida (italiano)

**A cosa serve.** Trascrive registrazioni audio e video in italiano (interviste, telefonate, riunioni, lezioni) in un
file di testo, **direttamente sul tuo computer**: dopo l'installazione non serve internet e nessun file esce dal
computer.

**Requisiti.** Mac con chip Apple (M1 o successivi) e macOS 14 o più recente (i Mac Intel non sono supportati),
oppure PC Windows (meglio con scheda video NVIDIA). Circa 5 GB di spazio libero su Mac.

**Installazione su Mac (una volta sola)**

1. Scarica il progetto: [questo link](https://github.com/Blimp3/LocalTransciber/archive/refs/heads/main.zip) scarica
   uno ZIP; aprilo e sposta la cartella in Documenti. (In alternativa: `git clone https://github.com/Blimp3/LocalTransciber.git`;
   la prima volta il Mac può chiedere di installare gli "strumenti da riga di comando": conferma con **Installa**.)
2. Apri **Terminale** (Cmd+Spazio, scrivi *Terminale*). Scrivi `cd ` (con lo spazio), trascina dentro la cartella del
   progetto e premi Invio.
3. Scrivi `bash setup_mac.sh` e premi Invio. Ci vogliono 10-20 minuti: scarica programmi e modelli (circa 3,5 GB in tutto).
   Il programma controlla il computer (memoria, spazio libero) e propone il modello adatto: premi Invio per accettare.
4. Per sapere quanta memoria ha il tuo Mac: menu Apple  > **Informazioni su questo Mac**. Con 8 GB o più viene usato
   il modello più accurato (misurato su un MacBook Air M2 da 8 GB).

**Uso su Mac.** Doppio clic su **`Trascrivi.command`**. La prima volta, se macOS non lo apre: clic destro sul file,
**Apri**, poi ancora **Apri**. Scegli uno o più file audio o video; puoi anche far separare due persone che parlano
(telefonate). Il testo viene salvato accanto a ogni file (stesso nome, estensione `.md`) e compare nel Finder. Se
trascrivi di nuovo lo stesso file, il testo precedente non va perso: resta come `<nome>.bak-<data>.md`. Se il
programma si interrompe a metà (batteria scarica, finestra chiusa), il testo fatto fino a quel punto è in
`<nome>.partial.md`; se il file `.md` è aperto in un altro programma e non si può sostituire, il nuovo testo viene
salvato come `<nome>.new-<data>.md` (consiglio: chiudi la trascrizione nell'editor prima di trascrivere di nuovo lo
stesso file o di salvare dalla pagina di revisione). Per il
testo unico il programma chiede anche se vuoi i **suggerimenti AI** per le parole incerte: un piccolo modello sul tuo
Mac propone correzioni (ci vuole circa il 10 % di tempo in più), e nulla cambia finché non le accetti tu.

**Rivedere il testo.** Doppio clic su **`Rivedi.command`** e scegli la registrazione: nel browser vedi il testo a sinistra
e l'audio a destra. Clicca su un orario per ascoltare quel punto, correggi il testo e premi **Salva**. Se hai chiesto
i suggerimenti AI, le parole con un suggerimento sono evidenziate: **Accetta** o **Rifiuta**. Nulla cambia finché non
premi **Salva**. Poco meno della metà dei suggerimenti è una vera correzione: controlla sempre ascoltando.

**Consigli per il MacBook Air.** Tienilo **collegato all'alimentazione** per i file lunghi (non ha ventola e rallenta
quando si scalda). Con 8 GB di memoria **chiudi i programmi pesanti** (browser con molte schede, videochiamate) prima
di trascrivere.

**Windows.** Doppio clic su `setup_windows.bat` (una volta sola), poi **trascina i file audio su `transcribe.bat`**.

**Controllo del computer.** Puoi ripetere il controllo quando vuoi con `check_hardware.bat` (Windows) o
`bash check_hardware.sh` (Mac).

**Suggerimento.** Dalla riga di comando puoi indicare nomi e termini tecnici che compaiono nella registrazione con
`--context "Mario Rossi, Politecnico di Milano"`: la loro trascrizione sarà più precisa. Per separare i parlanti
usa `--speakers 2`: il risultato è `[00:01:23] Parlante 1: ...` e `Parlante 2: ...`.

**Aggiornare.** Scarica il nuovo ZIP, aprilo ed esegui di nuovo `bash setup_mac.sh` nella nuova cartella, poi elimina
la vecchia. I modelli non vengono scaricati di nuovo e le tue trascrizioni restano accanto alle registrazioni.

**Disinstallare.** Elimina la cartella del programma e, per liberare altri 2-3 GB, i modelli nella cartella
`~/.cache/huggingface/hub` (nel Finder: **Vai > Vai alla cartella...**).

**Privacy.** Tutto avviene sul tuo computer. Internet serve solo durante l'installazione per scaricare i modelli.
Accanto a una trascrizione possono esserci file collegati (`.review.json`, `.bak-<data>.md`): se elimini o condividi
una trascrizione, ricordati anche di questi.

---

<sub>The repository name is <b>LocalTransc<i>i</i>ber</b>: an <i>r</i> went missing when the repository was created.
The name stays that way. The program is LocalTranscribe.</sub>
