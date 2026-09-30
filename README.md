<h1 align="center">LocalTranscribe</h1>

<p align="center">
  Italian speech-to-text that runs entirely on your own computer.<br>
  Drop in a recording, get a Markdown transcript next to it. Nothing is uploaded.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
  <img alt="Platforms" src="https://img.shields.io/badge/platform-macOS%20Apple%20Silicon%20%7C%20Windows%20NVIDIA%20%7C%20CPU-lightgrey">
  <img alt="Python 3.11" src="https://img.shields.io/badge/python-3.11-blue">
  <img alt="Model: Qwen3-ASR" src="https://img.shields.io/badge/model-Qwen3--ASR-orange">
  <img alt="Works offline" src="https://img.shields.io/badge/works-offline-brightgreen">
</p>

> **Italiano:** la [Guida rapida](#guida-rapida-italiano) è in fondo alla pagina.

LocalTranscribe turns Italian audio and video recordings (interviews, phone calls, meetings, lectures) into text with
the open speech model [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR). Models are downloaded once during setup;
after that everything runs offline on your Apple Silicon Mac, your NVIDIA GPU, or a plain CPU.

## Features

- **Local and offline.** After setup the model library is switched to offline mode. No account, no cloud service,
  no telemetry. Transcripts are plain Markdown files you control.
- **Any audio or video file.** wav, mp3, m4a, ogg, opus, flac, mp4, mkv, webm, ... FFmpeg is bundled; nothing to install.
- **Runs on what you have.** Apple Silicon (MLX on the Apple GPU), NVIDIA GPUs (PyTorch + CUDA), or CPU only.
- **Picks the model for your machine.** A [hardware check](#hardware-check) recommends *best* or *light*, explains
  why, and remembers the choice.
- **Speaker separation.** `--speakers 2` turns a two-person call into `[00:01:23] Parlante 1: ...` lines.
- **Context hints.** `--context "Mario Rossi, LoRaWAN"` helps names and jargon come out spelled right.
- **Word confidence** (Mac). `--confidence` saves each word's probability, flags the unsure ones and gives them
  whole-word alternatives from the speech model itself.
- **AI suggestions, checked and confirmed** (Mac). `--correct` lets a small local model choose among those
  alternatives at unsure words. A rule checker, proven correct in [Bend](https://github.com/HigherOrderCO/Bend),
  rejects anything else, and you accept or reject each suggestion on the review page. Nothing changes on its own.
- **Review page.** A local page with the editable transcript beside the audio; click a timestamp to jump there.
- **Never loses a transcript.** A new run keeps the transcript you edited as a backup, a crash or a closed laptop
  leaves the text done so far on disk, and a file that cannot be replaced is saved under another name
  (see [Your transcripts are safe](#your-transcripts-are-safe)).
- **Double-click launchers** for people who never open a terminal: `Trascrivi.command` and `Rivedi.command` on
  Mac, drag-and-drop onto `transcribe.bat` on Windows.

## Requirements

| | Apple Silicon Mac | Windows + NVIDIA GPU | Any PC, CPU only |
|---|---|---|---|
| System | macOS 14 (Sonoma) or newer, M1 or newer. Intel Macs are not supported. | Windows 10/11, NVIDIA driver 570.65 or newer, GeForce GTX 16 / RTX 20 series (2018) or newer | Windows 10/11 |
| GPU memory | 8 GB unified memory or more for *best* | 6 GB for *best*, 3 GB for *light* | - |
| Free disk space | about 5 GB (6 GB with both models) | about 12 GB (10 GB with one model) | about 4 GB |
| Internet | setup only | setup only | setup only |

Linux works with a manual install (see [Development](#development)) but is not officially supported.

## Quick start

Get the code: [download the ZIP](https://github.com/Blimp3/LocalTransciber/archive/refs/heads/main.zip) and unpack it
somewhere permanent (for example your Documents folder), or clone it:

```bash
git clone https://github.com/Blimp3/LocalTransciber.git
cd LocalTransciber
```

### macOS (Apple Silicon)

```bash
bash setup_mac.sh            # 10-20 minutes, once
./transcribe.sh recording.m4a
```

`setup_mac.sh` checks that the Mac is supported, installs the small helper tool [uv](https://docs.astral.sh/uv/) if
it is missing (official installer; it says so before doing it), runs the hardware check (press Enter to accept the
recommended model), creates a Python environment in `.venv`, installs about 1 GB of packages and downloads about 2.4 GB
of models. `bash setup_mac.sh light` (or `best`, or `both`) skips the question.

Prefer not to use Terminal? **Double-click `Trascrivi.command`** in Finder, pick one or more recordings, and the `.md`
transcripts appear next to them. For a single-speaker transcript it also offers the AI suggestions (`--correct`),
which you then accept or reject with `Rivedi.command`. The first time, macOS may refuse to open a downloaded script:
right-click it, choose **Open**, then **Open** again (only once).

### Windows (NVIDIA GPU or CPU)

1. Double-click **`setup_windows.bat`**. It installs uv if needed, runs the hardware check, creates `.venv`, installs
   the packages (1 - 5 GB; the NVIDIA build is the big one) and downloads the models. It falls back to the CPU-only
   packages when no usable NVIDIA card is found.
2. **Drag audio or video files onto `transcribe.bat`.**

From a command prompt: `setup_windows.bat light|both|cpu [--yes]` and `transcribe.bat recording.m4a [options]`.

If setup stops because of a network problem, run it again; it continues where it left off.

## Usage

```bash
./transcribe.sh interview.m4a --context "Mario Rossi, Politecnico di Milano, LoRaWAN"   # Mac
transcribe.bat  call.m4a --speakers 2                                                  # Windows
```

The transcript `<recording>.md` is written next to the recording (or into `--out-dir`). Long recordings are cut at
quiet moments into pieces of about 20 seconds; each piece becomes a paragraph that starts with its `[hh:mm:ss]`
timestamp. With `--speakers`, each turn starts with the timestamp and the speaker's label.

| Option | What it does |
|---|---|
| `--context "names, terms"` | names and technical words that occur in the recording; helps spell them correctly |
| `--speakers 2` | separate the speakers: `[00:01:23] Parlante 1: ...`. Reliable on calls of a few minutes or more; may merge very short replies such as "sì" into the other speaker |
| `--out-dir folder` | write the transcripts to this folder instead of next to the recordings |
| `--language auto` | detect the language instead of assuming Italian (or `English`, `French`, `Spanish`, ...) |
| `--model best`, `light`, or a Hugging Face repo id | choose the model (default: the choice saved by the hardware check) |
| `--device auto`, `cuda`, `mps`, `cpu` | choose the hardware (default: the saved choice). On a Mac `mps` is the Apple GPU |
| `--batch-size N` | pieces processed together (default: the saved choice, lowered if memory is tight; lower = less memory) |
| `--confidence` | also save each word-piece's confidence and the model's alternatives, plus per-word `"words"` with an `"unsure"` flag and whole-word `"cands"` for the unsure ones, to `<recording>.review.json` (Apple Silicon only for now) |
| `--correct` | suggest fixes for unsure words: a small local model (Qwen3-0.6B, downloaded at setup) chooses among the speech model's own alternatives, the Bend checker accepts or rejects each choice, and the accepted ones go to `<recording>.review.json` for the review page. The `.md` is not changed. About 10% slower; implies `--confidence`; Apple Silicon only for now |
| `--stats` | print the device, model, speed and peak memory at the end |
| `--chunk 20` | seconds per piece. Leave at 20: longer pieces are measurably less accurate |

### Your transcripts are safe

A transcript can take an hour to make and longer to correct, so the program never throws one away:

- **Transcribing the same recording again** keeps your edits: if the `.md` changed, the old one stays as
  `<recording>.bak-<date>.md`.
- **Before the model loads**, the program checks that every recording exists and that the output folder can be
  written, so a locked folder is reported in seconds, not after the work is done.
- **While a recording is being transcribed**, the text done so far is in `<recording>.partial.md` (it appears once
  the first pieces are done, usually within a minute, and grows every few pieces). If the computer crashes, the
  battery dies or you press Ctrl+C, that file stays; it is removed when the transcript is complete.
- **If the `.md` cannot be replaced** (on Windows: it is open in Word, marked read-only, or an antivirus or sync
  program is holding it), the program tries three times, then saves the new transcript as `<recording>.new-<date>.md` next to it (or in
  your home folder if that fails too) and says so. The old transcript is not touched.
- **If the model returns no text at all**, nothing is written: the program says that no speech was recognised and
  leaves any existing transcript alone. (On pure silence the speech model can still invent a short phrase; skipping
  silence is on the [roadmap](ROADMAP.md).)
- **The review page** keeps a `.bak` copy of the file on disk if it changed after the page was opened (a new
  transcription, a second browser tab) before it saves your version. If saving fails, your edits stay on the page
  and it says why.

### Review a transcript

Double-click `Rivedi.command` (Mac) and pick the recording, or run
`.venv/bin/python -m localtranscribe.review recording.m4a` (Windows:
`.venv\Scripts\python.exe -m localtranscribe.review recording.m4a`). The transcript is on the left and the audio on
the right; click a timestamp to jump the audio there, edit the text, and press *Salva* to write the `.md` back. The
page runs only on this computer (127.0.0.1).

If the recording was transcribed with `--correct`, each suggested word is highlighted: *Accetta* swaps it in (still
written only when you press *Salva*), *Rifiuta* hides the suggestion. A suggestion is offered only while the word is
still the original one.

How good are the suggestions? The settings were tuned on 100 FLEURS clips and then checked, unchanged, on 200 other
clips. On those 200, `--correct` made 25 suggestions: 11 fixed a mistake, 3 would have introduced one, and 11 changed
a wrong word into another wrong word. Accepting all of them would lower the word error rate from 3.77 % to 3.57 %
(on the tuning clips: 12 suggestions, 5 fixes, 1 new mistake, 3.51 % to 3.35 %). On its own the small model does
worse than the speech model, so it only breaks ties between the speech model's own guesses, and a bit less than half
of the suggestions are real fixes: that is why every suggestion waits for you.

## Models

Two presets of Qwen3-ASR. The hardware check picks one; `--model` overrides it.

| Preset | Model | Download | Memory (measured) | Word error rate, Italian | Speed |
|---|---|---|---|---|---|
| **best** | Qwen3-ASR-1.7B | NVIDIA 4.7 GB, Mac 1.6 GB (4-bit) | NVIDIA 4.0 GB (batch 1) to 5.2 GB (batch 8); Mac 2.5 GB | NVIDIA **2.4 - 2.7 %**; Mac 4-bit **3.5 %** | RTX 6000 **26x** real time (1 hour in 2.3 min); M2 Mac **6x** |
| **light** | Qwen3-ASR-0.6B | NVIDIA / CPU 1.9 GB, Mac 1.0 GB (8-bit) | NVIDIA 1.7 - 2.8 GB; CPU about 6 GB RAM; Mac 1.8 GB | NVIDIA **4.4 - 5.8 %**; Mac **5.7 %** | NVIDIA 27x; 6-core CPU **2.6x** (1 hour in 23 min); M2 Mac **19x** |

How the default is chosen (the hardware check and `--model auto` follow the same rules):

- **NVIDIA:** *best* with 6 GB of video memory or more, *light* from about 3 GB. Below that, on cards older than the
  GTX 16 / RTX 20 series, or with a driver older than 570.65 (Windows) / 570.26 (Linux), the CPU is used and the check
  says why.
- **Mac:** *best* (1.7B, 4-bit) with 8 GB of memory or more, otherwise *light* (0.6B, 8-bit). On an 8 GB M2 Air
  *best* makes about 40 % fewer mistakes than *light* and still runs at about 6x real time. The 8-bit 1.7B model
  (2.9 % WER, 2.3x, 3.3 GB) is available with `--model mlx-community/Qwen3-ASR-1.7B-8bit`.
- **CPU only:** always *light*.

## Accuracy and speed

Word error rate (WER, lower is better; case and punctuation ignored) on the FLEURS Italian test set: 100 read
sentences (about 25 minutes) and a 24-minute recording built from them.

| Model | 100 clips, set A | 100 clips, set B | 24-minute file | Speed | Peak GPU memory |
|---|---|---|---|---|---|
| best (1.7B), NVIDIA RTX 6000, float16 | **2.57 %** | 2.74 % | **2.44 %** | 25 - 26x | 5.2 GB (batch 8) |
| light (0.6B), NVIDIA RTX 6000, float16 | 4.38 % | 5.82 % | 5.18 % | 27x | 2.8 GB (batch 8) |
| light (0.6B), CPU, float32 | - | 4.68 % (first 10 clips) | - | 2.6x | about 6 GB RAM |
| best 4-bit (1.7B), M2 MacBook Air 8 GB, MLX | - | 3.51 % | - | 6.4x | 2.5 GB |
| 1.7B 8-bit, M2 MacBook Air 8 GB, MLX | - | 2.89 % | - | 2.3x | 3.3 GB |
| light 8-bit (0.6B), M2 MacBook Air 8 GB, MLX | - | 5.74 % | - | 18.7x | 1.8 GB |

Set B is what `benchmark/download_fleurs.py` downloads and what the Mac self-test uses, so the Mac rows compare directly
with the NVIDIA set B column. Mac speed varies with memory pressure (best measured 6 - 12x). On a real 6.6-minute phone
call the best model produced the same text as the previous version of this tool.

## Hardware check

The setup scripts run it first, before anything big is downloaded, so a full disk or an unsupported graphics card
shows up while it is still cheap to stop. It uses only Python's standard library. Repeat it any time:

```
check_hardware.bat                 (Windows)
bash check_hardware.sh             (Mac; Linux best effort)
check_hardware.bat --json          machine-readable; add --yes to save the recommendation without asking
```

<details>
<summary>What it looks at and what it recommends</summary>

It reads the operating system, the processor, total and available memory, the free disk space where the program and the
Hugging Face model cache live (`HF_HOME`, else `~/.cache/huggingface`), the NVIDIA cards (via `nvidia-smi`: memory,
driver, architecture) or the Apple chip (model, memory, GPU cores, Rosetta).

| Your computer | Recommended | Notes |
|---|---|---|
| NVIDIA card with 6 GB or more | **best** on the GPU | batch size 1 - 8 from the free video memory |
| NVIDIA card with about 3 - 6 GB | **light** on the GPU | |
| NVIDIA card below 3 GB, older than GTX 16 / RTX 20 (compute capability < 7.5), or driver older than 570.65 / 570.26 | **light** on the CPU | the check says which reason applies |
| No NVIDIA card or driver | **light** on the CPU | warns below 8 GB of memory |
| Apple Silicon Mac, 8 GB or more | **best** (1.7B, 4-bit) | batch size 1; "close memory-heavy apps" on 8 GB; "keep it plugged in" on a MacBook Air |
| Intel Mac, macOS older than 14, or Python under Rosetta | nothing | the setup stops and explains what to do |
| Not enough free disk space | nothing | the setup stops and says which choice would fit |

The saved choice lives in `localtranscribe_settings.json` next to the program (git-ignored): preset, device, batch size,
a hardware summary and the date. The launchers use it as their default; `--model`, `--device` and `--batch-size` still
win. If the file is missing or damaged the tool runs the check once, prints the recommendation and carries on. Delete
the file to be asked again.

</details>

## How it works

```
recording -> PyAV decoder (16 kHz mono) -> cut at quiet moments into ~20 s pieces -> Qwen3-ASR -> Markdown
                                           (same code on every platform)             NVIDIA: PyTorch + qwen-asr, float16
                                                                                     CPU:    PyTorch + qwen-asr, float32
                                                                                     Mac:    MLX (mlx-audio), 4-bit / 8-bit
--speakers N: Silero VAD -> WavLM voice embeddings -> clustering -> per-speaker turns -> the same speech model
```

- `localtranscribe/backends/` holds the MLX and PyTorch backends behind one interface; `mlx` is imported only on a Mac.
- `localtranscribe/textutil.py` cuts the audio and cleans the text identically on every platform. The chunker is
  adapted from `qwen-asr` with two changes: audio up to 25 s stays in one piece, and no cut leaves a final piece
  shorter than 3 s (the model used to invent a word on half-second scraps, raising WER from 2.6 % to 3.1 %).
- `localtranscribe/precheck.py` is the hardware check; `devices.py` applies the same rules, so `--model auto` and the
  check always agree. `config.py` holds every model name and threshold.
- `--context` becomes the model's system prompt on both backends.

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

On Windows use `.venv\Scripts\python.exe` and `transcribe.bat`. The self-test report contains no personal data (home
folder, user name and host name are removed); it is what the Mac rows above come from.

Linux (manual): `uv venv --python 3.11 .venv`, then `uv pip install --index-strategy unsafe-best-match -r
requirements-windows.txt` (NVIDIA) or `-r requirements-cpu.txt`, then `PYTHONPATH=. .venv/bin/python -m localtranscribe file.m4a`.

## Roadmap

The review page and the AI suggestions are done, measured on clips they were not tuned on, and offered from the
double-click launcher. Next: measure on real calls and lectures, reach more mistakes with fewer false alarms, and
skip silent stretches before they reach the model. Details in [ROADMAP.md](ROADMAP.md).

## Contributing

Issues and pull requests are welcome. Before opening one: run the unit tests, keep to the standard library where
possible (new dependencies are a discussion, not a default), and if you touch the chunker, the prompts or the model
presets, include `benchmark/bench.py` numbers before and after.

The AI suggestions are guarded by a rule checker written in [Bend](https://github.com/HigherOrderCO/Bend):
`LAWS.bend` states the rules (the maintainer's specification: propose changes to it in an issue first), `guard.bend`
is the checker, `PROOF.bend` proves that the checker obeys every rule, and `guard_cli.bend` is the small
command-line wrapper that Python calls. If you touch any of them, run `bend PROOF.bend` (it must print
`ALL PROOFS CHECK`), then `bash build_guard.sh`, which rebuilds `bin/guard-macos-arm64` only when the proofs pass.

## Troubleshooting

- **"has not been downloaded yet"**: run the setup script again while online.
- **Windows: "Could not create the Python environment"**: no internet, or a proxy blocking `github.com` (uv downloads
  Python from there). Install Python 3.11 first, then run the setup again.
- **Out of memory**: use `--model light`, or `--batch-size 1`, or close other programs.
- **"Not enough free disk space"** during setup: free some space (the message says how much), choose the smaller
  model (`setup_windows.bat light`), or move the model cache with the `HF_HOME` environment variable.
- **The NVIDIA card is not used**: run `check_hardware.bat`; it says whether the card is too old for the GPU build of
  PyTorch or the driver must be updated from nvidia.com/drivers. Then run the setup again.
- **"no speech was recognised"**: the model returned no text (a silent recording). Nothing is written and an
  existing transcript is left alone.
- **"could not read audio (no audio track found)"**: the file has no sound track, for example a silent video.
- **"could not replace ... The transcript was saved as ..."**: the `.md` is open in another program (on Windows,
  Word locks it). The new transcript is in `<recording>.new-<date>.md`; close the other program and keep the file
  you want.
- **The run stopped half-way** (crash, empty battery, Ctrl+C): the text done so far is in `<recording>.partial.md`.
  Run the transcription again for the whole text; the new run replaces the partial file, so copy it first if you
  corrected it by hand.
- **Mac: "cannot be opened because the developer cannot be verified"**: right-click the file, choose **Open**.

## Updating and uninstalling

**Update.** With a clone: `git pull`, then run the setup script again. With the ZIP: download the new ZIP, unpack it
and run the setup script in the new folder, then delete the old folder. The models are not downloaded again (they
live outside the program folder, see below) and your transcripts are next to your recordings, not in the program
folder.

**Uninstall.** Nothing is installed system-wide except the small tool uv. Delete:

- the program folder (it contains `.venv`, about 1 GB on a Mac and up to 5 GB with the NVIDIA packages);
- the models, 2 - 7 GB, in the Hugging Face cache: `~/.cache/huggingface/hub` on Mac and Linux,
  `%USERPROFILE%\.cache\huggingface\hub` on Windows (the folders named `models--mlx-community--Qwen3-...`,
  `models--Qwen--Qwen3-ASR-...` and `models--microsoft--wavlm-base-plus-sv`);
- optionally, and only if the setup installed uv for you (you did not use uv before): its download cache and
  the Python it fetched, with `uv cache clean` and `uv python uninstall 3.11`, then `uv` itself (`~/.local/bin/uv`
  on a Mac).

## Privacy

Everything runs on your computer. Setup downloads the models from Hugging Face and the Python packages from PyPI;
afterwards the launchers set `HF_HUB_OFFLINE=1`, so neither the audio nor the transcripts ever leave the machine.

A transcript can have companion files next to it: `<recording>.review.json` (word confidence and suggestions; it
repeats the whole text), `<recording>.bak-<date>.md` (earlier versions), and `.partial.md` or `.new-<date>.md` after
an interrupted or blocked save. When you delete or share a transcript, remember these too.

## License

LocalTranscribe is released under the [Apache License 2.0](LICENSE). Code from other projects that is part of this
repository (the notices are in [`NOTICE`](NOTICE)):

- The audio chunker and repetition filter in `localtranscribe/textutil.py` are adapted from
  [`qwen-asr`](https://github.com/QwenLM/Qwen3-ASR) (Apache-2.0, copyright The Alibaba Qwen team), with the changes
  described above.
- The decoding loop that records word confidence (`_single_recorded` in `localtranscribe/backends/mlx_qwen.py`) is
  adapted from [`mlx-audio`](https://github.com/Blaizzy/mlx-audio) (MIT, copyright Prince Canuma) and the
  [`mlx-lm`](https://github.com/ml-explore/mlx-lm) generation loop it includes (MIT, copyright Apple Inc.).
- `bin/guard-macos-arm64` is compiled from `guard_cli.bend` with [Bend](https://github.com/HigherOrderCO/Bend)
  (Apache-2.0, copyright HigherOrderCO) and contains Bend's runtime code.

What it downloads and uses:

| Component | License |
|---|---|
| Qwen3-ASR models (`Qwen/Qwen3-ASR-1.7B`, `-0.6B`) and the `qwen-asr` package | Apache-2.0 |
| MLX conversions (`mlx-community/Qwen3-ASR-...`) | Apache-2.0 (as the originals) |
| Correction model `mlx-community/Qwen3-0.6B-4bit` (used by `--correct`; a conversion of `Qwen/Qwen3-0.6B`) | Apache-2.0 |
| MLX, `mlx-audio`, `mlx-lm` | MIT |
| Silero VAD | MIT |
| WavLM speaker model `microsoft/wavlm-base-plus-sv` (used by `--speakers`) | The model card points to the license of Microsoft's UniSpeech repository (CC BY-SA 3.0); the WavLM code (microsoft/unilm) is MIT. The weights are not included here; setup downloads them from Hugging Face. |
| PyTorch, transformers, scikit-learn, PyAV | BSD / Apache-2.0 (PyAV bundles FFmpeg libraries under their own LGPL/GPL terms) |
| uv (installed by the setup scripts) | MIT or Apache-2.0 |
| Bend (only to rebuild the rule checker; not needed to use the program) | Apache-2.0 |
| FLEURS test clips (tests and benchmark) | CC-BY-4.0, Google; attribution in [`tests/README.md`](tests/README.md) |

## Acknowledgements

The Qwen team for Qwen3-ASR and the `qwen-asr` reference implementation; the MLX and `mlx-audio` maintainers and the
`mlx-community` for the Apple Silicon conversions; Silero for the VAD; Microsoft for WavLM; HigherOrderCO for
Bend; Google for FLEURS.

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
salvato come `<nome>.new-<data>.md`. Per il
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

<sub>Yes, the repository is called <b>LocalTransc<i>i</i>ber</b>: an <i>r</i> went missing when it was created, and it
stays that way. The program is LocalTranscribe.</sub>
