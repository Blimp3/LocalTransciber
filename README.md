# LocalTranscribe

Turn Italian audio and video recordings (interviews, phone calls, meetings, lectures) into text **on your own
computer**. Nothing is uploaded: after a one-time download of the speech models, it works fully offline.

It uses the open speech model **Qwen3-ASR** and runs on

- a **Mac with Apple Silicon** (M1, M2, M3, M4 ...) using the Apple GPU,
- a **Windows PC with an NVIDIA graphics card**,
- any other computer, on the processor only (slower).

You drag files in, you get a `.txt` file next to each recording. It can also split a two-person phone call into
`[00:01:23] Parlante 1: ...` / `Parlante 2: ...` lines.

> Italiano: vai alla [Guida rapida](#guida-rapida-italiano) in fondo.

---

## What you need

| | Apple Silicon Mac | Windows PC with NVIDIA GPU | Any PC, no GPU |
|---|---|---|---|
| System | macOS 14 (Sonoma) or newer, M1 or newer. **Intel Macs are not supported.** | Windows 10/11, current NVIDIA driver | Windows 10/11 |
| Free disk space | about 5 GB | about 12 GB (10 GB with one model) | about 4 GB |
| Internet | only for setup | only for setup | only for setup |

Two speech models are available. The [hardware check](#hardware-check) picks one for your machine and remembers it;
you can accept or change it, and force either later with `--model`.

| Model | Download | Memory it needs | Accuracy (Italian) | Speed |
|---|---|---|---|---|
| **best**: Qwen3-ASR-1.7B | 4.7 GB (NVIDIA) / 2.5 GB (Mac, 8-bit) | NVIDIA: **5.2 GB** video memory (measured, 4.0 GB at the slowest setting); Mac: about 3.5 GB *(estimate)* | word error rate **2.4 - 2.7 %** | NVIDIA RTX 6000: **26x** real time (1 hour of audio in 2.3 minutes) |
| **light**: Qwen3-ASR-0.6B | 1.9 GB (NVIDIA / CPU) / 1.0 GB (Mac, 8-bit) | NVIDIA: **2.8 GB** video memory (measured); CPU: about 6 GB RAM while loading (measured); Mac: about 2 GB *(estimate)* | word error rate **4.4 - 5.8 %** | NVIDIA: 27x; CPU (6-core Xeon): **2.6x** (1 hour in 23 minutes) |

Which one is chosen (by the hardware check, and by `--model auto`, which follows the same rules):

- **NVIDIA:** *best* if the card has 6 GB of video memory or more, *light* from about 3 GB. Below that, or when the
  card is older than the GeForce GTX 16 / RTX 20 series (2018), or the driver is older than 570.65 (Windows) /
  570.26 (Linux, the minimum for CUDA 12.8), the processor is used instead (*light*) and the check says why. (Until
  the hardware check existed the limit for *best* was 12 GB; measured, it needs 4.0 GB at the slowest setting and
  5.2 GB at batch 8.)
- **Mac:** *best* if the Mac has 16 GB of memory or more, otherwise *light*. **These Mac defaults are provisional**
  until the self-test below has been run on real hardware.
- **CPU only:** always *light*.

Numbers marked *(estimate)* are calculated from the model file sizes plus the per-recording overhead measured on the
NVIDIA card; they have not been measured on a Mac. "Speed 26x" means one minute of audio takes about 2.3 seconds.

---

## Setup

### Getting the files onto your computer

Either way you need access to the private GitHub repository (**LocalTransciber**).

- **Download a ZIP (easiest):** open the repository page in your browser while logged in to GitHub, click the green
  **Code** button, then **Download ZIP**. Double-click the ZIP to unpack it and move the folder somewhere permanent
  (for example your Documents folder).
- **Or use git:** `git clone https://github.com/<owner>/LocalTransciber.git`. On a Mac the first use of `git` may pop up
  a window offering to install the *command line developer tools* - click **Install** and wait a few minutes, then run
  the command again. Because the repository is private, git will ask you to log in to GitHub.

### Mac (Apple Silicon)

1. Open **Terminal** (press Cmd+Space, type *Terminal*, press Return).
2. Type `cd ` (with a space), drag the LocalTranscribe folder from Finder into the Terminal window, press Return.
3. Run:

   ```
   bash setup_mac.sh
   ```

   It checks that your Mac is supported, installs the small helper tool **uv** if it is missing (from the official
   installer at astral.sh; it says so before doing it), runs the [hardware check](#hardware-check) (press Enter to
   accept the recommended model), creates a private Python environment in the `.venv` folder, installs the packages
   (about 1 GB) and downloads the model for your Mac (1 - 2.5 GB). Allow 10 - 20 minutes.
   `bash setup_mac.sh light` (or `best`) skips the question, `bash setup_mac.sh both` downloads both models.

4. To transcribe: **double-click `Trascrivi.command`** in Finder. The first time, macOS may say it cannot check the
   file because it was downloaded: **right-click it, choose Open, then Open** (only once). A window asks you to pick
   one or more recordings, transcribes them, and shows the `.txt` files in Finder.

   Prefer Terminal? `./transcribe.sh recording.m4a`.

### Windows (NVIDIA GPU or plain CPU)

1. Double-click **`setup_windows.bat`**. It installs **uv** if needed (official installer, announced beforehand), runs the
   [hardware check](#hardware-check) (press Enter to accept the recommended model), creates the `.venv` folder,
   installs the packages (about 1 - 5 GB, the NVIDIA version is the big one) and downloads the models. It
   automatically uses the CPU-only packages when no usable NVIDIA card is found. Options (from a command prompt):
   `setup_windows.bat light`, `setup_windows.bat both`, `setup_windows.bat cpu` (force CPU-only), `--yes` (do not ask).
2. To transcribe: **drag audio or video files onto `transcribe.bat`**.

If the setup stops because of a network problem, just run it again; it continues where it left off.

*Linux (not tested, not officially supported):* create the environment yourself with
`uv venv --python 3.11 .venv` and `uv pip install --index-strategy unsafe-best-match -r requirements-windows.txt`
(NVIDIA) or `requirements-cpu.txt` (no GPU), then run `PYTHONPATH=. .venv/bin/python -m localtranscribe file.m4a`.

---

## Hardware check

The first time you use the project, LocalTranscribe looks at your computer, recommends the model that should run on
it, lets you accept or change the recommendation, and remembers the choice. The setup scripts do this **first**, before
anything big is downloaded, so a full disk or an unsupported graphics card shows up while it is still cheap to stop.
To repeat it at any time (it only looks and saves; it installs and downloads nothing):

```
check_hardware.bat              (Windows)
bash check_hardware.sh          (Mac; also Linux, best effort)
```

Example, on the NVIDIA PC where the tool was developed:

```
Hardware check
  Computer : Windows 11 (build 26200), Intel(R) Xeon(R) W-3235 CPU @ 3.30GHz (6 cores, 12 threads), 41 GB memory (32.7 GB free)
  Graphics : NVIDIA Quadro RTX 6000, 22.5 GB video memory, driver 597.06
  Disk     : 107 GB free (everything is already installed and downloaded)
Recommended: best - Qwen3-ASR 1.7B on the NVIDIA GPU (most accurate)
    about 2.4-2.7 mistakes per 100 words; about 26x real time on an RTX 6000 (1 hour of audio in a few minutes); needs about 6 GB of video memory
Also possible:
  light - Qwen3-ASR 0.6B (smaller and lighter): about twice as many mistakes (4.4-5.8 per 100 words), needs less video memory (about 3 GB), similar speed
Press Enter to use "best", or type "light":
```

It looks at the operating system (macOS must be 14 or newer; Windows and Linux versions are shown), the processor
(name, cores, threads), total and available memory, the free disk space where the program lives and where the Hugging
Face model cache lives (`HF_HOME`, else `~/.cache/huggingface`), the NVIDIA graphics cards (through `nvidia-smi`: memory,
driver, architecture) or the Apple chip (model, memory, GPU cores, Rosetta). It uses only Python's standard library, so it
runs before any package is installed.

| Your computer | Recommended | Notes |
|---|---|---|
| NVIDIA card with 6 GB of video memory or more | **best** on the GPU | batch size (1 to 8) from the free video memory |
| NVIDIA card with about 3 - 6 GB | **light** on the GPU | |
| NVIDIA card below 3 GB, older than GTX 16 / RTX 20 series (compute capability below 7.5), or driver older than 570.65 (Windows) / 570.26 (Linux) | **light** on the processor | the check says which reason applies; updating the driver enables the GPU |
| No NVIDIA card or driver | **light** on the processor | warns below 8 GB of memory; about 2.6x real time on a 6-core Xeon (1 hour of audio in about 23 minutes), yours may differ |
| Apple Silicon Mac, 16 GB or more | **best** | provisional, see the self-test below |
| Apple Silicon Mac, less memory | **light** | 8 GB: "close memory-heavy apps"; MacBook Air: "keep it plugged in for long files" |
| Intel Mac, macOS older than 14, or Python under Rosetta | nothing | the setup stops and explains what to do |
| Not enough free disk space | nothing | the setup stops; it says which choice would fit |

Disk space needed = program + models + 1 GB spare: NVIDIA program about 5 GB, CPU-only about 1.2 GB, Mac about 1 GB;
models 4.7 GB (best) / 1.9 GB (light) on PC, 2.5 GB / 1.0 GB on the Mac, plus the 0.4 GB speaker model. A program
folder and models that are already there are counted as zero, so running the setup again is never blocked by that.

Options (the same words work for `setup_windows.bat` and `setup_mac.sh`):

```
check_hardware.bat best         choose without being asked: best, light, both (download both) or cpu (force the processor)
check_hardware.bat --yes        accept the recommendation without asking (also when there is no keyboard, e.g. in a script)
check_hardware.bat --json       machine-readable result; nothing is saved unless --yes is added
check_hardware.bat --no-save    look only
```

An explicit choice is not asked about, but the check is still shown, with a warning if the choice will not fit (for
example `best` on a 4 GB card).

**The saved choice** is the file `localtranscribe_settings.json` in the program folder (ignored by git; it holds the
preset, device, batch size, a hardware summary and the date). `transcribe.bat`, `Trascrivi.command` and
`transcribe.sh` use it as their default; `--model`, `--device` and `--batch-size` on the command line still win. The
saved batch size is a ceiling: it is lowered when the video memory that is free at that moment cannot hold it. If the
file is missing (someone skipped the setup) or damaged, the tool runs the check once without asking, prints the
recommendation, saves it and carries on; if only one model is downloaded it uses that one. With several NVIDIA cards
the supported one with the most memory is used. Delete the file to have the check done again.

---

## Using it

Drag files onto `transcribe.bat` (Windows) or use `Trascrivi.command` (Mac). Any common format works (wav, mp3, m4a,
ogg, opus, flac, mp4, mkv, webm, ...): the decoder is built in, you do not need to install ffmpeg.

The transcript `<recording name>.txt` is saved next to the recording (long recordings are cut at quiet moments into
pieces of about 20 seconds; each piece becomes a paragraph).

From a terminal there are extra options:

```
transcribe.bat interview.m4a --context "Mario Rossi, Politecnico di Milano, LoRaWAN"
./transcribe.sh   interview.m4a --context "Mario Rossi, Politecnico di Milano, LoRaWAN"     # Mac
```

| Option | What it does |
|---|---|
| `--context "names, terms"` | names and technical words that occur in the recording; helps spell them correctly |
| `--speakers 2` | separate the speakers: `[00:01:23] Parlante 1: ...`. Reliable on calls of a few minutes or more; may merge very short replies such as "sì" into the other speaker |
| `--out-dir folder` | write the transcripts to this folder instead of next to the recordings |
| `--language auto` | detect the language instead of Italian (or `English`, `French`, `Spanish`, ...) |
| `--model best`, `light` or a repo id | choose the model (default: the choice saved by the hardware check; without one, automatic, see above) |
| `--device auto`, `cuda`, `mps`, `cpu` | choose the hardware (default: the saved choice, else automatic). On a Mac `mps` is the Apple GPU and `cpu` runs on the processor cores |
| `--batch-size N` | pieces processed together (default: the saved choice, lowered if memory is tight; lower = less memory) |
| `--stats` | at the end print the device, model, speed and peak memory used |
| `--chunk 20` | seconds per piece. Leave at 20: longer pieces are measurably less accurate |

### MacBook Air M2 notes

- **Which model you get.** The [hardware check](#hardware-check) reads your memory (you can also see it under
  **Apple menu > About This Mac**). With **16 GB or 24 GB** it recommends the *best* model (Qwen3-ASR-1.7B, 8-bit).
  With **8 GB** it recommends the *light* model (Qwen3-ASR-0.6B, 8-bit). (Provisional; the self-test decides the
  final rule.) You can try the best model on an 8 GB Air with `bash setup_mac.sh best` or
  `./transcribe.sh file.m4a --model best`.
- **Keep it plugged in** for long recordings. The Air has no fan, so it slows down when it gets hot; on battery
  it slows down further.
- **On 8 GB, close memory-hungry programs** (browsers with many tabs, video calls, Photos, Xcode) before transcribing.
  The speaker-separation step (`--speakers`) needs about 2 GB more for a short while.
- A one-hour recording takes several minutes; the window shows progress.

---

## Accuracy and speed

Measured on the FLEURS Italian test set (read sentences, 100 clips, about 24 minutes) and on a 24-minute
recording built from those clips, with the **NVIDIA Quadro RTX 6000** (float16). Word error rate (WER) ignores case and
punctuation; lower is better.

| Model | 100 clips (set A) | 100 clips (set B) | 24-minute file | Speed | Peak video memory |
|---|---|---|---|---|---|
| best (1.7B), NVIDIA | **2.57 %** | 2.74 % | **2.44 %** | 25 - 26x | 5.2 GB (batch 8) |
| light (0.6B), NVIDIA | 4.38 % | 5.82 % | 5.18 % | 27x | 2.8 GB (batch 8) |
| light (0.6B), CPU float32 | - | 4.68 % (first 10 clips of set B; same as GPU) | - | 2.6x | about 6 GB RAM |
| best 8-bit, Apple MLX | *not measured yet* | | | | |
| light 8-bit, Apple MLX | *not measured yet* | | | | |

Set A is the 100-clip set used to evaluate the previous version of this tool; set B is what
`benchmark/download_fleurs.py` downloads (the first 100 recordings of the test archive, the set the Mac
self-test uses). The Mac rows will be filled in from the self-test report. The 8-bit models are expected to be very
close to the numbers above but that is exactly what the self-test measures.

Peak video memory by batch size (best 1.7B / light 0.6B): 1: 4.0 / 1.7 GB, 2: 4.1 / 1.8, 4: 4.5 / 2.1, 8: 5.2 / 2.8,
16: 6.5 / 4.1. Speed (best): 4.7x / 7.4x / 12.9x / 22x / 32x for the same batch sizes.

On a real phone call (6.6 minutes) the best model produced exactly the same text as the previous version of this tool.

---

## Privacy

Everything runs on your computer. After setup (which downloads the models from Hugging Face and the Python packages from
the internet) the launchers switch the model library to **offline mode** (`HF_HUB_OFFLINE=1`), so the audio and the
transcripts never leave the machine and nothing is sent anywhere. No account, no cloud service, no telemetry from this
tool. Transcripts are plain `.txt` files that you control.

## Licences of what it uses

| Component | Licence |
|---|---|
| Qwen3-ASR models (`Qwen/Qwen3-ASR-1.7B`, `-0.6B`) and the `qwen-asr` package | Apache-2.0 |
| MLX conversions (`mlx-community/Qwen3-ASR-...-8bit`) | Apache-2.0 (as the originals) |
| MLX and `mlx-audio` | MIT |
| Silero VAD (voice activity detection, ships inside its Python package) | MIT |
| WavLM speaker model `microsoft/wavlm-base-plus-sv` (used by `--speakers`) | The model card names no licence and points to the licence file of Microsoft's UniSpeech repository, which is **Creative Commons Attribution-ShareAlike 3.0**; the WavLM code repository (microsoft/unilm) is MIT. The weights are not included here: the setup script downloads them from Hugging Face. Check the current terms before redistributing them. |
| PyTorch, transformers, scikit-learn, PyAV | BSD / Apache-2.0 (PyAV bundles FFmpeg libraries under their own LGPL/GPL terms) |
| FLEURS test clips (benchmark) | CC-BY-4.0, Google; attribution in `tests/README.md` |

The audio chunker and repetition filter in `localtranscribe/textutil.py` are adapted from `qwen-asr` (Apache-2.0,
copyright The Alibaba Qwen team), with two documented changes.

---

## How it works (for the curious)

```
file -> PyAV decoder (16 kHz mono) -> cut at quiet moments into ~20 s pieces -> speech model -> text
                                        (same code on every platform)      NVIDIA: PyTorch + qwen-asr, float16
                                                                            CPU:    PyTorch + qwen-asr, float32
                                                                            Mac:    MLX (mlx-audio), 8-bit, Apple GPU
--speakers N: Silero VAD -> WavLM voice fingerprints -> clustering -> per-speaker turns -> the same speech model
```

- `localtranscribe/backends/` holds the two backends behind one interface; `mlx` is imported only on a Mac.
- `localtranscribe/config.py` holds every model name and threshold in one place.
- `localtranscribe/precheck.py` is the hardware check (standard library only); `devices.py` uses its rules, so
  `--model auto` and the check always agree.
- `--context` becomes the model's system prompt on both backends. (On the Mac one extra line break follows the
  context text; without `--context` the prompts are identical.)
- The chunker differs from the one in `qwen-asr` in two ways: audio up to 25 s stays in one piece, and no cut leaves
  a final piece shorter than 3 s. Without this, 20-24 s clips were cut into a piece and a half-second scrap, and the
  model invented a word ("sì") on the scrap, raising the error rate on the benchmark from 2.6 % to 3.1 %.

### Benchmarks and tests

```
python benchmark/download_fleurs.py --n 100                       # clips into benchmark/data/ (about 90 MB)
python benchmark/bench.py --data benchmark/data/fleurs_it --model best       # WER, speed, peak memory
python benchmark/make_longform.py --data benchmark/data/fleurs_it --out long.wav
./transcribe.sh long.wav --out-dir out && python benchmark/score_long.py --data benchmark/data/fleurs_it --hyp out/long.txt
python benchmark/diar_test.py --data benchmark/data/fleurs_it     # two-speaker check of --speakers
python -m unittest discover -s tests -v                           # unit tests, no model needed (the hardware check runs on mocked machines)
```

(On Windows use `.venv\Scripts\python.exe` instead of `python`, and `transcribe.bat` instead of `./transcribe.sh`.)

### For the maintainer: the Mac self-test

The Apple/MLX path could not be run without a Mac. After `bash setup_mac.sh` on the MacBook Air run:

```
bash mac_selftest.sh          # about an hour; add --n 30 for a quicker run
```

It prints the download sizes first, then records the machine (chip, memory, macOS, package versions, and what the
hardware check detects and recommends there), and for the
presets `Qwen3-ASR-0.6B-8bit`, `1.7B-4bit` and `1.7B-8bit`: WER, real-time factor, peak MLX memory, peak process
memory and model load time; a comparison of batch sizes 1, 2 and 4; and a `--speakers 2` test (torch device used,
peak memory). Each step runs in its own process, so a failure or an out-of-memory kill is recorded and the run
continues. Everything is written to **`mac_selftest_report.txt`** (no personal data: home folder, user name and host
name are removed). Send that file back. Then adjust `MAC_BEST_MIN_RAM_GB`, `MAC_MODELS` and the batch sizes in
`localtranscribe/config.py`.

---

## Troubleshooting

- **"has not been downloaded yet"**: run the setup script again while online.
- **Windows: "Could not create the Python environment"**: no internet, or a company proxy blocking `github.com`
  (uv downloads Python from there). Install Python 3.11 first, then run the setup again.
- **Out of memory**: use `--model light`, or `--batch-size 1`, or close other programs.
- **"Not enough free disk space"** during setup: free some space (the message says how much), or choose the smaller
  model (`setup_windows.bat light`), or move the model cache with the `HF_HOME` environment variable.
- **The NVIDIA card is not used**: run `check_hardware.bat`; it says whether the card is too old for the GPU build of
  PyTorch, or the driver is older than 570.65 (Windows) / 570.26 (Linux) and must be updated from nvidia.com/drivers.
  Then run the check again and the setup again.
- **A recording gives an empty or short text**: the file may have no audio track; try another file.
- **Mac: "cannot be opened because the developer cannot be verified"**: right-click the file, choose **Open**.

---

## Guida rapida (italiano)

**A cosa serve.** Trascrive registrazioni audio e video in italiano (interviste, telefonate, riunioni) in un file di
testo, **direttamente sul tuo computer**: dopo l'installazione non serve internet e nessun file esce dal computer.

**Requisiti.** Mac con chip Apple (M1 o successivi) e macOS 14 o più recente (i Mac Intel non sono supportati),
oppure PC Windows (meglio con scheda video NVIDIA). Circa 5 GB di spazio libero su Mac.

**Installazione su Mac (una volta sola)**

1. Scarica il progetto da GitHub: pulsante verde **Code**, poi **Download ZIP** (devi aver fatto il login). Apri lo ZIP
   e sposta la cartella in Documenti. (In alternativa: `git clone`; alla prima volta il Mac può chiedere di installare
   gli "strumenti da riga di comando": conferma con **Installa**.)
2. Apri **Terminale** (Cmd+Spazio, scrivi *Terminale*). Scrivi `cd ` (con lo spazio), trascina dentro la cartella del
   progetto e premi Invio.
3. Scrivi `bash setup_mac.sh` e premi Invio. Ci vogliono 10-20 minuti: scarica programmi e modello (1-2,5 GB).
4. Per sapere quanta memoria ha il tuo Mac: menu Apple  > **Informazioni su questo Mac**. Con 16 GB o più viene usato
   il modello più accurato, con 8 GB quello più leggero (impostazione provvisoria).

**Uso su Mac.** Doppio clic su **`Trascrivi.command`**. La prima volta, se macOS non lo apre: clic destro sul file,
**Apri**, poi ancora **Apri**. Scegli uno o più file audio o video; puoi anche far separare due persone che parlano
(telefonate). Il testo viene salvato accanto a ogni file audio (stesso nome, estensione `.txt`) e compare nel Finder.

**Consigli per il MacBook Air.** Tienilo **collegato all'alimentazione** per i file lunghi (non ha ventola e rallenta
quando si scalda). Con 8 GB di memoria **chiudi i programmi pesanti** (browser con molte schede, videochiamate) prima
di trascrivere.

**Windows.** Doppio clic su `setup_windows.bat` (una volta sola), poi **trascina i file audio su `transcribe.bat`**.

**Controllo del computer.** All'installazione il programma controlla il computer (memoria, scheda video, spazio libero)
e propone il modello adatto: premi Invio per accettare. Puoi ripetere il controllo quando vuoi con `check_hardware.bat`
(Windows) o `bash check_hardware.sh` (Mac).

**Suggerimento.** Dalla riga di comando puoi indicare nomi e termini tecnici che compaiono nella registrazione con
`--context "Mario Rossi, Politecnico di Milano"`: la loro trascrizione sarà più precisa. Per separare i parlanti
usa `--speakers 2`: il risultato è `[00:01:23] Parlante 1: ...` e `Parlante 2: ...`.

**Privacy.** Tutto avviene sul tuo computer. Internet serve solo durante l'installazione per scaricare i modelli.
