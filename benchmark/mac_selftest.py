"""Self-test for a real Apple Silicon Mac: run it through mac_selftest.sh.

For every Mac model preset it measures word error rate on FLEURS Italian clips, speed (real-time factor),
peak MLX/GPU memory, peak process memory and model load time, then tests --speakers on a built two-speaker
recording. Everything goes into ONE file, mac_selftest_report.txt, which contains no personal data
(home folder, user name and host name are scrubbed). A failing preset never stops the run: the error is
recorded and the next step starts.

Every measured run is a separate process (so peaks are per preset and a crash cannot take the report
with it) and is started with HF_HUB_OFFLINE=1, exactly like the real launchers.
"""
import argparse
import getpass
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import wave
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import localtranscribe  # noqa: E402  (light imports: no torch/mlx)
from localtranscribe import config  # noqa: E402

DEFAULT_REPORT = os.path.join(ROOT, "mac_selftest_report.txt")
# smallest first: if a big model runs the Mac out of memory, the earlier results are already safe
DEFAULT_MODELS = sorted(config.MAC_SELFTEST_MODELS.values(),
                        key=lambda r: {"0.6B-8bit": 0, "1.7B-4bit": 1, "1.7B-8bit": 2}.get(r.split("Qwen3-ASR-")[-1], 3))


# ------------------------------------------------------------------------------ report helpers

class Report:
    def __init__(self, path):
        self.path = path
        self.lines = []
        self._scrub = self._build_scrub()

    @staticmethod
    def _build_scrub():
        items = [(ROOT, "<repo>"), (os.path.expanduser("~"), "~")]
        for name in (getpass.getuser(), socket.gethostname().split(".")[0]):
            if name and len(name) > 2:
                items.append((name, "<redacted>"))
        return [(a, b) for a, b in items if a]

    def clean(self, text):
        for a, b in self._scrub:
            text = text.replace(a, b)
        return text

    def add(self, text=""):
        text = self.clean(str(text))
        self.lines.append(text)
        print(text, flush=True)
        with open(self.path, "w", encoding="utf-8") as f:  # rewritten every time: a crash keeps what exists
            f.write("\n".join(self.lines) + "\n")

    def section(self, title):
        self.add("")
        self.add(title)
        self.add("-" * len(title))


def run(cmd, env=None, timeout=None):
    """(returncode, combined output, seconds). Never raises."""
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env=env, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or ""), time.perf_counter() - t0
    except subprocess.TimeoutExpired as e:
        return -1, f"TIMEOUT after {timeout} s\n" + str(e.stdout or "")[-500:], time.perf_counter() - t0
    except Exception as e:
        return -2, f"could not start: {type(e).__name__}: {e}", time.perf_counter() - t0


def shell(cmd, timeout=10):
    rc, out, _ = run(cmd, timeout=timeout)
    return out.strip() if rc == 0 else "n/a"


NOISE = ("Loading checkpoint", "Loading weights", "pad_token_id", "generation flags", "UserWarning",
         "key_padding_mask", "Fetching ", "it/s]", "s/it]")


def tail(text, n=12):
    lines = [l for l in text.strip().splitlines() if l.strip() and not any(x in l for x in NOISE)]
    return "\n".join("    " + l[:300] for l in lines[-n:])


def offline_env():
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["PYTHONPATH"] = ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def failure_text(rc, out):
    hint = ""
    if rc in (-9, 137):
        hint = " (killed by macOS: almost certainly out of memory)"
    elif rc == -1:
        hint = " (timed out)"
    return f"FAILED, exit code {rc}{hint}\n" + tail(out)


# ------------------------------------------------------------------------------ machine info

def package_versions():
    from importlib import metadata

    out = []
    for name in ("mlx", "mlx-audio", "torch", "transformers", "huggingface_hub", "av", "numpy", "scikit-learn", "silero-vad"):
        try:
            out.append(f"{name} {metadata.version(name)}")
        except Exception:
            out.append(f"{name} (not installed)")
    return ", ".join(out)


def machine_info(rep):
    rep.section("Machine")
    rep.add(f"Chip:            {shell(['sysctl', '-n', 'machdep.cpu.brand_string'])}")
    rep.add(f"Model id:        {shell(['sysctl', '-n', 'hw.model'])}")
    mem = shell(["sysctl", "-n", "hw.memsize"])
    rep.add(f"Total RAM:       {int(mem) / 1024 ** 3:.1f} GB (unified)" if mem.isdigit() else f"Total RAM:       {mem}")
    p_cores, e_cores = shell(["sysctl", "-n", "hw.perflevel0.physicalcpu"]), shell(["sysctl", "-n", "hw.perflevel1.physicalcpu"])
    rep.add(f"CPU cores:       {p_cores} performance + {e_cores} efficiency")
    gpu_cmd = 'system_profiler SPDisplaysDataType 2>/dev/null | grep -i "Total Number of Cores" | head -1'
    rep.add(f"GPU cores:       {shell(['bash', '-c', gpu_cmd], 20).strip()}")
    rep.add(f"macOS:           {platform.mac_ver()[0]} (build {shell(['sw_vers', '-buildVersion'])})")
    rep.add(f"Python:          {platform.python_version()} on {platform.machine()}")
    rep.add(f"Packages:        {package_versions()}")
    rep.add(f"Power source:    {shell(['bash', '-c', 'pmset -g batt | head -1'])}")
    therm_cmd = "pmset -g therm 2>&1 | tr '\n' ' '"
    rep.add(f"Thermal state:   {shell(['bash', '-c', therm_cmd])}")
    rep.add(f"Memory pressure: {shell(['bash', '-c', 'memory_pressure 2>/dev/null | tail -1'], 20)}")
    metal = shell([sys.executable, "-c",
                   "import mlx.core as mx; print('Metal available:', mx.metal.is_available(), '| default device:', mx.default_device())"], 60)
    rep.add(f"MLX:             {metal}")
    mps = shell([sys.executable, "-c",
                 "import torch; print('torch MPS available:', torch.backends.mps.is_available(), '| built:', torch.backends.mps.is_built())"], 60)
    rep.add(f"PyTorch:         {mps}")
    hardware_check(rep)


def hardware_check(rep):
    """What the tool's own hardware check (localtranscribe.precheck) detects and recommends on this Mac. This is the
    only place where its Mac detection runs on real hardware, so the raw values are recorded too. Never raises."""
    try:
        rc, out, _ = run([sys.executable, "-m", "localtranscribe.precheck", "--json", "--no-save"], env=offline_env(), timeout=180)
        if rc != 0:
            rep.add("Hardware check: " + failure_text(rc, out))
            return
        data = json.loads(out)
        hw, plan = data["hardware"], data["assessment"]
        rep.add("Hardware check (localtranscribe.precheck):")
        rep.add(f"  detected Apple values: {json.dumps(hw.get('apple'), sort_keys=True)}")
        rep.add(f"  OS: {hw['os'].get('name')} | CPU: {hw['cpu'].get('name')}, {hw['cpu'].get('cores')} cores, "
                f"{hw['cpu'].get('threads')} threads | memory total {hw['memory'].get('total_gb')} GB, "
                f"available {hw['memory'].get('available_gb')} GB | disk free {hw['disk'].get('repo_free_gb')} GB")
        rep.add(f"  recommended: {plan.get('recommended')} (batch "
                f"{(plan['options'].get(plan.get('recommended')) or {}).get('batch_size')}) | problems: {data.get('problems')}")
        rep.add(f"  warnings: {plan.get('warnings')} | notes: {plan.get('notes')}")
    except Exception as e:
        rep.add(f"Hardware check: could not be read ({type(e).__name__}: {e})")


def thermal_snapshot():
    text = shell(["bash", "-c", "pmset -g therm 2>&1 | grep -E 'Limit|warning' | tr '\\n' ';'"])
    return text if text != "n/a" else "n/a"


# ------------------------------------------------------------------------------ steps

def step_download_clips(rep, args):
    rep.section(f"Test clips: FLEURS Italian, {args.n} clips (CC-BY-4.0)")
    from download_fleurs import download

    try:
        rows = download(args.n, args.data, quiet=True)
        rep.add(f"OK: {len(rows)} clips in benchmark/data/fleurs_it")
        return True
    except Exception as e:
        rep.add(f"FAILED to download the clips: {type(e).__name__}: {e}")
        return False


def step_download_models(rep, models, with_speaker_model):
    rep.section("Model downloads (only the presets being tested)")
    cmd = [sys.executable, "-m", "localtranscribe.setup_models", "--model", *models]
    if not with_speaker_model:
        cmd.append("--no-diarization")
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT
    env.pop("HF_HUB_OFFLINE", None)
    # print the sizes first, before anything is fetched
    rc, out, _ = run(cmd + ["--dry-run"], env=env, timeout=120)
    rep.add(out.strip() if rc == 0 else failure_text(rc, out))
    rc, out, secs = run(cmd, env=env, timeout=3 * 3600)
    ok = rc == 0
    rep.add(f"download step {'OK' if ok else 'FINISHED WITH ERRORS'} in {secs:.0f}s")
    if not ok:
        rep.add(tail(out, 20))
    return ok


def step_smoke(rep):
    rep.section("Smoke test: command line tool on the committed test clip (saved hardware choice, else automatic)")
    clip = os.path.join(ROOT, "tests", "fleurs_it_sample.wav")
    ref_file = os.path.join(ROOT, "tests", "fleurs_it_sample.txt")
    if not (os.path.exists(clip) and os.path.exists(ref_file)):
        rep.add("skipped: tests/fleurs_it_sample.wav not found")
        return
    with tempfile.TemporaryDirectory() as out_dir:
        rc, out, secs = run([sys.executable, "-m", "localtranscribe", clip, "--out-dir", out_dir, "--stats"],
                            env=offline_env(), timeout=1800)
        hyp_path = os.path.join(out_dir, "fleurs_it_sample.txt")
        if rc != 0 or not os.path.exists(hyp_path):
            rep.add(failure_text(rc, out))
            return
        import wer as werlib

        hyp = open(hyp_path, encoding="utf-8").read()
        ref = open(ref_file, encoding="utf-8").read()
        rep.add(f"WER {100 * werlib.wer(ref, hyp):.2f}%  transcript: {hyp.strip()}")
        rep.add("tool output:")
        rep.add(tail(out, 14))


def bench_once(repo, args, tmp, n, batch=0, tag=""):
    """Run benchmark/bench.py in its own process. Returns (result dict | None, error text | None, hypotheses path)."""
    jf = os.path.join(tmp, f"bench_{tag or repo.split('/')[-1]}.json")
    hf = os.path.join(tmp, f"hyp_{tag or repo.split('/')[-1]}.csv")
    cmd = [sys.executable, os.path.join(HERE, "bench.py"), "--data", args.data, "--model", repo, "--n", str(n),
           "--json", jf, "--hyp-csv", hf]
    if batch:
        cmd += ["--batch-size", str(batch)]
    therm_before = thermal_snapshot()
    rc, out, secs = run(cmd, env=offline_env(), timeout=args.timeout_min * 60)
    therm_after = thermal_snapshot()
    if rc != 0 or not os.path.exists(jf):
        return None, failure_text(rc, out), None, (therm_before, therm_after)
    with open(jf, encoding="utf-8") as f:
        return json.load(f), None, hf, (therm_before, therm_after)


def step_presets(rep, args, models, tmp):
    rep.section(f"Model presets: {args.n} FLEURS clips each, greedy decoding, 20 s pieces")
    rows = []
    for repo in models:
        rep.add("")
        rep.add(f"* {repo}")
        res, err, _hyp, therm = bench_once(repo, args, tmp, args.n)
        rep.add(f"  thermal before: {therm[0]} | after: {therm[1]}")
        if err:
            rep.add("  " + err.replace("\n", "\n  "))
            rows.append((repo, None))
            continue
        rep.add(f"  device {res['device']}, batch {res['batch_size']}, WER {res['wer_percent']:.2f}%, CER {res['cer_percent']:.2f}%")
        rep.add(f"  {res['audio_seconds'] / 60:.1f} min of audio in {res['seconds']:.0f}s = {res['xrealtime']:.1f}x realtime "
                f"(real-time factor {res['rtf']:.3f})")
        rep.add(f"  model load {res['load_seconds']:.1f}s | peak MLX/GPU memory {res['peak_accelerator_gb']} GB | "
                f"peak process memory {res['peak_process_gb']} GB")
        rows.append((repo, res))
    return rows


def step_batch(rep, args, models, tmp):
    repo = next((m for m in models if m.endswith("1.7B-8bit")), models[-1])
    n = min(30, args.n)
    rep.section(f"Batch size comparison: {repo.split('/')[-1]}, first {n} clips")
    outputs = {}
    for bs in (1, 2, 4):
        res, err, hyp, _ = bench_once(repo, args, tmp, n, batch=bs, tag=f"batch{bs}")
        if err:
            rep.add(f"batch {bs}: " + err.replace("\n", "\n  "))
            continue
        import csv

        with open(hyp, encoding="utf-8", newline="") as f:
            outputs[bs] = [r["hypothesis"] for r in csv.DictReader(f)]
        rep.add(f"batch {bs}: WER {res['wer_percent']:.2f}% | {res['xrealtime']:.1f}x realtime | peak MLX/GPU memory "
                f"{res['peak_accelerator_gb']} GB | peak process {res['peak_process_gb']} GB")
    if 1 in outputs:
        for bs, texts in outputs.items():
            if bs != 1:
                diff = sum(1 for a, b in zip(outputs[1], texts) if a != b)
                rep.add(f"batch {bs} vs batch 1: {diff} of {len(texts)} clips differ in the transcript text")


def write_two_speaker_wav(path, data_dir, turns):
    import numpy as np

    import wer as werlib
    from diar_test import build

    wav, _spans, _refs = build(werlib.load_references(data_dir), data_dir, turns)
    pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())


def step_diarization(rep, args, tmp):
    rep.section("Speaker separation (--speakers 2) on a built two-speaker recording")
    jf = os.path.join(tmp, "diar.json")
    rc, out, secs = run([sys.executable, os.path.join(HERE, "diar_test.py"), "--data", args.data, "--turns", "6", "--json", jf],
                        env=offline_env(), timeout=3600)
    if rc in (0, 1) and os.path.exists(jf):
        with open(jf, encoding="utf-8") as f:
            r = json.load(f)
        rep.add(f"result: {'PASS' if r['pass'] else 'FAIL'} - speakers found {r['speakers_found']} (expected 2), "
                f"clips attributed consistently {r['clips_attributed_consistently']}/{r['clips']}, turns {r['turns']}")
        rep.add(f"torch device for the speaker model: {r['speaker_model_device']} | speech model device: {r['asr_device']} ({r['model']})")
        rep.add(f"audio {r['audio_seconds']} s | diarization {r['diarization_seconds']}s | transcription {r['transcription_seconds']}s | WER {r['wer_percent']}%")
        rep.add(f"peak process memory: {r['peak_process_gb_after_diarization']} GB after diarization, "
                f"{r['peak_process_gb_final']} GB at the end | peak MLX/GPU memory {r['peak_accelerator_gb']} GB")
    else:
        rep.add(failure_text(rc, out))
    # the same through the real command line, to exercise the launcher path and print its own stats
    wav_path = os.path.join(tmp, "two_speakers.wav")
    try:
        write_two_speaker_wav(wav_path, args.data, 6)
    except Exception as e:
        rep.add(f"could not build the CLI test recording: {type(e).__name__}: {e}")
        return
    rc, out, secs = run([sys.executable, "-m", "localtranscribe", wav_path, "--speakers", "2", "--out-dir", tmp, "--stats"],
                        env=offline_env(), timeout=3600)
    rep.add("")
    rep.add(f"command line run (--speakers 2 --stats), {secs:.0f}s, exit code {rc}:")
    rep.add(tail(out, 14))
    txt = os.path.join(tmp, "two_speakers.txt")
    if os.path.exists(txt):
        first = open(txt, encoding="utf-8").read().strip().splitlines()
        rep.add("first lines of the transcript:")
        rep.add("\n".join("    " + l[:200] for l in first[:6]))


def step_summary(rep, rows):
    rep.section("Summary")
    rep.add(f"{'model':<44} {'WER %':>6} {'xRT':>6} {'RTF':>6} {'MLX GB':>7} {'proc GB':>8} {'load s':>7}")
    for repo, r in rows:
        name = repo.split("/")[-1]
        if r is None:
            rep.add(f"{name:<44} {'FAILED':>6}")
        else:
            rep.add(f"{name:<44} {r['wer_percent']:>6.2f} {r['xrealtime']:>6.1f} {r['rtf']:>6.3f} "
                    f"{str(r['peak_accelerator_gb']):>7} {str(r['peak_process_gb']):>8} {r['load_seconds']:>7.1f}")
    rep.add("")
    rep.add("xRT = seconds of audio per second of processing (higher is faster); RTF = the inverse (lower is faster).")


# ------------------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="FLEURS clips per preset (default 100)")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Hugging Face repo ids to measure")
    ap.add_argument("--data", default=os.path.join(HERE, "data", "fleurs_it"), help="folder for the FLEURS clips")
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--timeout-min", type=int, default=120, help="give up on one measured run after this many minutes")
    ap.add_argument("--skip-diarization", action="store_true")
    args = ap.parse_args(argv)

    rep = Report(args.report)
    rep.add("LocalTranscribe Mac self-test report")
    rep.add("=" * 36)
    rep.add(f"Date (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}")
    rep.add(f"Repository version: {localtranscribe.__version__}")
    rep.add("Auto-selection rule in this version: best model if total RAM >= "
            f"{config.MAC_BEST_MIN_RAM_GB} GB, else light; Mac batch size {config.MAC_BATCH_SIZE_LOW_RAM} below / "
            f"{config.MAC_BATCH_SIZE_HIGH_RAM} at or above that.")
    rep.add(f"Presets under test: {', '.join(m.split('/')[-1] for m in args.models)}")
    machine_info(rep)

    started = time.perf_counter()
    rows = []
    try:
        have_clips = step_download_clips(rep, args)
        step_download_models(rep, args.models, with_speaker_model=not args.skip_diarization)
        step_smoke(rep)
        with tempfile.TemporaryDirectory(prefix="lt_selftest_") as tmp:
            if have_clips:
                rows = step_presets(rep, args, args.models, tmp)
                if len(args.models) and any(r for _, r in rows):
                    step_batch(rep, args, args.models, tmp)
                if not args.skip_diarization:
                    step_diarization(rep, args, tmp)
            else:
                rep.add("\nThe measured steps were skipped because the test clips are missing.")
    except KeyboardInterrupt:
        rep.add("\nInterrupted by the user; the report below is partial.")
    except Exception as e:  # the report must always be written
        rep.add(f"\nUNEXPECTED ERROR in the self-test script: {type(e).__name__}: {e}")
    if rows:
        step_summary(rep, rows)
    rep.section("Finished")
    rep.add(f"Total time: {(time.perf_counter() - started) / 60:.1f} minutes")
    rep.add(f"Thermal state at the end: {thermal_snapshot()}")
    print(f"\nThe report was written to:\n  {args.report}\nPlease send that file back to the maintainer.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
