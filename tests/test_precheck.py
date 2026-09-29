"""Tests of the hardware precheck with mocked hardware profiles: no GPU, Mac or network needed.
    python -m unittest discover -s tests -v
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from localtranscribe import cli, config, devices, precheck  # noqa: E402


# ----------------------------------------------------------------------------- mocked hardware

def gpu(total, free=None, name="NVIDIA Test GPU", cc=(7, 5), index=0, driver="597.06"):
    return {"index": index, "name": name, "total_gb": float(total), "free_gb": float(total if free is None else free) - 0.5,
            "driver": driver, "compute_capability": list(cc) if cc else None}


def pc(system="Windows", ram=32.0, gpus=(), driver=None, smi=None, disk_free=200.0, hf_free=None, cores=6, threads=12,
       version="10.0.26200"):
    """A Windows/Linux profile. `gpus` is a list of gpu(...) dicts."""
    driver = driver or (gpus[0]["driver"] if gpus else None)
    return {
        "os": {"system": system, "name": "Windows 11 (build 26200)" if system == "Windows" else "Ubuntu 22.04 (Linux)",
               "version": version, "machine": "AMD64", "mac_version": None, "linux_name": None},
        "cpu": {"name": "Test CPU @ 3.0GHz", "cores": cores, "threads": threads, "performance_cores": None,
                "efficiency_cores": None},
        "memory": {"total_gb": ram, "available_gb": ram * 0.6},
        "disk": {"repo_path": os.path.join(tempfile.gettempdir(), "lt_repo"), "repo_free_gb": disk_free,
                 "hf_path": os.path.join(tempfile.gettempdir(), "lt_hf"),
                 "hf_free_gb": disk_free if hf_free is None else hf_free, "same_volume": hf_free is None},
        "nvidia": {"smi": smi or ("found" if gpus else "missing"), "message": None, "driver": driver, "gpus": list(gpus)},
        "apple": None,
        "python": "3.11.8",
    }


def mac(ram=16.0, model="MacBook Air", chip="Apple M2", gpu_cores=8, macos="15.6", silicon=True, rosetta=False,
        disk_free=200.0):
    p = pc("Darwin", ram=ram, disk_free=disk_free)
    p["os"].update({"name": f"macOS {macos}", "mac_version": macos, "machine": "arm64" if silicon else "x86_64"})
    p["cpu"]["name"] = chip
    p["apple"] = {"apple_silicon": silicon, "python_arch": "x86_64" if rosetta or not silicon else "arm64", "rosetta": rosetta,
                  "chip": chip, "unified_gb": ram, "model_name": model, "model_id": "Mac14,2", "gpu_cores": gpu_cores,
                  "fanless": "air" in model.lower()}
    return p


def text_of(lines):
    return " ".join(" ".join(lines).split())


class Sandbox(unittest.TestCase):
    """Every test gets its own empty repo folder, Hugging Face cache and settings file."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.join(self.tmp.name, "repo")
        self.hub = os.path.join(self.tmp.name, "hub")
        os.makedirs(self.root)
        os.makedirs(self.hub)
        self.settings = os.path.join(self.root, config.SETTINGS_FILE)
        patcher = mock.patch.object(precheck, "hf_hub_dir", return_value=self.hub)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_check(self, profile, words=(), **kw):
        out = []
        kw.setdefault("interactive", False)
        code, settings = precheck.run(words, profile=profile, root=self.root, settings_file=self.settings, out=out.append, **kw)
        return code, settings, "\n".join(out)

    def fake_cache(self, repo):
        snap = os.path.join(self.hub, "models--" + repo.replace("/", "--"), "snapshots", "abc123")
        os.makedirs(snap)
        with open(os.path.join(snap, "model.safetensors"), "wb") as f:
            f.write(b"x")

    def fake_venv(self, torch_version):
        d = os.path.join(self.root, ".venv", "Lib", "site-packages", f"torch-{torch_version}.dist-info")
        os.makedirs(d)


# ----------------------------------------------------------------------------- rules

class RuleTests(unittest.TestCase):
    def test_nvidia_presets_by_memory(self):
        for vram, want in ((24, "best"), (12, "best"), (8, "best"), (6, "best"), (5.9, "best"), (4, "light"), (3, "light"),
                           (2, None), (1, None)):
            self.assertEqual(precheck.nvidia_preset_for(vram), want, vram)
        self.assertIsNone(precheck.nvidia_preset_for(None))

    def test_mac_presets_use_the_measured_rule(self):
        self.assertEqual((config.MAC_BEST_MIN_RAM_GB, config.MAC_BATCH_MIN_RAM_GB), (8, 16))  # from the M2 8 GB self-test
        for ram, want in ((8, "best"), (7.6, "best"), (16, "best"), (None, "light")):
            self.assertEqual(precheck.mac_preset_for(ram), want, ram)
        self.assertEqual((config.MAC_BATCH_SIZE_LOW_RAM, config.MAC_BATCH_SIZE_HIGH_RAM), (1, 2))
        self.assertEqual(precheck.mac_batch_size(8), 1)
        self.assertEqual(precheck.mac_batch_size(16), 2)
        self.assertEqual(precheck.mac_batch_size(24), 2)

    def test_batch_size_from_free_memory(self):
        self.assertEqual(precheck.nvidia_batch_size("best", 22.0), 8)
        self.assertEqual(precheck.nvidia_batch_size("best", 5.5), 2)
        self.assertEqual(precheck.nvidia_batch_size("best", 3.5), 1)
        self.assertEqual(precheck.nvidia_batch_size("light", 3.5), 4)
        self.assertEqual(precheck.nvidia_batch_size("light", 0.5), 1)
        # never asks for more than the measured peak: best 5.2 GB at batch 8 plus the reserve
        need8 = config.NVIDIA_PEAK_VRAM_GB["best"] + 7 * config.NVIDIA_VRAM_PER_EXTRA_PIECE_GB + config.NVIDIA_VRAM_RESERVE_GB
        self.assertEqual(precheck.nvidia_batch_size("best", need8 + 0.01), 8)
        self.assertLess(precheck.nvidia_batch_size("best", need8 - 0.2), 8)
        self.assertEqual(precheck.cpu_batch_size(16), 2)
        self.assertEqual(precheck.cpu_batch_size(6), 1)

    def test_driver_minimum(self):
        self.assertEqual(config.NVIDIA_MIN_DRIVER, {"windows": "570.65", "linux": "570.26"})
        self.assertTrue(precheck.driver_ok("597.06", "windows"))
        self.assertTrue(precheck.driver_ok("570.65", "windows"))
        self.assertFalse(precheck.driver_ok("570.64", "windows"))
        self.assertFalse(precheck.driver_ok("552.44", "windows"))
        self.assertTrue(precheck.driver_ok("570.124.06", "linux"))
        self.assertTrue(precheck.driver_ok("570.26", "linux"))
        self.assertFalse(precheck.driver_ok("550.54.14", "linux"))
        self.assertIsNone(precheck.driver_ok("", "windows"))
        self.assertIsNone(precheck.driver_ok(None, "linux"))

    def test_minimum_compute_capability_matches_the_torch_wheel(self):
        self.assertEqual(config.NVIDIA_MIN_COMPUTE_CAPABILITY, (7, 5))

    def test_cli_auto_rule_and_precheck_agree(self):
        with mock.patch.object(devices, "is_apple_silicon", return_value=False):
            for vram in (1, 2, 3, 4, 5.5, 6, 8, 12, 24):
                want = precheck.nvidia_preset_for(vram) or "light"
                self.assertEqual(devices.choose_model("auto", "cuda", vram), config.TORCH_MODELS[want], vram)
        with mock.patch.object(devices, "is_apple_silicon", return_value=True):
            for ram in (8, 16, 24):
                self.assertEqual(devices.choose_model("auto", "mps", ram), config.MAC_MODELS[precheck.mac_preset_for(ram)])

    def test_hf_cache_location(self):
        self.assertEqual(precheck.hf_hub_dir({"HF_HUB_CACHE": "/x/hub"}), "/x/hub")
        self.assertEqual(precheck.hf_hub_dir({"HF_HOME": "/x/hf"}), os.path.join("/x/hf", "hub"))
        self.assertEqual(precheck.hf_hub_dir({"XDG_CACHE_HOME": "/x/c"}), os.path.join("/x/c", "huggingface", "hub"))

    def test_fmt_gb(self):
        self.assertEqual([precheck.fmt_gb(v) for v in (8.0, 22.5, 16, 107.3, 0.4, None)],
                         ["8", "22.5", "16", "107", "0.4", "unknown"])


# ----------------------------------------------------------------------------- Apple Silicon

class MacTests(Sandbox):
    def test_m2_air_by_memory(self):
        for ram, preset, batch in ((8, "best", 1), (16, "best", 2), (24, "best", 2)):
            code, settings, out = self.run_check(mac(ram=ram))
            self.assertEqual(code, 0, out)
            self.assertEqual((settings["preset"], settings["device"], settings["batch_size"]), (preset, "mps", batch), ram)
            self.assertEqual(settings["model"], config.MAC_MODELS[preset])
            self.assertIn("Recommended: " + preset, out)
            self.assertIn("Also possible:", out)

    def test_air_notes(self):
        _, _, out = self.run_check(mac(ram=8))
        flat = text_of([out])
        self.assertIn("keep it plugged in for long files", flat)
        self.assertIn("close memory-heavy apps", flat)
        _, _, out = self.run_check(mac(ram=16))
        flat = text_of([out])
        self.assertIn("keep it plugged in for long files", flat)
        self.assertNotIn("close memory-heavy apps", flat)

    def test_fan_cooled_mac_gets_no_plugged_in_note(self):
        _, _, out = self.run_check(mac(ram=16, model="MacBook Pro"))
        self.assertNotIn("plugged in", out)
        # the memory advice is not specific to the Air: any 8 GB Mac gets it
        _, _, out = self.run_check(mac(ram=8, model="Mac mini"))
        self.assertNotIn("plugged in", out)
        self.assertIn("close memory-heavy apps", text_of([out]))

    def test_summary_line_names_the_machine(self):
        _, _, out = self.run_check(mac(ram=16))
        self.assertIn("Computer : MacBook Air (Apple M2, 8-core GPU), 16 GB memory, macOS 15.6", out)
        self.assertIn("Disk     : 200 GB free (needs about 4 GB", out)

    def test_best_on_8_gb_is_recommended(self):
        rep = precheck.assess(mac(ram=8))
        self.assertEqual((rep["recommended"], rep["options"]["best"]["fit"]), ("best", "good"))

    def test_best_on_4_gb_is_offered_but_tight(self):
        rep = precheck.assess(mac(ram=4))
        self.assertEqual(rep["recommended"], "light")
        self.assertEqual(rep["options"]["best"]["fit"], "tight")
        self.assertTrue(rep["options"]["best"]["offered"])
        self.assertTrue(rep["options"]["best"]["warnings"])

    def test_intel_mac_is_refused_clearly(self):
        code, settings, out = self.run_check(mac(silicon=False))
        self.assertEqual(code, 1)
        self.assertIsNone(settings)
        self.assertIn("Intel", out)
        self.assertIn("not supported", out)
        self.assertFalse(os.path.exists(self.settings))

    def test_rosetta_is_refused_clearly(self):
        code, settings, out = self.run_check(mac(rosetta=True))
        self.assertEqual(code, 1)
        self.assertIsNone(settings)
        self.assertIn("Rosetta", out)

    def test_macos_13_is_refused_and_14_accepted(self):
        code, settings, out = self.run_check(mac(macos="13.6.9"))
        self.assertEqual(code, 1)
        self.assertIsNone(settings)
        self.assertIn("macOS 13.6.9 is too old", out)
        self.assertIn("macOS 14", out)
        for ver in ("14.0", "14.7.1", "15.6", "26.0"):
            self.assertEqual(self.run_check(mac(macos=ver))[0], 0, ver)

    def test_cpu_word_is_not_a_mac_choice(self):
        code, settings, _ = self.run_check(mac(ram=16), ["cpu"])
        self.assertEqual(settings["device"], "mps")  # the Mac plan ignores the CPU-only install


# ----------------------------------------------------------------------------- NVIDIA

class NvidiaTests(Sandbox):
    def test_recommendation_by_video_memory(self):
        for vram, preset, device in ((24, "best", "cuda"), (12, "best", "cuda"), (8, "best", "cuda"), (6, "best", "cuda"),
                                     (4, "light", "cuda"), (3, "light", "cuda"), (2, "light", "cpu")):
            code, settings, out = self.run_check(pc(gpus=[gpu(vram)]))
            self.assertEqual(code, 0, out)
            self.assertEqual((settings["preset"], settings["device"]), (preset, device), vram)

    def test_batch_size_from_free_memory(self):
        _, settings, _ = self.run_check(pc(gpus=[gpu(24)]))
        self.assertEqual(settings["batch_size"], 8)
        # gpu() reports 0.5 GB less free than its total (the desktop's share)
        _, settings, _ = self.run_check(pc(gpus=[gpu(6)]))  # 5.5 free: 4.0 (batch 1) + 1.25 reserve leaves 0.25 GB = 1 more piece
        self.assertEqual((settings["preset"], settings["batch_size"]), ("best", 2))
        _, settings, _ = self.run_check(pc(gpus=[gpu(5.5)]))  # 5.0 free: not even the reserve fits: one piece at a time
        self.assertEqual((settings["preset"], settings["batch_size"]), ("best", 1))
        _, settings, _ = self.run_check(pc(gpus=[gpu(6.5)]))  # 6.0 free: 0.75 GB spare = 4 more pieces of 0.16 GB
        self.assertEqual(settings["batch_size"], 5)
        _, settings, _ = self.run_check(pc(gpus=[gpu(8)]))  # 7.5 free
        self.assertEqual(settings["batch_size"], 8)
        _, settings, _ = self.run_check(pc(gpus=[gpu(4)]))  # 3.5 free: light 1.7 + 1.25 reserve leaves 0.55 GB
        self.assertEqual((settings["preset"], settings["batch_size"]), ("light", 4))

    def test_2_gb_gpu_falls_back_to_cpu_with_a_reason(self):
        code, settings, out = self.run_check(pc(gpus=[gpu(2)]))
        self.assertEqual((settings["device"], settings["preset"]), ("cpu", "light"))
        self.assertIn("only 2 GB of video memory", text_of([out]))

    def test_unsupported_old_gpu_gets_cpu_mode_and_an_explanation(self):
        code, settings, out = self.run_check(pc(gpus=[gpu(8, name="NVIDIA GeForce GTX 1080", cc=(6, 1))]))
        self.assertEqual(code, 0)
        self.assertEqual((settings["device"], settings["preset"]), ("cpu", "light"))
        flat = text_of([out])
        self.assertIn("too old", flat)
        self.assertIn("6.1", flat)
        self.assertIn("7.5", flat)
        self.assertEqual(precheck.requirements_for(settings["device"]), "requirements-cpu.txt")

    def test_volta_is_below_the_wheel_minimum(self):
        _, settings, _ = self.run_check(pc(gpus=[gpu(16, name="NVIDIA Tesla V100", cc=(7, 0))]))
        self.assertEqual(settings["device"], "cpu")

    def test_turing_and_newer_are_supported(self):
        for cc in ((7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)):
            _, settings, _ = self.run_check(pc(gpus=[gpu(12, cc=cc)]))
            self.assertEqual(settings["device"], "cuda", cc)

    def test_old_driver_recommends_cpu_and_says_which_driver(self):
        code, settings, out = self.run_check(pc(gpus=[gpu(24, driver="552.44")]))
        self.assertEqual(code, 0)
        self.assertEqual(settings["device"], "cpu")
        flat = text_of([out])
        self.assertIn("driver 552.44 is too old", flat)
        self.assertIn("570.65", flat)
        self.assertIn('"best"', flat)  # what the card could run after updating
        code, settings, out = self.run_check(pc("Linux", gpus=[gpu(24, driver="550.54.14")]))
        self.assertEqual(settings["device"], "cpu")
        self.assertIn("570.26", text_of([out]))

    def test_driver_at_the_minimum_is_fine(self):
        self.assertEqual(self.run_check(pc(gpus=[gpu(8, driver="570.65")]))[1]["device"], "cuda")
        self.assertEqual(self.run_check(pc("Linux", gpus=[gpu(8, driver="570.26")]))[1]["device"], "cuda")

    def test_two_gpus_use_the_one_with_most_memory(self):
        gpus = [gpu(8, name="NVIDIA A", index=0), gpu(24, name="NVIDIA B", index=1)]
        _, settings, out = self.run_check(pc(gpus=gpus))
        self.assertEqual((settings["preset"], settings["gpu_index"], settings["gpu_count"]), ("best", 1, 2))
        self.assertIn("2 NVIDIA GPUs found; using GPU 1", out)
        gpus = [gpu(24, name="NVIDIA Old", index=0, cc=(6, 1)), gpu(6, name="NVIDIA New", index=1)]
        _, settings, out = self.run_check(pc(gpus=gpus))
        self.assertEqual((settings["device"], settings["gpu_index"]), ("cuda", 1))
        self.assertIn("GPU 0 (NVIDIA Old) is too old", out)

    def test_single_gpu_index_is_recorded_but_not_pinned_later(self):
        _, settings, _ = self.run_check(pc(gpus=[gpu(8)]))
        self.assertEqual((settings["gpu_index"], settings["gpu_count"]), (0, 1))

    def test_nvidia_smi_missing_means_cpu_with_a_hint(self):
        code, settings, out = self.run_check(pc(smi="missing"))
        self.assertEqual((code, settings["device"], settings["preset"]), (0, "cpu", "light"))
        self.assertIn("nvidia.com/drivers", out)

    def test_nvidia_smi_failing_means_cpu_and_shows_the_message(self):
        profile = pc(smi="failed")
        profile["nvidia"]["message"] = "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver."
        code, settings, out = self.run_check(profile)
        self.assertEqual((code, settings["device"]), (0, "cpu"))
        self.assertIn("couldn't communicate", out)

    def test_low_free_video_memory_is_a_warning(self):
        _, settings, out = self.run_check(pc(gpus=[gpu(24, free=3.0)]))
        self.assertEqual(settings["preset"], "best")  # the recommendation follows the hardware, not what is busy now
        self.assertEqual(settings["batch_size"], 1)
        self.assertIn("free right now", out)

    def test_explicit_best_on_a_small_gpu_warns_but_goes_ahead(self):
        code, settings, out = self.run_check(pc(gpus=[gpu(4)]), ["best"])
        self.assertEqual(code, 0)
        self.assertEqual(settings["preset"], "best")
        self.assertEqual(settings["chosen_by"], "setup argument")
        self.assertIn("will probably run out of memory", out)

    def test_explicit_cpu_forces_the_cpu_install(self):
        code, settings, out = self.run_check(pc(gpus=[gpu(24)]), ["cpu"])
        self.assertEqual((settings["device"], settings["preset"]), ("cpu", "light"))
        self.assertEqual(settings["gpu_index"], None)
        self.assertIn("CPU-only mode was requested", out)

    def test_explicit_choice_equal_to_the_recommendation_is_still_recorded_as_explicit(self):
        _, settings, out = self.run_check(pc(gpus=[gpu(24)]), ["best"])
        self.assertEqual((settings["preset"], settings["chosen_by"]), ("best", "setup argument"))
        self.assertNotIn("Using the recommended choice", out)

    def test_no_prompt_for_explicit_arguments(self):
        asked = []
        self.run_check(pc(gpus=[gpu(24)]), ["light"], interactive=True, ask=lambda p: asked.append(p) or "")
        self.assertEqual(asked, [])


# ----------------------------------------------------------------------------- CPU only

class CpuTests(Sandbox):
    def test_cpu_only_recommends_light(self):
        for ram in (6, 16):
            code, settings, out = self.run_check(pc(ram=ram))
            self.assertEqual((code, settings["device"], settings["preset"]), (0, "cpu", "light"), ram)
            self.assertEqual(precheck.requirements_for(settings["device"]), "requirements-cpu.txt")

    def test_low_ram_warns(self):
        _, _, out = self.run_check(pc(ram=6))
        self.assertIn("Only 6 GB of memory", out)
        _, _, out = self.run_check(pc(ram=16))
        self.assertNotIn("Only", out)

    def test_speed_hint_is_labelled_as_a_xeon_measurement(self):
        _, _, out = self.run_check(pc(ram=16))
        flat = text_of([out])
        self.assertIn("2.6x real time", flat)
        self.assertIn("1 hour of audio takes about 23 minutes", flat)
        self.assertIn("measured on a 6-core Xeon", flat)

    def test_best_on_cpu_only_offered_with_plenty_of_ram(self):
        self.assertTrue(precheck.assess(pc(ram=32))["options"]["best"]["offered"])
        self.assertFalse(precheck.assess(pc(ram=8))["options"]["best"]["offered"])
        self.assertIn("Also possible", self.run_check(pc(ram=32))[2])
        self.assertNotIn("Also possible", self.run_check(pc(ram=8))[2])

    def test_linux_is_best_effort(self):
        _, settings, out = self.run_check(pc("Linux", ram=16))
        self.assertEqual(settings["device"], "cpu")
        self.assertIn("Linux is not officially supported", out)

    def test_unknown_system_is_refused(self):
        profile = pc()
        profile["os"]["system"] = "FreeBSD"
        code, settings, out = self.run_check(profile)
        self.assertEqual(code, 1)
        self.assertIn("not supported", out)


# ----------------------------------------------------------------------------- disk

class DiskTests(Sandbox):
    def test_low_disk_blocks_and_saves_nothing(self):
        code, settings, out = self.run_check(pc(gpus=[gpu(24)], disk_free=3.0))
        self.assertEqual(code, 1)
        self.assertIsNone(settings)
        self.assertIn("Not enough free disk space", out)
        self.assertIn("3 GB free", out)
        self.assertFalse(os.path.exists(self.settings))

    def test_disk_need_matches_the_measured_sizes(self):
        expected = {("nvidia", "best"): 5.0 + 4.7 + 0.4 + 1.0, ("nvidia", "light"): 5.0 + 1.9 + 0.4 + 1.0,
                    ("cpu", "light"): 1.2 + 1.9 + 0.4 + 1.0, ("mac", "best"): 1.0 + 1.6 + 0.4 + 1.0,
                    ("mac", "light"): 1.0 + 1.0 + 0.4 + 1.0}
        for (kind, preset), gb in expected.items():
            profile = {"nvidia": pc(gpus=[gpu(24)]), "cpu": pc(), "mac": mac(ram=24)}[kind]
            option = precheck.assess(profile)["options"][preset]
            plan = precheck.disk_plan(profile, option, [preset], self.root)
            self.assertAlmostEqual(plan["needs_gb"], gb, places=2, msg=(kind, preset))

    def test_cheaper_choice_is_suggested_when_only_it_fits(self):
        code, settings, out = self.run_check(pc(gpus=[gpu(24)], disk_free=9.5))
        self.assertEqual(code, 1)
        self.assertIn('"light" needs only about 8 GB', out)
        code, settings, out = self.run_check(pc(gpus=[gpu(24)], disk_free=9.5), ["light"])
        self.assertEqual((code, settings["preset"]), (0, "light"))

    def test_the_space_needed_grows_with_both(self):
        profile = pc(gpus=[gpu(24)])
        rep = precheck.assess(profile)
        one = precheck.disk_plan(profile, rep["options"]["best"], ["best"], self.root)
        two = precheck.disk_plan(profile, rep["options"]["best"], ["best", "light"], self.root)
        self.assertAlmostEqual(two["needs_gb"] - one["needs_gb"], 1.9, places=2)
        code, settings, _ = self.run_check(profile, ["both"])
        self.assertEqual((settings["preset"], settings["download"]), ("best", "both"))

    def test_models_and_program_can_live_on_different_drives(self):
        code, _, out = self.run_check(pc(gpus=[gpu(24)], disk_free=50.0, hf_free=2.0))
        self.assertEqual(code, 1)
        self.assertIn("free where the models are stored", out)
        # the program alone needs 5 GB + 1 GB spare on its own drive
        self.assertEqual(self.run_check(pc(gpus=[gpu(24)], disk_free=5.9, hf_free=50.0))[0], 1)
        self.assertEqual(self.run_check(pc(gpus=[gpu(24)], disk_free=6.1, hf_free=50.0))[0], 0)

    def test_an_existing_install_needs_no_space(self):
        self.fake_venv("2.11.0+cu128")
        self.fake_cache(config.TORCH_MODELS["best"])
        self.fake_cache(config.SPEAKER_MODEL)
        profile = pc(gpus=[gpu(24)], disk_free=1.5)
        code, settings, out = self.run_check(profile)
        self.assertEqual(code, 0, out)
        self.assertIn("already installed", out)

    def test_a_cpu_venv_does_not_count_for_an_nvidia_install(self):
        self.fake_venv("2.11.0+cpu")
        profile = pc(gpus=[gpu(24)])
        plan = precheck.disk_plan(profile, precheck.assess(profile)["options"]["best"], ["best"], self.root)
        self.assertEqual(plan["venv_gb"], 5.0)
        self.assertTrue(precheck.venv_ready(self.root, "cpu"))
        self.assertFalse(precheck.venv_ready(self.root, "nvidia"))


# ----------------------------------------------------------------------------- interaction, JSON, settings

class RunTests(Sandbox):
    def test_non_tty_accepts_the_recommendation_without_asking(self):
        asked = []
        code, settings, out = self.run_check(pc(gpus=[gpu(24)]), interactive=False, ask=lambda p: asked.append(p))
        self.assertEqual((code, settings["preset"], asked), (0, "best", []))
        self.assertIn('Using the recommended choice: "best"', out)
        self.assertTrue(os.path.exists(self.settings))

    def test_yes_skips_the_question(self):
        asked = []
        _, settings, _ = self.run_check(pc(gpus=[gpu(24)]), interactive=True, yes=True, ask=lambda p: asked.append(p))
        self.assertEqual((settings["preset"], asked), ("best", []))

    def test_interactive_enter_keeps_the_recommendation(self):
        prompts = []
        _, settings, out = self.run_check(pc(gpus=[gpu(24)]), interactive=True, ask=lambda p: prompts.append(p) or "")
        self.assertEqual(settings["preset"], "best")
        self.assertEqual(prompts, ['Press Enter to use "best", or type "light": '])
        self.assertEqual(settings["chosen_by"], "recommended")

    def test_interactive_override(self):
        _, settings, out = self.run_check(pc(gpus=[gpu(24)]), interactive=True, ask=lambda p: " Light ")
        self.assertEqual((settings["preset"], settings["chosen_by"]), ("light", "user"))
        self.assertIn('Using "light" (your choice)', out)

    def test_interactive_typo_asks_again_then_gives_up(self):
        answers = iter(["blah", "light"])
        _, settings, out = self.run_check(pc(gpus=[gpu(24)]), interactive=True, ask=lambda p: next(answers))
        self.assertEqual(settings["preset"], "light")
        self.assertIn("Please press Enter or type one of", out)
        _, settings, _ = self.run_check(pc(gpus=[gpu(24)]), interactive=True, ask=lambda p: "nope")
        self.assertEqual(settings["preset"], "best")

    def test_end_of_input_at_the_prompt_takes_the_default(self):
        def eof(_prompt):
            raise EOFError

        _, settings, _ = self.run_check(pc(gpus=[gpu(24)]), interactive=True, ask=eof)
        self.assertEqual(settings["preset"], "best")

    def test_override_to_a_choice_that_does_not_fit_warns(self):
        _, settings, out = self.run_check(pc(ram=8), interactive=True, ask=lambda p: "best")
        self.assertEqual(settings["preset"], "best")
        self.assertIn("will probably fail or crawl", out)

    def test_json_output_is_machine_readable_and_does_not_save(self):
        code, settings, out = self.run_check(pc(gpus=[gpu(24)]), as_json=True)
        data = json.loads(out)
        self.assertEqual((code, data["selected"], data["saved"]), (0, "best", False))
        self.assertEqual(data["assessment"]["recommended"], "best")
        self.assertEqual(data["problems"], [])
        self.assertEqual(data["hardware"]["nvidia"]["gpus"][0]["compute_capability"], [7, 5])
        self.assertEqual(sorted(data["assessment"]["options"]), ["best", "light"])
        self.assertFalse(os.path.exists(self.settings))
        self.assertNotIn("Hardware check", out)

    def test_json_on_the_real_stdout_is_not_reflowed(self):
        buf = io.StringIO()
        with mock.patch.object(precheck, "detect", return_value=pc(gpus=[gpu(24)])),                 mock.patch.object(precheck, "settings_path", return_value=self.settings), contextlib.redirect_stdout(buf):
            code = precheck.main(["--json", "--no-save"])
        self.assertEqual(code, 0)
        data = json.loads(buf.getvalue())  # the whole stdout is one JSON document
        self.assertEqual(data["selected"], "best")
        self.assertGreater(len(buf.getvalue().splitlines()), 50)

    def test_json_with_yes_saves(self):
        _, _, out = self.run_check(pc(gpus=[gpu(24)]), as_json=True, yes=True)
        self.assertTrue(json.loads(out)["saved"])
        self.assertTrue(os.path.exists(self.settings))

    def test_json_reports_blocking_problems(self):
        code, _, out = self.run_check(mac(silicon=False), as_json=True)
        data = json.loads(out)
        self.assertEqual(code, 1)
        self.assertTrue(any("Intel" in p for p in data["problems"]))
        self.assertIsNone(data["selected"])

    def test_no_save(self):
        code, settings, _ = self.run_check(pc(gpus=[gpu(24)]), save=False)
        self.assertEqual(code, 0)
        self.assertFalse(os.path.exists(self.settings))

    def test_saved_file_has_the_agreed_fields(self):
        self.run_check(pc(gpus=[gpu(24, name="NVIDIA Quadro RTX 6000")]))
        with open(self.settings, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual((data["preset"], data["device"], data["batch_size"]), ("best", "cuda", 8))
        self.assertEqual(data["model"], "Qwen/Qwen3-ASR-1.7B")
        self.assertIn("Quadro RTX 6000", data["hardware"])
        self.assertRegex(data["date"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertEqual(data["recommended"], "best")

    def test_unwritable_settings_location_is_a_warning_not_a_crash(self):
        bad = os.path.join(self.tmp.name, "no", "such", "dir", "s.json")
        out = []
        code, settings = precheck.run([], profile=pc(gpus=[gpu(24)]), root=self.root, settings_file=bad, interactive=False,
                                      out=out.append)
        self.assertEqual(code, 0)
        self.assertIn("could not save", "\n".join(out))

    def test_parse_words(self):
        self.assertEqual(precheck.parse_words([]), ("auto", False))
        self.assertEqual(precheck.parse_words(["Best", "CPU"]), ("best", True))
        self.assertEqual(precheck.parse_words(["auto", "cpu"]), ("auto", True))
        for bad in (["fast"], ["best", "light"]):
            with self.assertRaises(ValueError):
                precheck.parse_words(bad)

    def test_main_rejects_unknown_arguments(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
            precheck.main(["turbo"])
        self.assertEqual(cm.exception.code, 2)


class SettingsFileTests(Sandbox):
    def test_missing_file(self):
        self.assertEqual(precheck.read_settings(self.settings), (None, None))
        self.assertIsNone(precheck.load_settings(self.settings))

    def test_round_trip(self):
        settings = {"version": 1, "preset": "light", "device": "cpu", "model": "m", "batch_size": 2, "download": "both",
                    "gpu_index": None, "gpu_count": 0, "hardware": "x", "date": "2026-01-01"}
        self.assertIsNone(precheck.save_settings(settings, self.settings))
        read, problem = precheck.read_settings(self.settings)
        self.assertIsNone(problem)
        self.assertEqual((read["preset"], read["device"], read["batch_size"], read["download"]), ("light", "cpu", 2, "both"))
        self.assertFalse(os.path.exists(self.settings + ".tmp"))

    def test_corrupt_files_are_reported_not_raised(self):
        cases = ["{not json", "", "[1, 2]", "null", '{"preset": "best"}',
                 '{"preset": "huge", "device": "cuda", "batch_size": 2}',
                 '{"preset": "best", "device": "tpu", "batch_size": 2}',
                 '{"preset": "best", "device": "cuda", "batch_size": 0}',
                 '{"preset": "best", "device": "cuda", "batch_size": "8"}',
                 '{"preset": "best", "device": "cuda", "batch_size": true}']
        for text in cases:
            with open(self.settings, "w", encoding="utf-8") as f:
                f.write(text)
            settings, problem = precheck.read_settings(self.settings)
            self.assertIsNone(settings, text)
            self.assertTrue(problem and "localtranscribe_settings.json" in problem, text)

    def test_binary_garbage(self):
        with open(self.settings, "wb") as f:
            f.write(b"\xff\xfe\x00garbage\x80")
        self.assertIsNone(precheck.read_settings(self.settings)[0])

    def test_extra_or_odd_optional_fields_are_tolerated(self):
        with open(self.settings, "w", encoding="utf-8") as f:
            json.dump({"preset": "best", "device": "cuda", "batch_size": 4, "download": "everything", "gpu_index": -3,
                       "gpu_count": "two", "future_field": 1}, f)
        settings, problem = precheck.read_settings(self.settings)
        self.assertIsNone(problem)
        self.assertEqual((settings["download"], settings["gpu_index"], settings["gpu_count"]), ("best", None, 1))

    def test_print_mode(self):
        self.run_check(pc(gpus=[gpu(24)]))
        outputs = {}
        for key in ("preset", "device", "models", "requirements", "batch"):
            buf = io.StringIO()
            with mock.patch.object(precheck, "settings_path", return_value=self.settings), contextlib.redirect_stdout(buf):
                self.assertEqual(precheck.main(["--print", key]), 0)
            outputs[key] = buf.getvalue().strip()
        self.assertEqual(outputs, {"preset": "best", "device": "cuda", "models": "best",
                                   "requirements": "requirements-windows.txt", "batch": "8"})
        os.remove(self.settings)
        with mock.patch.object(precheck, "settings_path", return_value=self.settings), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(precheck.main(["--print", "preset"]), 1)


class FirstRunTests(Sandbox):
    def first_run(self, profile):
        out = []
        settings = precheck.first_run_check(root=self.root, settings_file=self.settings, out=out.append, profile=profile)
        return settings, "\n".join(out)

    def test_saves_the_recommendation(self):
        settings, out = self.first_run(pc(gpus=[gpu(24)]))
        self.assertEqual(settings["preset"], "best")
        self.assertIn("Recommended: best", out)
        self.assertNotIn("Disk", out)  # the program is installed already: disk space is not the question
        self.assertEqual(precheck.load_settings(self.settings)["preset"], "best")

    def test_prefers_a_model_that_is_already_downloaded(self):
        self.fake_cache(config.TORCH_MODELS["light"])
        settings, out = self.first_run(pc(gpus=[gpu(24)]))
        self.assertEqual(settings["preset"], "light")
        self.assertIn('"best" is recommended', out)
        self.assertFalse(os.path.exists(self.settings))  # not saved: it checks again after the better model is downloaded
        self.fake_cache(config.TORCH_MODELS["best"])
        settings, _ = self.first_run(pc(gpus=[gpu(24)]))
        self.assertEqual(settings["preset"], "best")

    def test_nothing_to_recommend_returns_none(self):
        settings, out = self.first_run(mac(silicon=False))
        self.assertIsNone(settings)


# ----------------------------------------------------------------------------- detection with mocked programs

NVIDIA_TWO = "NVIDIA GeForce RTX 3060, 12288 MiB, 11800 MiB, 572.16, 8.6\nNVIDIA GeForce GT 1030, 2048 MiB, 1900 MiB, 572.16, 6.1\n"
HARDWARE_JSON = json.dumps({"SPHardwareDataType": [{"_name": "hardware_overview", "chip_type": "Apple M2",
                                                    "machine_model": "Mac14,2", "machine_name": "MacBook Air",
                                                    "physical_memory": "8 GB"}]})
DISPLAYS_JSON = json.dumps({"SPDisplaysDataType": [{"_name": "Apple M2", "sppci_cores": "10", "sppci_model": "Apple M2"}]})


class DetectionTests(unittest.TestCase):
    def test_parse_nvidia_smi_two_gpus(self):
        gpus = precheck.parse_nvidia_smi(NVIDIA_TWO)
        self.assertEqual([g["name"] for g in gpus], ["NVIDIA GeForce RTX 3060", "NVIDIA GeForce GT 1030"])
        self.assertEqual([g["index"] for g in gpus], [0, 1])
        self.assertEqual(gpus[0]["total_gb"], 12.0)
        self.assertAlmostEqual(gpus[0]["free_gb"], 11800 / 1024)
        self.assertEqual([g["compute_capability"] for g in gpus], [[8, 6], [6, 1]])
        self.assertEqual(gpus[0]["driver"], "572.16")

    def test_parse_nvidia_smi_odd_output(self):
        self.assertEqual(precheck.parse_nvidia_smi(""), [])
        self.assertEqual(precheck.parse_nvidia_smi("Failed to initialize NVML: Driver/library version mismatch"), [])
        gpus = precheck.parse_nvidia_smi("Quadro RTX 6000, 23040 MiB, 22132 MiB, 597.06, 7.5")
        self.assertEqual(gpus[0]["total_gb"], 22.5)
        self.assertEqual(precheck.parse_nvidia_smi("X, 4096 MiB, 4000 MiB, 470.1\n")[0]["compute_capability"], None)
        self.assertEqual(precheck.parse_nvidia_smi("X, 4096 MiB, [N/A], 570.1, [N/A]")[0]["free_gb"], None)

    def detect_with(self, results, system="Windows", exe="C:/nvidia-smi.exe"):
        calls = []

        def fake_exec(cmd, timeout=10):
            calls.append(cmd)
            return results.pop(0)

        with mock.patch.object(precheck, "_find_nvidia_smi", return_value=exe), mock.patch.object(precheck, "_exec", fake_exec):
            return precheck.detect_nvidia(system), calls

    def test_detect_nvidia_uses_the_documented_query(self):
        info, calls = self.detect_with([(0, NVIDIA_TWO, "")])
        self.assertEqual(calls[0][1:], ["--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
                                        "--format=csv,noheader"])
        self.assertEqual((info["smi"], info["driver"], len(info["gpus"])), ("found", "572.16", 2))

    def test_detect_nvidia_missing(self):
        with mock.patch.object(precheck, "_find_nvidia_smi", return_value=None):
            info = precheck.detect_nvidia("Windows")
        self.assertEqual((info["smi"], info["gpus"]), ("missing", []))
        self.assertEqual(precheck.detect_nvidia("Darwin")["smi"], "missing")

    def test_detect_nvidia_failure_and_timeout(self):
        info, _ = self.detect_with([(9, "", "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n")])
        self.assertEqual(info["smi"], "failed")
        self.assertIn("couldn't communicate", info["message"])
        info, _ = self.detect_with([(None, "", "no answer within 15 seconds")])
        self.assertEqual((info["smi"], info["message"]), ("failed", "no answer within 15 seconds"))
        info, _ = self.detect_with([(0, "", "")])
        self.assertEqual(info["smi"], "failed")

    def test_detect_nvidia_old_driver_without_compute_cap(self):
        info, calls = self.detect_with([(2, "", 'Field "compute_cap" is not a valid field to query.\n'),
                                        (0, "Quadro K2200, 4096 MiB, 3900 MiB, 452.06\n", "")])
        self.assertEqual(len(calls), 2)
        self.assertNotIn("compute_cap", calls[1][1])
        self.assertEqual(info["gpus"][0]["compute_capability"], None)
        self.assertEqual(info["driver"], "452.06")

    def test_parse_system_profiler(self):
        self.assertEqual(precheck.parse_system_profiler_hardware(HARDWARE_JSON), ("MacBook Air", "Mac14,2", "Apple M2"))
        self.assertEqual(precheck.parse_system_profiler_gpu_cores(DISPLAYS_JSON), 10)
        text = "Hardware Overview:\n\n      Model Name: MacBook Air\n      Model Identifier: Mac14,2\n      Chip: Apple M2\n"
        self.assertEqual(precheck.parse_system_profiler_hardware(text), ("MacBook Air", "Mac14,2", "Apple M2"))
        self.assertEqual(precheck.parse_system_profiler_gpu_cores("Chipset Model: Apple M2\n      Total Number of Cores: 8\n"), 8)
        self.assertEqual(precheck.parse_system_profiler_hardware("Nome del modello: MacBook Air"), (None, None, None))
        self.assertIsNone(precheck.parse_system_profiler_gpu_cores("garbage"))

    def mac_detect(self, machine="arm64", sysctl=None, hardware=HARDWARE_JSON, displays=DISPLAYS_JSON):
        table = {"hw.optional.arm64": "1", "sysctl.proc_translated": "0", "machdep.cpu.brand_string": "Apple M2",
                 "hw.memsize": str(8 * 1024 ** 3), "hw.model": "Mac14,2"}
        table.update(sysctl or {})

        def fake_run(cmd, timeout=10):
            if cmd[0] == "system_profiler":
                return hardware if cmd[-1] == "SPHardwareDataType" else displays
            return None

        with mock.patch.object(sys, "platform", "darwin"), mock.patch("platform.machine", return_value=machine), \
                mock.patch.object(precheck, "_sysctl", lambda name: table.get(name)), mock.patch.object(precheck, "_run", fake_run):
            return precheck.detect_apple()

    def test_detect_apple_m2_air(self):
        apple = self.mac_detect()
        self.assertEqual((apple["apple_silicon"], apple["rosetta"], apple["fanless"]), (True, False, True))
        self.assertEqual((apple["chip"], apple["model_name"], apple["gpu_cores"], apple["unified_gb"]),
                         ("Apple M2", "MacBook Air", 10, 8.0))

    def test_detect_apple_rosetta(self):
        apple = self.mac_detect(machine="x86_64", sysctl={"sysctl.proc_translated": "1"})
        self.assertEqual((apple["apple_silicon"], apple["rosetta"]), (True, True))

    def test_detect_apple_intel(self):
        apple = self.mac_detect(machine="x86_64", sysctl={"hw.optional.arm64": None, "sysctl.proc_translated": None,
                                                          "machdep.cpu.brand_string": "Intel(R) Core(TM) i7"})
        self.assertEqual((apple["apple_silicon"], apple["rosetta"]), (False, False))

    def test_detect_apple_survives_missing_system_profiler(self):
        apple = self.mac_detect(hardware=None, displays=None)
        self.assertEqual((apple["chip"], apple["model_name"], apple["gpu_cores"], apple["fanless"]),
                         ("Apple M2", None, None, False))
        self.assertTrue(apple["apple_silicon"])
        # the model identifier alone also identifies an Air
        apple = self.mac_detect(hardware=None, sysctl={"hw.model": "MacBookAir10,1"})
        self.assertTrue(apple["fanless"])

    def test_detect_apple_off_a_mac_is_none(self):
        with mock.patch.object(sys, "platform", "win32"):
            self.assertIsNone(precheck.detect_apple())

    def test_detect_os_reads_the_mac_version_from_sw_vers(self):
        with mock.patch("platform.system", return_value="Darwin"), mock.patch.object(precheck, "_run", return_value="15.6"):
            self.assertEqual(precheck.detect_os()["name"], "macOS 15.6")

    def test_detect_os_names_windows_11(self):
        with mock.patch("platform.system", return_value="Windows"), mock.patch("platform.version", return_value="10.0.26200"):
            self.assertEqual(precheck.detect_os()["name"], "Windows 11 (build 26200)")
        with mock.patch("platform.system", return_value="Windows"), mock.patch("platform.version", return_value="10.0.19045"):
            self.assertEqual(precheck.detect_os()["name"], "Windows 10 (build 19045)")

    def test_real_detection_returns_a_serialisable_profile(self):
        profile = precheck.detect()
        json.dumps(profile)
        self.assertIn(profile["os"]["system"], ("Windows", "Darwin", "Linux"))
        self.assertTrue(profile["cpu"]["threads"])
        self.assertTrue(profile["memory"]["total_gb"])
        self.assertTrue(profile["disk"]["repo_free_gb"])

    def test_precheck_needs_only_the_standard_library(self):
        code = ("import sys; sys.path.insert(0, %r); import localtranscribe.precheck as p; p.detect();"
                "print([m for m in ('numpy','torch','mlx','mlx_audio','transformers','huggingface_hub','av','qwen_asr') "
                "if m in sys.modules])" % ROOT)
        out = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True)  # -S: no site-packages at all
        self.assertEqual(out.stdout.strip(), "[]", out.stderr)


# ----------------------------------------------------------------------------- the command line tool and devices

SAVED_CUDA = {"preset": "best", "device": "cuda", "batch_size": 4, "model": None, "download": "best", "gpu_index": None,
              "gpu_count": 1}


class RunConfigTests(unittest.TestCase):
    def resolve(self, saved, device="auto", model=None, batch=0, cuda=(24.0, 22.0), notes=None):
        """resolve_run_config on a mocked NVIDIA machine (cuda=None: no usable GPU)."""
        with mock.patch.object(devices, "is_apple_silicon", return_value=False), \
                mock.patch.object(devices, "cuda_memory_gb", return_value=cuda), \
                mock.patch.object(devices, "total_ram_gb", return_value=32.0):
            return devices.resolve_run_config(device, model, batch, saved=saved, notes=notes)

    def test_saved_choice_fills_the_defaults(self):
        device, repo, batch, _ = self.resolve(dict(SAVED_CUDA, preset="light", batch_size=3), device=None)
        self.assertEqual((device, repo, batch), ("cuda", config.TORCH_MODELS["light"], 3))

    def test_no_saved_choice_is_the_old_automatic_behaviour(self):
        device, repo, batch, _ = self.resolve(None, device=None)
        self.assertEqual((device, repo, batch), ("cuda", config.TORCH_MODELS["best"], 8))
        device, repo, batch, _ = self.resolve(None, device="auto", model="auto", cuda=(4.0, 3.5))
        self.assertEqual((repo, batch), (config.TORCH_MODELS["light"], 4))

    def test_explicit_flags_win(self):
        _, repo, batch, _ = self.resolve(SAVED_CUDA, device=None, model="light", batch=6)
        self.assertEqual((repo, batch), (config.TORCH_MODELS["light"], 6))
        _, repo, _, _ = self.resolve(dict(SAVED_CUDA, preset="light"), device=None, model="auto")  # explicit auto = the rule
        self.assertEqual(repo, config.TORCH_MODELS["best"])
        _, repo, _, _ = self.resolve(SAVED_CUDA, device=None, model="acme/other")
        self.assertEqual(repo, "acme/other")

    def test_saved_batch_is_a_ceiling_lowered_when_memory_is_tight(self):
        _, _, batch, _ = self.resolve(dict(SAVED_CUDA, batch_size=8), device=None, cuda=(24.0, 5.5))
        self.assertEqual(batch, 2)
        _, _, batch, _ = self.resolve(dict(SAVED_CUDA, batch_size=1), device=None)
        self.assertEqual(batch, 1)

    def test_saved_batch_belongs_to_the_saved_model(self):
        _, _, batch, _ = self.resolve(dict(SAVED_CUDA, batch_size=2), device=None, model="light")
        self.assertEqual(batch, 8)  # light needs less memory: the batch saved for best does not apply

    def test_saved_cpu_choice_forces_the_cpu_unless_a_device_is_given(self):
        saved = dict(SAVED_CUDA, device="cpu", preset="light", batch_size=1)
        device, repo, batch, _ = self.resolve(saved, device=None)
        self.assertEqual((device, repo, batch), ("cpu", config.TORCH_MODELS["light"], 1))
        device, _, _, _ = self.resolve(saved, device="auto")
        self.assertEqual(device, "cuda")

    def test_saved_gpu_choice_is_ignored_when_there_is_no_gpu(self):
        notes = []
        device, repo, _, _ = self.resolve(SAVED_CUDA, device=None, cuda=None, notes=notes)
        self.assertEqual((device, repo), ("cpu", config.TORCH_MODELS["light"]))  # not "best on the CPU"
        self.assertTrue(notes and "saved hardware choice" in notes[0])

    def test_explicit_device_that_differs_from_the_saved_one_uses_the_rule_quietly(self):
        notes = []
        device, repo, _, _ = self.resolve(SAVED_CUDA, device="cpu", notes=notes)
        self.assertEqual((device, repo), ("cpu", config.TORCH_MODELS["light"]))
        self.assertEqual(notes, [])

    def test_mac_saved_choice(self):
        saved = dict(SAVED_CUDA, device="mps", preset="best", batch_size=2)
        with mock.patch.object(devices, "is_apple_silicon", return_value=True), \
                mock.patch.object(devices, "total_ram_gb", return_value=8.0), mock.patch.object(devices, "check_platform"):
            device, repo, batch, _ = devices.resolve_run_config(None, None, 0, saved=saved)
        self.assertEqual((device, repo, batch), ("mps", config.MAC_MODELS["best"], 1))  # batch lowered by the 8 GB rule


class CliTests(Sandbox):
    def args(self, *argv):
        return cli.build_parser().parse_args(["a.wav", *argv])

    def test_defaults_are_unset_so_that_the_saved_choice_can_fill_them(self):
        args = self.args()
        self.assertEqual((args.model, args.device, args.batch_size), (None, None, 0))

    def test_saved_choice_is_read_without_checking_again(self):
        precheck.save_settings({"preset": "light", "device": "cpu", "batch_size": 1}, self.settings)
        with mock.patch.object(precheck, "settings_path", return_value=self.settings), \
                mock.patch.object(precheck, "first_run_check", side_effect=AssertionError("must not run")):
            self.assertEqual(cli._saved_choice(self.args())["preset"], "light")

    def test_nothing_to_look_up_when_everything_is_explicit(self):
        with mock.patch.object(precheck, "first_run_check", side_effect=AssertionError("must not run")):
            self.assertIsNone(cli._saved_choice(self.args("--model", "light", "--device", "cpu", "--batch-size", "2")))

    def test_missing_file_runs_the_check_once(self):
        buf = io.StringIO()
        with mock.patch.object(precheck, "settings_path", return_value=self.settings), \
                mock.patch.object(precheck, "first_run_check", return_value=dict(SAVED_CUDA)) as check, \
                contextlib.redirect_stdout(buf):
            self.assertEqual(cli._saved_choice(self.args())["preset"], "best")
        self.assertEqual(check.call_count, 1)
        self.assertIn("No saved hardware check yet", buf.getvalue())

    def test_corrupt_file_runs_the_check_again_and_says_why(self):
        with open(self.settings, "w", encoding="utf-8") as f:
            f.write("{oops")
        buf = io.StringIO()
        with mock.patch.object(precheck, "settings_path", return_value=self.settings), \
                mock.patch.object(precheck, "first_run_check", return_value=None) as check, contextlib.redirect_stdout(buf):
            self.assertIsNone(cli._saved_choice(self.args()))
        self.assertEqual(check.call_count, 1)
        self.assertIn("could not be read", buf.getvalue())

    def test_a_failing_check_never_stops_the_run(self):
        buf = io.StringIO()
        with mock.patch.object(precheck, "settings_path", return_value=self.settings), \
                mock.patch.object(precheck, "first_run_check", side_effect=RuntimeError("boom")), contextlib.redirect_stdout(buf):
            self.assertIsNone(cli._saved_choice(self.args()))
        self.assertIn("using the automatic choice", buf.getvalue())

    def test_first_run_end_to_end_writes_the_file(self):
        buf = io.StringIO()
        profile = pc(gpus=[gpu(8)])
        with mock.patch.object(precheck, "settings_path", return_value=self.settings), \
                mock.patch.object(precheck, "detect", return_value=profile), contextlib.redirect_stdout(buf):
            saved = cli._saved_choice(self.args())
        self.assertEqual((saved["preset"], saved["device"]), ("best", "cuda"))
        self.assertTrue(os.path.exists(self.settings))
        self.assertIn("Recommended: best", buf.getvalue())

    def test_several_gpus_pin_the_chosen_one_before_torch_loads(self):
        saved = dict(SAVED_CUDA, gpu_index=1, gpu_count=2)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            cli._pin_gpu(saved)
            self.assertEqual((os.environ["CUDA_VISIBLE_DEVICES"], os.environ["CUDA_DEVICE_ORDER"]), ("1", "PCI_BUS_ID"))
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            cli._pin_gpu(saved)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")  # the user's own setting wins
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            cli._pin_gpu(dict(SAVED_CUDA, gpu_index=0, gpu_count=1))  # one GPU: nothing to pin
            cli._pin_gpu(None)
            self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)


if __name__ == "__main__":
    unittest.main()
