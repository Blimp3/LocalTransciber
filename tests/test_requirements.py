"""The Windows requirement files must pin torch to an exact local version (+cu128 / +cpu) and must not
use an extra index.   python -m unittest discover -s tests -v

The local version is what makes setup_windows.bat switch a PC between the CUDA and CPU builds of torch;
--extra-index-url with unsafe-best-match made uv ask the PyTorch index for every package."""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


def pins(text, package):
    """Active (non-comment) requirement lines for one package."""
    return [line.strip() for line in text.splitlines()
            if re.match(rf"{re.escape(package)}\s*[=<>~!]", line.strip())]


class RequirementFiles(unittest.TestCase):
    def test_windows_pins_torch_cuda_build(self):
        found = pins(read("requirements-windows.txt"), "torch")
        self.assertEqual(len(found), 1, found)
        self.assertRegex(found[0], r"^torch==\d+\.\d+\.\d+\+cu128(\s+#.*)?$")

    def test_cpu_pins_torch_cpu_build(self):
        found = pins(read("requirements-cpu.txt"), "torch")
        self.assertEqual(len(found), 1, found)
        self.assertRegex(found[0], r"^torch==\d+\.\d+\.\d+\+cpu(\s+#.*)?$")

    def test_same_torch_release_in_both_files(self):
        def release(name):
            return re.match(r"torch==(\d+\.\d+\.\d+)\+", pins(read(name), "torch")[0]).group(1)
        self.assertEqual(release("requirements-windows.txt"), release("requirements-cpu.txt"))

    def test_no_extra_index_or_unsafe_best_match(self):
        for name in ("requirements-windows.txt", "requirements-cpu.txt"):
            text = read(name)
            for bad in ("--extra-index-url", "unsafe-best-match"):
                self.assertNotIn(bad, text, f"{name} contains {bad}")


if __name__ == "__main__":
    unittest.main()
