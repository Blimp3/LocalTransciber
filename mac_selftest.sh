#!/usr/bin/env bash
# Self-test for a real Apple Silicon Mac. Measures accuracy, speed and memory of every Mac model preset
# and of speaker separation, and writes ONE report file: mac_selftest_report.txt (no personal data).
#
#   bash mac_selftest.sh            # full test: 100 FLEURS clips per preset (about 1 hour on a MacBook Air)
#   bash mac_selftest.sh --n 30     # quicker test with 30 clips per preset
#   bash mac_selftest.sh --models mlx-community/Qwen3-ASR-0.6B-8bit    # only some presets
#
# Needs an internet connection (downloads the test clips and the models being tested).
# Keep the Mac plugged in and do not run other heavy programs meanwhile.
cd "$(dirname "$0")" || exit 1
DIR="$(pwd)"

if [ "$(uname -s)" != "Darwin" ]; then
  echo "This self-test is meant for a Mac. (It also runs elsewhere, but the numbers are only useful on the target Mac.)" >&2
fi
if [ ! -x "$DIR/.venv/bin/python" ]; then
  echo "LocalTranscribe is not set up yet. Run first:  bash setup_mac.sh" >&2
  exit 1
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export PYTHONPATH="$DIR${PYTHONPATH:+:$PYTHONPATH}"
unset HF_HUB_OFFLINE   # this script downloads; the measured runs themselves are started offline

# keep the Mac awake while the test runs
if command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -i "$DIR/.venv/bin/python" "$DIR/benchmark/mac_selftest.py" "$@"
fi
exec "$DIR/.venv/bin/python" "$DIR/benchmark/mac_selftest.py" "$@"
