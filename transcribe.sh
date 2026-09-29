#!/usr/bin/env bash
# Transcribe audio/video files with LocalTranscribe (macOS/Linux command line).
#   ./transcribe.sh recording.m4a [video.mp4 ...] [--context "names, terms"] [--speakers 2] [--out-dir folder]
# Everything runs on this computer; the models were downloaded by setup_mac.sh.
DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export PYTHONPATH="$DIR${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -x "$DIR/.venv/bin/python" ]; then
  echo "LocalTranscribe is not set up yet. Run:  bash setup_mac.sh" >&2
  exit 1
fi
if [ "$#" -eq 0 ]; then
  echo "Usage: ./transcribe.sh recording.m4a [more files] [--context \"names, terms\"] [--speakers 2] [--out-dir folder]" >&2
  echo "Run ./transcribe.sh --help for all options." >&2
  exit 1
fi
exec "$DIR/.venv/bin/python" -m localtranscribe "$@"
