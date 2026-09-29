#!/usr/bin/env bash
# LocalTranscribe - setup for Apple Silicon Macs (M1 or newer, macOS 14 Sonoma or newer).
#
# Usage:   bash setup_mac.sh [auto|best|light|both] [--yes]
#   auto   (default) checks this Mac, recommends a model and asks you to accept it or choose another
#   best   Qwen3-ASR-1.7B, 4-bit (1.6 GB download)      light  Qwen3-ASR-0.6B, 8-bit (1.0 GB download)
#   both   downloads both       --yes  accept the recommendation without asking
# An explicit best/light/both skips the question, but the hardware check is still shown and warns if it will not fit.
# The choice is saved in localtranscribe_settings.json; check_hardware.sh repeats the check at any time.
set -euo pipefail
cd "$(dirname "$0")"

say() { printf '%s\n' "$*"; }
die() { printf '\nError: %s\n' "$*" >&2; exit 1; }

say "============================================================"
say "  LocalTranscribe - setup for Mac"
say "============================================================"

[ "$(uname -s)" = "Darwin" ] || die "This script is for macOS. On Windows run setup_windows.bat instead."

# --- Apple Silicon only -------------------------------------------------------------------
if [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" != "1" ]; then
  die "This Mac has an Intel processor. LocalTranscribe needs an Apple Silicon Mac (M1 or newer); Intel Macs are not supported."
fi
if [ "$(uname -m)" != "arm64" ]; then
  # Terminal started in Intel (Rosetta) mode on an Apple Silicon Mac: restart this script natively.
  exec arch -arm64 /bin/bash "$0" "$@"
fi

# --- macOS version: the MLX wheels need macOS 14 or newer ------------------------------------
MACOS_VERSION="$(sw_vers -productVersion)"
MACOS_MAJOR="${MACOS_VERSION%%.*}"
if [ "$MACOS_MAJOR" -lt 14 ]; then
  die "macOS $MACOS_VERSION is too old. LocalTranscribe needs macOS 14 (Sonoma) or newer: Apple menu > System Settings > General > Software Update."
fi
RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
say "Mac: $(sysctl -n machdep.cpu.brand_string), ${RAM_GB} GB memory, macOS ${MACOS_VERSION}"
say

say "This will:"
say "  1. install 'uv' (a small tool that fetches Python and the packages) if it is missing"
say "  2. check this Mac (memory, disk space) and recommend the model that fits it"
say "  3. create a private Python environment in the folder .venv next to this file"
say "  4. install the packages (about 1 GB)"
say "  5. download the speech models once, so that later runs work without internet"
say "Nothing outside this folder, your home folder and the Hugging Face model cache (~/.cache/huggingface) is touched."
say

# Files downloaded as a ZIP lose their executable bit and get a 'quarantine' flag; undo both so that
# transcribe.sh and Trascrivi.command work (a double-click may still ask once: right-click > Open).
chmod +x setup_mac.sh transcribe.sh Trascrivi.command Rivedi.command mac_selftest.sh check_hardware.sh bin/guard-macos-arm64 2>/dev/null || true
xattr -dr com.apple.quarantine . 2>/dev/null || true

# --- uv -----------------------------------------------------------------------------------
for d in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
  if [ -x "$d/uv" ]; then export PATH="$d:$PATH"; fi
done
if ! command -v uv >/dev/null 2>&1; then
  say "'uv' is not installed. Installing it now for your user account with the official installer"
  say "from https://astral.sh/uv  (it downloads the uv program into ~/.local/bin and edits your shell profile)."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "Could not install uv automatically. Install it from https://docs.astral.sh/uv/ and run this setup again."
say "Using $(uv --version)"

# --- hardware check: standard library only, so it runs with the bare uv Python before any big download ---
BASEPY="$(uv python find 3.11 2>/dev/null || true)"
if [ -z "$BASEPY" ]; then
  say "Fetching Python 3.11 (about 30 MB) for the hardware check..."
  uv python install 3.11 || die "Could not fetch Python 3.11. Check your internet connection and run this setup again."
  BASEPY="$(uv python find 3.11 2>/dev/null || true)"
fi
[ -n "$BASEPY" ] || die "Could not find Python 3.11. Check your internet connection and run this setup again."
say
PYTHONPATH="$PWD" "$BASEPY" -m localtranscribe.precheck "$@" \
  || die "Setup stopped: see the message above. No packages or models have been downloaded."
REQ="$(PYTHONPATH="$PWD" "$BASEPY" -m localtranscribe.precheck --print requirements)" \
  || die "The hardware check did not save a choice. Run this setup again."
MODELS="$(PYTHONPATH="$PWD" "$BASEPY" -m localtranscribe.precheck --print models)" || MODELS=auto
say

# --- environment and packages ---------------------------------------------------------------
if [ ! -x .venv/bin/python ]; then
  say "Creating the Python 3.11 environment..."
  uv venv --python 3.11 .venv || die "Could not create the Python environment. Check your internet connection and run this setup again."
fi
say "Installing packages (a few minutes the first time)..."
uv pip install --python .venv/bin/python -r "$REQ" \
  || die "Package installation failed. Check your internet connection and run this setup again."

# --- models ---------------------------------------------------------------------------------
say
say "Downloading models..."
PYTHONPATH="$PWD" HF_HUB_DISABLE_SYMLINKS_WARNING=1 .venv/bin/python -m localtranscribe.setup_models --model "$MODELS" \
  || die "Some models could not be downloaded. Check your internet connection and run this setup again."

say
say "============================================================"
say "  Setup complete."
say "  Double-click  Trascrivi.command  in Finder to transcribe audio or video files,"
say "  or use  ./transcribe.sh file.m4a  in Terminal."
say "  Double-click  Rivedi.command  to review a transcript next to its audio."
say "  To change the model later, run  bash check_hardware.sh  and then this setup again."
say "  (If macOS refuses to open Trascrivi.command: right-click it > Open > Open, once.)"
say "============================================================"
