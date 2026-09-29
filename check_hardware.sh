#!/usr/bin/env bash
# Check this computer (memory, graphics, disk space), recommend the speech model that fits it and save your choice
# in localtranscribe_settings.json. It only looks and saves; it installs and downloads nothing.
#
#   bash check_hardware.sh                 look, recommend, ask (press Enter to accept)
#   bash check_hardware.sh best            choose without asking:  best | light | both | cpu (cpu: PCs only)
#   bash check_hardware.sh --yes           accept the recommendation without asking
#   bash check_hardware.sh --json          machine-readable result, nothing is saved (add --yes to save)
#   bash check_hardware.sh --no-save       look only
#
# Works before setup_mac.sh has run (it then needs uv or a Python 3 on the PATH), and again at any time afterwards.
cd "$(dirname "$0")" || exit 1

# An Apple Silicon Mac whose Terminal runs in Intel (Rosetta) mode: restart this script natively.
if [ "$(uname -s)" = "Darwin" ] && [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = "1" ] \
    && [ "$(uname -m)" != "arm64" ]; then
  exec arch -arm64 /bin/bash "$0" "$@"
fi

# Any Python does: the check needs only the standard library.
PY=""
if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  for d in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    if [ -x "$d/uv" ]; then PATH="$d:$PATH"; fi
  done
  if command -v uv >/dev/null 2>&1; then
    PY="$(uv python find 3.11 2>/dev/null || true)"
  fi
  # On a Mac /usr/bin/python3 is a stub that offers to install developer tools, so it is not used there.
  if [ -z "$PY" ] && [ "$(uname -s)" != "Darwin" ]; then
    PY="$(command -v python3 || command -v python || true)"
  fi
fi
if [ -z "$PY" ]; then
  echo "Python was not found. Run  bash setup_mac.sh  first: it installs everything, including this check." >&2
  exit 1
fi

PYTHONPATH="$PWD" "$PY" -m localtranscribe.precheck "$@"
STATUS=$?

QUIET=0
for a in "$@"; do
  if [ "$a" = "--json" ]; then QUIET=1; fi
done
if [ "$QUIET" -eq 0 ] && [ "$STATUS" -eq 0 ] && [ "$(uname -s)" = "Darwin" ]; then
  echo
  if [ ! -x .venv/bin/python ]; then
    echo "Next: run  bash setup_mac.sh  to install the program and download the model."
  else
    echo "If you changed the choice, run  bash setup_mac.sh  again to install and download what it needs."
  fi
fi
exit "$STATUS"
