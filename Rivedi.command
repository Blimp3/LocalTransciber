#!/bin/bash
# Doppio clic in Finder: scegli una registrazione gia trascritta e rivedi il testo (.md) nel browser,
# con l'audio accanto. Tutto resta sul tuo Mac (la pagina e visibile solo da questo computer).
cd "$(dirname "$0")" || exit 1
DIR="$(pwd)"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export PYTHONPATH="$DIR${PYTHONPATH:+:$PYTHONPATH}"
PY="$DIR/.venv/bin/python"

if [ ! -x "$PY" ]; then
  osascript -e 'display dialog "Il programma non e ancora installato. Apri il Terminale in questa cartella ed esegui:  bash setup_mac.sh" buttons {"OK"} default button 1 with icon caution' >/dev/null 2>&1
  echo "Programma non installato: esegui  bash setup_mac.sh"
  exit 1
fi

f="$(osascript -e 'POSIX path of (choose file with prompt "Scegli la registrazione da rivedere (il file .md deve essere accanto)")' 2>/dev/null)"
if [ -z "$f" ]; then
  echo "Nessun file scelto."
  exit 0
fi

echo "La pagina di revisione si apre nel browser. Quando hai finito, chiudi questa finestra (oppure premi Ctrl+C)."
echo
"$PY" -m localtranscribe.review "$f"
STATUS=$?
if [ $STATUS -ne 0 ]; then
  echo
  echo "Premi Invio per chiudere questa finestra."
  read -r _
fi
exit $STATUS
