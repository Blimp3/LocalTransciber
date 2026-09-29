#!/bin/bash
# Doppio clic in Finder: scegli uno o piu file audio/video, li trascrive (tutto sul tuo Mac)
# e mostra i file .md nel Finder. Il .md viene creato accanto a ogni file audio.
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

TMP_SCRIPT="$(mktemp "${TMPDIR:-/tmp}/trascrivi_pick.XXXXXX")"
cat > "$TMP_SCRIPT" <<'APPLESCRIPT'
set theFiles to choose file with prompt "Scegli i file audio o video da trascrivere (tieni premuto Cmd per sceglierne diversi)" with multiple selections allowed
set out to ""
repeat with f in theFiles
	set out to out & (POSIX path of f) & linefeed
end repeat
return out
APPLESCRIPT
PICKED="$(osascript "$TMP_SCRIPT" 2>/dev/null)"
PICK_STATUS=$?
rm -f "$TMP_SCRIPT"
if [ $PICK_STATUS -ne 0 ] || [ -z "$PICKED" ]; then
  echo "Nessun file scelto."
  exit 0
fi

FILES=()
while IFS= read -r line; do
  [ -n "$line" ] && FILES+=("$line")
done <<< "$PICKED"

EXTRA=()
CHOICE="$(osascript -e 'button returned of (display dialog "Vuoi separare chi parla? Utile per una telefonata tra due persone." with title "Trascrivi" buttons {"Annulla", "Separa 2 persone", "Testo unico"} default button "Testo unico")' 2>/dev/null)"
case "$CHOICE" in
  "Separa 2 persone") EXTRA=(--speakers 2) ;;
  "Testo unico")
    AI="$(osascript -e 'button returned of (display dialog "Vuoi i suggerimenti AI per le parole incerte? Un piccolo modello sul tuo Mac propone correzioni, che poi accetti o rifiuti con Rivedi.command. Ci vuole circa un quarto di tempo in più." with title "Trascrivi" buttons {"Annulla", "Sì, con suggerimenti", "No"} default button "No")' 2>/dev/null)"
    case "$AI" in
      "Sì, con suggerimenti") EXTRA=(--correct) ;;
      "No") ;;
      *) echo "Annullato."; exit 0 ;;
    esac ;;
  *) echo "Annullato."; exit 0 ;;
esac

echo "Trascrizione di ${#FILES[@]} file. La prima volta il programma ci mette qualche istante a partire..."
echo
"$PY" -m localtranscribe "${FILES[@]}" "${EXTRA[@]}"
STATUS=$?

OUTS=()
for f in "${FILES[@]}"; do
  base="$(basename "$f")"
  out="$(dirname "$f")/${base%.*}.md"
  [ -f "$out" ] && OUTS+=("$out")
done
if [ ${#OUTS[@]} -gt 0 ]; then
  open -R "${OUTS[@]}"
fi

echo
if [ $STATUS -eq 0 ]; then
  echo "Fatto. I file .md sono accanto ai file audio (li trovi selezionati nel Finder)."
else
  echo "Finito con qualche problema (vedi i messaggi sopra)."
fi
if [ $STATUS -eq 0 ] && [ "${EXTRA[0]}" = "--correct" ]; then
  echo "Per vedere e accettare o rifiutare i suggerimenti, fai doppio clic su Rivedi.command."
fi
echo "Premi Invio per chiudere questa finestra."
read -r _
exit $STATUS
