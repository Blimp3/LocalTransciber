"""Review page: the transcript (editable) on the left, the audio on the right, served only on this computer.

    python -m localtranscribe.review <recording> [--md PATH] [--port N] [--no-browser]
"""
import argparse
import html
import io
import json
import os
import re
import secrets
import sys
import threading
import wave
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from . import config
from .textutil import fmt_time

STAMP = re.compile(r"^\[(\d{2,}):(\d{2}):(\d{2})\] ?")
MAX_BODY = 20 * 1024 * 1024


def parse_md(text):
    """Paragraphs of a transcript .md: [{"start": seconds or None, "stamp": "hh:mm:ss" or None, "text": str}]."""
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        m = STAMP.match(block)
        if m:
            h, mi, s = (int(g) for g in m.groups())
            out.append({"start": h * 3600 + mi * 60 + s, "stamp": m.group(0).strip()[1:-1], "text": block[m.end():]})
        elif block:
            out.append({"start": None, "stamp": None, "text": block})
    return out


def join_md(paragraphs):
    """The .md text for edited paragraphs [{"stamp", "text"}]; emptied paragraphs are dropped."""
    parts = []
    for p in paragraphs:
        t = " ".join(str(p.get("text", "")).split())
        if t:
            parts.append(f"[{p['stamp']}] {t}" if p.get("stamp") else t)
    return "\n\n".join(parts) + "\n"


def add_suggestions(paragraphs, sidecar_path):
    """Add "sugg": [{"i", "from", "to"}] to each paragraph from the .review.json; a bad sidecar means none."""
    try:
        with open(sidecar_path, encoding="utf-8") as f:
            side = {fmt_time(q["offset"]): q for q in json.load(f)["paragraphs"]}
    except Exception:  # also OverflowError (Infinity offset) and RecursionError (deep JSON)
        side = {}
    for p in paragraphs:
        p["sugg"] = []
        try:
            words, have = p["text"].split(), side.get(p["stamp"], {}).get("words") or []
            for i, w in enumerate(have):
                if isinstance(w.get("suggest"), str) and w["suggest"] and isinstance(w.get("w"), str) \
                        and i < len(words) and words[i] == w["w"]:
                    p["sugg"].append({"i": i, "from": w["w"], "to": w["suggest"]})
        except Exception:
            p["sugg"] = []
    return paragraphs


def wav_bytes(samples):
    """float32 mono 16 kHz samples -> in-memory 16-bit PCM WAV."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(config.SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def make_server(audio_bytes, md_path, title, port=0):
    """A server on 127.0.0.1 only; every route lives under a random /<token>/. Returns (server, url)."""
    token = secrets.token_urlsafe(16)
    base = f"/{token}/"
    lock = threading.Lock()
    sidecar = os.path.splitext(md_path)[0] + ".review.json"
    page = PAGE.replace("__TITLE__", html.escape(title)).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body=b"", ctype="text/plain; charset=utf-8", **headers):
            try:  # browsers abort audio requests on every seek
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                for k, v in headers.items():
                    self.send_header(k.replace("_", "-"), v)
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def route(self):
            port = self.server.server_address[1]  # DNS rebinding: only our own host names
            if self.headers.get("Host") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                return None
            path = self.path.split("?", 1)[0]
            return path[len(base):] if path.startswith(base) else None

        def do_GET(self):
            r = self.route()
            if r == "":
                self.send(200, page, "text/html; charset=utf-8")
            elif r == "transcript.json":
                try:
                    with open(md_path, encoding="utf-8") as f:
                        data = add_suggestions(parse_md(f.read()), sidecar)
                except OSError:
                    return self.send(500, b"cannot read the .md file")
                self.send(200, json.dumps(data).encode(), "application/json")
            elif r == "audio.wav":
                self.audio()
            else:
                self.send(404, b"not found")

        do_HEAD = do_GET

        def audio(self):
            total = len(audio_bytes)
            rng = self.headers.get("Range")
            if not rng:
                return self.send(200, audio_bytes, "audio/wav", Accept_Ranges="bytes")
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
            if not m or not (m.group(1) or m.group(2)):
                return self.send(416, b"", Content_Range=f"bytes */{total}")
            if m.group(1):
                a = int(m.group(1))
                b = min(int(m.group(2)), total - 1) if m.group(2) else total - 1
            else:  # suffix: the last n bytes
                a, b = max(total - int(m.group(2)), 0), total - 1
            if a > b or a >= total:
                return self.send(416, b"", Content_Range=f"bytes */{total}")
            self.send(206, audio_bytes[a:b + 1], "audio/wav", Accept_Ranges="bytes",
                      Content_Range=f"bytes {a}-{b}/{total}")

        def do_POST(self):
            if self.route() != "save":
                return self.send(404, b"not found")
            try:
                n = int(self.headers.get("Content-Length", 0))
                if not 0 < n <= MAX_BODY:
                    raise ValueError
                paragraphs = json.loads(self.rfile.read(n))
                text = join_md([{"stamp": p.get("stamp"), "text": p.get("text", "")} for p in paragraphs])
            except (ValueError, AttributeError, TypeError):
                return self.send(400, b"bad request")
            if not text.strip():
                return self.send(400, b"empty transcript")
            tmp = f"{md_path}.{os.getpid()}.tmp"
            try:
                with lock:
                    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                        f.write(text)
                    os.replace(tmp, md_path)
            except OSError:
                return self.send(500, b"cannot save")
            self.send(200, b'{"ok": true}', "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return server, f"http://127.0.0.1:{server.server_address[1]}{base}"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m localtranscribe.review",
                                 description="Review a transcript next to its audio, in the browser.")
    ap.add_argument("recording")
    ap.add_argument("--md", help="the transcript (default: the recording's name with .md)")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = ap.parse_args(argv)
    md = args.md or os.path.splitext(args.recording)[0] + ".md"
    for path in (args.recording, md):
        if not os.path.isfile(path):
            print(f"Cannot find {path}. Transcribe the recording first.")
            return 1
    from .audio import load_audio

    print("Loading audio...")
    try:
        audio = wav_bytes(load_audio(args.recording))
    except Exception as e:
        print(f"Cannot read the audio of {args.recording}: {e}")
        return 1
    server, url = make_server(audio, md, os.path.basename(args.recording), args.port)
    print(url)
    print("Press Ctrl+C to stop.", flush=True)  # a launcher or log reading stdout needs the URL now
    if not args.no_browser:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


PAGE = """<!doctype html>
<html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fff;--fg:#1d1d1f;--mute:#6e6e73;--panel:#f2f2f4;--line:#d2d2d7;--acc:#0a5fd6}
@media(prefers-color-scheme:dark){:root{--bg:#1c1c1e;--fg:#f2f2f7;--mute:#98989f;--panel:#2c2c2e;--line:#48484a;--acc:#6cb0ff}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
main{display:flex;gap:24px;max-width:1200px;margin:0 auto;padding:16px}
#text{flex:1;min-width:0;max-width:70ch}
aside{width:340px;flex:none;align-self:flex-start;position:sticky;top:16px;background:var(--panel);
  border:1px solid var(--line);border-radius:10px;padding:14px}
aside h1{font-size:15px;margin:0 0 10px;word-break:break-all}
audio{width:100%}
#save{margin-top:10px;font:inherit;padding:6px 16px;border-radius:8px;border:1px solid var(--acc);
  background:var(--acc);color:var(--bg);cursor:pointer}
#status{margin:8px 0 0;font-size:14px;color:var(--mute)}
.p{display:flex;gap:10px;margin:0 0 14px;align-items:baseline}
.ts{flex:none;font:14px ui-monospace,Menlo,Consolas,monospace;color:var(--acc);background:none;border:0;
  padding:2px 4px;cursor:pointer}
.tx{flex:1;min-width:0;padding:2px 6px;border-radius:6px;white-space:pre-wrap}
button:focus-visible,.tx:focus-visible{outline:3px solid var(--acc);outline-offset:2px}
.tx:focus{background:var(--panel)}
.body{flex:1;min-width:0}
mark{background:#ffe27a;color:#1d1d1f;border-radius:3px;padding:0 2px}
.sg{font-size:14px;color:var(--mute);margin:2px 6px 0;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.sg button{font:inherit;padding:1px 10px;border-radius:6px;border:1px solid var(--acc);background:none;
  color:var(--acc);cursor:pointer}
.sg[hidden]{display:none}
.sg .note{color:#b3261e}
@media(max-width:800px){main{flex-direction:column-reverse;padding:0 16px 16px}
  aside{width:auto;align-self:stretch;position:sticky;top:0;z-index:1;border-radius:0 0 10px 10px;margin:0 -16px;padding:10px 16px}
  #text{max-width:none}}
</style></head><body>
<main>
<div id="text"><p id="msg">Caricamento...</p></div>
<aside>
<h1>__TITLE__</h1>
<audio id="a" controls src="audio.wav" preload="auto"></audio>
<button id="save" type="button">Salva</button>
<p id="status" role="status"></p>
<p id="count" role="status"></p>
</aside>
</main>
<script>
const $ = id => document.getElementById(id);
const audio = $("a"), status = $("status"), box = $("text");
let dirty = false, saving = 0, chain = Promise.resolve();
function setStatus(s) { status.textContent = s; }
function markDirty() { dirty = true; setStatus("Modifiche non salvate"); }
function countOpen() {
  const n = box.querySelectorAll(".sg:not([hidden])").length;
  $("count").textContent = n ? n + (n === 1 ? " suggerimento" : " suggerimenti") : "";
}
function unwrap(m) { m.replaceWith(document.createTextNode(m.textContent)); }
function paragraphs() {
  return [...box.querySelectorAll(".p")].map(p => ({stamp: p.dataset.stamp || null, text: p.querySelector(".tx").innerText}));
}
function save() { saving++; return chain = chain.then(doSave); }  // one request at a time, snapshot taken when its turn comes
async function doSave() {
  try {
    const body = JSON.stringify(paragraphs());
    dirty = false;
    const r = await fetch("save", {method: "POST", body});
    if (!r.ok) throw new Error(r.status);
    if (!dirty) setStatus("Salvato " + new Date().toLocaleTimeString("it-IT", {hour: "2-digit", minute: "2-digit"}));
  } catch (e) { dirty = true; setStatus("Errore nel salvataggio"); }
  finally { saving--; }
}
function show(items) {
  box.textContent = "";
  for (const it of items) {
    const p = document.createElement("div");
    p.className = "p";
    if (it.stamp) {
      p.dataset.stamp = it.stamp;
      const b = document.createElement("button");
      b.className = "ts"; b.type = "button";
      b.textContent = "[" + it.stamp + "]";
      b.setAttribute("aria-label", "Vai a " + it.stamp);
      b.onclick = () => { audio.currentTime = it.start; audio.play(); };
      p.append(b);
    }
    const t = document.createElement("div");
    t.className = "tx"; t.contentEditable = "plaintext-only";
    const sugg = new Map((it.sugg || []).map(g => [g.i, g]));
    let n = 0;
    for (const part of it.text.match(/\\S+|\\s+/g) || []) {
      const g = /\\S/.test(part) ? sugg.get(n++) : null;
      if (g) { const m = document.createElement("mark"); m.textContent = part; g.mark = m; t.append(m); }
      else t.append(part);
    }
    t.oninput = markDirty;
    t.onkeydown = e => { if (e.key === "Enter") e.preventDefault(); };
    const body = document.createElement("div");
    body.className = "body";
    body.append(t);
    for (const g of sugg.values()) {
      const row = document.createElement("div");
      row.className = "sg";
      const label = document.createElement("span");
      label.textContent = "\u00ab" + g.from + "\u00bb \u2192 \u00ab" + g.to + "\u00bb";
      const ok = document.createElement("button"), no = document.createElement("button");
      ok.type = no.type = "button";
      ok.textContent = "Accetta"; no.textContent = "Rifiuta";
      ok.setAttribute("aria-label", "Accetta: " + g.from + " diventa " + g.to);
      no.setAttribute("aria-label", "Rifiuta il suggerimento per " + g.from);
      const note = document.createElement("span");
      note.className = "note";
      ok.onclick = () => {
        const m = g.mark;
        if (!m.isConnected || m.textContent !== g.from) { note.textContent = "testo modificato"; return; }
        m.textContent = g.to; unwrap(m); row.hidden = true; markDirty(); countOpen();
      };
      no.onclick = () => { if (g.mark.isConnected) unwrap(g.mark); row.hidden = true; countOpen(); };
      row.append(label, ok, no, note);
      body.append(row);
    }
    p.append(body);
    box.append(p);
  }
  countOpen();
}
fetch("transcript.json").then(r => { if (!r.ok) throw 0; return r.json(); }).then(show)
  .catch(() => { box.textContent = "Impossibile caricare la trascrizione (file .md mancante o illeggibile)."; });
$("save").onclick = save;
addEventListener("keydown", e => { if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") { e.preventDefault(); save(); } });
addEventListener("beforeunload", e => { if (dirty || saving) { e.preventDefault(); e.returnValue = ""; } });
</script></body></html>
"""


if __name__ == "__main__":
    sys.exit(main())
