"""Review page: the .md parsing/joining and the local server (no model, no browser)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from localtranscribe.review import PAGE, join_md, main, make_server, parse_md, wav_bytes  # noqa: E402

PLAIN = "[00:00:00] primo testo\n\n[00:00:19] secondo testo\n"
SPEAKERS = "[00:01:23] Parlante 1: ciao\n\n[00:01:30] Parlante 2: buongiorno\n"
OLD = "senza orario\n\n[00:00:20] con orario\n"


class ParseJoin(unittest.TestCase):
    def test_round_trip(self):
        for t in (PLAIN, SPEAKERS, OLD):
            self.assertEqual(join_md(parse_md(t)), t)

    def test_crlf(self):
        ps = parse_md("[00:00:00] a\r\nb\r\n\r\n[00:00:05] c\r\n")
        self.assertEqual([p["text"] for p in ps], ["a\nb", "c"])

    def test_start(self):
        p = parse_md("[01:02:05] ciao")[0]
        self.assertEqual((p["start"], p["stamp"], p["text"]), (3725, "01:02:05", "ciao"))
        self.assertIsNone(parse_md("solo testo")[0]["start"])

    def test_join_drops_empty_and_collapses(self):
        out = join_md([{"stamp": "00:00:01", "text": "a\n b  c"}, {"stamp": "00:00:02", "text": " \n "}])
        self.assertEqual(out, "[00:00:01] a b c\n")


class Server(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.md = os.path.join(self.dir.name, "r.md")
        with open(self.md, "w", encoding="utf-8") as f:
            f.write(PLAIN)
        self.audio = wav_bytes(np.zeros(16000, np.float32))
        self.server, self.url = make_server(self.audio, self.md, "r<1>.wav")
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.dir.cleanup()

    def req(self, path="", data=None, headers=None, base=None):
        r = urllib.request.Request((base or self.url) + path, data=data, headers=headers or {})
        try:
            return urllib.request.urlopen(r)
        except urllib.error.HTTPError as e:
            return e

    def test_page_and_token(self):
        r = self.req()
        self.assertEqual(r.status, 200)
        self.assertIn("r&lt;1&gt;.wav", r.read().decode())
        self.assertEqual(self.req(base=self.url.rsplit("/", 2)[0] + "/nottoken/").code, 404)

    def test_audio_and_range(self):
        r = self.req("audio.wav")
        self.assertEqual(len(r.read()), len(self.audio))
        self.assertEqual(r.headers["Accept-Ranges"], "bytes")
        r = self.req("audio.wav", headers={"Range": "bytes=0-99"})
        self.assertEqual(r.status, 206)
        self.assertEqual(len(r.read()), 100)
        self.assertEqual(r.headers["Content-Range"], f"bytes 0-99/{len(self.audio)}")
        r = self.req("audio.wav", headers={"Range": f"bytes={len(self.audio) - 10}-"})
        self.assertEqual(len(r.read()), 10)
        self.assertEqual(self.req("audio.wav", headers={"Range": "bytes=999999999-"}).code, 416)

    def test_transcript_and_save(self):
        self.assertEqual(json.load(self.req("transcript.json"))[1]["stamp"], "00:00:19")
        body = json.dumps([{"stamp": "00:00:00", "text": "corretto"}, {"stamp": None, "text": ""}]).encode()
        self.assertEqual(self.req("save", data=body).status, 200)
        with open(self.md, encoding="utf-8") as f:
            self.assertEqual(f.read(), "[00:00:00] corretto\n")

    def test_save_refused(self):
        body = b'[{"stamp": null, "text": "x"}]'
        self.assertEqual(self.req("save", data=body, base=self.url.rsplit("/", 2)[0] + "/nottoken/").code, 404)
        self.assertEqual(self.req("save", data=b"not json").code, 400)
        self.assertEqual(self.req("save", data=b"[]").code, 400)
        with open(self.md, encoding="utf-8") as f:
            self.assertEqual(f.read(), PLAIN)

    def test_foreign_host_refused(self):
        self.assertEqual(self.req(headers={"Host": "evil.example"}).code, 404)
        self.assertEqual(self.req("transcript.json", headers={"Host": "evil.example:80"}).code, 404)
        port = self.server.server_address[1]
        self.assertEqual(self.req(headers={"Host": f"localhost:{port}"}).status, 200)

    def save(self, text="x", version=None):
        body = json.dumps([{"stamp": None, "text": text}]).encode()
        return self.req("save", data=body, headers={"X-Md-Version": version} if version is not None else {})

    def read(self, path=None):
        with open(path or self.md, encoding="utf-8") as f:
            return f.read()

    def baks(self):
        return [n for n in os.listdir(self.dir.name) if ".bak-" in n]

    def test_same_version_saves_without_backup(self):
        v = self.req("transcript.json").headers["X-Md-Version"]
        r = self.save("uno", v)
        j = json.load(r)
        self.assertEqual((r.status, j["ok"], j["backup"]), (200, True, None))
        self.assertEqual(self.save("due", j["version"]).status, 200)
        self.assertEqual((self.read(), self.baks()), ("due\n", []))

    def test_stale_version_keeps_disk_copy(self):
        v = self.req("transcript.json").headers["X-Md-Version"]
        with open(self.md, "w", encoding="utf-8") as f:
            f.write("scritto da un altro processo, più lungo\n")
        j = json.load(self.save("mio", v))
        self.assertEqual(self.read(), "mio\n")
        self.assertEqual(self.baks(), [j["backup"]])
        self.assertEqual(self.read(os.path.join(self.dir.name, j["backup"])), "scritto da un altro processo, più lungo\n")

    def test_missing_header_is_stale(self):
        j = json.load(self.save("mio"))
        self.assertEqual(self.read(os.path.join(self.dir.name, j["backup"])), PLAIN)
        self.assertEqual(self.read(), "mio\n")

    def test_md_gone_saves_without_backup(self):
        os.remove(self.md)
        self.assertEqual(json.load(self.save("mio", "1-1"))["backup"], None)
        self.assertEqual(self.read(), "mio\n")

    def test_failed_replace_is_clean(self):
        with mock.patch("os.replace", side_effect=PermissionError("open elsewhere")):
            self.assertEqual(self.save("x", self.req("transcript.json").headers["X-Md-Version"]).code, 500)
        self.assertEqual((self.read(), os.listdir(self.dir.name)), (PLAIN, ["r.md"]))

    def test_failed_backup_does_not_overwrite(self):
        with mock.patch("shutil.copyfile", side_effect=OSError("disk full")):
            self.assertEqual(self.save("x").code, 500)
        self.assertEqual((self.read(), os.listdir(self.dir.name)), (PLAIN, ["r.md"]))

    def test_failed_save_after_a_backup_leaves_nothing(self):
        with mock.patch("os.replace", side_effect=PermissionError("open elsewhere")):
            self.assertEqual(self.save("x").code, 500)
        self.assertEqual((self.read(), os.listdir(self.dir.name)), (PLAIN, ["r.md"]))

    def test_half_written_backup_removed(self):
        def half(src, dst):
            with open(dst, "w") as f:
                f.write("[00:00")
            raise OSError("disk full")

        with mock.patch("shutil.copyfile", half):
            self.assertEqual(self.save("x").code, 500)
        self.assertEqual(os.listdir(self.dir.name), ["r.md"])

    def test_saved_text_and_backup_on_disk_before_the_swap(self):
        events, real_fsync, real_replace = [], os.fsync, os.replace

        def fsync(fd):
            events.append(("synced", os.fstat(fd).st_ino))
            real_fsync(fd)

        def replace(a, b):
            events.append(("replaced", os.stat(a).st_ino, b))
            real_replace(a, b)

        with mock.patch("os.fsync", fsync), mock.patch("os.replace", replace):
            j = json.load(self.save("mio"))
        (swap,) = [i for i, e in enumerate(events) if e[0] == "replaced" and e[2] == self.md]
        self.assertIn(("synced", events[swap][1]), events[:swap])
        self.assertIn(("synced", os.stat(os.path.join(self.dir.name, j["backup"])).st_ino), events[:swap])

    def test_file_system_without_fsync_still_saves(self):
        with mock.patch("os.fsync", side_effect=OSError(22, "Invalid argument")):
            self.assertEqual(self.save("mio").status, 200)
        self.assertEqual(self.read(), "mio\n")

    def test_tmp_name_has_pid(self):
        seen = []
        real = os.replace
        with mock.patch("os.replace", side_effect=lambda a, b: (seen.append(a), real(a, b))):
            self.save("x", self.req("transcript.json").headers["X-Md-Version"])
        self.assertEqual(seen, [f"{self.md}.{os.getpid()}.tmp"])
        self.assertEqual(os.listdir(self.dir.name), ["r.md"])


NODE_RACE = """
const vm = require('vm');
let visible = 'older edit', pending = [], listeners = {}, bodies = [];
const els = {};
const el = id => els[id] || (els[id] = {id, textContent: '', querySelectorAll: () => [{dataset: {stamp: '00:00:00'},
  querySelector: () => ({get innerText() { return visible; }})}]});
const ctx = {document: {getElementById: el}, addEventListener: (t, f) => listeners[t] = f, Date,
  fetch: (u, o) => u === 'save' ? new Promise(res => { bodies.push(JSON.parse(o.body)[0].text); pending.push(res); }) : Promise.reject(0)};
vm.createContext(ctx); vm.runInContext(process.argv[1], ctx);
const okay = {ok: true, json: async () => ({ok: true, version: 'v', backup: null})};
const tick = () => new Promise(r => setImmediate(r));
(async () => {
  const s1 = ctx.save(); await tick();
  const ev = {preventDefault() {}}; listeners.beforeunload(ev);
  const warned = ev.returnValue === '';
  visible = 'newer edit'; vm.runInContext('markDirty()', ctx); const s2 = ctx.save();
  await tick();
  const started = pending.length;                 // the second request must wait for the first
  pending[0](okay); await s1; await tick();
  pending[1](okay); await s2;
  console.log(JSON.stringify({warned, started, bodies, dirty: vm.runInContext('dirty', ctx), saving: vm.runInContext('saving', ctx)}));
})();
"""


@unittest.skipUnless(shutil.which("node"), "node not installed")
class SaveRace(unittest.TestCase):
    def test_saves_are_serialized_and_newest_lands_last(self):
        script = PAGE.split("<script>", 1)[1].split("</script>", 1)[0]
        out = subprocess.run(["node", "-e", NODE_RACE, script], capture_output=True, text=True, timeout=30, check=True)
        r = json.loads(out.stdout)
        self.assertTrue(r["warned"])
        self.assertEqual(r["started"], 1)
        self.assertEqual(r["bodies"], ["older edit", "newer edit"])
        self.assertEqual((r["dirty"], r["saving"]), (False, 0))


NODE_PAGE = """
const vm = require('vm');
let pending = [], sent = [];
const els = {};
const el = id => els[id] || (els[id] = {id, textContent: '', querySelectorAll: () => [{dataset: {stamp: '00:00:00'},
  querySelector: () => ({innerText: 'testo'})}]});
const ctx = {document: {getElementById: el}, addEventListener: () => {}, Date,
  fetch: (u, o) => u === 'save' ? new Promise(res => { sent.push(o.headers['X-Md-Version']); pending.push(res); })
    : Promise.resolve({ok: true, headers: {get: n => n === 'X-Md-Version' ? 'v1' : null}, json: async () => []})};
vm.createContext(ctx); vm.runInContext(process.argv[1], ctx);
const tick = () => new Promise(r => setImmediate(r));
const answer = j => ({ok: true, json: async () => j});
const out = [];
async function step(reply) {
  const s = ctx.save(); await tick();
  pending.shift()(reply); await s;
  out.push({status: els.status.textContent, dirty: vm.runInContext('dirty', ctx)});
}
(async () => {
  await tick(); await tick();
  await step(answer({ok: true, version: 'v2', backup: null}));
  await step({ok: false, status: 500});
  await step(answer({ok: true, version: 'v3', backup: 'r.bak-20260930-101010.md'}));
  await step({ok: false, status: 400});
  console.log(JSON.stringify({sent, out}));
})();
"""


@unittest.skipUnless(shutil.which("node"), "node not installed")
class PageVersion(unittest.TestCase):
    def test_version_header_messages_and_dirty(self):
        script = PAGE.split("<script>", 1)[1].split("</script>", 1)[0]
        r = json.loads(subprocess.run(["node", "-e", NODE_PAGE, script], capture_output=True, text=True, timeout=30,
                                      check=True).stdout)
        self.assertEqual(r["sent"], ["v1", "v2", "v2", "v3"])
        (ok, locked, backup, bad) = r["out"]
        self.assertTrue(ok["status"].startswith("Salvato ") and not ok["dirty"])
        self.assertIn("aperto in un altro programma", locked["status"])
        self.assertTrue(locked["dirty"])
        self.assertIn("conservata come r.bak-20260930-101010.md", backup["status"])
        self.assertFalse(backup["dirty"])
        self.assertEqual((bad["status"], bad["dirty"]), ("Errore nel salvataggio", True))


class Suggestions(Server):
    def sidecar(self, data):
        with open(os.path.join(self.dir.name, "r.review.json"), "w", encoding="utf-8") as f:
            f.write(data if isinstance(data, str) else json.dumps(data))

    def side(self, word="secondo", offset=19.4):
        words = [{"w": "secondo", "p": 0.4, "unsure": True, "suggest": "secondi"}, {"w": "testo", "p": 0.99}]
        words[0]["w"] = word
        return {"version": 1, "paragraphs": [{"text": "x", "offset": offset, "words": words}]}

    def sugg(self):
        r = self.req("transcript.json")
        self.assertEqual(r.status, 200)
        return [p["sugg"] for p in json.load(r)]

    def test_from_sidecar_matched_by_stamp(self):
        self.sidecar(self.side())
        self.assertEqual(self.sugg(), [[], [{"i": 0, "from": "secondo", "to": "secondi"}]])

    def test_stale_word_or_other_stamp(self):
        self.sidecar(self.side(word="terzo"))
        self.assertEqual(self.sugg(), [[], []])
        self.sidecar(self.side(offset=5))
        self.assertEqual(self.sugg(), [[], []])

    def test_missing_or_bad_sidecar(self):
        self.assertEqual(self.sugg(), [[], []])
        for bad in ("not json", "[]", '{"paragraphs": 3}', '{"paragraphs": [{"offset": "x"}]}'):
            self.sidecar(bad)
            self.assertEqual(self.sugg(), [[], []])

    def test_hostile_sidecar_never_breaks_transcript(self):
        deep = "[" * 100000 + "]" * 100000
        bad_word = self.side()
        bad_word["paragraphs"][0]["words"] = [{"w": "secondo", "suggest": 5}]
        for bad in ('{"paragraphs": [{"text": "x", "offset": Infinity, "words": []}]}',
                    '{"paragraphs": ' + deep + "}", json.dumps(bad_word)):
            self.sidecar(bad)
            self.assertEqual(self.sugg(), [[], []])

    def test_save_unchanged(self):
        self.sidecar(self.side())
        body = json.dumps([{"stamp": "00:00:19", "text": "secondi testo", "sugg": []}]).encode()
        self.assertEqual(self.req("save", data=body).status, 200)
        with open(self.md, encoding="utf-8") as f:
            self.assertEqual(f.read(), "[00:00:19] secondi testo\n")


class Main(unittest.TestCase):
    def test_unreadable_audio(self):
        with tempfile.TemporaryDirectory() as d:
            txt, md = os.path.join(d, "a.txt"), os.path.join(d, "a.md")
            for path in (txt, md):
                with open(path, "w") as f:
                    f.write("not audio")
            self.assertEqual(main([txt, "--md", md, "--no-browser"]), 1)


if __name__ == "__main__":
    unittest.main()
