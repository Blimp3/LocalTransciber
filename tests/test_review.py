"""Review page: the .md parsing/joining and the local server (no model, no browser)."""
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from localtranscribe.review import join_md, main, make_server, parse_md, wav_bytes  # noqa: E402

PLAIN = "[00:00:00] primo testo\n\n[00:00:19] secondo testo\n"
SPEAKERS = "[00:01:23] Parlante 1: ciao\n\n[00:01:30] Parlante 2: buongiorno\n"
OLD = "senza orario\n\n[00:00:20] con orario\n"


class ParseJoin(unittest.TestCase):
    def test_round_trip(self):
        for t in (PLAIN, SPEAKERS, OLD):
            self.assertEqual(join_md(parse_md(t)), t)

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
