"""Offline tests for demo mode: the events it streams, using the local test site and the scripted model."""

import functools
import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from urllib.parse import urlencode

from demo.app import make_server, run_demo

try:
    from tests.test_agent import SITE_DIR, QuietHandler, ScriptedLLM
except ImportError:  # when unittest discovery imports the tests folder as the top level
    from test_agent import SITE_DIR, QuietHandler, ScriptedLLM

QUESTION = "What is the MTech application deadline?"


def deadline_model():
    return ScriptedLLM(r"Last date to apply: ([^\n]+)", ["mtech admissions", "admissions"])


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        handler = functools.partial(QuietHandler, directory=str(SITE_DIR))
        cls.site = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.base = f"http://127.0.0.1:{cls.site.server_port}/"
        threading.Thread(target=cls.site.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.site.shutdown()
        cls.site.server_close()

    def test_run_demo_reports_every_step(self):
        events = []
        final = run_demo(self.base + "index.html", QUESTION, events.append, llm=deadline_model())
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds[0], "start")
        self.assertEqual(kinds[-1], "result")
        pages = [e for e in events if e["type"] == "page"]
        self.assertEqual([p["step"] for p in pages], [1, 2, 3])
        self.assertTrue(all(p["screenshot"].startswith("data:image/jpeg;base64,") for p in pages))
        self.assertTrue(pages[-1]["url"].endswith("admissions/mtech.html"))
        decisions = [e for e in events if e["type"] == "decision"]
        self.assertEqual([d["action"] for d in decisions], ["visit", "visit", "answer"])
        self.assertIn("Admissions", decisions[0]["detail"])  # the link's text, not just its number
        self.assertEqual(final["answer"], "15 December 2026")
        self.assertTrue(final["verified"])

    def test_web_page_and_event_stream(self):
        server = make_server(0, llm_factory=deadline_model)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        address = f"http://127.0.0.1:{server.server_port}"
        try:
            page = urllib.request.urlopen(address + "/", timeout=10).read().decode()
            self.assertIn('<form id="form">', page)

            query = urlencode({"url": self.base + "index.html", "q": QUESTION})
            with urllib.request.urlopen(f"{address}/run?{query}", timeout=60) as stream:
                self.assertEqual(stream.headers["Content-Type"], "text/event-stream")
                events = [json.loads(line[6:]) for line in stream.read().decode().splitlines()
                          if line.startswith("data: ")]
            self.assertEqual(events[-1]["type"], "result")
            self.assertEqual(events[-1]["answer"], "15 December 2026")

            with urllib.request.urlopen(f"{address}/run?{urlencode({'url': 'not a url', 'q': 'x'})}", timeout=10) as bad:
                self.assertEqual(json.loads(bad.read().decode().split("data: ", 1)[1])["type"], "fatal")
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
