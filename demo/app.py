"""Demo mode: a local web page that shows the agent working, step by step, with a screenshot of every page.

    python demo/app.py                  # opens http://localhost:8765
    python demo/app.py --show-browser   # also shows the real Chromium window while it browses

The agent code is not changed. The demo hands it two stand-ins with the same interface as the real parts:
- DemoBrowser opens pages like webagent.Browser, but also takes a screenshot of each one;
- ReportingLLM passes every call through to the real model and reports the decision it made.
Both report events through `emit`, which the web server streams to the page (Server-Sent Events).
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `python demo/app.py` finds webagent/

from playwright.sync_api import Error as PlaywrightError  # noqa: E402
from playwright.sync_api import TimeoutError as PlaywrightTimeout  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from main import build_agent  # noqa: E402
from webagent import FetchError  # noqa: E402
from webagent.browser import USER_AGENT  # noqa: E402
from webagent.config import load_settings  # noqa: E402

PAGE_HTML = Path(__file__).with_name("index.html")


class DemoBrowser:
    """Same interface as webagent.Browser (a context manager with fetch(url) -> (final_url, html)), plus a
    screenshot of every page. Unlike the agent's browser it loads images, so screenshots look like the real site."""

    def __init__(self, emit, timeout_ms: int = 20000, headless: bool = True):
        self.emit = emit
        self.timeout_ms = timeout_ms
        self.headless = headless
        self.pages = 0

    def __enter__(self) -> "DemoBrowser":
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(
            user_agent=USER_AGENT, viewport={"width": 1280, "height": 800}, ignore_https_errors=True
        )
        return self

    def __exit__(self, *exc) -> None:
        for closer in (self._context, self._browser):
            try:
                closer.close()
            except PlaywrightError:
                pass
        self._pw.stop()

    def fetch(self, url: str) -> tuple[str, str]:
        page = self._context.new_page()
        try:
            response = page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=3000)
            except PlaywrightTimeout:
                pass
            if response is not None and response.status >= 400:
                raise FetchError(f"HTTP {response.status} for {url}")
            shot = page.screenshot(type="jpeg", quality=60)
            self.pages += 1
            self.emit({"type": "page", "step": self.pages, "url": page.url, "title": page.title(),
                       "screenshot": "data:image/jpeg;base64," + base64.b64encode(shot).decode()})
            return page.url, page.content()
        except PlaywrightTimeout as exc:
            self.emit({"type": "error", "url": url, "detail": f"Timeout loading {url}"})
            raise FetchError(f"Timeout loading {url}") from exc
        except PlaywrightError as exc:
            detail = f"Could not load {url}: {str(exc).splitlines()[0]}"
            self.emit({"type": "error", "url": url, "detail": detail})
            raise FetchError(detail) from exc
        except FetchError as exc:
            self.emit({"type": "error", "url": url, "detail": str(exc)})
            raise
        finally:
            page.close()


class ReportingLLM:
    """Passes decide() through to the real model and reports what it decided, in words a viewer can follow."""

    def __init__(self, llm, emit):
        self.llm = llm
        self.emit = emit

    def decide(self, messages: list[dict]) -> dict:
        decision = self.llm.decide(messages)
        links = dict(re.findall(r"^\[(\d+)\] (.+) \(\S+\)$", messages[1]["content"].split("\nLINKS:\n", 1)[-1], re.M))
        action = str(decision.get("action", "")).lower()
        if action == "visit":
            link_id = str(decision.get("link_id", "")).strip("[] ")
            detail = f"{links.get(link_id, 'link ' + link_id)}: {decision.get('reason', '')}"
        elif action == "answer":
            detail = f"{decision.get('answer')} (evidence: \"{decision.get('evidence', '')}\")"
        else:
            detail = str(decision.get("reason", ""))
        # More than system + user message means the agent asked the model to re-check its previous decision.
        self.emit({"type": "decision", "action": action or "?", "detail": detail, "recheck": len(messages) > 2})
        return decision


def run_demo(url: str, query: str, emit, settings=None, llm=None, headless: bool = True) -> dict:
    """Run the agent once, reporting each step through `emit(event_dict)`. Returns the final result event."""
    settings = settings or load_settings()
    emit({"type": "start", "url": url, "query": query, "model": settings.llm_model})
    with DemoBrowser(emit, timeout_ms=settings.page_timeout_ms, headless=headless) as browser:
        agent = build_agent(settings, browser, verbose=False)
        agent.llm = ReportingLLM(llm or agent.llm, emit)
        result = agent.run(url, query)
    final = {"type": "result", "found": result.found, "answer": result.answer if result.found else None,
             "source": result.source, "evidence": result.evidence, "verified": result.verified,
             "pages_opened": result.pages_opened, "seconds": result.seconds, "error": result.error}
    emit(final)
    return final


class DemoHandler(BaseHTTPRequestHandler):
    llm_factory = None  # tests swap in a scripted model; None means the model from .env
    headless = True

    def log_message(self, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = PAGE_HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/run":
            params = parse_qs(parsed.query)
            self.stream(params.get("url", [""])[0].strip(), params.get("q", [""])[0].strip())
        else:
            self.send_error(404)

    def stream(self, url: str, query: str) -> None:
        """Server-Sent Events: one `data: {json}` message per step, so the page updates while the agent works."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(event: dict) -> None:
            self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()

        try:
            if not urlparse(url).scheme.startswith("http") or not query:
                emit({"type": "fatal", "detail": "Give a full URL (https://...) and a question."})
                return
            llm = self.llm_factory() if self.llm_factory else None
            run_demo(url, query, emit, llm=llm, headless=self.headless)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the viewer pressed Stop or closed the tab
        except Exception as exc:  # show the problem on the page instead of a silent hang
            try:
                emit({"type": "fatal", "detail": f"{type(exc).__name__}: {exc}"})
            except (BrokenPipeError, ConnectionResetError):
                pass


def make_server(port: int, headless: bool = True, llm_factory=None) -> ThreadingHTTPServer:
    handler = type("Handler", (DemoHandler,), {"headless": headless, "llm_factory": staticmethod(llm_factory)
                                               if llm_factory else None})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)  # local only: it browses whatever URL it is given


def main() -> int:
    parser = argparse.ArgumentParser(description="Live demo of the web agent")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--show-browser", action="store_true", help="Also show the Chromium window while it browses")
    parser.add_argument("--no-open", action="store_true", help="Don't open the demo page in your browser")
    args = parser.parse_args()

    server = make_server(args.port, headless=not args.show_browser)
    address = f"http://localhost:{server.server_port}"
    settings = load_settings()
    print(f"Demo running at {address}  (model: {settings.llm_model} at {settings.llm_base_url})  Ctrl+C to stop", flush=True)
    if not args.no_open:
        threading.Timer(0.5, webbrowser.open, args=(address,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
