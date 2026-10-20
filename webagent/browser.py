"""The agent's hands: opens pages in a real (headless) Chromium so JavaScript sites work."""

from __future__ import annotations

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)

# Resources we never need to read text: skipping them makes pages load much faster.
BLOCKED_RESOURCES = {"image", "media", "font"}


class FetchError(Exception):
    """A page could not be loaded (timeout, HTTP error, DNS failure...)."""


class Browser:
    """Use as a context manager:

        with Browser() as browser:
            final_url, html = browser.fetch("https://example.com")
    """

    def __init__(self, timeout_ms: int = 20000, headless: bool = True):
        self.timeout_ms = timeout_ms
        self.headless = headless
        self._pw = None
        self._browser = None
        self._context = None

    def __enter__(self) -> "Browser":
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(
            user_agent=USER_AGENT, ignore_https_errors=True, java_script_enabled=True
        )
        self._context.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in BLOCKED_RESOURCES
            else route.continue_(),
        )
        return self

    def __exit__(self, *exc) -> None:
        for closer in (self._context, self._browser):
            try:
                if closer:
                    closer.close()
            except PlaywrightError:
                pass
        if self._pw:
            self._pw.stop()

    def fetch(self, url: str) -> tuple[str, str]:
        """Open `url` and return (final_url_after_redirects, rendered_html)."""
        if self._context is None:
            raise RuntimeError("Browser not started; use `with Browser() as b:`")
        page = self._context.new_page()
        try:
            response = page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            # Give JavaScript a moment to render content, but don't wait forever.
            try:
                page.wait_for_load_state("networkidle", timeout=3000)
            except PlaywrightTimeout:
                pass
            if response is not None and response.status >= 400:
                raise FetchError(f"HTTP {response.status} for {url}")
            return page.url, page.content()
        except PlaywrightTimeout as exc:
            raise FetchError(f"Timeout loading {url}") from exc
        except PlaywrightError as exc:
            raise FetchError(f"Could not load {url}: {str(exc).splitlines()[0]}") from exc
        finally:
            page.close()
