"""Show exactly what the model sees for one page + question (debugging tool).

    python scripts/inspect_page.py https://docs.python.org/3/library/stdtypes.html "What does str.removeprefix do?"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webagent import Browser  # noqa: E402
from webagent.config import load_settings  # noqa: E402
from webagent.extractor import parse_html, rank_links, top_passages, trim_text  # noqa: E402


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    url, query = sys.argv[1], sys.argv[2]
    settings = load_settings()
    with Browser(timeout_ms=settings.page_timeout_ms) as browser:
        final_url, html = browser.fetch(url)
    page = parse_html(html, final_url)
    print(f"URL: {final_url}\nTITLE: {page.title}\nFULL TEXT: {len(page.text)} chars, {len(page.links)} links\n")
    print("=== MOST RELEVANT PASSAGES ===\n" + (top_passages(page.text, query) or "(none)"))
    print("\n=== PAGE TEXT SENT TO MODEL ===\n" + trim_text(page.text, query, settings.max_page_chars))
    print("\n=== TOP LINKS ===")
    for link in rank_links(page.links, query, 15):
        print(f"[{link.id}] {link.text} ({link.url})")


if __name__ == "__main__":
    main()
