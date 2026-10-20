"""The agent's eyes: turns raw HTML into clean text + a numbered list of links."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from urllib.parse import urldefrag, urljoin, urlparse

from bs4 import BeautifulSoup, Comment

SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".zip", ".gz",
    ".mp3", ".mp4", ".mov", ".avi", ".css", ".js", ".xml", ".rss", ".doc", ".docx",
    ".xls", ".xlsx", ".ppt", ".pptx", ".exe", ".dmg",
)

BLOCK_TAGS = [
    "p", "div", "section", "article", "main", "aside", "header", "footer", "li", "ul", "ol", "dl", "dt", "dd",
    "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote", "table", "figure", "figcaption", "details",
    "summary", "form", "fieldset", "address", "hr", "caption",
]
HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "dt"]

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "of", "to", "in", "on", "for",
    "and", "or", "at", "by", "with", "what", "which", "who", "whom", "when", "where",
    "why", "how", "does", "do", "did", "this", "that", "it", "its", "as", "from", "can",
    "i", "me", "my", "you", "your", "find", "tell", "give", "please", "about", "there",
    "much", "many", "any", "all", "get", "has", "have", "www", "http", "https", "html",
    "htm", "php", "com", "org", "net", "index",
}


@dataclass
class Link:
    id: int
    text: str
    url: str


@dataclass
class Page:
    url: str
    title: str
    text: str
    links: list[Link] = field(default_factory=list)


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in STOPWORDS]


def normalize_url(url: str) -> str:
    """Canonical form used to decide whether two URLs are the same page."""
    url = urldefrag(url)[0]
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/") or "/"
    for suffix in ("/index.html", "/index.htm", "/index.php"):
        if path.endswith(suffix):
            path = path[: -len(suffix)] or "/"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{host}{path}{query}"


def parse_html(html: str, base_url: str) -> Page:
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    # 1) Links (collected before cleanup so navigation menus still count).
    links: list[Link] = []
    seen = {normalize_url(base_url)}
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
            continue
        url = urldefrag(urljoin(base_url, href))[0]
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            continue
        if parsed.path.lower().endswith(SKIP_EXTENSIONS):
            continue
        key = normalize_url(url)
        if key in seen:
            continue
        seen.add(key)
        text = " ".join(a.get_text(" ", strip=True).split())
        if not text:
            text = a.get("title") or a.get("aria-label") or ""
        if not text:
            text = parsed.path.rstrip("/").rsplit("/", 1)[-1] or url
        links.append(Link(id=len(links) + 1, text=text[:120], url=url))

    # 2) Clean text, laid out the way a browser shows it: line breaks only between blocks
    #    (paragraphs, headings, list items...), never inside a sentence or a line of code.
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "template", "nav"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()
    # Like a browser: line breaks in the HTML source are just spaces (except inside <pre>).
    for node in soup.find_all(string=True):
        text = str(node)
        if "\n" in text or "\t" in text or "  " in text:
            if node.find_parent(["pre", "textarea"]) is None:
                node.replace_with(re.sub(r"\s+", " ", text))
    # Keep table rows readable: "cell | cell | cell" on one line.
    for row in soup.find_all("tr"):
        cells = [" ".join(c.get_text(" ", strip=True).split()) for c in row.find_all(["th", "td"])]
        row.replace_with(soup.new_string("\n" + " | ".join(c for c in cells if c) + "\n"))
    for br in soup.find_all("br"):
        br.replace_with(soup.new_string("\n"))
    # Mark headings and definition terms ("## ...") so the model and the ranking can see the structure.
    for tag in soup.find_all(HEADING_TAGS):
        tag.insert(0, soup.new_string("## "))
    for tag in soup.find_all(["a", "button", "label"]):
        tag.insert_after(soup.new_string(" "))  # side-by-side links shouldn't merge ("HomeAbout")
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_before(soup.new_string("\n"))
        tag.insert_after(soup.new_string("\n"))

    body = soup.body or soup
    lines: list[str] = []
    for raw in body.get_text().splitlines():
        line = " ".join(raw.replace("\u00b6", " ").split())
        if line in ("", "##"):
            continue
        if not lines or lines[-1] != line:
            lines.append(line)
    return Page(url=base_url, title=title, text="\n".join(lines), links=links)


def _split_chunks(text: str, size: int = 400) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for line in text.splitlines():
        current.append(line)
        length += len(line) + 1
        if length >= size:
            chunks.append("\n".join(current))
            current, length = [], 0
    if current:
        chunks.append("\n".join(current))
    return chunks


def score_chunks(chunks: list[str], query: str) -> list[float]:
    """Relevance of each chunk to the query, weighting words by how rare they are on the page (IDF)."""
    n = len(chunks)
    chunk_words = [set(tokenize(c)) for c in chunks]
    query_words = set(tokenize(query))

    def idf(df: int) -> float:
        return math.log((n + 1) / (df + 0.5)) if df else 0.0

    exact_idf = {w: idf(sum(1 for cw in chunk_words if w in cw)) for w in query_words}
    partial_hits = [
        {w for w in query_words if len(w) >= 4 and w not in cw and any(w in word for word in cw)}
        for cw in chunk_words
    ]
    partial_idf = {w: idf(sum(1 for hits in partial_hits if w in hits)) for w in query_words}

    scores = []
    for chunk, words, hits in zip(chunks, chunk_words, partial_hits):
        score = sum(exact_idf[w] for w in query_words if w in words)
        score += 0.5 * sum(partial_idf[w] for w in hits)
        # A heading or definition term that names the thing asked about is where its details live.
        heading_words = set()
        for line in chunk.splitlines():
            if line.startswith("## "):
                heading_words |= set(tokenize(line))
        score += sum(exact_idf[w] for w in query_words if w in heading_words)
        scores.append(score)
    return scores


def top_passages(text: str, query: str, k: int = 3, max_chars: int = 1500) -> str:
    """The few passages that best match the question, shown to the model first so it can't miss them."""
    chunks = _split_chunks(text, size=300)
    if len(chunks) <= 2:
        return ""
    scores = score_chunks(chunks, query)
    best = [i for i in sorted(range(len(chunks)), key=lambda i: -scores[i]) if scores[i] > 0.05][:k]
    out, used = [], 0
    for i in best:  # best match first
        piece = chunks[i].strip()
        if used + len(piece) > max_chars:
            piece = piece[: max(0, max_chars - used)]
        if piece:
            out.append(piece)
            used += len(piece)
    return "\n---\n".join(out)


def trim_text(text: str, query: str, max_chars: int) -> str:
    """Fit a long page into `max_chars`, keeping the start plus the parts most related to the query.

    Words are weighted by how RARE they are on the page (IDF): on a page where "str" and "method"
    appear in every paragraph, they count for almost nothing, while a rare word like "removeprefix"
    decides which paragraphs are kept.
    """
    if len(text) <= max_chars:
        return text

    chunks = _split_chunks(text)
    n = len(chunks)
    scores = score_chunks(chunks, query)

    keep = {0}  # the page start usually holds the heading/context
    budget = max_chars - len(chunks[0])
    ranked = [i for i in sorted(range(1, n), key=lambda i: (-scores[i], i)) if scores[i] > 0.05]
    for rank, i in enumerate(ranked):
        # For the two best matches also keep the following chunk, in case the answer runs across the boundary.
        group = [i, i + 1] if rank < 2 and i + 1 < n else [i]
        for j in group:
            if j in keep or len(chunks[j]) + 6 > budget:
                continue
            keep.add(j)
            budget -= len(chunks[j]) + 6

    out: list[str] = []
    previous = -1
    for i in sorted(keep):
        if previous != -1 and i != previous + 1:
            out.append("[...]")
        out.append(chunks[i])
        previous = i
    if previous != n - 1:
        out.append("[...]")
    return "\n".join(out)[:max_chars]


def link_score(link: Link, query: str) -> int:
    query_words = set(tokenize(query))
    words = set(tokenize(link.text)) | set(tokenize(urlparse(link.url).path))
    exact = len(query_words & words)
    partial = sum(1 for q in query_words for w in words if q != w and (q in w or w in q) and min(len(q), len(w)) >= 3)
    return exact * 3 + partial


def rank_links(links: list[Link], query: str, limit: int) -> list[Link]:
    """Most query-relevant links first; original page order breaks ties."""
    ranked = sorted(links, key=lambda link: (-link_score(link, query), link.id))
    return ranked[:limit]
