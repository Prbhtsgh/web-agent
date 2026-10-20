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
# A table row containing any of these holds page layout, not data (see _is_layout_row).
LAYOUT_TAGS = ["table", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "main", "header", "footer", "form"]

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


def _is_layout_row(row) -> bool:
    """Older sites wrap the whole page in a <table>. Flattening such a row into "a | b" would put the
    entire page on one line (seen on nps.gov: 13,000 of 14,700 characters), so only data rows are flattened."""
    return row.find(LAYOUT_TAGS) is not None


def _link_label(a, url: str) -> tuple[str, bool]:
    """What a person would read as the link's name. Returns (label, is_fallback); the fallback is the URL's
    last path part, used only when the page gives no name at all."""
    text = " ".join(a.get_text(" ", strip=True).split())
    named = a.get("aria-label") or a.get("title") or ""
    if not re.search(r"\w", text):  # empty or only symbols like "›" / "»": prefer the accessible name
        img = a.find("img", alt=True)
        text = named or (img["alt"].strip() if img else "") or text
    if re.fullmatch(r"\d{1,3}", text) and _in_pagination(a, url):
        text = f"Page {text}"
    if text:
        return " ".join(text.split()), False
    return urlparse(url).path.rstrip("/").rsplit("/", 1)[-1] or url, True


def _in_pagination(a, url: str) -> bool:
    if re.search(r"(^|&)(page|p|pg)=\d", urlparse(url).query):
        return True
    for parent in list(a.parents)[:3]:
        marks = " ".join(parent.get("class") or []) + " " + (parent.get("id") or "") + " " + (parent.get("aria-label") or "")
        if "pag" in marks.lower():
            return True
    return False


def _context(a, label: str, limit: int = 60) -> str:
    """Nearby text that tells identical links apart: "(about)" -> "by Steve Martin"."""
    for parent in list(a.parents)[:3]:
        text = " ".join(parent.get_text(" ", strip=True).split())
        if len(text) > 300:
            break
        rest = " ".join(text.replace(label, " ").split())
        if len(rest) >= 3:
            return rest[:limit]
    return ""


RECORD_TAGS = ["div", "li", "article"]  # not <section>: sections are document structure, not records


def _flatten_records(soup, max_chars: int = 300) -> None:
    """Put each small repeated record (a product card, a quote card...) on one line: "$88.99 · IdeaTab A3500L".

    Without this, records run together line by line and neighbours bleed into each other: on webscraper.io the
    model paired a price with the product listed before it, and on quotes.toscrape a passage started with the
    previous quote's "Tags: ...eleanor-roosevelt" right above a Steve Martin quote. A record is one of 3+
    siblings with the same tag and class. Only the innermost records are flattened (scrapethissite.com puts
    countries in rows of 3; flattening the rows would merge 3 countries). Runs after line layout, so inline
    text is never split."""
    records = []
    for parent in soup.find_all(True):
        children = parent.find_all(RECORD_TAGS, recursive=False)
        counts: dict[tuple, int] = {}
        for child in children:
            key = (child.name, tuple(child.get("class") or []))
            counts[key] = counts.get(key, 0) + 1
        records += [c for c in children if counts[(c.name, tuple(c.get("class") or []))] >= 3]
    record_ids = {id(r) for r in records}
    for record in records:
        if any(id(d) in record_ids for d in record.find_all(RECORD_TAGS)):
            continue  # contains smaller records: flatten those instead
        parts = [" ".join(line.split()).removeprefix("## ").strip() for line in record.get_text().splitlines()]
        parts = [part for part in parts if part and part != "##"]
        line = " · ".join(parts)
        if len(parts) >= 2 and len(line) <= max_chars:
            record.replace_with(soup.new_string(f"\n{line}\n"))


def parse_html(html: str, base_url: str) -> Page:
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    # 1) Links (collected before cleanup so navigation menus still count).
    links: list[Link] = []
    anchors = []  # the <a> element behind each link, for context below
    weak: set[int] = set()  # indexes of links whose label is only a URL fallback
    seen = {normalize_url(base_url): None}
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
        label, is_fallback = _link_label(a, url)
        if key in seen:
            index = seen[key]
            # Same page linked twice (e.g. a product image, then its title): keep the better name.
            if index is not None and index in weak and not is_fallback:
                links[index].text, anchors[index] = label[:120], a
                weak.discard(index)
            continue
        seen[key] = len(links)
        if is_fallback:
            weak.add(len(links))
        links.append(Link(id=len(links) + 1, text=label[:120], url=url))
        anchors.append(a)

    # Many links with the same generic name ("(about)", "Read more") are indistinguishable to the model.
    counts: dict[str, int] = {}
    for link in links:
        counts[link.text.lower()] = counts.get(link.text.lower(), 0) + 1
    for link, a in zip(links, anchors):
        if counts[link.text.lower()] > 1:
            context = _context(a, link.text)
            if context:
                link.text = f"{link.text} [{context}]"

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
    # Keep data-table rows readable: "cell | cell | cell" on one line. Layout rows keep their structure.
    for row in soup.find_all("tr"):
        if _is_layout_row(row):
            for cell in row.find_all(["th", "td"], recursive=False):
                cell.insert_before(soup.new_string("\n"))
                cell.insert_after(soup.new_string("\n"))
            continue
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
    _flatten_records(soup)

    body = soup.body or soup
    lines: list[str] = []
    for raw in body.get_text().splitlines():
        line = " ".join(raw.replace("\u00b6", " ").split())
        if line in ("", "##"):
            continue
        if not lines or lines[-1] != line:
            lines.append(line)
    return Page(url=base_url, title=title, text="\n".join(lines), links=links)


def _split_long_line(line: str, size: int) -> list[str]:
    """Cut a very long line at spaces into pieces of about `size` characters."""
    pieces, current = [], ""
    for word in line.split(" "):
        if current and len(current) + len(word) + 1 > size:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        pieces.append(current)
    return pieces


def _split_chunks(text: str, size: int = 400) -> list[str]:
    """Group lines into ~`size`-character chunks. A single huge line is cut up first, otherwise it becomes
    one chunk bigger than the whole trim budget and is silently dropped."""
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    lines = [piece for line in text.splitlines()
             for piece in (_split_long_line(line, size) if len(line) > 2 * size else [line])]
    for line in lines:
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


def _stem(word: str) -> str:
    """Crude plural folding so "pass" matches "Passes" and "fee" matches "fees"."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("sses", "ches", "shes", "xes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def link_score(link: Link, query: str, title: str = "") -> float:
    """How well a link's text and URL match the question.

    Question words that also appear in the current page's title (usually the site's own name, e.g.
    "Yellowstone National Park") describe every link on the site, so they count a third as much. Without
    this, on nps.gov "Birth of a National Park" outranked "Fees & Passes" (rank 47 of 327, never shown)."""
    title_words = {_stem(w) for w in tokenize(title)}
    words = {_stem(w) for w in tokenize(link.text) + tokenize(urlparse(link.url).path)}
    score = 0.0
    for q in {_stem(w) for w in tokenize(query)}:
        weight = 1.0 if q in title_words else 3.0
        if q in words:
            score += weight
        elif any((q in w or w in q) and min(len(q), len(w)) >= 3 for w in words):
            score += weight / 3
    return score


def rank_links(links: list[Link], query: str, limit: int, title: str = "") -> list[Link]:
    """Most query-relevant links first; original page order breaks ties."""
    ranked = sorted(links, key=lambda link: (-link_score(link, query, title), link.id))
    return ranked[:limit]
