"""The manager: runs the open → read → decide → act loop until the answer is found."""

from __future__ import annotations

import heapq
import itertools
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from urllib.parse import urlparse

from .browser import FetchError
from .extractor import link_score, normalize_url, parse_html, rank_links, top_passages, trim_text
from .prompts import EVIDENCE_RETRY, SYSTEM_PROMPT, build_user_prompt


@dataclass
class AgentResult:
    query: str
    start_url: str
    found: bool = False
    answer: str | None = None
    source: str | None = None
    evidence: str | None = None
    verified: bool = False  # True when the evidence/answer was found word-for-word on a visited page
    notes: str = ""
    pages_opened: int = 0
    visited: list[str] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _site(host: str | None) -> str:
    """'admissions.example.edu' -> 'example.edu' (good enough to keep the agent on one website)."""
    host = (host or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if re.fullmatch(r"[\d.]+|localhost|\[.*\]", host):
        return host
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _clean(text: str) -> str:
    text = (text or "").lower()
    text = text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                                         "–": "-", "—": "-", " ": " "}))
    return " ".join(text.split()).strip(" .,:;\"'")


def _as_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def _as_int(value) -> int | None:
    try:
        return int(str(value).strip().strip("[]"))
    except (TypeError, ValueError):
        return None


def is_supported(evidence: str | None, answer: str | None, current_text: str, all_texts: list[str]) -> bool:
    current = _clean(current_text)
    if evidence and len(_clean(evidence)) >= 3 and _clean(evidence) in current:
        return True
    if answer and len(_clean(answer)) >= 2:
        needle = _clean(answer)
        return any(needle in _clean(t) for t in all_texts)
    return False


class WebAgent:
    def __init__(self, llm, browser, max_steps=8, max_page_chars=6000, max_links=40,
                 same_domain_only=True, verbose=True):
        self.llm = llm
        self.browser = browser
        self.max_steps = max_steps
        self.max_page_chars = max_page_chars
        self.max_links = max_links
        self.same_domain_only = same_domain_only
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, file=sys.stderr, flush=True)

    def _allowed(self, url: str, start_site: str) -> bool:
        return not self.same_domain_only or _site(urlparse(url).hostname) == start_site

    def _decide(self, messages: list[dict], current_text: str, all_texts: list[str]) -> tuple[dict, bool]:
        decision = self.llm.decide(messages)
        if decision.get("action") != "answer":
            return decision, False
        if is_supported(_as_text(decision.get("evidence")), _as_text(decision.get("answer")), current_text, all_texts):
            return decision, True
        # The quote didn't match the page: give the model one chance to correct itself.
        self._log("    evidence not found on page; asking the model to re-check")
        retry = messages + [
            {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)},
            {"role": "user", "content": EVIDENCE_RETRY},
        ]
        second = self.llm.decide(retry)
        if second.get("action") != "answer":
            return second, False
        ok = is_supported(_as_text(second.get("evidence")), _as_text(second.get("answer")), current_text, all_texts)
        return second, ok

    def run(self, url: str, query: str) -> AgentResult:
        started = time.time()
        result = AgentResult(query=query, start_url=url)
        start_site = _site(urlparse(url).hostname)

        visited_keys: set[str] = set()
        page_texts: list[str] = []
        frontier: list[tuple[int, int, str]] = []  # (-relevance, order, url): best unvisited links for backtracking
        queued: set[str] = set()
        order = itertools.count()
        notes = ""
        next_url: str | None = url

        self._log(f"QUESTION: {query}\nSTART:    {url}")
        while result.pages_opened < self.max_steps:
            if next_url is None:
                next_url = None
                while frontier:
                    _, _, candidate = heapq.heappop(frontier)
                    if normalize_url(candidate) not in visited_keys:
                        next_url = candidate
                        break
                if next_url is None:
                    result.steps.append({"step": result.pages_opened, "action": "stop", "detail": "no unvisited links left"})
                    break

            key = normalize_url(next_url)
            if key in visited_keys:
                next_url = None
                continue
            visited_keys.add(key)
            result.pages_opened += 1
            step = result.pages_opened

            try:
                final_url, html = self.browser.fetch(next_url)
            except FetchError as exc:
                self._log(f"[{step}] ERROR {exc}")
                result.steps.append({"step": step, "url": next_url, "action": "error", "detail": str(exc)})
                next_url = None
                continue

            visited_keys.add(normalize_url(final_url))
            result.visited.append(final_url)
            page = parse_html(html, final_url)
            page_texts.append(page.text)
            self._log(f"[{step}] {final_url}  ({page.title[:60]})")

            candidates = [
                link for link in page.links
                if self._allowed(link.url, start_site) and normalize_url(link.url) not in visited_keys
            ]
            for link in candidates:
                link_key = normalize_url(link.url)
                if link_key not in queued:
                    queued.add(link_key)
                    heapq.heappush(frontier, (-link_score(link, query), next(order), link.url))
            shown = rank_links(candidates, query, self.max_links)

            prompt = build_user_prompt(
                query=query, page_url=final_url, page_title=page.title,
                page_text=trim_text(page.text, query, self.max_page_chars), links=shown,
                visited=result.visited[:-1], notes=notes, step=step, max_steps=self.max_steps,
                highlights=top_passages(page.text, query),
            )
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]

            try:
                decision, verified = self._decide(messages, page.text, page_texts)
            except Exception as exc:  # model unreachable or unusable: stop and report it
                result.error = f"Model error: {exc}"
                self._log(f"[{step}] {result.error}")
                break

            action = str(decision.get("action", "")).lower()
            if decision.get("notes"):
                notes = _as_text(decision["notes"])[:500]
            record = {"step": step, "url": final_url, "action": action,
                      "detail": _as_text(decision.get("reason") or decision.get("answer")) or ""}

            if action == "answer":
                result.found = True
                result.answer = _as_text(decision.get("answer"))
                result.evidence = _as_text(decision.get("evidence"))
                result.source = final_url
                result.verified = verified
                result.steps.append(record)
                self._log(f"    ANSWER: {result.answer}  (verified={verified})")
                break

            if action == "give_up":
                result.steps.append(record)
                self._log(f"    GIVE UP: {record['detail']}")
                break

            if action == "visit":
                link = {l.id: l for l in shown}.get(_as_int(decision.get("link_id")))
                if link and normalize_url(link.url) not in visited_keys:
                    next_url = link.url
                    record["detail"] = f"-> {link.text}: {record['detail']}"
                    self._log(f"    visit [{link.id}] {link.text}")
                else:
                    next_url = None
                    record["detail"] = f"invalid link_id {decision.get('link_id')}; backtracking"
                    self._log(f"    {record['detail']}")
            else:  # "back" or anything unexpected
                next_url = None
                self._log(f"    back: {record['detail']}")
            result.steps.append(record)
        else:
            result.steps.append({"step": result.pages_opened, "action": "stop", "detail": "step limit reached"})

        result.notes = notes
        result.seconds = round(time.time() - started, 2)
        return result
