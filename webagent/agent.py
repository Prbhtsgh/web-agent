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
from .extractor import (Page, link_score, normalize_url, parse_html, rank_links, stem, tokenize, top_passages,
                        trim_text)
from .prompts import EVIDENCE_RETRY, GIVE_UP_RETRY, SUBJECT_RETRY, SYSTEM_PROMPT, build_user_prompt


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


NON_ANSWER = re.compile(
    r"(none|null|nil|n/?a|unknown|not (found|mentioned|available|specified|stated|provided|listed|given)"
    r"|no (information|answer|data)( (found|available))?)",
)


def is_non_answer(answer: str | None) -> bool:
    """'None', 'N/A', 'Not mentioned'... Small models sometimes use the answer action to say they found nothing
    (scrapethissite.com: asked about a country not in the list, it answered "None" from another row)."""
    return NON_ANSWER.fullmatch(_clean(answer or "")) is not None or not _clean(answer or "")


def _numbers(text: str) -> list[str]:
    """'£1,250.50 on 3 May' -> ['1250.50', '3']"""
    return [n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text or "")]


def is_supported(evidence: str | None, answer: str | None, current_text: str, all_texts: list[str]) -> bool:
    """True when the answer is backed by the website: the evidence is a real quote from the current page that
    contains every number in the answer, or else the answer itself appears on a visited page.

    The number rule matters: small models sometimes pair a real quote with a number they made up (sqlite.org:
    answer "500", quote about column limits that says nothing of 500, and 500 nowhere on the page)."""
    current = _clean(current_text)
    if evidence and len(_clean(evidence)) >= 3 and _clean(evidence) in current:
        quoted = _clean(evidence).replace(",", "")
        if all(n in quoted for n in _numbers(answer)):
            return True
    if answer and len(_clean(answer)) >= 2:
        needle = _clean(answer)
        return any(needle in _clean(t) for t in all_texts)
    return False


QUESTION_WORDS = {"what", "who", "where", "when", "which", "how", "why", "in", "on", "since", "according", "is",
                  "are", "does", "do", "the", "a", "an", "of", "for"}


def question_names(query: str) -> dict[str, str]:
    """Things the question names: capitalised words after the first word, and words like "iPad".
    Quoted text ('A Light in the Attic', a quote) is skipped: it is matched elsewhere and often not repeated.
    Returns {stemmed token: word as written}."""
    words = re.findall(r"[A-Za-z][\w.+-]*", re.sub(r"'[^']*'|\"[^\"]*\"|‘[^’]*’|“[^”]*”", " ", query))
    names: dict[str, str] = {}
    for i, word in enumerate(words):
        if i == 0 or word.lower() in QUESTION_WORDS:
            continue
        if word[0].isupper() or re.search(r"[a-z][A-Z]", word):
            for token in tokenize(word):
                names[stem(token)] = word
    return names


def missing_names(query: str, evidence: str | None, page: Page) -> list[str]:
    """Names from the question that appear neither in the evidence, nor within 2 lines of it, nor in the page title.

    Catches answers about the wrong item: asked for the "Apple iPad Air", the model quoted the iPad Mini's price
    (webscraper.io), or a nationwide fee table that never mentions Yellowstone (nps.gov). On all 38 correct
    answers from past evals this rejected none."""
    names = question_names(query)
    if not names:
        return []
    lines = page.text.splitlines()
    first = _clean((evidence or "").splitlines()[0]) if evidence and evidence.strip() else ""
    near = []
    if first:
        for i, line in enumerate(lines):
            if first[:40] in _clean(line):
                near += lines[max(0, i - 2): i + 3]
    words = {stem(t) for t in tokenize(" ".join(near) + " " + page.title + " " + (evidence or ""))}
    return sorted({word for token, word in names.items() if token not in words})


class WebAgent:
    def __init__(self, llm, browser, max_steps=8, max_page_chars=6000, max_links=40,
                 same_domain_only=True, verbose=True, max_errors=8):
        self.llm = llm
        self.browser = browser
        self.max_steps = max_steps
        self.max_page_chars = max_page_chars
        self.max_links = max_links
        self.same_domain_only = same_domain_only
        self.verbose = verbose
        self.max_errors = max_errors  # pages that fail to load don't use up max_steps, but this caps them

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, file=sys.stderr, flush=True)

    def _allowed(self, url: str, start_site: str) -> bool:
        return not self.same_domain_only or _site(urlparse(url).hostname) == start_site

    def _problem(self, decision: dict, query: str, page: Page, all_texts: list[str]) -> str | None:
        """None when the answer is backed by the page; otherwise the message asking the model to re-check."""
        evidence, answer = _as_text(decision.get("evidence")), _as_text(decision.get("answer"))
        if not is_supported(evidence, answer, page.text, all_texts):
            return EVIDENCE_RETRY
        missing = missing_names(query, evidence, page)
        if missing:
            return SUBJECT_RETRY.format(names=", ".join(missing))
        return None

    def _decide(self, messages: list[dict], query: str, page: Page, all_texts: list[str],
                can_backtrack: bool = False) -> tuple[dict, bool]:
        decision = self.llm.decide(messages)
        if decision.get("action") == "give_up" and can_backtrack:
            # Small models often say "this page doesn't mention it" and give up on the whole site.
            # Like answers, a give_up gets one re-check while there are still places left to look.
            self._log("    give_up with unvisited links left; asking the model to confirm")
            decision = self.llm.decide(messages + [
                {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)},
                {"role": "user", "content": GIVE_UP_RETRY.format(pages=len(all_texts))},
            ])
        if decision.get("action") == "answer" and is_non_answer(_as_text(decision.get("answer"))):
            # "Nothing here" is not an answer: treat it as leaving this page and keep searching.
            return {"action": "back", "notes": decision.get("notes"),
                    "reason": f"answered {_as_text(decision.get('answer'))!r}, i.e. not found on this page"}, False
        if decision.get("action") != "answer":
            return decision, False
        problem = self._problem(decision, query, page, all_texts)
        if problem is None:
            return decision, True
        # The quote isn't on the page, or isn't about what was asked: give the model one chance to correct itself.
        self._log(f"    answer not backed by the page; asking the model to re-check ({problem[:60]}...)")
        second = self.llm.decide(messages + [
            {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)},
            {"role": "user", "content": problem},
        ])
        if second.get("action") != "answer":
            return second, False
        return second, self._problem(second, query, page, all_texts) is None

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
        errors = 0

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

            try:
                final_url, html = self.browser.fetch(next_url)
                if not self._allowed(final_url, start_site):
                    # e.g. an nps.gov link that redirected to a Microsoft login page
                    raise FetchError(f"Redirected off the website to {urlparse(final_url).hostname}")
            except FetchError as exc:
                # Broken, blocked (e.g. HTTP 403 from bot protection) or off-site pages teach us nothing,
                # so they don't use up the page budget (ox.ac.uk: 6 of 8 pages were 403s).
                errors += 1
                self._log(f"[-] ERROR {exc}")
                result.steps.append({"step": result.pages_opened, "url": next_url, "action": "error", "detail": str(exc)})
                next_url = None
                if errors >= self.max_errors:
                    result.error = f"Stopped: {errors} pages could not be loaded (last: {exc})"
                    result.steps.append({"step": result.pages_opened, "action": "stop", "detail": result.error})
                    break
                continue

            result.pages_opened += 1
            step = result.pages_opened
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
                    heapq.heappush(frontier, (-link_score(link, query, page.title), next(order), link.url))
            shown = rank_links(candidates, query, self.max_links, page.title)

            prompt = build_user_prompt(
                query=query, page_url=final_url, page_title=page.title,
                page_text=trim_text(page.text, query, self.max_page_chars), links=shown,
                visited=result.visited[:-1], notes=notes, step=step, max_steps=self.max_steps,
                highlights=top_passages(page.text, query),
            )
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]

            try:
                can_backtrack = result.pages_opened < self.max_steps and any(
                    normalize_url(u) not in visited_keys for _, _, u in frontier)
                decision, verified = self._decide(messages, query, page, page_texts, can_backtrack)
            except Exception as exc:  # model unreachable or unusable: stop and report it
                result.error = f"Model error: {exc}"
                self._log(f"[{step}] {result.error}")
                break

            action = str(decision.get("action", "")).lower()
            if decision.get("notes"):
                notes = _as_text(decision["notes"])[:500]
            record = {"step": step, "url": final_url, "action": action,
                      "detail": _as_text(decision.get("reason") or decision.get("answer")) or ""}

            if action == "answer" and not verified:
                # Not backed by the page even after a re-check: likely made up. Keep looking instead.
                record["action"] = "rejected_answer"
                record["detail"] = f"unsupported answer {_as_text(decision.get('answer'))!r}; backtracking"
                result.steps.append(record)
                self._log(f"    {record['detail']}")
                next_url = None
                continue

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
