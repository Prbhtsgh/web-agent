"""The exam: runs every test case and reports accuracy, so changes are measured, not guessed.

    python evaluate.py                          # tests/cases.json, 3 cases at a time
    python evaluate.py --only 15,19,1,5         # just some cases
    python evaluate.py --workers 1              # one at a time (old behaviour)

Running cases in parallel needs Ollama started with OLLAMA_NUM_PARALLEL=3, otherwise the model still
answers one request at a time and only the page loading overlaps.
"""

import argparse
import json
import os
import queue
import sys
import threading
import time

from main import build_agent
from webagent import Browser
from webagent.agent import AgentResult
from webagent.config import load_settings


def norm(text) -> str:
    return " ".join(str(text or "").lower().replace(",", "").split())


def is_correct(answer, expected) -> bool:
    """`expected` is a list of accepted options. An option is a string that must appear in the answer, or a
    list of strings that must ALL appear (e.g. ["20", "35"] for "$20.00-$35.00"). [] means the site does not
    contain the answer, so the right result is "not found"."""
    if answer is None:
        return expected == []
    options = expected if isinstance(expected, list) else [expected]
    for option in options:
        parts = option if isinstance(option, list) else [option]
        if all(norm(part) in norm(answer) for part in parts):
            return True
    return False


def breakdown(rows: list[dict], key: str) -> str:
    """'docs 3/4, shop 5/5, ...' so weak spots by site type or question kind are visible."""
    groups: dict[str, list[bool]] = {}
    for row in rows:
        groups.setdefault(row.get(key) or "-", []).append(row["ok"])
    return ", ".join(f"{name} {sum(oks)}/{len(oks)}" for name, oks in sorted(groups.items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="tests/cases.json")
    parser.add_argument("--only", help="Comma-separated case ids to run")
    parser.add_argument("--verbose", action="store_true", help="Show each agent step")
    parser.add_argument("--workers", type=int, default=3, help="Cases run at the same time (each has its own browser)")
    args = parser.parse_args()

    with open(args.cases, encoding="utf-8") as f:
        cases = json.load(f)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        cases = [c for c in cases if str(c["id"]) in wanted]

    settings = load_settings()
    rows: list[dict] = []
    todo: queue.Queue = queue.Queue()
    for case in cases:
        todo.put(case)
    lock = threading.Lock()

    def worker() -> None:
        """Each worker has its own browser (Playwright objects can't be shared between threads)."""
        with Browser(timeout_ms=settings.page_timeout_ms, headless=settings.headless) as browser:
            agent = build_agent(settings, browser, verbose=args.verbose)
            while True:
                try:
                    case = todo.get_nowait()
                except queue.Empty:
                    return
                try:
                    result = agent.run(case["url"], case["query"])
                except Exception as exc:  # a crash in one case must not silently drop it from the results
                    result = AgentResult(query=case["query"], start_url=case["url"], error=f"Crashed: {exc!r}")
                ok = is_correct(result.answer if result.found else None, case["expected"])
                with lock:
                    rows.append({"id": case["id"], "ok": ok, "category": case.get("category"),
                                 "kind": case.get("kind"), "query": case["query"], "expected": case["expected"],
                                 "result": result.to_dict()})
                    mark = "PASS" if ok else "FAIL"
                    print(f"{mark}  #{case['id']:<3} pages={result.pages_opened:<2} {result.seconds:>6.1f}s  "
                          f"answer={result.answer!r}" + (f"  error={result.error}" if result.error else ""), flush=True)

    started = time.time()
    threads = [threading.Thread(target=worker) for _ in range(max(1, min(args.workers, len(cases))))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.time() - started
    rows.sort(key=lambda r: r["id"])

    passed = sum(r["ok"] for r in rows)
    total = len(rows) or 1
    avg_pages = sum(r["result"]["pages_opened"] for r in rows) / total
    avg_secs = sum(r["result"]["seconds"] for r in rows) / total
    print(f"\nAccuracy: {passed}/{len(rows)} ({100 * passed / total:.0f}%)   "
          f"avg pages: {avg_pages:.1f}   avg time: {avg_secs:.1f}s   model: {settings.llm_model}")
    print(f"Wall time: {wall:.0f}s with {len(threads)} worker(s)  (per-case times grow when cases share the model)")
    print(f"By site type:     {breakdown(rows, 'category')}")
    print(f"By question kind: {breakdown(rows, 'kind')}")

    os.makedirs("results", exist_ok=True)
    path = f"results/eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": settings.llm_model, "passed": passed, "total": len(rows), "workers": len(threads),
                   "wall_seconds": round(wall, 1), "cases": rows}, f, indent=2)
    print(f"Details saved to {path}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
