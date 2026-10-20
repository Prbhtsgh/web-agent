"""Front door. The ONLY file that should change when the official Match 4 input/output format is published.

One question:
    python main.py --url https://books.toscrape.com --query "What is the price of A Light in the Attic?"
    (add --show-browser to watch the Chromium window while it browses)

A batch of tasks (JSON list of {"id", "url", "query"}):
    python main.py --input tasks.json --output results.json
"""

import argparse
import json
import sys

from webagent import LLM, Browser, WebAgent
from webagent.config import load_settings


def build_agent(settings, browser, verbose=True) -> WebAgent:
    llm = LLM(base_url=settings.llm_base_url, api_key=settings.llm_api_key, model=settings.llm_model)
    return WebAgent(
        llm=llm, browser=browser, max_steps=settings.max_steps, max_page_chars=settings.max_page_chars,
        max_links=settings.max_links, same_domain_only=settings.same_domain_only, verbose=verbose,
    )


def to_output(task_id, result) -> dict:
    """Shape of what we return. Adapt this to the official format when it is published."""
    return {
        "id": task_id,
        "answer": result.answer if result.found else None,
        "source": result.source,
        "found": result.found,
        "verified": result.verified,
        "pages_opened": result.pages_opened,
        "seconds": result.seconds,
        "error": result.error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Web information retrieval agent")
    parser.add_argument("--url", help="Website to start from")
    parser.add_argument("--query", help="Question in plain English")
    parser.add_argument("--input", help="JSON file with a list of {id, url, query}")
    parser.add_argument("--output", help="Where to save results (JSON). Default: print to screen")
    parser.add_argument("--max-steps", type=int, help="Override MAX_STEPS")
    parser.add_argument("--full", action="store_true", help="Include the full step-by-step trace")
    parser.add_argument("--quiet", action="store_true", help="Hide the live step log")
    parser.add_argument("--show-browser", action="store_true", help="Show the Chromium window while the agent browses")
    args = parser.parse_args()

    if args.input:
        with open(args.input, encoding="utf-8") as f:
            tasks = json.load(f)
    elif args.url and args.query:
        tasks = [{"id": 1, "url": args.url, "query": args.query}]
    else:
        parser.error("give --url and --query, or --input")

    settings = load_settings()
    if args.max_steps:
        settings.max_steps = args.max_steps
    if args.show_browser:
        settings.headless = False

    outputs = []
    with Browser(timeout_ms=settings.page_timeout_ms, headless=settings.headless) as browser:
        agent = build_agent(settings, browser, verbose=not args.quiet)
        for i, task in enumerate(tasks, 1):
            result = agent.run(task["url"], task["query"])
            out = result.to_dict() if args.full else to_output(task.get("id", i), result)
            outputs.append(out)
            if result.error and "Connection" in result.error:
                print(f"Hint: no model server answered at {settings.llm_base_url}. "
                      "Is Ollama / vLLM running, and is LLM_BASE_URL in .env correct?", file=sys.stderr)
                break

    payload = outputs[0] if len(outputs) == 1 and not args.input else outputs
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"Saved {len(outputs)} result(s) to {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
