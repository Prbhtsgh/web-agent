# Web Information Retrieval Agent

An AI agent that answers a plain-English question about a website by **browsing it page by page**, then returns the exact answer and the page it came from.

Built for the **AMD AI League — Match 4: Intelligent Web Information Retrieval** (lablab.ai).

```bash
python main.py --url https://books.toscrape.com --query "What is the price of A Light in the Attic?"
```
Example of the output shape:
```json
{"answer": "£51.77", "source": "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
 "found": true, "verified": true, "pages_opened": 2}
```

## How it works

```
question + start URL
        │
        ▼
 ┌──────────────┐   ┌───────────────┐   ┌──────────────────────────┐
 │ browser.py   │──▶│ extractor.py  │──▶│ LLM decides (one JSON):  │
 │ open page    │   │ clean text +  │   │ answer / visit link N /  │
 │ (Playwright) │   │ numbered links│   │ back / give_up           │
 └──────────────┘   └───────────────┘   └────────────┬─────────────┘
        ▲                                             │
        └────────────── agent.py loop ◀───────────────┘
```

Reliability features:
- **Real browser** (headless Chromium) so JavaScript-rendered pages work
- **Visited set**: never opens the same page twice (handles `/`, `index.html`, `#anchors`, `www.`)
- **Step limit** so it always finishes
- **Link ranking**: the most query-relevant links are shown to the model first
- **Smart trimming**: long pages are cut down to the parts related to the question
- **Tables kept readable** (`cell | cell | cell`)
- **Same-site rule**: doesn't wander to other websites
- **Answer verification**: the model must quote evidence; if the quote isn't on the page, it's asked to re-check
- **Backtracking**: dead ends and bad link choices fall back to the best unvisited link seen so far
- **Model-agnostic**: any OpenAI-compatible endpoint (Ollama, vLLM on AMD GPUs, hosted APIs)

## Project layout

| File | Role |
|---|---|
| `webagent/browser.py` | Opens pages with Playwright |
| `webagent/extractor.py` | HTML → clean text + numbered links; link ranking; trimming |
| `webagent/prompts.py` | Instructions given to the model |
| `webagent/llm.py` | Model connection + robust JSON parsing |
| `webagent/agent.py` | The decision loop |
| `webagent/config.py` | Settings from `.env` |
| `main.py` | Command-line entry point (input/output format lives here) |
| `evaluate.py` | Runs `tests/cases.json` and reports accuracy |
| `tests/` | Offline tests with a local test website |
| `scripts/start_vllm.sh` | Serves a model on the AMD GPU with vLLM |

## Setup (Mac)

```bash
cd web-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
cp .env.example .env
```

Free local model with [Ollama](https://ollama.com):
```bash
brew install ollama      # or download the app
ollama serve &           # skip if the Ollama app is already running
ollama pull qwen2.5:7b
```

Check everything:
```bash
python -m unittest discover tests     # offline tests, no model needed
python main.py --url https://books.toscrape.com --query "What is the price of A Light in the Attic?"
python evaluate.py                     # accuracy on tests/cases.json
```

## Running on the AMD GPU notebook

1. Open `notebooks.amd.com/hackathon` → Launch Notebook → open a Terminal.
2. Work inside the persistent folder (`/persistent` or `/workspace`, see the League FAQ) and clone this repo there.
3. `pip install -r requirements.txt && python -m playwright install chromium`
   (if Chromium complains about missing system libraries: `python -m playwright install --with-deps chromium`)
4. `bash scripts/start_vllm.sh` — serves `Qwen/Qwen2.5-7B-Instruct` on the AMD GPU.
5. In `.env`, switch to Option B, then run `python evaluate.py`.
6. **Turn the session off when finished** — the 3 h/day GPU quota counts idle time.

## Batch mode

```bash
python main.py --input tasks.json --output results.json      # tasks.json: [{"id", "url", "query"}, ...]
python main.py --url ... --query ... --full                  # full step-by-step trace
```

## Status

- [x] Core agent loop, browser, extractor, verification, backtracking
- [x] Offline test suite (13 tests)
- [ ] Results with a real model on `tests/cases.json`
- [ ] Run on AMD GPU via vLLM
- [ ] Adapt `main.py` to the official Match 4 input/output format
