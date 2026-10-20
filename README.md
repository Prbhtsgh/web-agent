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

## Running the eval on the AMD GPU notebook (notebooks.amd.com)

The League gives **3 GPU hours per day, and idle time counts**, so prepare everything before you start the model,
and shut the session down when you're done. Only about **25 GB** is persistent; the model weights (14B: ~30 GB,
32B: ~65 GB) go on a bigger, non-persistent disk and are re-downloaded each session (a few minutes).

**First session (setup, ~10 min before the model starts):**

1. Open `notebooks.amd.com/hackathon` → *Launch Notebook* → open a *Terminal*.
2. Pick the persistent folder from the notebook's URL: `jupyter-hack-…` → `/persistent`; `rgapi-hackathon-…` →
   `/workspace`. Files anywhere else are lost when the session ends.
   ```bash
   export PERSIST=/persistent            # or /workspace (see above)
   cd $PERSIST && git clone https://github.com/Prbhtsgh/web-agent.git && cd web-agent
   ```
3. A Python environment that persists and reuses the notebook's preinstalled PyTorch + vLLM:
   ```bash
   python -m venv --system-site-packages $PERSIST/venv && source $PERSIST/venv/bin/activate
   pip install -r requirements.txt
   export PLAYWRIGHT_BROWSERS_PATH=$PERSIST/ms-playwright && python -m playwright install chromium
   ```
   If Chromium later complains about missing system libraries: `python -m playwright install-deps chromium`.
4. Choose a disk for the model weights: run `df -h /tmp $HOME /workspace /persistent` and pick the mount with the
   most free space (needs ~35 GB for 14B, ~70 GB for 32B), e.g. `export HF_HOME=/tmp/hf`.
5. Check the notebook can do the job: `bash scripts/check_notebook.sh`. It checks the GPU, vLLM, disk, internet
   access to the eval websites and Hugging Face, and opens a real page with the headless browser. **If the internet
   checks fail, stop**: the agent can't browse, so the eval can't run there. Shut down so you don't waste quota.

**Every session (after setup):**

```bash
export PERSIST=/persistent                                   # or /workspace
cd $PERSIST/web-agent && git pull && source $PERSIST/venv/bin/activate
export PLAYWRIGHT_BROWSERS_PATH=$PERSIST/ms-playwright HF_HOME=/tmp/hf    # HF_HOME: the big disk from step 4
bash scripts/start_vllm.sh            # picks Qwen2.5-32B (>=120 GB GPU) or 14B (48 GB GPU); prints an export line
export LLM_BASE_URL=http://localhost:8000/v1 LLM_API_KEY=EMPTY LLM_MODEL=<model it printed>
python main.py --url https://books.toscrape.com --query "What is the price of A Light in the Attic?"   # smoke test
python evaluate.py 2>&1 | tee results/eval_amd_$(date +%Y%m%d_%H%M).log                                 # all 30 cases
```

Then save and stop:
- The eval JSON is in `$PERSIST/web-agent/results/` (persistent). Download it from the Jupyter file browser
  (right-click → *Download*) into `results/` on your own machine; `results/` is not committed to git.
- `pkill -f "vllm serve"`, then shut the session down (*File → Hub Control Panel → Stop My Server*), because idle
  time still counts against the 3 hours.

## Batch mode

```bash
python main.py --input tasks.json --output results.json      # tasks.json: [{"id", "url", "query"}, ...]
python main.py --url ... --query ... --full                  # full step-by-step trace
```

## Status

- [x] Core agent loop, browser, extractor, verification, backtracking
- [x] Offline test suite (33 tests, local test website + scripted model)
- [x] 30 verified eval cases across 9 site types: **28/30 with qwen2.5:7b** on a Mac
- [ ] Eval on the AMD GPU via vLLM (larger model)
- [ ] Adapt `main.py` to the official Match 4 input/output format
