# Match 4 Build Plan — Web Information Retrieval Agent

**Match:** AMD AI League · Match 4 · Intelligent Web Information Retrieval
**Deadline:** Oct 27, 2026 · **Points:** 1,800 · **Team:** mustang (solo)

---

## 1. Goal

Build an AI agent that takes a **website + a natural-language question**, browses the site on its own, and returns the **exact answer + the page it came from**.

```
Input:  {"url": "https://example.edu", "query": "What is the MTech application deadline?"}
Output: {"answer": "December 15, 2026", "source": "https://example.edu/admissions/mtech", "steps": 3}
```

> The official brief isn't published yet. The input/output format above is a placeholder and will be adapted once the brief arrives. Only `main.py` should need to change.

---

## 2. Architecture

```
            ┌──────────────────────────────────────────┐
 query+url →│  AGENT LOOP (agent.py)                    │
            │                                            │
            │  1. browser.open(url)        ← browser.py │
            │  2. extractor.clean(page)    ← extractor.py│
            │  3. llm.decide(query, page)  ← llm.py     │
            │        │                                   │
            │        ├─ "answer"  → return result        │
            │        ├─ "visit"   → next url, go to 1    │
            │        └─ "give_up" → return not found     │
            │                                            │
            │  limits: max steps, visited set, timeouts  │
            └──────────────────────────────────────────┘
                                 → {"answer", "source"}
```

---

## 3. Files

| File | Job |
|---|---|
| `browser.py` | Opens pages with Playwright (headless Chromium), handles JS sites, timeouts, errors |
| `extractor.py` | Turns raw HTML into clean text + a numbered list of links; trims long pages |
| `prompts.py` | The instructions we give the LLM (system prompt + decision format) |
| `llm.py` | Talks to the model through an OpenAI-compatible API; swappable model/endpoint |
| `agent.py` | The loop: open → clean → decide → act; tracks visited pages and step count |
| `main.py` | Entry point: reads input, runs the agent, prints/saves output in the required format |
| `tests/cases.json` | Our own test set: URL + question + expected answer |
| `evaluate.py` | Runs all test cases and reports accuracy |
| `requirements.txt` | Python packages |
| `README.md` | How to run, design, results (for submission) |

---

## 4. The LLM decision format

Every step, the model must reply with JSON only:

```json
{"action": "answer",  "answer": "...", "source": "<url>", "reason": "..."}
{"action": "visit",   "link_id": 7, "reason": "Admissions page likely has deadlines"}
{"action": "give_up", "reason": "Site has no information about this"}
```

Strict JSON → our code can act on it reliably. Invalid JSON → retry once with a correction message.

---

## 5. Model choice

- **Primary:** open model served on the AMD GPU in the hackathon notebook via **vLLM** (OpenAI-compatible server), e.g. a Qwen or Llama instruct model (7B–14B). Shows AMD usage.
- **Fallback:** any OpenAI-compatible API, for local development on the Mac.
- Switching = change two settings (`BASE_URL`, `MODEL`) in `.env`. No code change.

---

## 6. Build stages

| Stage | When | Output |
|---|---|---|
| 1. Skeleton | Day 1 | Project structure, browser opens a page, extractor prints clean text + links |
| 2. First loop | Day 1–2 | Agent answers simple questions on 2–3 test sites |
| 3. Test set | Day 2 | 15–20 test cases across different site types |
| 4. Reliability | Day 3–4 | Better link choice, long-page handling, retries, dead links, loop prevention |
| 5. Run on AMD | Day 4 | Model served on AMD GPU via vLLM, agent runs end-to-end in notebook |
| 6. Adapt to brief | When published | Match official input/output + evaluation format |
| 7. Package | Day 5–6 | README, demo video, slides, submission form |
| Buffer | Oct 26–27 | Fixes, final submit before deadline |

---

## 7. Reliability tricks (where we beat other entries)

- **Visited set** — never open the same page twice
- **Step limit** — stop after N pages (default 8)
- **Link ranking** — show the LLM the most relevant links first (keyword match with the query) when a page has hundreds
- **Page trimming** — keep the text most related to the query when a page is too long for the model
- **Same-site rule** — don't wander off to external domains unless needed
- **Answer check** — before returning, confirm the answer text actually appears on the source page
- **Backtracking** — if a path dead-ends, return to the best unvisited link from earlier pages

---

## 8. Risks

| Risk | Plan |
|---|---|
| Official brief changes format | Keep I/O isolated in `main.py` |
| 3 h/day GPU quota | Develop on Mac with API; use GPU for final runs only; turn session off when idle |
| Sites block bots / need login | Out of scope; return "not found" gracefully |
| Model too slow on big pages | Trimming + smaller model |
| Files lost on session end | Save in the correct persistent folder; push to GitHub often |

---

## 9. Submission checklist

- [ ] Public GitHub repo
- [ ] README (what, how, results)
- [ ] Demo video
- [ ] Slides / presentation
- [ ] lablab submission form (title, description 100+ words, links)
