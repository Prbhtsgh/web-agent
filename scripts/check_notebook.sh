#!/usr/bin/env bash
# Is this AMD notebook ready for the eval? Checks GPU, vLLM, disk, internet and the headless browser.
# Run from the repo folder after setup:  bash scripts/check_notebook.sh
# Every line should say OK. Read the README checklist for what to do when one doesn't.

ok()   { echo "  OK    $*"; }
fail() { echo "  FAIL  $*"; failed=1; }
failed=0

echo "GPU"
if rocm-smi --showproductname > /dev/null 2>&1; then ok "rocm-smi sees a GPU"; else fail "rocm-smi sees no GPU"; fi
python -c "import torch; assert torch.cuda.is_available(); p = torch.cuda.get_device_properties(0); print(f'  OK    PyTorch sees {p.name}, {p.total_memory // 2**30} GB')" 2> /dev/null \
  || fail "PyTorch can't see the GPU"
python -c "import vllm; print(f'  OK    vLLM {vllm.__version__} installed')" 2> /dev/null || fail "vLLM is not installed in this Python"

echo "Disk (persistent quota is ~25 GB; model weights need a bigger disk)"
df -h "${PERSIST:-.}" "${HF_HOME:-$HOME/.cache/huggingface}" 2> /dev/null | sed 's/^/        /'

echo "Internet (the agent browses live websites; vLLM downloads from Hugging Face)"
for url in https://books.toscrape.com/ https://quotes.toscrape.com/ https://en.wikipedia.org/wiki/AMD https://www.gov.uk/ https://huggingface.co/; do
  code=$(curl -s -o /dev/null -m 15 -w "%{http_code}" "$url")
  if [ "$code" = "200" ]; then ok "$url"; else fail "$url (HTTP ${code:-none})"; fi
done

echo "Headless browser (what the agent really uses)"
python - << 'EOF' || failed=1
from webagent import Browser
from webagent.extractor import parse_html
try:
    with Browser(timeout_ms=30000) as browser:
        url, html = browser.fetch("https://books.toscrape.com/")
        page = parse_html(html, url)
    assert "51.77" in page.text, "page loaded but the expected text is missing"
    print(f"  OK    Playwright opened {url} ({len(page.links)} links)")
except Exception as exc:
    print(f"  FAIL  Playwright: {exc}")
    raise SystemExit(1)
EOF

if [ "$failed" -eq 0 ]; then echo "All checks passed."; else echo "Some checks failed: see the README checklist."; exit 1; fi
