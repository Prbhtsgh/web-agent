"""Settings, read from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # Any OpenAI-compatible endpoint: Ollama, vLLM on the AMD GPU, Fireworks, OpenAI...
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "EMPTY")
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:7b")

    max_steps: int = int(os.getenv("MAX_STEPS", "8"))            # pages the agent may open
    max_page_chars: int = int(os.getenv("MAX_PAGE_CHARS", "6000"))  # page text sent to the model
    max_links: int = int(os.getenv("MAX_LINKS", "40"))           # links shown to the model
    same_domain_only: bool = _bool("SAME_DOMAIN_ONLY", True)
    page_timeout_ms: int = int(os.getenv("PAGE_TIMEOUT_MS", "20000"))
    headless: bool = _bool("HEADLESS", True)


def load_settings() -> Settings:
    return Settings()
