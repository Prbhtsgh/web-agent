"""Web information retrieval agent: browses a website to answer a natural-language question."""

from .agent import WebAgent, AgentResult
from .browser import Browser, FetchError
from .llm import LLM

__all__ = ["WebAgent", "AgentResult", "Browser", "FetchError", "LLM"]
