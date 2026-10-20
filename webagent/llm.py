"""Connection to the brain: any OpenAI-compatible chat endpoint (Ollama, vLLM on AMD, Fireworks, OpenAI...)."""

from __future__ import annotations

import json
import re

from openai import BadRequestError, OpenAI

from .prompts import JSON_RETRY


def parse_json(raw: str) -> dict:
    """Pull the first JSON object out of a model reply, tolerating code fences and <think> blocks."""
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"No JSON object in model reply: {raw[:200]!r}")


class LLM:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.0, max_tokens: int = 700):
        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._json_mode = True  # turned off automatically if the server doesn't support it

    def chat(self, messages: list[dict]) -> str:
        kwargs = dict(model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens)
        if self._json_mode:
            try:
                response = self.client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
                return response.choices[0].message.content or ""
            except BadRequestError:
                self._json_mode = False
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def decide(self, messages: list[dict]) -> dict:
        """Ask for a decision; if the reply isn't valid JSON, ask once more."""
        raw = self.chat(messages)
        try:
            return parse_json(raw)
        except ValueError:
            retry = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": JSON_RETRY},
            ]
            return parse_json(self.chat(retry))
