"""Minimal prompt -> Ollama reply, reusing agentrx's OllamaProvider. Edit PROMPT/MODEL, then: python ask_ollama.py"""
from agentrx.providers import OllamaProvider

PROMPT = "What fault attribution is targetted in AgentRx?"
MODEL = "qwen2.5-coder:7b"

p = OllamaProvider(MODEL, num_ctx=4096, settings={"temperature": 0})
# ponytail: uses private _post; complete() is JSON-mode + stage-specific, so it can't take a free-form prompt
raw = p._post({"model": p.model, "messages": [{"role": "user", "content": PROMPT}], "stream": False,
               "keep_alive": p.keep_alive, "options": {"num_ctx": p.num_ctx, **p.settings}})
print(raw["message"]["content"])
