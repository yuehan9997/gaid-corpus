"""Language models used in each role of the pipeline (Table 2 of the Data Descriptor).

DISPLAY also names the models that were tried in the pilot (pilot_llm.py, pilot_round2.py).
"""
from __future__ import annotations

PRIMARY = "deepseek-v4.1-flash"                      # annotates every distinct passage
SECOND = "qwen3.8-flash"                             # independent second annotator, different developer
# reference consensus (majority vote): three developers, none shared with PRIMARY
REF_MODELS = ["qwen3.6-35b-a3b", "glm-5.3", "doubao-seed-2-1-pro"]
# open-weight models served on a local OpenAI-compatible server (LOCAL_BASE_URL / LOCAL_API_KEY in .env)
LOCAL_MODELS = {"qwen3.6-35b-a3b", "qwen3.5-9b"}
AUDIT_MODEL = PRIMARY                                # recall audits re-read article text with the primary model ...
AUDIT_SCREENER2 = "qwen-plus-latest"                 # ... and, independently, with a second screener (final audits)
AUDIT_SCREENERS = [PRIMARY, AUDIT_SCREENER2]

# request parameters per model (reasoning off for the annotators and screeners, on for the reference models;
# large output budgets for reasoning models)
EXTRA = {
    "deepseek-v4.1-flash": {"thinking": {"type": "disabled"}},
    "qwen3.8-flash": {"enable_thinking": False},
    "qwen-plus-latest": {"enable_thinking": False},
    "qwen3.6-35b-a3b": {"chat_template_kwargs": {"enable_thinking": True}},   # reasoning on, like the other reference models
}
MAX_TOKENS = {"qwen3.8-max": 16000, "qwen3.7-max": 16000, "qwen3.6-35b-a3b": 16000, "glm-5.3": 16000, "kimi-k3.1-pro": 16000, "glm-5.3-flash": 16000, "doubao-seed-2-1-pro": 16000}

DISPLAY = {
    "qwen3.8-flash": "Qwen3.8-Flash (Alibaba Cloud)",
    "qwen3.7-flash": "Qwen3.7-Flash (Alibaba Cloud)",
    "qwen3.8-max": "Qwen3.8-Max (Alibaba Cloud)",
    "qwen3.7-max": "Qwen3.7-Max (Alibaba Cloud)",
    "qwen3.6-35b-a3b": "Qwen3.6-35B-A3B (Alibaba Cloud)",
    "qwen3.5-9b": "Qwen3.5-9B (Alibaba Cloud)",
    "glm-5.3-flash": "GLM-5.3-Flash (Zhipu AI)",
    "glm-5.3": "GLM-5.3 (Zhipu AI)",
    "deepseek-v4.1-flash": "DeepSeek-V4.1-Flash (DeepSeek)",
    "deepseek-v4-flash": "DeepSeek-V4-Flash (DeepSeek)",
    "deepseek-v4-pro-0813": "DeepSeek-V4-Pro (DeepSeek)",
    "kimi-k3.1-pro": "Kimi K3.1 Pro (Moonshot AI)",
    "doubao-seed-2-1-pro": "Doubao Seed 2.1 Pro (ByteDance)",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite (Google)",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro (Google)",
    "gpt-5.5": "GPT-5.5 (OpenAI)",
    "gpt-4.1-nano": "GPT-4.1 nano (OpenAI)",
    "gpt-4.1-mini": "GPT-4.1 mini (OpenAI)",
    "gpt-4o-mini": "GPT-4o mini (OpenAI)",
    "gpt-5-nano": "GPT-5 nano (OpenAI)",
    "gpt-5-mini": "GPT-5 mini (OpenAI)",
    "gpt-5.4-nano": "GPT-5.4 nano (OpenAI)",
    "qwen-flash": "Qwen-Flash (Alibaba Cloud)",
    "qwen-plus-latest": "Qwen-Plus (Alibaba Cloud)",
    "claude-haiku-5-5": "Claude Haiku 5.5 (Anthropic)",
    "claude-opus-5-5": "Claude Opus 5.5 (Anthropic)",
}


def show(model: str) -> str:
    return DISPLAY.get(model, model)


def extra_for(model: str) -> dict:
    return dict(EXTRA.get(model, {}))
