"""Shared paths, constants and small helpers for the GAID corpus pipeline."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
RELEASE = DATA / "release"
VALID = DATA / "validation"
LOGS = ROOT / "logs"
for _p in (RAW, INTERIM, RELEASE, VALID, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

# ---- corpus definition -------------------------------------------------
DATE_START = "2023-01-01"
DATE_END = "2026-10-31"
# Europe PMC query scope: open-access full text held in Europe PMC, excluding preprints (records must be in PMC)
EPMC_SCOPE = "OPEN_ACCESS:y AND IN_EPMC:y AND IN_PMC:y"
EPMC_SEARCH_POST = "https://www.ebi.ac.uk/europepmc/webservices/rest/searchPOST"
S3_BASE = "https://pmc-oa-opendata.s3.amazonaws.com"
USER_AGENT = "gaid-corpus-builder/0.1 (academic research; python-aiohttp)"

# ---- candidate retrieval vocabulary (Europe PMC full-text search, high recall) ----
TOOL_TERMS = [
    "ChatGPT", "Chat GPT", "GPT", "GPT-3.5", "GPT-4", "GPT-4o", "GPT-5", "OpenAI", "Grammarly", "QuillBot", "DeepL",
    "Paperpal", "Writefull", "DeepSeek", "Copilot", "Anthropic", "Gemini", "Claude", "Perplexity", "NotebookLM",
    "SciSpace", "Wordtune", "Trinka", "Midjourney", "DALL-E", "Ernie Bot", "Kimi", "Doubao", "Qwen", "Llama",
    "Mistral", "Bard", "Elicit", "Scite", "Jenni", "Curie", "ProWritingAid", "LanguageTool", "Wordvice", "InstaText",
    "Scholarcy", "Humata", "Stable Diffusion", "Grok", "Consensus AI", "Research Rabbit", "Rayyan", "Napkin AI",
]
GENERIC_TERMS = [
    "generative AI", "generative artificial intelligence", "GenAI", "Gen AI", "large language model",
    "large language models", "LLM", "LLMs", "AI-assisted", "AI tool", "AI tools", "AI technologies", "AI-generated",
    "AI-based", "language model", "AI chatbot", "chatbot", "artificial intelligence", "AI", "A.I.",
    "machine translation", "inteligencia artificial", "inteligência artificial", "intelligence artificielle",
    "künstliche Intelligenz", "IA generativa", "IA générative",
]


def candidate_query() -> str:
    return " OR ".join(f'"{t}"' for t in TOOL_TERMS + GENERIC_TERMS)


def months(start: str = DATE_START, end: str = DATE_END):
    """Yield (label, first_day, last_day) for each calendar month in [start, end]."""
    import calendar
    y, m = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    while (y, m) <= (ey, em):
        last = calendar.monthrange(y, m)[1]
        yield f"{y:04d}-{m:02d}", f"{y:04d}-{m:02d}-01", f"{y:04d}-{m:02d}-{last:02d}"
        m += 1
        if m == 13:
            y, m = y + 1, 1


def load_env() -> dict:
    env = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                env[k] = v
    env.update({k: v for k, v in os.environ.items() if k.startswith("AIHUBMIX_")})
    return env


def read_jsonl(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows, mode="wt"):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, mode, encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
