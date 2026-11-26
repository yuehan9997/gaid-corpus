"""Thin async client for the OpenAI-compatible aihubmix endpoint, with retries and a persistent cost ledger."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import aiohttp

from common import LOGS, ROOT, load_env

LEDGER = LOGS / "llm_ledger.jsonl"
PRICE_CACHE = ROOT / "data" / "raw" / "aihubmix_prices.json"

# USD per million tokens (input, cached_input, output) as published by aihubmix on 2026-11-14 (/api/v1/models).
# Filled / refreshed by refresh_prices(); these defaults are only a fallback.
PRICES: dict[str, tuple[float, float, float]] = {}


def refresh_prices(force: bool = False) -> dict:
    global PRICES
    if PRICE_CACHE.exists() and not force:
        PRICES = {k: tuple(v) for k, v in json.loads(PRICE_CACHE.read_text()).items()}
        return PRICES
    import urllib.request
    env = load_env()
    req = urllib.request.Request("https://aihubmix.com/api/v1/models", headers={
        "Authorization": "Bearer " + env["AIHUBMIX_API_KEY"], "User-Agent": "gaid-corpus-builder/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    items = data.get("data", data)
    out = {}
    for it in items:
        p = it.get("pricing") or {}
        if "input" in p and "output" in p:
            out[it["model_id"]] = (float(p["input"]), float(p.get("cache_read", p["input"])), float(p["output"]))
    PRICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PRICE_CACHE.write_text(json.dumps(out))
    PRICES = out
    return out


def cost_usd(model: str, usage: dict) -> float:
    from models import LOCAL_MODELS
    if model in LOCAL_MODELS:          # local server: no charge
        return 0.0
    if not PRICES:
        refresh_prices()
    pin, pcache, pout = PRICES.get(model, (0.0, 0.0, 0.0))
    prompt = usage.get("prompt_tokens", 0) or 0
    comp = usage.get("completion_tokens", 0) or 0
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) or 0
    return ((prompt - cached) * pin + cached * pcache + comp * pout) / 1e6


def ledger_total() -> dict:
    tot: dict = {}
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            t = tot.setdefault(r["model"], {"calls": 0, "prompt": 0, "completion": 0, "usd": 0.0})
            t["calls"] += r.get("calls", 1)
            t["prompt"] += r.get("prompt", 0)
            t["completion"] += r.get("completion", 0)
            t["usd"] += r.get("usd", 0.0)
    return tot


class LLM:
    def __init__(self, model: str, tag: str, conc: int = 8, max_tokens: int = 1500, extra: dict | None = None,
                 timeout: int = 180):
        env = load_env()
        from models import LOCAL_MODELS
        if model in LOCAL_MODELS:      # open-weight model on a local server
            self.url = env["LOCAL_BASE_URL"].rstrip("/") + "/chat/completions"
            self.key = env["LOCAL_API_KEY"]
        else:
            self.url = env["AIHUBMIX_BASE_URL"].rstrip("/") + "/chat/completions"
            self.key = env["AIHUBMIX_API_KEY"]
        self.model, self.tag, self.max_tokens = model, tag, max_tokens
        self.extra = dict(extra or {})
        self.sem = asyncio.Semaphore(conc)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.acc = {"calls": 0, "prompt": 0, "completion": 0, "cached": 0, "usd": 0.0, "errors": 0}
        self._session: aiohttp.ClientSession | None = None
        refresh_prices()

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, *exc):
        await self._session.close()
        self.flush()

    def flush(self):
        if self.acc["calls"]:
            with open(LEDGER, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "model": self.model, "tag": self.tag,
                                     **self.acc}) + "\n")
            self.acc = {"calls": 0, "prompt": 0, "completion": 0, "cached": 0, "usd": 0.0, "errors": 0}

    async def chat(self, system: str, user: str, json_mode: bool = True, retries: int = 5,
                   cache_system: bool = False) -> tuple[str | None, dict]:
        sys_content = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}] if cache_system else system
        body = {"model": self.model, "messages": [{"role": "system", "content": sys_content}, {"role": "user", "content": user}],
                "max_tokens": self.max_tokens, "temperature": 0}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(self.extra)
        last_err = None
        async with self.sem:
            for attempt in range(retries):
                try:
                    async with self._session.post(self.url, json=body, headers={
                            "Authorization": "Bearer " + self.key, "Content-Type": "application/json"}) as r:
                        txt = await r.text()
                        if r.status == 200:
                            d = json.loads(txt)
                            usage = d.get("usage") or {}
                            self.acc["calls"] += 1
                            self.acc["prompt"] += usage.get("prompt_tokens", 0) or 0
                            self.acc["completion"] += usage.get("completion_tokens", 0) or 0
                            self.acc["cached"] += ((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) or 0
                            self.acc["usd"] += cost_usd(self.model, usage)
                            content = (d.get("choices") or [{}])[0].get("message", {}).get("content")
                            return content, usage
                        last_err = f"HTTP {r.status}: {txt[:300]}"
                        # parameter not supported by this model -> drop it and retry immediately
                        low = txt.lower()
                        if r.status == 400:
                            changed = False
                            if "temperature" in low and "temperature" in body:
                                body.pop("temperature"); changed = True
                            if "max_tokens" in low and "max_completion_tokens" in low and "max_tokens" in body:
                                body["max_completion_tokens"] = body.pop("max_tokens"); changed = True
                            if "response_format" in low and "response_format" in body:
                                body.pop("response_format"); changed = True
                            if "reasoning_effort" in low and "reasoning_effort" in body:
                                body.pop("reasoning_effort"); changed = True
                            if changed:
                                continue
                            break
                        if r.status in (401, 403):
                            break
                except Exception as e:  # noqa: BLE001
                    last_err = f"{type(e).__name__}: {e}"
                await asyncio.sleep(min(30, 2 ** attempt))
        self.acc["errors"] += 1
        return None, {"error": last_err}
