"""Model-selection pilot.

1. build   : passages from the pilot XMLs (2025-2026 articles, 15 publisher styles) -> stratified sample
2. run M.. : annotate the sample with the listed models (batched), store raw + parsed outputs
3. report  : agreement of each model with the consensus of the strong reference models
"""
from __future__ import annotations

import asyncio
import collections
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annot_schema import PURPOSES, SYSTEM_PROMPT, build_user, parse_response  # noqa: E402
from common import RAW, VALID  # noqa: E402
from llm_lib import LLM, ledger_total  # noqa: E402

PDIR = VALID / "pilot"
PDIR.mkdir(parents=True, exist_ok=True)
SAMPLE = PDIR / "pilot_sample.jsonl"
BATCH = 8

MODEL_CFG = {
    # reference (strong) models
    "claude-opus-5-5": {"extra": {}, "max_tokens": 3000, "conc": 4},
    "gpt-5.5": {"extra": {"reasoning_effort": "low"}, "max_tokens": 6000, "conc": 4},
    "gemini-3.1-pro-preview": {"extra": {}, "max_tokens": 8000, "conc": 4},
    "claude-sonnet-5-5": {"extra": {}, "max_tokens": 3000, "conc": 4},
    # candidate production models
    "claude-haiku-5-5": {"extra": {}, "max_tokens": 3000, "conc": 6},
    "gpt-4.1-nano": {"extra": {}, "max_tokens": 3000, "conc": 6},
    "gpt-4.1-mini": {"extra": {}, "max_tokens": 3000, "conc": 6},
    "gpt-4o-mini": {"extra": {}, "max_tokens": 3000, "conc": 6},
    "gpt-5-nano": {"extra": {"reasoning_effort": "minimal"}, "max_tokens": 6000, "conc": 6},
    "gpt-5-mini": {"extra": {"reasoning_effort": "minimal"}, "max_tokens": 6000, "conc": 6},
    "gpt-5.4-nano": {"extra": {"reasoning_effort": "none"}, "max_tokens": 4000, "conc": 6},
    "gemini-3.1-flash-lite": {"extra": {}, "max_tokens": 4000, "conc": 6},
    "deepseek-v4-flash": {"extra": {}, "max_tokens": 3000, "conc": 6},
    "deepseek-v4.1-flash": {"extra": {}, "max_tokens": 4000, "conc": 6},
    "qwen-flash": {"extra": {}, "max_tokens": 3000, "conc": 6},
    "qwen3.8-flash": {"extra": {}, "max_tokens": 4000, "conc": 6},
    "glm-5.3-flash": {"extra": {}, "max_tokens": 4000, "conc": 6},
    # added in the second round (codebook v1.1)
    "qwen3.7-flash": {"extra": {}, "max_tokens": 4000, "conc": 6},
    "qwen-plus-latest": {"extra": {}, "max_tokens": 4000, "conc": 6},
    "qwen3.8-max": {"extra": {}, "max_tokens": 6000, "conc": 4},
    "glm-5.3": {"extra": {}, "max_tokens": 6000, "conc": 4},
    "deepseek-v4-pro-0813": {"extra": {}, "max_tokens": 6000, "conc": 4},
    "kimi-k3.1-pro": {"extra": {}, "max_tokens": 6000, "conc": 4},
    "doubao-seed-2-0-lite-260428": {"extra": {}, "max_tokens": 4000, "conc": 6},
}


def build(n_ai=110, n_decl=150, n_body=90, seed=11):
    from extract_lib import extract
    from passage_rules import passages_from_article
    random.seed(seed)
    pool = collections.defaultdict(list)
    for f in sorted((RAW / "pilot_xml").glob("*.xml")):
        res = extract(f.read_bytes())
        if not res:
            continue
        ps, _ = passages_from_article(res)
        for p in ps:
            pool[p["route"]].append({"pmcid": f.stem, "journal": res["meta"].get("journal"), "loc": p["route"],
                                     "heading": p["title"], "text": p["text"], "key": p["key"]})
    out, seen = [], set()
    for route, n in (("ai_section", n_ai), ("decl_zone", n_decl), ("body", n_body)):
        items = pool[route]
        random.shuffle(items)
        k = 0
        for it in items:
            if it["key"] in seen:
                continue
            seen.add(it["key"])
            out.append(it)
            k += 1
            if k >= n:
                break
        print(route, "pool", len(items), "sampled", k)
    random.shuffle(out)
    for i, it in enumerate(out, 1):
        it["id"] = i
    with open(SAMPLE, "w", encoding="utf-8") as fh:
        for it in out:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print("sample size", len(out), "->", SAMPLE)


def load_sample():
    return [json.loads(l) for l in open(SAMPLE, encoding="utf-8")]


async def run_model(model: str, batch_size: int = BATCH, cache: bool = False, suffix: str = ""):
    cfg = MODEL_CFG[model]
    items = load_sample()
    BATCH_N = batch_size
    outp = PDIR / f"out_{model.replace('/', '_')}{suffix}.jsonl"
    done = {}
    if outp.exists():
        for l in open(outp, encoding="utf-8"):
            r = json.loads(l)
            done[r["id"]] = r
    todo = [it for it in items if it["id"] not in done]
    if not todo:
        print(model, "already complete")
        return
    batches = [todo[i:i + BATCH_N] for i in range(0, len(todo), BATCH_N)]
    t0 = time.time()
    async with LLM(model, "pilot" + suffix, conc=cfg["conc"], max_tokens=cfg["max_tokens"] * max(1, BATCH_N // 8),
                   extra=cfg["extra"]) as llm:
        async def do(batch):
            ids = [b["id"] for b in batch]
            for attempt in range(2):
                content, usage = await llm.chat(SYSTEM_PROMPT, build_user(batch), cache_system=cache)
                if attempt == 0 and batch is batches[0]:
                    print("   usage sample:", json.dumps(usage)[:400])
                got = parse_response(content, ids)
                if len(got) == len(ids):
                    return got
            return got
        results = await asyncio.gather(*[do(b) for b in batches])
        acc = dict(llm.acc)
    n = 0
    with open(outp, "a", encoding="utf-8") as fh:
        for got in results:
            for r in got.values():
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    print(f"{model}: {n}/{len(todo)} parsed in {time.time()-t0:.0f}s calls={acc['calls']} in={acc['prompt']} out={acc['completion']} "
          f"cached={acc['cached']} usd={acc['usd']:.4f} errors={acc['errors']}")


def load_out(model):
    p = PDIR / f"out_{model.replace('/', '_')}.jsonl"
    if not p.exists():
        return {}
    return {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8")}


def tool_set(r):
    return frozenset(t[0].lower().strip() for t in r["tools"])


def report(refs: list[str]):
    items = load_sample()
    ref_outs = [load_out(m) for m in refs]
    # consensus reference: majority across reference models (ties -> first reference model)
    cons = {}
    for it in items:
        rs = [o[it["id"]] for o in ref_outs if it["id"] in o]
        if not rs:
            continue
        c = {}
        for f in ("decl", "pol", "resp", "rev", "nonuse"):
            vals = [r[f] for r in rs]
            c[f] = collections.Counter(vals).most_common(1)[0][0]
        c["purp"] = sorted([p for p in PURPOSES if sum(p in r["purp"] for r in rs) * 2 > len(rs)], key=PURPOSES.index)
        ts = collections.Counter(tool_set(r) for r in rs)
        c["tools"] = ts.most_common(1)[0][0]
        cons[it["id"]] = c
    print(f"reference = majority of {refs}; n={len(cons)}; decl=1 in {sum(c['decl'] for c in cons.values())}")
    # pairwise agreement among references
    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            a, b = ref_outs[i], ref_outs[j]
            ids = [k for k in a if k in b]
            if ids:
                print(f"  {refs[i]} vs {refs[j]}: decl agree {sum(a[k]['decl']==b[k]['decl'] for k in ids)/len(ids):.3f}, "
                      f"pol agree {sum(a[k]['pol']==b[k]['pol'] for k in ids)/len(ids):.3f} (n={len(ids)})")
    print(f"\n{'model':26s} {'n':>4s} {'decl':>6s} {'pol':>6s} {'tools':>6s} {'purpEx':>6s} {'purpF1':>6s} {'resp':>6s} {'rev':>6s} {'nonuse':>6s} {'ALL':>6s}")
    tot = ledger_total()
    for model in MODEL_CFG:
        o = load_out(model)
        ids = [k for k in o if k in cons]
        if not ids:
            continue
        n = len(ids)
        decl = sum(o[k]["decl"] == cons[k]["decl"] for k in ids) / n
        pos = [k for k in ids if cons[k]["decl"] == 1 and o[k]["decl"] == 1]
        m = max(1, len(pos))
        pol = sum(o[k]["pol"] == cons[k]["pol"] for k in pos) / m
        tools = sum(tool_set(o[k]) == cons[k]["tools"] for k in pos) / m
        pex = sum(o[k]["purp"] == cons[k]["purp"] for k in pos) / m
        tp = sum(len(set(o[k]["purp"]) & set(cons[k]["purp"])) for k in pos)
        fp = sum(len(set(o[k]["purp"]) - set(cons[k]["purp"])) for k in pos)
        fn = sum(len(set(cons[k]["purp"]) - set(o[k]["purp"])) for k in pos)
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        resp = sum(o[k]["resp"] == cons[k]["resp"] for k in pos) / m
        rev = sum(o[k]["rev"] == cons[k]["rev"] for k in pos) / m
        non = sum(o[k]["nonuse"] == cons[k]["nonuse"] for k in pos) / m
        allok = sum(o[k]["decl"] == cons[k]["decl"] and (cons[k]["decl"] == 0 or (
            o[k]["pol"] == cons[k]["pol"] and tool_set(o[k]) == cons[k]["tools"] and o[k]["purp"] == cons[k]["purp"]
            and o[k]["resp"] == cons[k]["resp"] and o[k]["rev"] == cons[k]["rev"] and o[k]["nonuse"] == cons[k]["nonuse"]))
            for k in ids) / n
        usd = tot.get(model, {}).get("usd", 0)
        print(f"{model:26s} {n:4d} {decl:6.3f} {pol:6.3f} {tools:6.3f} {pex:6.3f} {f1:6.3f} {resp:6.3f} {rev:6.3f} {non:6.3f} {allok:6.3f}  ${usd:.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "run":
        async def go():
            for m in sys.argv[2:]:
                await run_model(m)
        asyncio.run(go())
    elif cmd == "run2":  # run2 <model> <batch> <cache 0|1> <suffix> [extra-json]
        if len(sys.argv) > 6:
            MODEL_CFG[sys.argv[2]]["extra"] = json.loads(sys.argv[6].replace("'", '"'))
        asyncio.run(run_model(sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1", sys.argv[5]))
    elif cmd == "report":
        report(sys.argv[2:] or ["claude-opus-5-5"])
