"""Second pilot round (codebook v1.1, batch 16): cost, speed and agreement of candidate annotators.

Agreement is computed against the majority label of the three reference runs of round 1 on the same 336 passages.
Usage: python -I pilot_round2.py <suffix> <model> [extra-json]   (extra uses single quotes for JSON strings)
"""
from __future__ import annotations

import asyncio
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annot_schema import PURPOSES  # noqa: E402
from llm_lib import ledger_total  # noqa: E402
from pilot_llm import MODEL_CFG, PDIR, load_out, load_sample, run_model  # noqa: E402
from tool_lexicon import canonicalise  # noqa: E402

REFS = ["claude-opus-5-5", "gpt-5.5", "gemini-3.1-pro-preview"]


def tset(r):
    return frozenset(canonicalise(t[0], t[1])["tool"] for t in r["tools"])


def consensus():
    items = load_sample()
    outs = [load_out(m) for m in REFS]
    cons = {}
    for it in items:
        rs = [o[it["id"]] for o in outs if it["id"] in o]
        c = {f: collections.Counter(r[f] for r in rs).most_common(1)[0][0] for f in ("decl", "pol", "resp", "rev", "nonuse")}
        c["purp"] = [p for p in PURPOSES if sum(p in r["purp"] for r in rs) * 2 > len(rs)]
        c["tools"] = collections.Counter(tset(r) for r in rs).most_common(1)[0][0]
        cons[it["id"]] = c
    return cons


def score(name):
    cons = consensus()
    o = load_out(name)
    ids = [i for i in o if i in cons]
    if not ids:
        return None
    pos = [i for i in ids if cons[i]["decl"] == 1 and o[i]["decl"] == 1]
    m = max(1, len(pos))
    tp = sum(len(set(o[i]["purp"]) & set(cons[i]["purp"])) for i in pos)
    fp = sum(len(set(o[i]["purp"]) - set(cons[i]["purp"])) for i in pos)
    fn = sum(len(set(cons[i]["purp"]) - set(o[i]["purp"])) for i in pos)
    return {"n": len(ids), "decl": sum(o[i]["decl"] == cons[i]["decl"] for i in ids) / len(ids),
            "pol": sum(o[i]["pol"] == cons[i]["pol"] for i in pos) / m,
            "tools": sum(tset(o[i]) == cons[i]["tools"] for i in pos) / m,
            "purp_exact": sum(o[i]["purp"] == cons[i]["purp"] for i in pos) / m, "purp_f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "resp": sum(o[i]["resp"] == cons[i]["resp"] for i in pos) / m, "rev": sum(o[i]["rev"] == cons[i]["rev"] for i in pos) / m,
            "nonuse": sum(o[i]["nonuse"] == cons[i]["nonuse"] for i in pos) / m}


def main():
    if sys.argv[1] == "table":
        print(f"{'run':46s} {'n':>4s} {'decl':>6s} {'pol':>6s} {'tools':>6s} {'purpEx':>6s} {'purpF1':>6s} {'resp':>6s} {'rev':>6s} {'nonuse':>6s}")
        for f in sorted(PDIR.glob("out_*.jsonl")):
            name = f.stem[4:]
            s = score(name)
            if s:
                print(f"{name:46s} {s['n']:4d} {s['decl']:6.3f} {s['pol']:6.3f} {s['tools']:6.3f} {s['purp_exact']:6.3f} {s['purp_f1']:6.3f} "
                      f"{s['resp']:6.3f} {s['rev']:6.3f} {s['nonuse']:6.3f}")
        return
    suffix, model = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3:
        MODEL_CFG[model]["extra"] = json.loads(sys.argv[3].replace("'", '"'))
    before = ledger_total().get(model, {}).get("usd", 0.0)
    t0 = time.time()
    asyncio.run(run_model(model, 16, False, suffix))
    dt = time.time() - t0
    usd = ledger_total().get(model, {}).get("usd", 0.0) - before
    s = score(model + suffix)
    print(f"RESULT {model}{suffix}: {dt:.0f}s for 336 items (conc {MODEL_CFG[model]['conc']}), ${usd:.4f} -> projected ${usd / 336 * 130738:.1f} "
          f"for 130,738 texts | {json.dumps({k: round(v, 3) for k, v in s.items()}) if s else 'no output'}")


if __name__ == "__main__":
    main()
