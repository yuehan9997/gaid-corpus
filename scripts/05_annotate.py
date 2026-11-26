"""Step 5. LLM annotation of unique passage texts with the codebook prompt (annot_schema.SYSTEM_PROMPT).

  python -I 05_annotate.py --model deepseek-v4.1-flash --out annot_primary
  python -I 05_annotate.py --model qwen3.8-flash --out annot_second

Input rows: {key, n, loc, heading, text}. Output (append-only, resumable): one JSON line per key with the normalised
labels plus provenance (model, codebook version). Items that a batch fails to return are retried in smaller batches.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annot_schema import CODEBOOK_VERSION, SYSTEM_PROMPT, build_user, parse_response  # noqa: E402
from common import INTERIM, ROOT, read_jsonl  # noqa: E402
from llm_lib import LLM  # noqa: E402

MAX_TEXT = 2600


async def amain(args):
    src = Path(args.input) if args.input else INTERIM / "unique_texts.jsonl.gz"
    rows = list(read_jsonl(src))
    if args.keys_file:
        keep = set(Path(args.keys_file).read_text().split())
        rows = [r for r in rows if r["key"] in keep]
    outp = INTERIM / f"{args.out}.jsonl"
    done = set()
    if outp.exists():
        for l in open(outp, encoding="utf-8"):
            try:
                done.add(json.loads(l)["key"])
            except Exception:  # noqa: BLE001
                pass
    todo = [r for r in rows if r["key"] not in done]
    random.Random(13).shuffle(todo)
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(rows)} rows, {len(done)} done, {len(todo)} to annotate with {args.model} (batch {args.batch}, conc {args.conc})", flush=True)
    if not todo:
        return
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    from models import extra_for
    extra = json.loads(args.extra) if args.extra else extra_for(args.model)
    print(f"request parameters: {extra}", flush=True)
    t0 = time.time()
    n_ok = n_fail = 0
    lock = asyncio.Lock()
    fh = open(outp, "a", encoding="utf-8")
    stop = False

    async with LLM(args.model, args.out, conc=args.conc, max_tokens=args.max_tokens, extra=extra, timeout=240) as llm:
        async def annotate(batch, depth=0):
            nonlocal n_ok, n_fail, stop
            if stop:
                return
            items = [{"id": i + 1, "loc": r["loc"], "heading": (r.get("heading") or "")[:200], "text": r["text"][:MAX_TEXT]}
                     for i, r in enumerate(batch)]
            content, usage = await llm.chat(SYSTEM_PROMPT, build_user(items), cache_system=args.cache)
            got = parse_response(content, [it["id"] for it in items])
            missing = []
            async with lock:
                for i, r in enumerate(batch):
                    res = got.get(i + 1)
                    if res is None:
                        missing.append(r)
                        continue
                    res.pop("id", None)
                    fh.write(json.dumps({"key": r["key"], **res, "model": args.model, "cb": CODEBOOK_VERSION}, ensure_ascii=False) + "\n")
                    n_ok += 1
                fh.flush()
                if llm.acc["usd"] > args.max_usd:
                    stop = True
                # the endpoint is refusing requests (disabled key, exhausted quota): stop instead of burning through the queue
                if llm.acc["errors"] >= 20 and llm.acc["errors"] > llm.acc["calls"]:
                    if not stop:
                        print(f"ABORT: {llm.acc['errors']} failed requests vs {llm.acc['calls']} successful; last error: {usage.get('error')}", flush=True)
                    stop = True
            if missing and not stop:
                if depth >= 3:
                    n_fail += len(missing)
                    return
                half = max(1, len(missing) // 2)
                await asyncio.gather(*[annotate(missing[i:i + half], depth + 1) for i in range(0, len(missing), half)])

        async def reporter():
            while True:
                await asyncio.sleep(60)
                el = time.time() - t0
                print(f"  {n_ok}/{len(todo)} ok, {n_fail} failed, {el/60:.1f} min, {n_ok/el:.1f} items/s, usd={llm.acc['usd']:.3f}, "
                      f"calls={llm.acc['calls']} errors={llm.acc['errors']} in={llm.acc['prompt']} out={llm.acc['completion']} "
                      f"cached={llm.acc['cached']}", flush=True)

        rep = asyncio.create_task(reporter())
        # bounded scheduling so that a stop request takes effect quickly
        sem = asyncio.Semaphore(args.conc * 2)

        async def guarded(b):
            async with sem:
                await annotate(b)
        await asyncio.gather(*[guarded(b) for b in batches])
        rep.cancel()
        acc = dict(llm.acc)
    fh.close()
    print(f"DONE {args.model}: ok={n_ok} failed={n_fail} in {(time.time()-t0)/60:.1f} min; calls={acc['calls']} in={acc['prompt']} "
          f"out={acc['completion']} cached={acc['cached']} usd={acc['usd']:.3f} errors={acc['errors']} stopped_by_budget={stop}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input")
    ap.add_argument("--keys-file")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-usd", type=float, default=20.0)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--extra")
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
