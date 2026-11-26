"""Step 3. Download JATS full text from the PMC Open Access bucket on AWS S3 and keep the block superset.

For every PMCID: GET {S3}/{PMCID}.1/{PMCID}.1.xml  (fallback: Europe PMC fullTextXML), parse with extract_lib.extract,
write one JSON line {pmcid, status, src, meta, blocks, back_titles}. Full texts are not stored.

Usage:
  python -I 03_fetch_extract.py --source candidates --outdir blocks_v2      # all candidate PMCIDs
  python -I 03_fetch_extract.py --idfile ids.txt --outdir <name>            # any list of PMCIDs
Resumable: IDs are cut into fixed shards; finished shards are skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import INTERIM, RAW, S3_BASE, USER_AGENT, read_jsonl  # noqa: E402
from extract_lib import extract  # noqa: E402

SHARD = 5000
EPMC_FT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"


def parse_worker(args):
    pmcid, src, data = args
    res = extract(data)
    if res is None:
        return {"pmcid": pmcid, "status": "unparseable", "src": src, "bytes": len(data)}
    return {"pmcid": pmcid, "status": "ok", "src": src, "bytes": len(data), **res}


async def fetch_one(session, sem, epmc_sem, pmcid, use_fallback):
    url = f"{S3_BASE}/{pmcid}.1/{pmcid}.1.xml"
    last = None
    async with sem:
        for attempt in range(5):
            try:
                async with session.get(url) as r:
                    if r.status == 200:
                        return pmcid, "s3", await r.read()
                    if r.status in (403, 404):
                        last = f"s3_{r.status}"
                        break
                    last = f"s3_{r.status}"
            except Exception as e:  # noqa: BLE001
                last = f"s3_err_{type(e).__name__}"
            await asyncio.sleep(1.5 * (attempt + 1))
    if use_fallback and last in ("s3_403", "s3_404"):
        async with epmc_sem:
            for attempt in range(2):
                try:
                    async with session.get(EPMC_FT.format(pmcid=pmcid)) as r:
                        if r.status == 200:
                            return pmcid, "epmc", await r.read()
                        last = f"{last}|epmc_{r.status}"
                        if r.status == 404:
                            break
                except Exception as e:  # noqa: BLE001
                    last = f"{last}|epmc_err_{type(e).__name__}"
                await asyncio.sleep(2)
    return pmcid, last or "fail", None


async def run_shard(session, pool, ids, out_path, conc, use_fallback):
    loop = asyncio.get_running_loop()
    sem, epmc_sem = asyncio.Semaphore(conc), asyncio.Semaphore(2)
    stats = {"ok": 0, "fail": 0, "unparseable": 0, "bytes": 0, "epmc": 0, "blocks": 0, "with_blocks": 0}
    tmp = out_path.with_suffix(".tmp")
    parse_futs = []

    async def handle(pmcid):
        pid, src, data = await fetch_one(session, sem, epmc_sem, pmcid, use_fallback)
        if data is None:
            return {"pmcid": pid, "status": f"fetch_fail:{src}"}
        return await loop.run_in_executor(pool, parse_worker, (pid, src, data))

    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=5) as fh:
        for coro in asyncio.as_completed([handle(p) for p in ids]):
            rec = await coro
            st = rec["status"]
            if st == "ok":
                stats["ok"] += 1
                stats["bytes"] += rec.get("bytes", 0)
                stats["blocks"] += len(rec["blocks"])
                stats["with_blocks"] += 1 if rec["blocks"] else 0
                if rec.get("src") == "epmc":
                    stats["epmc"] += 1
            elif st == "unparseable":
                stats["unparseable"] += 1
            else:
                stats["fail"] += 1
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(out_path)
    return stats


def load_ids(args) -> list[str]:
    if args.idfile:
        ids = [l.strip() for l in Path(args.idfile).read_text().splitlines() if l.strip()]
    else:
        ids = set()
        for f in sorted((RAW / "epmc_candidates").glob("*.jsonl.gz")):
            for r in read_jsonl(f):
                if r.get("pmcid"):
                    ids.add(r["pmcid"])
        ids = list(ids)
    return sorted(set(ids), key=lambda s: int(s[3:]))


async def amain(args):
    ids = load_ids(args)
    outdir = INTERIM / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    shards = [ids[i:i + SHARD] for i in range(0, len(ids), SHARD)]
    print(f"{len(ids)} ids -> {len(shards)} shards in {outdir}", flush=True)
    timeout = aiohttp.ClientTimeout(total=150, connect=30)
    conn = aiohttp.TCPConnector(limit=args.conc + 10, ttl_dns_cache=600)
    t_all = time.time()
    done_n = 0
    with ProcessPoolExecutor(args.procs) as pool:
        async with aiohttp.ClientSession(timeout=timeout, connector=conn, headers={"User-Agent": USER_AGENT}) as session:
            for k, shard in enumerate(shards):
                out_path = outdir / f"shard_{k:04d}.jsonl.gz"
                if out_path.exists():
                    done_n += len(shard)
                    continue
                t0 = time.time()
                st = await run_shard(session, pool, shard, out_path, args.conc, not args.no_fallback)
                dt = time.time() - t0
                done_n += len(shard)
                print(f"shard {k:04d}/{len(shards)-1} n={len(shard)} ok={st['ok']} fail={st['fail']} unp={st['unparseable']} "
                      f"epmc={st['epmc']} with_blocks={st['with_blocks']} blocks={st['blocks']} {st['bytes']/1e6:.0f}MB "
                      f"{dt:.0f}s ({len(shard)/dt:.0f}/s, {st['bytes']/1e6/dt:.1f}MB/s) total_done={done_n} "
                      f"elapsed={(time.time()-t_all)/60:.1f}min", flush=True)
    print("ALL DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="candidates")
    ap.add_argument("--idfile")
    ap.add_argument("--outdir", default="blocks")
    ap.add_argument("--conc", type=int, default=140)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--no-fallback", action="store_true")
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
