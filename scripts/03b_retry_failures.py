"""Step 3b. Retry PMCIDs that were missing at {PMCID}.1 on S3: look for another version, then Europe PMC fullTextXML.

Writes data/interim/blocks_v2/shard_retry.jsonl.gz in the same format as the main shards.
"""
from __future__ import annotations

import asyncio
import collections
import gzip
import json
import re
import sys
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import INTERIM, S3_BASE, USER_AGENT  # noqa: E402
from extract_lib import extract  # noqa: E402

EPMC_FT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"


async def one(session, sem, pid):
    async with sem:
        # 1) any other version on S3?
        try:
            async with session.get(f"{S3_BASE}/?list-type=2&max-keys=20&delimiter=/&prefix={pid}.") as r:
                txt = await r.text()
            vers = sorted(set(re.findall(rf"<Prefix>({pid}\.\d+)/</Prefix>", txt)), key=lambda s: int(s.split(".")[1]))
            for v in vers:
                async with session.get(f"{S3_BASE}/{v}/{v}.xml") as r:
                    if r.status == 200:
                        return pid, f"s3:{v}", await r.read()
        except Exception:  # noqa: BLE001
            pass
        # 2) Europe PMC
        for attempt in range(3):
            try:
                async with session.get(EPMC_FT.format(pmcid=pid)) as r:
                    if r.status == 200:
                        return pid, "epmc", await r.read()
                    if r.status == 404:
                        return pid, "not_found", None
            except Exception:  # noqa: BLE001
                await asyncio.sleep(3)
        return pid, "fail", None


async def main():
    ids = (INTERIM / "fetch_failures.txt").read_text().split()
    sem = asyncio.Semaphore(4)
    stats = collections.Counter()
    import time
    out = INTERIM / "blocks_v2" / f"shard_retry_{time.strftime('%Y%m%d%H%M%S')}.jsonl.gz"  # never overwrite earlier retries
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120), headers={"User-Agent": USER_AGENT}) as session:
        res = await asyncio.gather(*[one(session, sem, p) for p in ids])
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        for pid, src, data in res:
            if data is None:
                stats[src] += 1
                fh.write(json.dumps({"pmcid": pid, "status": f"fetch_fail:{src}"}) + "\n")
                continue
            ex = extract(data)
            if ex is None:
                stats["unparseable"] += 1
                fh.write(json.dumps({"pmcid": pid, "status": "unparseable", "src": src}) + "\n")
                continue
            stats["ok:" + src.split(":")[0]] += 1
            fh.write(json.dumps({"pmcid": pid, "status": "ok", "src": src, "bytes": len(data), **ex}, ensure_ascii=False) + "\n")
    print(dict(stats))


if __name__ == "__main__":
    asyncio.run(main())
