"""Step 3c. Use the latest version of every article in the PMC Open Access collection.

Step 3 requested {PMCID}.1 (the first version) and looked for other versions only when it was absent. Europe PMC indexes
the current version, and about 1% of candidates have later versions in the S3 collection whose declaration can differ
from the first one. This step lists the versions of every candidate and re-parses those whose latest version is not
the one that was parsed.

  python -I 03c_latest_versions.py list      -> data/interim/pmc_versions.jsonl   (resumable)
  python -I 03c_latest_versions.py fetch     -> data/interim/blocks_v2/shard_latest.jsonl.gz
Step 4 then uses the record in shard_latest instead of the earlier one.
"""
from __future__ import annotations

import asyncio
import collections
import gzip
import json
import re
import sys
import time
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import INTERIM, RAW, S3_BASE, USER_AGENT, read_jsonl  # noqa: E402

VERS = INTERIM / "pmc_versions.jsonl"
LATEST_SHARD = INTERIM / "blocks_v2" / "shard_latest.jsonl.gz"


def candidate_ids() -> list[str]:
    ids = set()
    for f in sorted((RAW / "epmc_candidates").glob("*.jsonl.gz")):
        for r in read_jsonl(f):
            if r.get("pmcid"):
                ids.add(r["pmcid"])
    return sorted(ids, key=lambda s: int(s[3:]))


def vnum(v: str) -> int:
    return int(v.rsplit(".", 1)[1])


async def list_versions(session, pid: str) -> list[str] | None:
    for attempt in range(5):
        try:
            async with session.get(f"{S3_BASE}/?list-type=2&max-keys=50&delimiter=/&prefix={pid}.") as r:
                if r.status == 200:
                    txt = await r.text()
                    return sorted(set(re.findall(rf"<Prefix>({pid}\.\d+)/</Prefix>", txt)), key=vnum)
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(1.5 * (attempt + 1))
    return None


def load_versions() -> dict:
    out = {}
    if VERS.exists():
        for l in open(VERS, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                out[r["pmcid"]] = r["versions"]
    return out


async def cmd_list(ids: list[str] | None = None, conc: int = 160):
    ids = ids or candidate_ids()
    done = load_versions()
    todo = [p for p in ids if p not in done]
    print(f"{len(ids)} ids, {len(done)} listed, {len(todo)} to list", flush=True)
    sem = asyncio.Semaphore(conc)
    t0, n = time.time(), 0
    conn = aiohttp.TCPConnector(limit=conc + 20, ttl_dns_cache=600)
    with open(VERS, "a", encoding="utf-8") as fh:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60), connector=conn,
                                         headers={"User-Agent": USER_AGENT}) as session:
            async def one(pid):
                nonlocal n
                async with sem:
                    v = await list_versions(session, pid)
                if v is not None:
                    fh.write(json.dumps({"pmcid": pid, "versions": v}) + "\n")
                n += 1
                if n % 20000 == 0:
                    fh.flush()
                    print(f"  {n}/{len(todo)} {n/(time.time()-t0):.0f}/s", flush=True)
            for i in range(0, len(todo), 20000):
                await asyncio.gather(*[one(p) for p in todo[i:i + 20000]])
    vers = load_versions()
    print("versions per article:", sorted(collections.Counter(len(v) for v in vers.values()).items()))


def used_version(a: dict) -> int | None:
    """Version number of the parsed copy: 's3' = .1, 's3:PMCx.N' = N; Europe PMC copies are current (None)."""
    s = a.get("src") or ""
    if s == "s3":
        return 1
    if s.startswith("s3:"):
        return vnum(s[3:])
    return None


async def cmd_fetch(conc: int = 60):
    from concurrent.futures import ProcessPoolExecutor
    import importlib
    fx = importlib.import_module("03_fetch_extract")
    vers = load_versions()
    todo = []
    for a in read_jsonl(INTERIM / "articles.jsonl.gz"):
        u = used_version(a)
        v = vers.get(a["pmcid"]) or []
        if u is not None and v and vnum(v[-1]) != u:
            todo.append((a["pmcid"], v[-1]))
    print(f"{len(todo)} articles whose latest version differs from the parsed one", flush=True)
    sem = asyncio.Semaphore(conc)
    loop = asyncio.get_running_loop()
    stats = collections.Counter()
    tmp = LATEST_SHARD.with_suffix(".tmp")
    with ProcessPoolExecutor(6) as pool, gzip.open(tmp, "wt", encoding="utf-8") as fh:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=150), headers={"User-Agent": USER_AGENT}) as session:
            async def one(pid, v):
                data = None
                async with sem:
                    for attempt in range(5):
                        try:
                            async with session.get(f"{S3_BASE}/{v}/{v}.xml") as r:
                                if r.status == 200:
                                    data = await r.read()
                                    break
                        except Exception:  # noqa: BLE001
                            pass
                        await asyncio.sleep(1.5 * (attempt + 1))
                if data is None:
                    stats["fail"] += 1
                    return  # keep the earlier copy
                rec = await loop.run_in_executor(pool, fx.parse_worker, (pid, f"s3:{v}", data))
                stats[rec["status"]] += 1
                if rec["status"] == "ok":
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            await asyncio.gather(*[one(p, v) for p, v in todo])
    tmp.replace(LATEST_SHARD)
    print(dict(stats))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        asyncio.run(cmd_list())
    elif cmd == "fetch":
        asyncio.run(cmd_fetch())
