"""Step 1. Enumerate Europe PMC.

mode=index      : every open-access PMC article first published in the study window (the denominator), 'lite' metadata.
mode=candidates : the subset whose full text matches the high-recall AI vocabulary (idlist only).

Queries are sliced by publication day (index) or month (candidates); rows are de-duplicated on (source, id) because
deep cursor pagination occasionally repeats records. Output: data/raw/epmc_{mode}/{YYYY-MM}.jsonl.gz, resumable.
"""
from __future__ import annotations

import calendar
import gzip
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EPMC_SCOPE, EPMC_SEARCH_POST, RAW, USER_AGENT, candidate_query, months  # noqa: E402

KEEP = ["id", "source", "pmid", "pmcid", "doi", "journalTitle", "journalIssn", "pubYear", "pubType",
        "firstPublicationDate", "firstIndexDate", "isOpenAccess", "inPMC"]


def post(params: dict, retries: int = 40) -> dict:
    body = urllib.parse.urlencode(params).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(EPMC_SEARCH_POST, data=body, headers={
                "User-Agent": USER_AGENT, "Content-Type": "application/x-www-form-urlencoded"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            wait = min(90, 2 ** min(a, 7))
            if a >= 2:
                print(f"  retry {a+1} after error: {e!r}; sleeping {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("Europe PMC request failed repeatedly")


def run_slice(mode: str, q: str) -> tuple[dict, int]:
    """Page through one query; return ({(source,id): row}, hitCount)."""
    rows: dict = {}
    hit = 0
    for attempt in range(3):
        cursor = "*"
        while True:
            d = post({"query": q, "format": "json", "pageSize": "1000", "cursorMark": cursor,
                      "resultType": "idlist" if mode == "candidates" else "lite"})
            hit = d.get("hitCount", hit) or 0
            res = d.get("resultList", {}).get("result", [])
            for r in res:
                keys = KEEP if mode == "index" else ("id", "source", "pmid", "pmcid")
                rows[(r.get("source"), r.get("id"))] = {k: r.get(k) for k in keys if r.get(k) is not None}
            nxt = d.get("nextCursorMark")
            if not res or not nxt or nxt == cursor:
                break
            cursor = nxt
        if len(rows) >= hit:
            break
    return rows, hit


def run_month(mode: str, label: str, d0: str, d1: str) -> int:
    outdir = RAW / {"index": "epmc_index_daily", "candidates": "epmc_candidates"}[mode]
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{label}.jsonl.gz"
    done = outdir / f"{label}.done"
    if done.exists():
        return int(done.read_text().split()[0])
    y, m = int(label[:4]), int(label[5:7])
    if mode == "index":
        slices = [(f"{label}-{d:02d}", f"{label}-{d:02d}") for d in range(1, calendar.monthrange(y, m)[1] + 1)]
    else:
        slices = [(d0, d1)]
    allrows: dict = {}
    hits = 0
    short = 0
    for s0, s1 in slices:
        q = f"{EPMC_SCOPE} AND FIRST_PDATE:[{s0} TO {s1}]"
        if mode == "candidates":
            q = f"({candidate_query()}) AND {q}"
        rows, hit = run_slice(mode, q)
        hits += hit
        short += max(0, hit - len(rows))
        allrows.update(rows)
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for r in allrows.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out)
    done.write_text(f"{len(allrows)} {hits} {short}\n")
    print(f"[{mode}] {label}: {len(allrows)} unique records (sum hitCount {hits}, shortfall {short})", flush=True)
    return len(allrows)


def main():
    mode = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    jobs = list(months())
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        totals = list(ex.map(lambda j: run_month(mode, *j), jobs))
    print(f"[{mode}] DONE total={sum(totals)} months={len(jobs)} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
