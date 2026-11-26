"""Step 4. Apply the passage rules to the block superset.

Outputs (data/interim):
  articles.jsonl.gz   one row per successfully parsed candidate article (metadata, n_passages, publisher notices)
  passages.jsonl.gz   one row per candidate passage (pmcid, pidx, route, zone, title, text, key)
  unique_texts.jsonl.gz  one row per distinct (key) passage text with its frequency and modal loc/heading  -> LLM input
  fetch_failures.txt  PMCIDs that could not be fetched from S3
"""
from __future__ import annotations

import collections
import gzip
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import INTERIM, read_jsonl  # noqa: E402
from passage_rules import RULES_VERSION, passages_from_article  # noqa: E402


def do_shard(path: str, replaced: frozenset = frozenset()):
    """replaced: PMCIDs re-parsed from their latest version (shard_latest, step 3c); earlier copies are skipped."""
    arts, passages, fails = [], [], []
    stats = collections.Counter()
    own = Path(path).name.startswith("shard_latest")
    for rec in read_jsonl(path):
        if not own and rec.get("pmcid") in replaced:
            continue
        st = rec.get("status")
        if st != "ok":
            stats["status:" + (st or "?").split(":")[0]] += 1
            if st and st.startswith("fetch_fail"):
                fails.append(rec["pmcid"])
            continue
        ps, notices = passages_from_article(rec)
        m = rec["meta"]
        m["pmcid"] = rec["pmcid"]
        arts.append({**m, "src": rec.get("src"), "n_blocks": len(rec["blocks"]), "n_passages": len(ps), "notices": notices,
                     "back_titles": rec.get("back_titles", [])[:40]})
        stats["articles"] += 1
        stats["articles_with_passage"] += 1 if ps else 0
        for i, p in enumerate(ps):
            passages.append({"pmcid": rec["pmcid"], "pidx": i, "route": p["route"], "zone": p["zone"], "title": p["title"],
                             "text": p["text"], "key": p["key"], "trunc": p.get("trunc", 0)})
            stats["passages:" + p["route"]] += 1
    return arts, passages, fails, stats


def main():
    src = INTERIM / (sys.argv[1] if len(sys.argv) > 1 else "blocks_v2")
    files = sorted(str(p) for p in src.glob("shard_*.jsonl.gz"))
    latest = src / "shard_latest.jsonl.gz"
    replaced = frozenset(r["pmcid"] for r in read_jsonl(latest) if r.get("status") == "ok") if latest.exists() else frozenset()
    print(f"{len(replaced)} articles taken from their latest version (shard_latest)", flush=True)
    t0 = time.time()
    tot = collections.Counter()
    uniq: dict = {}
    fails_all = []
    with gzip.open(INTERIM / "articles.jsonl.gz", "wt", encoding="utf-8") as fa, \
            gzip.open(INTERIM / "passages.jsonl.gz", "wt", encoding="utf-8") as fp, \
            ProcessPoolExecutor(8) as pool:
        for k, (arts, passages, fails, stats) in enumerate(pool.map(do_shard, files, [replaced] * len(files))):
            tot.update(stats)
            fails_all.extend(fails)
            for a in arts:
                fa.write(json.dumps(a, ensure_ascii=False) + "\n")
            for p in passages:
                fp.write(json.dumps(p, ensure_ascii=False) + "\n")
                u = uniq.get(p["key"])
                if u is None:
                    uniq[p["key"]] = {"key": p["key"], "text": p["text"], "n": 1,
                                      "locs": collections.Counter([p["route"]]), "titles": collections.Counter([p["title"]])}
                else:
                    u["n"] += 1
                    u["locs"][p["route"]] += 1
                    if len(u["titles"]) < 50:
                        u["titles"][p["title"]] += 1
            if k % 20 == 0:
                print(f"  shard {k}/{len(files)} passages={sum(v for kk, v in tot.items() if kk.startswith('passages:'))} "
                      f"unique={len(uniq)} t={time.time()-t0:.0f}s", flush=True)
    with gzip.open(INTERIM / "unique_texts.jsonl.gz", "wt", encoding="utf-8") as fu:
        for u in sorted(uniq.values(), key=lambda u: -u["n"]):
            fu.write(json.dumps({"key": u["key"], "n": u["n"], "loc": u["locs"].most_common(1)[0][0],
                                 "heading": u["titles"].most_common(1)[0][0], "text": u["text"]}, ensure_ascii=False) + "\n")
    # an article counts as failed only if no shard (including the retry shard) holds a parsed copy
    ok_ids = {json.loads(l)["pmcid"] for l in gzip.open(INTERIM / "articles.jsonl.gz", "rt", encoding="utf-8")}
    fails_all = sorted(set(fails_all) - ok_ids)
    (INTERIM / "fetch_failures.txt").write_text("\n".join(fails_all))
    print(f"rules v{RULES_VERSION}; {len(files)} shards in {time.time()-t0:.0f}s")
    for kk, v in sorted(tot.items()):
        print(f"  {kk}: {v}")
    print(f"  unique passage texts: {len(uniq)}")
    ns = sorted((u["n"] for u in uniq.values()), reverse=True)
    print(f"  top multiplicities: {ns[:12]}")
    chars = sum(len(u["text"]) for u in uniq.values())
    print(f"  total chars in unique texts: {chars} (~{chars/4/1e6:.1f}M tokens)")
    print(f"  fetch failures: {len(fails_all)}")


if __name__ == "__main__":
    main()
