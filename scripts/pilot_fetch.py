"""Pilot: fetch a few hundred candidate XMLs of different publisher styles and summarise where AI terms occur."""
from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EPMC_SCOPE, RAW, S3_BASE, USER_AGENT  # noqa: E402

PILOT = RAW / "pilot_xml"
PILOT.mkdir(parents=True, exist_ok=True)

QUERIES = {
    "frontiers": '"Generative AI statement"',
    "elsevier": '"Declaration of generative AI"',
    "ack_chatgpt": 'ACK_FUND:"ChatGPT"',
    "grammarly": '"Grammarly"',
    "mdpi": 'PUBLISHER:"MDPI" AND "ChatGPT" AND ("used" OR "manuscript")',
    "springer": '(JOURNAL:"Scientific Reports" OR JOURNAL:"BMC Medical Education") AND "ChatGPT" AND "manuscript"',
    "wiley": 'PUBLISHER:"Wiley" AND "generative AI" AND "manuscript"',
    "plos": 'JOURNAL:"PLoS One" AND "ChatGPT" AND "writing"',
    "no_ai": '"no artificial intelligence" OR "no AI tools" OR "without the use of AI"',
    "llm_generic": '"large language model" AND "preparation of this manuscript"',
    "cureus": 'JOURNAL:"Cureus" AND "artificial intelligence" AND "declare"',
    "sage_tandf": '(PUBLISHER:"SAGE Publications" OR PUBLISHER:"Taylor & Francis") AND "AI" AND "manuscript" AND "ChatGPT"',
    "bmj_jama": '(PUBLISHER:"BMJ Publishing Group" OR JOURNAL:"JAMA Netw Open") AND "AI" AND "ChatGPT"',
    "nonenglish": '("inteligencia artificial" OR "inteligência artificial") AND "ChatGPT"',
    "ai_only": '"AI" AND NOT "artificial intelligence" AND NOT "ChatGPT" AND "statement"',
}


def get(url, data=None, timeout=90):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def search(q, n=40, year="2025"):
    full = f"({q}) AND {EPMC_SCOPE} AND FIRST_PDATE:[{year}-03-01 TO 2026-09-30]"
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&resultType=idlist&pageSize=%d&query=%s"
           % (n, urllib.parse.quote(full)))
    d = json.loads(get(url))
    return d.get("hitCount"), [r["pmcid"] for r in d["resultList"]["result"] if r.get("pmcid")]


def fetch(pmcid):
    out = PILOT / f"{pmcid}.xml"
    if out.exists():
        return "cached"
    try:
        out.write_bytes(get(f"{S3_BASE}/{pmcid}.1/{pmcid}.1.xml"))
        return "s3"
    except Exception as e:  # noqa: BLE001
        try:
            out.write_bytes(get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"))
            return "epmc"
        except Exception as e2:  # noqa: BLE001
            return f"fail {e!r} / {e2!r}"


def main():
    manifest = {}
    for name, q in QUERIES.items():
        try:
            hit, ids = search(q)
        except Exception as e:  # noqa: BLE001
            print(name, "search ERR", e)
            continue
        with ThreadPoolExecutor(16) as ex:
            res = list(ex.map(fetch, ids))
        manifest[name] = ids
        src = {k: sum(1 for r in res if r.startswith(k)) for k in ("s3", "epmc", "cached", "fail")}
        print(f"{name:14s} hits={hit} fetched={len(ids)} {src}")
    (PILOT.parent / "pilot_manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
