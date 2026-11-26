"""Step 6. Recall audit: how many AI-use declarations does the retrieval + rule pipeline miss?

Three strata, each a simple random sample; together they cover every article of the frame that is not in the corpus:
  A  candidate articles (full text matched the AI vocabulary) for which the rules extracted NO passage
  B  non-candidate open-access articles (full text did not match the vocabulary in Europe PMC)
  C  candidate articles with extracted passages that were all labelled non-declarations
For every sampled article the complete declaration-zone text (all back matter, acknowledgements, notes, footnotes,
front notes) plus every main-text paragraph containing an AI term is re-read by an LLM that applies the codebook
definition and must quote the declaration verbatim. Positives are grounded (quote must occur in the text).

  python -I 06_audit_recall.py sample A3 3000
  python -I 06_audit_recall.py fetch A3
  python -I 06_audit_recall.py screen A3 deepseek-v4.1-flash
  python -I 06_audit_recall.py screen A3 qwen-plus-latest
  python -I 06_audit_recall.py combine A3 deepseek-v4.1-flash qwen-plus-latest
  python -I 06_audit_recall.py licences A3
Samples A and A2 were used to develop the extraction rules; A3, B and C are the final samples. Full texts are taken
from the latest version of each article in the PMC Open Access collection (`refresh` re-reads sampled articles whose
text was first read from version 1).
"""
from __future__ import annotations

import asyncio
import collections
import gzip
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import aiohttp
from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATE_END, DATE_START, INTERIM, RAW, S3_BASE, USER_AGENT, VALID, read_jsonl  # noqa: E402
from extract_lib import BROAD_RE, PARSER_KW, SKIP_ANCESTORS, direct_title, el_text, lname  # noqa: E402
from llm_lib import LLM  # noqa: E402

ADIR = VALID / "audit"
ADIR.mkdir(parents=True, exist_ok=True)

AUDIT_PROMPT = """You are auditing a text-mining pipeline that collects AI-use declarations from scientific articles.

An AI-use declaration is a statement by the article's own authors disclosing whether or how generative AI or AI-assisted tools (chatbots / large language models such as ChatGPT, writing or translation aids such as Grammarly or DeepL, image generators, coding assistants, or "AI" in general) were used AS AN AID in preparing the article or carrying out the work behind it (writing, editing, translating, literature search, coding, data analysis, figures). Statements of NON-use ("No AI tools were used") count, and so do minimal answers under an AI-declaration heading ("None", "Not applicable").
NOT a declaration: AI as the object, method, material or intervention of the research itself; AI in the name of a funder, grant, institution, company, journal or person; competing-interest or employment disclosures; descriptions of other people's work; general discussion of AI; author initials such as "AI"; publisher or editor notices (e.g. a publisher stating that alt text was generated with AI).

You receive excerpts of ONE article: section headings, the full declaration-type text (acknowledgements, back matter, notes, footnotes) and main-text paragraphs that mention AI terms. Decide whether the excerpts contain an AI-use declaration by the authors.

Return minified JSON: {"found":0|1,"quote":"<the declaration copied verbatim, at most 60 words; empty if none>","heading":"<heading under which it appears, or empty>","pol":"use"|"no_use"|"na"|"unclear"|null}"""

DECL_CONTAINERS = {"ack", "notes", "fn-group", "author-notes", "back", "fn"}


def audit_text(xml_bytes: bytes) -> dict | None:
    try:
        root = etree.fromstring(xml_bytes, etree.XMLParser(**PARSER_KW))
    except Exception:  # noqa: BLE001
        return None
    art = root if lname(root) == "article" else root.find(".//article")
    if art is None:
        return None
    decl_parts, body_parts, titles = [], [], []
    for el in art.iter("p"):
        skip, part, in_decl, title = False, None, False, ""
        cur = el.getparent()
        nested = False
        while cur is not None:
            n = lname(cur)
            if n == "p":
                nested = True
            if n in SKIP_ANCESTORS:
                skip = True
                break
            if n in DECL_CONTAINERS:
                in_decl = True
            if not title and n in ("sec", "ack", "notes", "fn-group", "app", "fn"):
                title = direct_title(cur)
            if n in ("front", "body", "back") and part is None:
                part = n
            cur = cur.getparent()
        if skip or nested or part is None:
            continue
        txt = el_text(el)
        if not txt:
            continue
        if part in ("back", "front") or in_decl:
            decl_parts.append((title, txt))
        elif BROAD_RE.search(txt):
            body_parts.append((title, txt))
    for el in art.iter("sec", "ack", "notes", "app"):
        t = direct_title(el)
        if t and len(titles) < 80:
            titles.append(t[:90])
    out, used, last = [], 0, None
    for title, txt in decl_parts:
        if used > 9000:
            break
        if title != last:
            out.append(f"## {title or '(untitled)'}")
            last = title
        out.append(txt[:1500])
        used += min(len(txt), 1500)
    body_out, used_b = [], 0
    for title, txt in body_parts[:25]:
        if used_b > 6000:
            break
        m = BROAD_RE.search(txt)
        a = max(0, m.start() - 350)
        seg = txt[a:a + 800]
        body_out.append(f"[{title[:60]}] {seg}")
        used_b += len(seg)
    return {"headings": titles, "decl_text": "\n".join(out), "body_text": "\n".join(body_out),
            "n_decl_p": len(decl_parts), "n_body_ai_p": len(body_parts)}


def stratum_b_pool() -> set:
    """Articles of the sampling frame not returned by the full-text query: distinct PMCIDs with at least one index record
    dated inside the window."""
    cand = set()
    for f in (RAW / "epmc_candidates").glob("*.jsonl.gz"):
        for r in read_jsonl(f):
            if r.get("pmcid"):
                cand.add(r["pmcid"])
    pool = set()
    for f in sorted((RAW / "epmc_index_daily").glob("*.jsonl.gz")):
        for r in read_jsonl(f):
            pid = r.get("pmcid")
            if pid and pid not in cand and DATE_START <= (r.get("firstPublicationDate") or "")[:10] <= DATE_END:
                pool.add(pid)
    return pool


def cmd_repop_b():
    """Recompute the size of stratum B (distinct PMCIDs, see stratum_b_pool), check that every sampled article belongs
    to it, and update the population and the extrapolated counts; the sample and the model judgements are unchanged."""
    pool = stratum_b_pool()
    sample = (ADIR / "sample_B.txt").read_text().split()
    outside = [p for p in sample if p not in pool]
    if outside:
        raise SystemExit(f"{len(outside)} sampled articles are not in the corrected stratum: {outside[:10]}")
    meta_f, ver_f = ADIR / "sample_B.meta.json", ADIR / "verified_B.json"
    meta = json.loads(meta_f.read_text())
    old = meta["population"]
    meta["population"], meta["population_v1"] = len(pool), old
    meta_f.write_text(json.dumps(meta))
    if ver_f.exists():
        v = json.loads(ver_f.read_text())
        v.update({"population": len(pool), "est": v["rate"] * len(pool), "est_lo": v["lo"] * len(pool), "est_hi": v["hi"] * len(pool)})
        ver_f.write_text(json.dumps(v, indent=1), encoding="utf-8")
    print(f"stratum B: population {old} -> {len(pool)}; all {len(sample)} sampled articles are in the corrected stratum")


def stratum_a_pool() -> set:
    """Candidate articles without any extracted passage (all of them, including the development samples)."""
    return {a["pmcid"] for a in read_jsonl(INTERIM / "articles.jsonl.gz") if a["n_passages"] == 0}


def stratum_c_pool() -> set:
    """Candidate articles with extracted passages none of which was labelled a declaration."""
    import pandas as pd
    from common import RELEASE
    s = pd.read_parquet(RELEASE / "gaid_screened_articles.parquet", columns=["pmcid", "n_candidate_passages", "has_declaration"])
    return set(s.pmcid[(s.n_candidate_passages > 0) & (s.has_declaration == 0)])


def current_pool(stratum: str) -> set | None:
    if stratum.startswith("A"):
        return stratum_a_pool()
    if stratum.startswith("C"):
        return stratum_c_pool()
    return None


_VERS: dict | None = None


async def latest_version(session, pid: str) -> str:
    """Latest version of an article in the S3 collection (list of candidates from step 3c, else listed now)."""
    global _VERS
    if _VERS is None:
        import importlib
        _VERS = importlib.import_module("03c_latest_versions").load_versions()
    v = _VERS.get(pid)
    if v is None:
        import importlib
        v = await importlib.import_module("03c_latest_versions").list_versions(session, pid)
        _VERS[pid] = v or []
    return v[-1] if v else f"{pid}.1"


def cmd_sample(stratum: str, n: int, seed: int = 2026):
    rng = random.Random(seed + (len(stratum) - 1) * 101 + (int(stratum[1:]) if stratum[1:].isdigit() else 0))
    if stratum.startswith("C"):
        pool = list(stratum_c_pool())
    elif stratum.startswith("A"):
        # candidate articles without any extracted passage under the CURRENT rules; articles of earlier audit samples
        # (which were used to improve the rules) are excluded so that this sample is an independent test
        used = set()
        for f in ADIR.glob("sample_A*.txt"):
            used.update(f.read_text().split())
        pool = [a["pmcid"] for a in read_jsonl(INTERIM / "articles.jsonl.gz") if a["n_passages"] == 0 and a["pmcid"] not in used]
    else:
        pool = sorted(stratum_b_pool())
    pool.sort()
    ids = rng.sample(pool, n)
    (ADIR / f"sample_{stratum}.txt").write_text("\n".join(ids))
    (ADIR / f"sample_{stratum}.meta.json").write_text(json.dumps({"stratum": stratum, "population": len(pool), "n": n, "seed": seed}))
    print(f"stratum {stratum}: population {len(pool)}, sampled {n}")


async def cmd_fetch(stratum: str):
    ids = (ADIR / f"sample_{stratum}.txt").read_text().split()
    outp = ADIR / f"texts_{stratum}.jsonl.gz"
    sem = asyncio.Semaphore(60)
    res = []

    async def one(session, pid):
        async with sem:
            v = await latest_version(session, pid)
            for attempt in range(4):
                try:
                    async with session.get(f"{S3_BASE}/{v}/{v}.xml") as r:
                        if r.status == 200:
                            data = await r.read()
                            t = audit_text(data)
                            return {"pmcid": pid, "status": "ok" if t else "unparseable", "version": v, **(t or {})}
                        if r.status in (403, 404):
                            return {"pmcid": pid, "status": f"s3_{r.status}"}
                except Exception as e:  # noqa: BLE001
                    await asyncio.sleep(2 * (attempt + 1))
            return {"pmcid": pid, "status": "fail"}

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=150), headers={"User-Agent": USER_AGENT}) as session:
        res = await asyncio.gather(*[one(session, p) for p in ids])
    with gzip.open(outp, "wt", encoding="utf-8") as fh:
        for r in res:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = sum(r["status"] == "ok" for r in res)
    print(f"fetched {ok}/{len(ids)} ok; statuses: { {s: sum(r['status']==s for r in res) for s in set(r['status'] for r in res)} }")


async def cmd_screen(stratum: str, model: str, conc: int = 24):
    rows = [r for r in read_jsonl(ADIR / f"texts_{stratum}.jsonl.gz") if r["status"] == "ok"]
    outp = ADIR / f"screen_{stratum}_{model}.jsonl"
    done = set()
    if outp.exists():
        # keep successfully screened rows only; rows that failed (API or parse error) are re-screened
        good = [l for l in open(outp, encoding="utf-8") if l.strip() and not json.loads(l).get("parse_error")]
        outp.write_text("".join(good), encoding="utf-8")
        done = {json.loads(l)["pmcid"] for l in good}
    todo = [r for r in rows if r["pmcid"] not in done]
    print(f"{len(rows)} texts, {len(todo)} to screen with {model}")
    t0 = time.time()
    fh = open(outp, "a", encoding="utf-8")
    from models import extra_for
    async with LLM(model, f"audit_{stratum}", conc=conc, max_tokens=4000, extra=extra_for(model)) as llm:
        async def one(r):
            if not r["decl_text"] and not r["body_text"]:
                rec = {"pmcid": r["pmcid"], "found": 0, "quote": "", "heading": "", "pol": None, "empty": 1}
            else:
                user = ("SECTION HEADINGS: " + " | ".join(r["headings"][:60]) + "\n\nDECLARATION-TYPE TEXT:\n" + (r["decl_text"] or "(none)")
                        + "\n\nMAIN-TEXT PARAGRAPHS MENTIONING AI TERMS:\n" + (r["body_text"] or "(none)"))
                content, usage = await llm.chat(AUDIT_PROMPT, user)
                try:
                    d = json.loads(re.search(r"\{.*\}", content or "", re.S).group(0))
                except Exception:  # noqa: BLE001
                    d = {"found": None, "quote": "", "heading": "", "pol": None, "parse_error": 1}
                rec = {"pmcid": r["pmcid"], **{k: d.get(k) for k in ("found", "quote", "heading", "pol")}}
                if d.get("parse_error"):
                    rec["parse_error"] = 1
                q = re.sub(r"\W+", " ", (rec.get("quote") or "").lower()).strip()
                hay = re.sub(r"\W+", " ", (r["decl_text"] + " " + r["body_text"]).lower())
                rec["grounded"] = int(bool(q) and q[:80] in hay)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
        await asyncio.gather(*[one(r) for r in todo])
        print(f"screened in {time.time()-t0:.0f}s usd={llm.acc['usd']:.3f} calls={llm.acc['calls']} errors={llm.acc['errors']}")
    fh.close()


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def cmd_report(stratum: str, model: str):
    meta = json.loads((ADIR / f"sample_{stratum}.meta.json").read_text())
    rows = [json.loads(l) for l in open(ADIR / f"screen_{stratum}_{model}.jsonl", encoding="utf-8")]
    n = len(rows)
    pos = [r for r in rows if r.get("found") == 1]
    grounded = [r for r in pos if r.get("grounded")]
    lo, hi = wilson(len(pos), n)
    print(f"stratum {stratum}: population {meta['population']}, screened {n}, flagged {len(pos)} ({len(pos)/n:.4%}; 95% CI {lo:.4%}-{hi:.4%}), "
          f"grounded {len(grounded)}; parse errors {sum(1 for r in rows if r.get('parse_error'))}")
    print(f"  -> estimated missed articles in stratum: {len(pos)/n*meta['population']:.0f} (CI {lo*meta['population']:.0f}-{hi*meta['population']:.0f})")
    with open(ADIR / f"positives_{stratum}_{model}.txt", "w", encoding="utf-8") as fh:
        for r in pos:
            fh.write(f"{r['pmcid']} | pol={r.get('pol')} | grounded={r.get('grounded')} | heading='{r.get('heading')}' | {r.get('quote')}\n")


def cmd_verify(stratum: str, model: str, batch: int = 8):
    """The screening prompt is deliberately liberal. Each flagged quotation is therefore re-judged with the full
    annotation codebook by the three reference models; a finding is confirmed when the majority calls it a declaration."""
    import importlib
    from models import REF_MODELS
    val = importlib.import_module("08_validation")
    from passage_rules import DECL_TITLE_RE
    rows = [json.loads(l) for l in open(ADIR / f"screen_{stratum}_{model}.jsonl", encoding="utf-8")]
    pos = [r for r in rows if r.get("found") == 1 and r.get("grounded") and (r.get("quote") or "").strip()]
    items = [{"key": r["pmcid"], "loc": "decl_zone" if DECL_TITLE_RE.search(r.get("heading") or "") else "body",
              "heading": r.get("heading") or "", "text": r["quote"]} for r in pos]
    labels = {}
    for m in REF_MODELS:
        outp = ADIR / f"verify_{stratum}_{m}.jsonl"
        asyncio.run(val.run_model_on(items, m, outp, f"audit_verify_{stratum}", batch=batch, conc=8))
        labels[m] = {json.loads(l)["key"]: json.loads(l) for l in open(outp, encoding="utf-8")}
    confirmed = []
    for it in items:
        votes = [labels[m][it["key"]]["decl"] for m in REF_MODELS if it["key"] in labels[m]]
        if votes and sum(votes) * 2 > len(votes):
            confirmed.append(it["key"])
    meta = json.loads((ADIR / f"sample_{stratum}.meta.json").read_text())
    n = sum(1 for r in rows if not r.get("parse_error"))
    lo, hi = wilson(len(confirmed), n)
    res = {"stratum": stratum, "population": meta["population"], "screened": n, "flagged": len(pos), "confirmed": len(confirmed),
           "rate": len(confirmed) / n, "lo": lo, "hi": hi, "est": len(confirmed) / n * meta["population"],
           "est_lo": lo * meta["population"], "est_hi": hi * meta["population"], "screen_model": model, "verify_models": REF_MODELS,
           "confirmed_pmcids": confirmed}
    (ADIR / f"verified_{stratum}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "confirmed_pmcids"}, indent=1))


async def cmd_licences(stratum: str):
    """Licence of every article in the sample whose screening output contains a quotation (stratum B articles are not
    in the candidate table, so the release step cannot look their licence up there). -> licences_<stratum>.json"""
    from extract_lib import license_from
    ids = set()
    for f in ADIR.glob(f"screen_{stratum}_*.jsonl"):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if (r.get("quote") or "").strip():
                ids.add(r["pmcid"])
    out = {}

    async def one(session, pid):
        v = await latest_version(session, pid)
        for attempt in range(4):
            try:
                async with session.get(f"{S3_BASE}/{v}/{v}.xml") as r:
                    if r.status == 200:
                        root = etree.fromstring(await r.read(), etree.XMLParser(**PARSER_KW))
                        art = root if lname(root) == "article" else root.find(".//article")
                        out[pid] = license_from(art)[1] if art is not None else None
                        return
                    out[pid] = None
                    return
            except Exception:  # noqa: BLE001
                await asyncio.sleep(2 * (attempt + 1))
        out[pid] = None

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=150), headers={"User-Agent": USER_AGENT}) as session:
        await asyncio.gather(*[one(session, p) for p in sorted(ids)])
    (ADIR / f"licences_{stratum}.json").write_text(json.dumps(dict(sorted(out.items())), indent=1), encoding="utf-8")
    print(f"{stratum}: licences of {len(out)} quoted articles:", collections.Counter(v or "none" for v in out.values()))


def screener_items(stratum: str, sm: str):
    from passage_rules import DECL_TITLE_RE
    rows = [json.loads(l) for l in open(ADIR / f"screen_{stratum}_{sm}.jsonl", encoding="utf-8") if l.strip()]
    pos = [r for r in rows if r.get("found") == 1 and r.get("grounded") and (r.get("quote") or "").strip()]
    return rows, pos, [{"key": r["pmcid"], "loc": "decl_zone" if DECL_TITLE_RE.search(r.get("heading") or "") else "body",
                        "heading": r.get("heading") or "", "text": r["quote"]} for r in pos]


def verify_path(stratum: str, sm: str, m: str):
    from models import AUDIT_MODEL
    # the primary screener's quotations were verified first and keep their original file names
    return ADIR / (f"verify_{stratum}_{m}.jsonl" if sm == AUDIT_MODEL else f"verify_{stratum}_{sm}_{m}.jsonl")


async def cmd_verify_parallel(strata: list[str], sm: str, batch: int = 4):
    """Judge the flagged quotations of screener `sm` with all reference models at once (all strata in parallel)."""
    import importlib
    from models import REF_MODELS
    val = importlib.import_module("08_validation")
    jobs = []
    for s in strata:
        _, _, items = screener_items(s, sm)
        for m in REF_MODELS:
            jobs.append(val.run_model_on(items, m, verify_path(s, sm, m), f"audit_verify_{s}", batch=batch, conc=8))
    await asyncio.gather(*jobs)


def cmd_combine(stratum: str, models: list[str], batch: int = 8):
    """Recall estimate from two independent screeners. The flagged quotations of every screener are judged with the full
    codebook by the three reference models (majority vote); an article counts as a missed declaration when the quotation
    of at least one screener is confirmed. The overlap between the screeners gives a capture-recapture (Chapman) estimate
    of the declarations that both would have missed."""
    import importlib
    from models import AUDIT_MODEL, REF_MODELS
    val = importlib.import_module("08_validation")
    from passage_rules import DECL_TITLE_RE
    meta = json.loads((ADIR / f"sample_{stratum}.meta.json").read_text())
    screened, flagged, confirmed = {}, {}, {}
    for sm in models:
        rows = [json.loads(l) for l in open(ADIR / f"screen_{stratum}_{sm}.jsonl", encoding="utf-8") if l.strip()]
        screened[sm] = {r["pmcid"] for r in rows if not r.get("parse_error")}
        pos = [r for r in rows if r.get("found") == 1 and r.get("grounded") and (r.get("quote") or "").strip()]
        flagged[sm] = {r["pmcid"] for r in pos}
        items = [{"key": r["pmcid"], "loc": "decl_zone" if DECL_TITLE_RE.search(r.get("heading") or "") else "body",
                  "heading": r.get("heading") or "", "text": r["quote"]} for r in pos]
        labels = {}
        for m in REF_MODELS:
            # the primary screener's quotations were verified first and keep their original file names
            outp = ADIR / (f"verify_{stratum}_{m}.jsonl" if sm == AUDIT_MODEL else f"verify_{stratum}_{sm}_{m}.jsonl")
            if items:
                asyncio.run(val.run_model_on(items, m, outp, f"audit_verify_{stratum}", batch=batch, conc=8))
            labels[m] = {json.loads(l)["key"]: json.loads(l) for l in open(outp, encoding="utf-8")} if outp.exists() else {}
        confirmed[sm] = set()
        for it in items:
            votes = [labels[m][it["key"]]["decl"] for m in REF_MODELS if it["key"] in labels[m]]
            if votes and sum(votes) * 2 > len(votes):
                confirmed[sm].add(it["key"])
    both_screened = set.intersection(*screened.values())
    pool = current_pool(stratum)
    population = meta["population"]
    left = 0
    if pool is not None:
        left = len(set((ADIR / f"sample_{stratum}.txt").read_text().split()) - pool)
        # strata A and C are defined on the final data: sampled articles that left the stratum (for example because
        # their latest version yielded a passage) are dropped, and estimates are scaled to the whole current stratum
        both_screened &= pool
        population = len(pool)
    n = len(both_screened)
    union = set.union(*confirmed.values()) & both_screened
    lo, hi = wilson(len(union), n)
    c1, c2 = (confirmed[m] & both_screened for m in models[:2])
    m12 = len(c1 & c2)
    chapman = (len(c1) + 1) * (len(c2) + 1) / (m12 + 1) - 1
    res = {"stratum": stratum, "population": population, "population_at_sampling": meta["population"], "sampled": meta["n"],
           "left_stratum": left, "screened": n, "screened_by": {m: len(screened[m]) for m in models},
           "flagged": len(set.union(*flagged.values()) & both_screened), "flagged_by": {m: len(flagged[m] & both_screened) for m in models},
           "confirmed": len(union), "confirmed_by": {m: len(confirmed[m] & both_screened) for m in models}, "confirmed_by_both": m12,
           "chapman_total": chapman, "chapman_missed_by_both": max(0.0, chapman - len(union)),
           "rate": len(union) / n, "lo": lo, "hi": hi, "est": len(union) / n * population,
           "est_lo": lo * population, "est_hi": hi * population, "screen_models": models,
           "verify_models": REF_MODELS, "confirmed_pmcids": sorted(union),
           "refreshed_to_latest_version": len(json.loads((ADIR / f"refresh_{stratum}.json").read_text()))
           if (ADIR / f"refresh_{stratum}.json").exists() else 0}
    (ADIR / f"verified_{stratum}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "confirmed_pmcids"}, indent=1))


async def cmd_refresh(stratum: str):
    """Audit texts were first read from version 1. Re-read every sampled article whose latest version is a later one,
    and drop its screening and verification outputs so that both screeners and the verifiers judge the new text."""
    ids = (ADIR / f"sample_{stratum}.txt").read_text().split()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=150), headers={"User-Agent": USER_AGENT}) as session:
        sem = asyncio.Semaphore(60)

        async def lv(pid):
            async with sem:
                return pid, await latest_version(session, pid)
        latest = dict(await asyncio.gather(*[lv(p) for p in ids]))
        texts = {r["pmcid"]: r for r in read_jsonl(ADIR / f"texts_{stratum}.jsonl.gz")}
        todo = [p for p in ids if latest[p] != f"{p}.1" and texts.get(p, {}).get("version") != latest[p]]
        print(f"{stratum}: {len(todo)} of {len(ids)} sampled articles have a later version", flush=True)

        async def one(pid):
            v = latest[pid]
            async with sem:
                for attempt in range(4):
                    try:
                        async with session.get(f"{S3_BASE}/{v}/{v}.xml") as r:
                            if r.status == 200:
                                t = audit_text(await r.read())
                                return {"pmcid": pid, "status": "ok" if t else "unparseable", "version": v, **(t or {})}
                    except Exception:  # noqa: BLE001
                        pass
                    await asyncio.sleep(2 * (attempt + 1))
            return None
        new = [r for r in await asyncio.gather(*[one(p) for p in todo]) if r]
    for r in new:
        texts[r["pmcid"]] = r
    with gzip.open(ADIR / f"texts_{stratum}.jsonl.gz", "wt", encoding="utf-8") as fh:
        for p in ids:
            if p in texts:
                fh.write(json.dumps(texts[p], ensure_ascii=False) + "\n")
    changed = {r["pmcid"] for r in new}
    for f in list(ADIR.glob(f"screen_{stratum}_*.jsonl")) + list(ADIR.glob(f"verify_{stratum}_*.jsonl")):
        keep = [l for l in open(f, encoding="utf-8") if l.strip()
                and (json.loads(l).get("pmcid") or json.loads(l).get("key")) not in changed]
        f.write_text("".join(keep), encoding="utf-8")
    rf = ADIR / f"refresh_{stratum}.json"
    old = json.loads(rf.read_text()) if rf.exists() else {}
    old.update({r["pmcid"]: r["version"] for r in new})
    rf.write_text(json.dumps(old, indent=1), encoding="utf-8")
    print(f"{stratum}: re-read {len(new)} articles from their latest version; their screening results were removed")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "refresh":
        asyncio.run(cmd_refresh(sys.argv[2]))
    elif cmd == "combine":
        cmd_combine(sys.argv[2], sys.argv[3:])
    elif cmd == "licences":
        asyncio.run(cmd_licences(sys.argv[2]))
    elif cmd == "verify_parallel":     # verify_parallel <screener> <stratum> [<stratum> ...] [batch]
        args = sys.argv[3:]
        b = int(args.pop()) if args and args[-1].isdigit() else 4
        asyncio.run(cmd_verify_parallel(args, sys.argv[2], b))
    elif cmd == "verify":
        cmd_verify(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 8)
    elif cmd == "sample":
        cmd_sample(sys.argv[2], int(sys.argv[3]))
    elif cmd == "fetch":
        asyncio.run(cmd_fetch(sys.argv[2]))
    elif cmd == "screen":
        asyncio.run(cmd_screen(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 24))
    elif cmd == "report":
        cmd_report(sys.argv[2], sys.argv[3])
    elif cmd == "repop_b":
        cmd_repop_b()
