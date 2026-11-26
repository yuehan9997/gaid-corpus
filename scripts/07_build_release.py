"""Step 7. Assemble the release tables (CSV + Parquet) in data/release/.

  gaid_statements          one row per AI-use declaration statement (annotator: decl = 1)
  gaid_statement_tools     long table: one row per (statement, tool mention), raw + canonical
  gaid_articles            one row per article carrying >= 1 declaration, article-level labels + metadata
  gaid_screened_articles   every candidate article that was downloaded and screened (with outcome)
  gaid_nondeclarations     candidate passages the annotator judged NOT to be declarations (hard negatives)
  gaid_denominators_journal_month / gaid_denominators_publisher_month
                           number of open-access articles, candidates, and declaring articles per stratum
  build_summary.json       headline numbers used in the manuscript
Works on partial annotation (reports coverage) so that the pipeline can be tested end-to-end.
"""
from __future__ import annotations

import collections
import csv
import gzip
import json
import re
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annot_schema import CODEBOOK_VERSION, PURPOSES  # noqa: E402
from common import DATE_END, DATE_START, INTERIM, RAW, RELEASE, read_jsonl  # noqa: E402
from passage_rules import RULES_VERSION  # noqa: E402
from publisher_map import publisher_from_doi  # noqa: E402
from tool_lexicon import LEXICON_VERSION, canonicalise  # noqa: E402

NON_PRIMARY_TYPES = {"correction", "retraction", "expression-of-concern", "addendum", "corrected-article",
                     "partial-retraction", "withdrawal", "erratum"}
OPEN_LICENSE_RE = re.compile(r"^(CC0|CC BY)")
POL_RANK = {"use": 0, "no_use": 1, "na": 2, "unclear": 3}


def alnum(s: str) -> str:
    return re.sub(r"[\W_]+", "", (s or "").lower())


def load_annotations(name="annot_primary"):
    ann = {}
    p = INTERIM / f"{name}.jsonl"
    if p.exists():
        for l in open(p, encoding="utf-8"):
            try:
                r = json.loads(l)
                ann[r["key"]] = r
            except Exception:  # noqa: BLE001
                pass
    return ann


def write_table(name: str, rows: list[dict], cols: list[str], gz: bool = True):
    RELEASE.mkdir(parents=True, exist_ok=True)
    path = RELEASE / f"{name}.csv{'.gz' if gz else ''}"
    op = (lambda p: gzip.open(p, "wt", encoding="utf-8", newline="")) if gz else (lambda p: open(p, "w", encoding="utf-8", newline=""))
    with op(path) as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    table = pa.Table.from_pylist([{c: r.get(c) for c in cols} for r in rows]) if rows else pa.table({c: [] for c in cols})
    pq.write_table(table, RELEASE / f"{name}.parquet", compression="zstd")
    print(f"  wrote {name}: {len(rows)} rows, {len(cols)} columns")


def main():
    t0 = time.time()
    ann = load_annotations()
    ann2 = load_annotations("annot_second")      # independent second annotator (different developer)
    print(f"annotations: {len(ann)} unique texts; second annotator: {len(ann2)}")

    # ---- articles (candidates that were screened) ----------------------------------------------
    arts = {}
    for a in read_jsonl(INTERIM / "articles.jsonl.gz"):
        arts[a["pmcid"]] = a
    print(f"screened candidate articles: {len(arts)}")

    # ---- Europe PMC index: denominators + authoritative first publication date -------------------
    idx_dir = RAW / "epmc_index_daily"
    idx_files = sorted(idx_dir.glob("*.jsonl.gz")) if idx_dir.exists() else []
    idx_months = len(list(idx_dir.glob("*.done"))) if idx_dir.exists() else 0
    denom_j = collections.Counter()   # (journal, issn, ym) -> n
    denom_p = collections.Counter()   # (brand, group, ym) -> n
    idx_of = {}
    # A few articles are listed twice, once as a PMC record (placeholder date, no DOI) and once as a PubMed record.
    # Keep one record per PMCID: in-window first, then the PubMed (MED) record, then a record with a DOI.
    best, n_rows = {}, 0

    def rank(r):
        d = (r.get("firstPublicationDate") or "")[:10]
        return (DATE_START <= d <= DATE_END, r.get("source") == "MED", bool(r.get("doi")))
    for f in idx_files:
        if not (idx_dir / (f.name.replace(".jsonl.gz", "") + ".done")).exists():
            continue
        for r in read_jsonl(f):
            pid = r.get("pmcid")
            if not pid:
                continue  # non-journal records (e.g. NCBI Bookshelf chapters) carry no PMCID
            n_rows += 1
            if pid not in best or rank(r) > rank(best[pid]):
                best[pid] = r
    n_index = len(best)
    n_index_duplicates = n_rows - n_index
    n_outside = 0
    for pid, r in best.items():
        d = (r.get("firstPublicationDate") or "")[:10]
        ym = d[:7]
        j = r.get("journalTitle") or ""
        issn = (r.get("journalIssn") or "").split(";")[0].strip()
        brand, group = publisher_from_doi(r.get("doi"))
        if pid in arts:
            idx_of[pid] = (r.get("firstPublicationDate"), j, issn, r.get("pubType"), r.get("doi"))
        if not (DATE_START <= d <= DATE_END):
            n_outside += 1      # returned by a day query but dated outside the window: not part of the frame
            continue
        denom_j[(j, issn, ym)] += 1
        denom_p[(brand, group, ym)] += 1
    print(f"index: {n_index} OA articles ({n_index_duplicates} duplicate records dropped) in {idx_months}/46 months; "
          f"matched to candidates: {len(idx_of)}")

    def art_date(a):
        ix = idx_of.get(a["pmcid"])
        if ix and ix[0]:
            return ix[0]
        d = [v for k, v in (a.get("pub_dates") or {}).items() if k != "pmc-release"]
        return min(d) if d else None

    # ---- passages -------------------------------------------------------------------------------
    template_n = collections.Counter()
    passages = []
    for p in read_jsonl(INTERIM / "passages.jsonl.gz"):
        passages.append(p)
        template_n[p["key"]] += 1
    n_unann = sum(1 for p in passages if p["key"] not in ann)
    print(f"passages: {len(passages)}; un-annotated: {n_unann} ({n_unann/len(passages):.1%})")

    statements, tools_long, nondecl = [], [], []
    by_art = collections.defaultdict(list)
    tool_ground = collections.Counter()
    for p in passages:
        a = ann.get(p["key"])
        if a is None:
            continue
        art = arts[p["pmcid"]]
        lic = art.get("license")
        text_ok = bool(lic and lic.startswith("CC"))
        sid = f"{p['pmcid']}_{p['pidx'] + 1}"
        a2 = ann2.get(p["key"])
        if not a["decl"]:
            nondecl.append({"passage_id": sid, "pmcid": p["pmcid"], "capture_route": p["route"], "location": p["zone"],
                            "section_heading": p["title"], "passage_text": p["text"] if text_ok else "",
                            "text_withheld": int(not text_ok), "source_license": lic, "text_key": p["key"],
                            "annotator2_decl": a2["decl"] if a2 else None})
            continue
        canon, cats, vers, raw = [], [], [], []
        hay = alnum(p["text"])
        for name, ver in a["tools"]:
            c = canonicalise(name, ver)
            verb = int(alnum(name) in hay) if name else 0
            tool_ground["verbatim" if verb else "not_verbatim"] += 1
            tools_long.append({"statement_id": sid, "pmcid": p["pmcid"], "tool_raw": name, "version_raw": ver, "tool": c["tool"],
                               "developer": c["developer"], "category": c["category"], "model_version": c["model_version"],
                               "lexicon_mapped": c["mapped"], "name_verbatim_in_text": verb})
            raw.append([name, ver])
            if c["tool"] not in canon:
                canon.append(c["tool"])
            if c["category"] not in cats:
                cats.append(c["category"])
            if c["model_version"] and c["model_version"] not in vers:
                vers.append(c["model_version"])
        row = {"statement_id": sid, "pmcid": p["pmcid"], "capture_route": p["route"], "location": p["zone"],
               "section_heading": p["title"], "statement_text": p["text"] if text_ok else "", "text_withheld": int(not text_ok),
               "text_truncated": int(bool(p.get("trunc"))), "source_license": lic, "text_key": p["key"],
               "template_count": template_n[p["key"]], "polarity": a["pol"], "n_tools": len(canon),
               "tools": "|".join(canon), "tool_categories": "|".join(cats), "gpt_model_versions": "|".join(vers),
               "tools_raw": json.dumps(raw, ensure_ascii=False), "purposes": "|".join(a["purp"]),
               "responsibility": a["resp"], "human_review": a["rev"], "nonuse_scope": a["nonuse"],
               "annotator_model": a.get("model"), "codebook_version": a.get("cb"),
               "annotator2_model": a2.get("model") if a2 else None,
               "annotator2_decl": a2["decl"] if a2 else None,
               "annotator2_polarity": (a2["pol"] if a2["decl"] else None) if a2 else None,
               "annotators_agree": (int(a2["decl"] == 1 and a2["pol"] == a["pol"]) if a2 else None)}
        for pu in PURPOSES:
            row["purpose_" + pu] = int(pu in a["purp"])
        statements.append(row)
        by_art[p["pmcid"]].append(row)

    # ---- article-level table --------------------------------------------------------------------
    art_rows, screened = [], []
    for pid, a in arts.items():
        sts = by_art.get(pid, [])
        date = art_date(a)
        ix = idx_of.get(pid)
        doi = a.get("doi") or (ix[4] if ix else None)
        brand, group = publisher_from_doi(doi)
        atype = (a.get("article_type") or "").lower()
        base = {"pmcid": pid, "pmid": a.get("pmid"), "doi": doi, "journal": a.get("journal") or (ix[1] if ix else None),
                "journal_nlm_ta": a.get("journal_nlm_ta"), "issn_print": a.get("issn_p"), "issn_electronic": a.get("issn_e"),
                # journal title and ISSN as in the Europe PMC index: the keys of gaid_denominators_journal_month
                "journal_epmc": ix[1] if ix else None, "issn_epmc": ix[2] if ix else None,
                "publisher_jats": a.get("publisher"), "publisher_brand": brand, "publisher_group": group,
                "article_type": atype or None, "is_correction_or_retraction": int(atype in NON_PRIMARY_TYPES),
                "language": (a.get("lang") or "").lower() or None, "first_publication_date": date,
                "pub_year": int(date[:4]) if date else None,
                "pub_month": date[:7] if date and len(date) >= 7 else None, "license": a.get("license"),
                "n_authors": a.get("n_authors"), "aff_countries": "|".join(a.get("aff_countries") or []),
                "n_body_paragraphs": a.get("n_body_p"), "n_body_ai_paragraphs": a.get("n_body_ai_p"),
                "in_epmc_index": int(ix is not None)}
        if sts:
            pols = sorted({s["polarity"] for s in sts}, key=POL_RANK.get)
            tools, cats, purps = [], [], []
            for s in sts:
                for t in filter(None, s["tools"].split("|")):
                    if t not in tools:
                        tools.append(t)
                for c in filter(None, s["tool_categories"].split("|")):
                    if c not in cats:
                        cats.append(c)
                for pu in filter(None, s["purposes"].split("|")):
                    if pu not in purps:
                        purps.append(pu)
            row = {**base, "title": a.get("title"), "n_statements": len(sts), "polarity": pols[0],
                   "polarity_conflict": int("use" in pols and "no_use" in pols),
                   "tools": "|".join(tools), "tool_categories": "|".join(cats),
                   "purposes": "|".join(sorted(purps, key=PURPOSES.index)),
                   "responsibility_any": int(any(s["responsibility"] for s in sts)),
                   "human_review_any": int(any(s["human_review"] for s in sts)),
                   "nonuse_scope_any": int(any(s["nonuse_scope"] for s in sts)),
                   "capture_routes": "|".join(sorted({s["capture_route"] for s in sts})),
                   "has_dedicated_section": int(any(s["capture_route"] == "ai_section" for s in sts)),
                   "publisher_alt_text_notice": int("frontiers_alt_text" in (a.get("notices") or []))}
            art_rows.append(row)
        screened.append({"pmcid": pid, "doi": doi, "journal": base["journal"], "journal_epmc": base["journal_epmc"],
                         "issn_epmc": base["issn_epmc"], "publisher_brand": brand,
                         "article_type": base["article_type"], "pub_month": base["pub_month"], "license": a.get("license"),
                         "n_candidate_passages": a["n_passages"], "has_declaration": int(bool(sts)),
                         "polarity": art_rows[-1]["polarity"] if sts else None})

    # ---- denominators ---------------------------------------------------------------------------
    num_j = collections.defaultdict(lambda: [0, 0, 0, 0])   # candidates, declaring, use, no_use
    num_p = collections.defaultdict(lambda: [0, 0, 0, 0])
    pol_of = {r["pmcid"]: r["polarity"] for r in art_rows}
    for pid, ix in idx_of.items():
        ym = (ix[0] or "")[:7]
        brand, group = publisher_from_doi(ix[4])
        for store, key in ((num_j, (ix[1] or "", ix[2] or "", ym)), (num_p, (brand, group, ym))):
            v = store[key]
            v[0] += 1
            pol = pol_of.get(pid)
            if pol:
                v[1] += 1
                v[2] += pol == "use"
                v[3] += pol == "no_use"
    dj = [{"journal": j, "issn": issn, "month": ym, "n_oa_articles": n, "n_candidates": num_j[(j, issn, ym)][0],
           "n_declaring": num_j[(j, issn, ym)][1], "n_use": num_j[(j, issn, ym)][2], "n_no_use": num_j[(j, issn, ym)][3]}
          for (j, issn, ym), n in sorted(denom_j.items())]
    dp = [{"publisher_brand": b, "publisher_group": g, "month": ym, "n_oa_articles": n, "n_candidates": num_p[(b, g, ym)][0],
           "n_declaring": num_p[(b, g, ym)][1], "n_use": num_p[(b, g, ym)][2], "n_no_use": num_p[(b, g, ym)][3]}
          for (b, g, ym), n in sorted(denom_p.items())]

    # ---- write ----------------------------------------------------------------------------------
    st_cols = ["statement_id", "pmcid", "capture_route", "location", "section_heading", "statement_text", "text_withheld",
               "text_truncated", "source_license", "text_key", "template_count", "polarity", "n_tools", "tools", "tool_categories",
               "gpt_model_versions", "tools_raw", "purposes"] + ["purpose_" + p for p in PURPOSES] + [
               "responsibility", "human_review", "nonuse_scope", "annotator_model", "codebook_version",
               "annotator2_model", "annotator2_decl", "annotator2_polarity", "annotators_agree"]
    write_table("gaid_statements", statements, st_cols)
    write_table("gaid_statement_tools", tools_long, ["statement_id", "pmcid", "tool_raw", "version_raw", "tool", "developer",
                                                    "category", "model_version", "lexicon_mapped", "name_verbatim_in_text"])
    art_cols = ["pmcid", "pmid", "doi", "title", "journal", "journal_nlm_ta", "issn_print", "issn_electronic", "journal_epmc",
                "issn_epmc", "publisher_jats",
                "publisher_brand", "publisher_group", "article_type", "is_correction_or_retraction", "language",
                "first_publication_date", "pub_year", "pub_month", "license", "n_authors", "aff_countries", "n_body_paragraphs",
                "n_body_ai_paragraphs", "in_epmc_index", "n_statements", "polarity", "polarity_conflict", "tools",
                "tool_categories", "purposes", "responsibility_any", "human_review_any", "nonuse_scope_any", "capture_routes",
                "has_dedicated_section", "publisher_alt_text_notice"]
    write_table("gaid_articles", art_rows, art_cols)
    write_table("gaid_screened_articles", screened, ["pmcid", "doi", "journal", "journal_epmc", "issn_epmc", "publisher_brand",
                                                      "article_type", "pub_month",
                                                      "license", "n_candidate_passages", "has_declaration", "polarity"])
    write_table("gaid_nondeclarations", nondecl, ["passage_id", "pmcid", "capture_route", "location", "section_heading",
                                                  "passage_text", "text_withheld", "source_license", "text_key", "annotator2_decl"])
    write_table("gaid_denominators_journal_month", dj, ["journal", "issn", "month", "n_oa_articles", "n_candidates", "n_declaring",
                                                        "n_use", "n_no_use"])
    write_table("gaid_denominators_publisher_month", dp, ["publisher_brand", "publisher_group", "month", "n_oa_articles",
                                                          "n_candidates", "n_declaring", "n_use", "n_no_use"])

    pol_st = collections.Counter(s["polarity"] for s in statements)
    pol_art = collections.Counter(r["polarity"] for r in art_rows)
    summary = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"), "window": [DATE_START, DATE_END],
        "versions": {"rules": RULES_VERSION, "codebook": CODEBOOK_VERSION, "tool_lexicon": LEXICON_VERSION},
        "index_months_done": idx_months, "n_oa_articles_index": n_index, "n_index_duplicate_records": n_index_duplicates,
        "n_index_records_outside_window": n_outside, "n_oa_articles_frame": n_index - n_outside,
        "n_candidates_screened": len(arts), "n_candidates_in_index": len(idx_of),
        "n_candidate_passages": len(passages), "n_unique_passage_texts": len(template_n),
        "annotation_coverage_passages": 1 - n_unann / len(passages),
        "n_statements": len(statements), "n_nondeclaration_passages": len(nondecl),
        "n_declaring_articles": len(art_rows), "statement_polarity": dict(pol_st), "article_polarity": dict(pol_art),
        "n_tool_mentions": len(tools_long), "tool_name_grounding": dict(tool_ground),
        "tool_lexicon_mapped_share": (sum(t["lexicon_mapped"] for t in tools_long) / len(tools_long)) if tools_long else None,
        "n_journals_declaring": len({r["journal_epmc"] for r in art_rows if r["journal_epmc"] is not None}),
        "n_journal_titles_jats_declaring": len({r["journal"] for r in art_rows}),
        "text_withheld_statements": sum(s["text_withheld"] for s in statements),
        "annotator_models": sorted({s["annotator_model"] for s in statements if s["annotator_model"]}),
        "annotator2_models": sorted({s["annotator2_model"] for s in statements if s["annotator2_model"]}),
        "statements_with_annotator2": sum(1 for s in statements if s["annotators_agree"] is not None),
        "statements_annotators_agree": sum(1 for s in statements if s["annotators_agree"] == 1),
        "nondecl_with_annotator2": sum(1 for r in nondecl if r["annotator2_decl"] is not None),
        "nondecl_annotator2_also_rejects": sum(1 for r in nondecl if r["annotator2_decl"] == 0),
    }
    (RELEASE / "build_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
