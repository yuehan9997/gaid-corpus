"""Step 11. Export documentation and lookup tables that accompany the release (data/release/docs)."""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annot_schema import CODEBOOK_VERSION, PURPOSES, SYSTEM_PROMPT  # noqa: E402
from common import DATE_END, DATE_START, EPMC_SCOPE, GENERIC_TERMS, RELEASE, ROOT, TOOL_TERMS, VALID  # noqa: E402
from passage_rules import RULES_VERSION  # noqa: E402
from publisher_map import DOI_PREFIX, PUBLISHER_MAP_VERSION  # noqa: E402
from tool_lexicon import LEXICON_VERSION, RULES  # noqa: E402

DOCS = RELEASE / "docs"

DICT = [
    # table, variable, type, description
    ("gaid_statements", "statement_id", "string", "PMCID and running number of the passage within the article, e.g. PMC12345678_1"),
    ("gaid_statements", "pmcid", "string", "PubMed Central identifier of the source article"),
    ("gaid_statements", "capture_route", "category", "How the passage was captured: ai_section (section whose heading announces an AI declaration), decl_zone (acknowledgements, declarations, notes, footnotes), body (main text)"),
    ("gaid_statements", "location", "category", "JATS zone of the passage: front, ack, back, body, appendix"),
    ("gaid_statements", "section_heading", "string", "Heading of the nearest enclosing section (may be empty)"),
    ("gaid_statements", "statement_text", "string", "Verbatim passage (publisher notices removed; capped at 2,500 / 1,800 / 1,200 characters for ai_section / decl_zone / body). Empty when text_withheld = 1"),
    ("gaid_statements", "text_withheld", "0/1", "1 if the source article carries no Creative Commons licence and the text is therefore not redistributed"),
    ("gaid_statements", "text_truncated", "0/1", "1 if the passage was shortened by a length cap"),
    ("gaid_statements", "source_license", "string", "Licence of the source article; applies to statement_text"),
    ("gaid_statements", "text_key", "string", "First 20 hex digits of SHA-1 of the normalised text (lower case, every run of non-alphanumeric characters replaced by one space); identical keys = identical wording"),
    ("gaid_statements", "template_count", "integer", "Number of candidate passages in the whole corpus that share text_key"),
    ("gaid_statements", "polarity", "category", "use (AI or AI-assisted tool used), no_use, na (bare 'not applicable'), unclear"),
    ("gaid_statements", "n_tools", "integer", "Number of distinct canonical tools named as used"),
    ("gaid_statements", "tools", "string", "Canonical tool names, '|' separated"),
    ("gaid_statements", "tool_categories", "string", "Categories of the tools, '|' separated"),
    ("gaid_statements", "gpt_model_versions", "string", "GPT model generations stated for ChatGPT/GPT mentions, '|' separated (e.g. GPT-4o)"),
    ("gaid_statements", "tools_raw", "JSON", "List of [name, version] pairs as extracted by the annotator"),
    ("gaid_statements", "purposes", "string", "Purpose codes, '|' separated: " + ", ".join(PURPOSES)),
    ("gaid_statements", "purpose_<CODE>", "0/1", "One indicator column per purpose code"),
    ("gaid_statements", "responsibility", "0/1", "Authors explicitly assume responsibility or accountability for the content"),
    ("gaid_statements", "human_review", "0/1", "Authors state that they reviewed, edited, checked, verified, validated, approved or supervised the AI-assisted output"),
    ("gaid_statements", "nonuse_scope", "0/1", "In a declaration of use, authors also explicitly negate AI involvement in some task or content"),
    ("gaid_statements", "annotator_model", "string", "Model that produced the labels"),
    ("gaid_statements", "codebook_version", "string", "Version of the annotation codebook"),
    ("gaid_statements", "annotator2_model", "string", "Independent second annotator (model from another developer)"),
    ("gaid_statements", "annotator2_decl", "0/1", "Second annotator's decision on whether the passage is an AI-use declaration"),
    ("gaid_statements", "annotator2_polarity", "category", "Second annotator's polarity (empty if it judged the passage not to be a declaration)"),
    ("gaid_statements", "annotators_agree", "0/1", "1 if the second annotator also judged the passage a declaration and assigned the same polarity"),
    ("gaid_statement_tools", "statement_id / pmcid", "string", "Keys to gaid_statements / gaid_articles"),
    ("gaid_statement_tools", "tool_raw / version_raw", "string", "Tool name and version string as extracted"),
    ("gaid_statement_tools", "tool / developer / category", "string", "Canonical tool, its developer and category from tool_lexicon.csv"),
    ("gaid_statement_tools", "model_version", "string", "Normalised GPT model generation, if stated"),
    ("gaid_statement_tools", "lexicon_mapped", "0/1", "1 if a lexicon rule matched; otherwise tool = raw name"),
    ("gaid_statement_tools", "name_verbatim_in_text", "0/1", "1 if the raw tool name occurs verbatim (alphanumeric match) in the statement text"),
    ("gaid_articles", "pmcid / pmid / doi", "string", "Article identifiers"),
    ("gaid_articles", "title", "string", "Article title (max 500 characters)"),
    ("gaid_articles", "journal / journal_nlm_ta / issn_print / issn_electronic", "string", "Journal title, NLM abbreviation and ISSNs from JATS"),
    ("gaid_articles", "journal_epmc / issn_epmc", "string", "Journal title and first ISSN in the Europe PMC index; with pub_month they are the keys of gaid_denominators_journal_month (empty if in_epmc_index = 0)"),
    ("gaid_articles", "publisher_jats", "string", "Publisher name as given in JATS (often missing)"),
    ("gaid_articles", "publisher_brand / publisher_group", "string", "Publisher brand and parent group assigned from the DOI prefix (publisher_map.csv)"),
    ("gaid_articles", "article_type", "string", "JATS article-type attribute"),
    ("gaid_articles", "is_correction_or_retraction", "0/1", "1 for corrections, retractions, expressions of concern and similar notices"),
    ("gaid_articles", "language", "string", "xml:lang of the article"),
    ("gaid_articles", "first_publication_date / pub_year / pub_month", "date / integer / YYYY-MM", "First publication date according to Europe PMC (JATS dates if not in the index)"),
    ("gaid_articles", "license", "string", "Creative Commons licence of the article"),
    ("gaid_articles", "n_authors", "integer", "Number of contributors of type author"),
    ("gaid_articles", "aff_countries", "string", "Countries tagged in affiliations, '|' separated (present only when the publisher tags countries)"),
    ("gaid_articles", "n_body_paragraphs / n_body_ai_paragraphs", "integer", "Main-text paragraphs, and those mentioning an AI term; their ratio indicates articles about AI"),
    ("gaid_articles", "in_epmc_index", "0/1", "1 if the article is in the denominator index"),
    ("gaid_articles", "n_statements", "integer", "Number of declaration statements in the article"),
    ("gaid_articles", "polarity", "category", "Article-level polarity: use if any statement declares use, else no_use, na, unclear"),
    ("gaid_articles", "polarity_conflict", "0/1", "1 if the article has both a use and a no_use statement"),
    ("gaid_articles", "tools / tool_categories / purposes", "string", "Unions over the article's statements, '|' separated"),
    ("gaid_articles", "responsibility_any / human_review_any / nonuse_scope_any", "0/1", "1 if any statement of the article has the indicator"),
    ("gaid_articles", "capture_routes", "string", "Capture routes of the article's statements"),
    ("gaid_articles", "has_dedicated_section", "0/1", "1 if a statement comes from a section with an AI-declaration heading"),
    ("gaid_articles", "publisher_alt_text_notice", "0/1", "1 if a publisher notice about AI-generated alternative text was found (and removed)"),
    ("gaid_screened_articles", "pmcid / doi / journal / journal_epmc / issn_epmc / publisher_brand / article_type / pub_month / license", "string", "Identifiers and basic metadata of every candidate article that was downloaded"),
    ("gaid_screened_articles", "n_candidate_passages", "integer", "Number of passages extracted by the rules"),
    ("gaid_screened_articles", "has_declaration", "0/1", "1 if at least one passage was annotated as a declaration"),
    ("gaid_screened_articles", "polarity", "category", "Article-level polarity if has_declaration = 1"),
    ("gaid_nondeclarations", "passage_id / pmcid / capture_route / location / section_heading / passage_text / text_withheld / source_license / text_key", "as in gaid_statements", "Candidate passages that the annotator judged not to be AI-use declarations"),
    ("gaid_denominators_journal_month", "journal / issn / month", "string / string / YYYY-MM", "Stratum: Europe PMC journal title, first ISSN, month of first publication"),
    ("gaid_denominators_journal_month", "n_oa_articles", "integer", "Open-access articles in the sampling frame"),
    ("gaid_denominators_journal_month", "n_candidates", "integer", "Of these, articles retrieved by the candidate query and screened"),
    ("gaid_denominators_journal_month", "n_declaring / n_use / n_no_use", "integer", "Articles with any declaration / declaring use / declaring non-use"),
    ("gaid_denominators_publisher_month", "publisher_brand / publisher_group / month", "string", "Stratum by publisher brand and group (DOI prefix) and month; counts as above"),
]

README = f"""# GAID - Generative-AI Declarations corpus

Authors' declarations about the use (or non-use) of generative AI and AI-assisted tools, extracted from the open-access
full text in Europe PMC / PubMed Central for articles first published {DATE_START} to {DATE_END}.

## Files
* `gaid_statements` - one row per declaration statement (verbatim text + labels)
* `gaid_statement_tools` - one row per tool mention (raw and canonical)
* `gaid_articles` - one row per article with at least one declaration
* `gaid_screened_articles` - every candidate article that was screened, with outcome
* `gaid_nondeclarations` - candidate passages judged not to be declarations (hard negatives)
* `gaid_denominators_journal_month`, `gaid_denominators_publisher_month` - counts of all open-access articles per stratum
* `build_summary.json` - main counts of the build (sampling frame, candidates, passages, statements, articles, journals)
* `docs/` - codebook, data dictionary, tool lexicon, publisher map, candidate vocabulary
* `annotations/` - complete outputs of the primary and the second annotator model, one line per annotated passage text (join on `text_key`; a few texts produced only by earlier rule versions do not occur in the tables)
* `validation/` - validation sample (`validation_sample.jsonl`), all agreement statistics (`validation_results.json`), human labels (`human_labels.jsonl`), reference-model labels (`ref_*.jsonl`), test-retest labels, the context check (`context_*.jsonl`: texts that occur under more than one capture route, re-labelled with each other route and its heading; key = `text_key|route`), and the recall audits in `validation/audit/`: final samples `A3` (stratum A of the Data Descriptor: candidates without a passage), `B` (articles not retrieved) and `C` (candidates whose passages were all labelled non-declarations), each screened independently by the primary annotator model and by Qwen-Plus (`screen_<sample>_<model>.jsonl`), with the reference-model judgements of the flagged quotations (`verify_*.jsonl`) and the result (`verified_<sample>.json`); `refresh_<sample>.json` lists sampled articles whose text was read from a later PMC version; `licences_<sample>.json` gives the licence of articles quoted in strata A3 and B, used to withhold text from sources without a Creative Commons licence; `A` and `A2` are the two development samples used to revise the extraction rules

Every table is provided as gzip-compressed UTF-8 CSV and as Parquet (identical content). In the repository record the three
folders are packed as `docs.zip`, `annotations.zip` and `validation.zip`.

## Versions
passage rules {RULES_VERSION}; codebook {CODEBOOK_VERSION}; tool lexicon {LEXICON_VERSION}; publisher map {PUBLISHER_MAP_VERSION}.
Sampling frame query: `{EPMC_SCOPE} AND FIRST_PDATE:[date TO date]`.

## Licence
The record has two layers. Annotations, metadata and denominators: CC BY 4.0. `statement_text` / `passage_text` are short
verbatim excerpts of open-access articles and remain under the licence of the source article given in `source_license`
(most are CC BY; some sources carry non-commercial, no-derivatives or share-alike terms, which then apply to that text);
credit the source article via its DOI. Text is withheld where the source has no Creative Commons licence (also in the
validation sample and the audit quotations).

## Reading the data
```python
import pandas as pd
st = pd.read_parquet("gaid_statements.parquet")
ar = pd.read_parquet("gaid_articles.parquet")
den = pd.read_parquet("gaid_denominators_publisher_month.parquet")
monthly = den.groupby("month")[["n_oa_articles", "n_declaring", "n_use"]].sum()
monthly["share_declaring"] = monthly.n_declaring / monthly.n_oa_articles

# by journal: link articles to the journal denominators through the Europe PMC title, ISSN and month
dj = pd.read_parquet("gaid_denominators_journal_month.parquet")
chatgpt = (ar[ar.tools.str.contains("ChatGPT", na=False)]
           .groupby(["journal_epmc", "issn_epmc", "pub_month"]).size().rename("n_chatgpt").reset_index())
dj = dj.merge(chatgpt, how="left", left_on=["journal", "issn", "month"],
              right_on=["journal_epmc", "issn_epmc", "pub_month"]).fillna({{"n_chatgpt": 0}})
```

## Caveats
A declaration records what authors state, not what they did. Recent months are incomplete (deposit lag). Statements in the
main text are captured less completely than those in dedicated sections. Labels were produced by a language model; see the
Data Descriptor for measured agreement with human annotators.
"""


def export_human_labels(vdir: Path):
    """Labels of the human annotators (A, B, ... in the order used in validation_results.json), one line per item."""
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location("validation08", Path(__file__).with_name("08_validation.py"))
    v08 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v08)
    files = v08.human_files()
    if not files:
        return
    out = vdir / "human_labels.jsonl"
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for i, f in enumerate(files):
            labels = v08.load_human_xlsx(f)[0] if f.suffix == ".xlsx" else v08.load_human(f)
            for key, r in labels.items():
                fh.write(json.dumps({"key": key, "annotator": chr(65 + i), "decl": r["decl"], "pol": r["pol"],
                                     "tools": [t[0] for t in r["tools"]], "purp": r["purp"], "resp": r["resp"],
                                     "rev": r["rev"], "nonuse": r["nonuse"]}, ensure_ascii=False) + "\n")
                n += 1
    print(f"human labels: {n} rows from {len(files)} annotators -> {out}")


def withhold_noncc_text(vdir: Path):
    """Apply the licence rule of the tables to the validation copies: no text from articles without a CC licence.
    A text is withheld when it occurs only in such articles; an audit quotation when its candidate article has no CC licence."""
    import json

    import pandas as pd
    st = pd.read_parquet(RELEASE / "gaid_statements.parquet", columns=["text_key", "text_withheld"])
    nd = pd.read_parquet(RELEASE / "gaid_nondeclarations.parquet", columns=["text_key", "text_withheld"])
    sc = pd.read_parquet(RELEASE / "gaid_screened_articles.parquet", columns=["pmcid", "license"])
    keys_nc = (set(st.text_key[st.text_withheld == 1]) | set(nd.text_key[nd.text_withheld == 1])) - (
        set(st.text_key[st.text_withheld == 0]) | set(nd.text_key[nd.text_withheld == 0]))
    arts_nc = set(sc.pmcid[~sc.license.fillna("").str.startswith("CC")])
    # audit articles outside the candidate table (stratum B): licences read from their JATS XML by 06_audit_recall.py licences
    for f in (VALID / "audit").glob("licences_*.json"):
        arts_nc |= {k for k, v in json.loads(f.read_text(encoding="utf-8")).items() if not (v or "").startswith("CC")}
    quoted_unknown = set()
    for f in (vdir / "audit").glob("screen_*.jsonl"):
        for l in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(l) if l.strip() else {}
            if (r.get("quote") or "").strip() and r["pmcid"] not in set(sc.pmcid):
                quoted_unknown.add(r["pmcid"])
    known = set()
    for f in (VALID / "audit").glob("licences_*.json"):
        known |= set(json.loads(f.read_text(encoding="utf-8")))
    arts_nc |= quoted_unknown - known          # licence never checked: treat as not CC
    mark = "[text withheld: source has no Creative Commons licence]"
    n = 0
    p = vdir / "validation_sample.jsonl"
    if p.exists():
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:
            r["text_withheld"] = int(r["key"] in keys_nc)
            if r["text_withheld"]:
                r["text"], n = "", n + 1
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    for f in (vdir / "audit").glob("screen_*.jsonl"):
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:
            if r.get("pmcid") in arts_nc and (r.get("quote") or "").strip():
                r["quote"], n = mark, n + 1
        f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    for f in (vdir / "audit").glob("positives_*.txt"):
        out = []
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.split(" | ")
            if parts[0] in arts_nc and len(parts) >= 5:
                line, n = " | ".join(parts[:4] + [mark]), n + 1
            out.append(line)
        f.write_text("".join(l + "\n" for l in out), encoding="utf-8")
    print(f"validation copies: text withheld in {n} places (sources without a Creative Commons licence)")


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    guide = SYSTEM_PROMPT.split("Return a single minified JSON object")[0].strip()
    (DOCS / "codebook.md").write_text(
        f"# GAID annotation codebook, version {CODEBOOK_VERSION}\n\n"
        "The text below is the instruction given verbatim to the annotator model and to the human annotators.\n\n```\n"
        + guide + "\n```\n", encoding="utf-8")
    with open(DOCS / "data_dictionary.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["table", "variable", "type", "description"])
        w.writerows(DICT)
    with open(DOCS / "tool_lexicon.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["order", "regex_on_lowercased_name", "tool", "developer", "category"])
        for i, (pat, tool, dev, cat) in enumerate(RULES, 1):
            w.writerow([i, pat, tool, dev or "", cat])
    with open(DOCS / "publisher_map.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["doi_prefix", "publisher_brand", "publisher_group"])
        for pre, (brand, group) in sorted(DOI_PREFIX.items()):
            w.writerow([pre, brand, group])
    (DOCS / "candidate_vocabulary.txt").write_text(
        "Europe PMC full-text query terms (each searched as a quoted phrase, combined with OR)\n\n"
        "## Product names\n" + "\n".join(TOOL_TERMS) + "\n\n## Generic expressions\n" + "\n".join(GENERIC_TERMS) + "\n", encoding="utf-8")
    (RELEASE / "README.md").write_text(README, encoding="utf-8")
    # validation material that can be shared (the human labels are exported as JSONL, without the workbooks)
    vdir = RELEASE / "validation"
    vdir.mkdir(exist_ok=True)
    if (VALID / "validation_sample.jsonl").exists():
        shutil.copy2(VALID / "validation_sample.jsonl", vdir / "validation_sample.jsonl")
    if (VALID / "validation_results.json").exists():
        vr = json.loads((VALID / "validation_results.json").read_text(encoding="utf-8"))
        (vdir / "validation_results.json").write_text(json.dumps(vr, indent=1), encoding="utf-8")
    for sub in ("human", "audit"):
        src = VALID / sub
        if src.exists():
            for f in src.glob("*"):
                if f.name.startswith("positives_"):      # readable printouts of the screening files (06 report)
                    continue
                if f.suffix in (".json", ".jsonl", ".txt") and f.stat().st_size < 30_000_000:
                    (vdir / sub).mkdir(exist_ok=True)
                    shutil.copy2(f, vdir / sub / f.name)
    export_human_labels(vdir)
    for f in VALID.glob("ref_*.jsonl"):
        shutil.copy2(f, vdir / f.name)
    for f in list(VALID.glob("second_*.jsonl")) + list(VALID.glob("retest_*.jsonl")) + list(VALID.glob("context_*.jsonl")):
        shutil.copy2(f, vdir / f.name)
    withhold_noncc_text(vdir)
    # complete model outputs, keyed by text_key (one line per distinct passage text)
    import gzip
    from common import INTERIM
    adir = RELEASE / "annotations"
    adir.mkdir(exist_ok=True)
    for src, dst in (("annot_primary.jsonl", "labels_primary_annotator.jsonl.gz"),
                     ("annot_second.jsonl", "labels_second_annotator.jsonl.gz")):
        p = INTERIM / src
        if p.exists():
            with open(p, "rb") as fi, gzip.open(adir / dst, "wb", compresslevel=6) as fo:
                shutil.copyfileobj(fi, fo)
    # deposit metadata draft (Zenodo reads .zenodo.json); creators must be filled in by the authors
    summ_p = RELEASE / "build_summary.json"
    summ = json.loads(summ_p.read_text(encoding="utf-8")) if summ_p.exists() else {}
    zen = {
        "title": "GAID: a corpus of generative-AI use declarations in open-access research articles, 2023–2026",
        "upload_type": "dataset",
        "description": (
            "<p>GAID (Generative-AI Declarations) contains authors' declarations about the use or non-use of generative AI and "
            f"AI-assisted tools, extracted from the open-access full text in Europe PMC / PubMed Central for articles first published "
            f"{DATE_START} to {DATE_END}. It holds {summ.get('n_statements', 0):,} declaration statements from "
            f"{summ.get('n_declaring_articles', 0):,} articles, each with verbatim text, location, polarity, named tools, declared "
            "purposes and accountability indicators, together with article metadata, all screened candidate articles, rejected "
            "passages, and journal-by-month and publisher-by-month denominators.</p>"
            "<p>Annotations, metadata and denominators are released under CC BY 4.0. Statement texts are short verbatim excerpts "
            "of open-access articles and remain under the licence of the source article recorded in each row.</p>"
            "<p>See README.md and docs/ for the data dictionary and the annotation codebook.</p>"),
        "creators": [{"name": "[FAMILY NAME, Given name]", "affiliation": "[affiliation]", "orcid": "[0000-0000-0000-0000]"}],
        "keywords": ["generative artificial intelligence", "large language models", "disclosure", "research integrity",
                     "scholarly publishing", "meta-research", "text corpus", "PubMed Central"],
        "license": "cc-by-4.0",
        "access_right": "open",
        "language": "eng",
        "version": "1.0.0",
        "notes": f"Passage rules {RULES_VERSION}; codebook {CODEBOOK_VERSION}; tool lexicon {LEXICON_VERSION}; "
                 f"publisher map {PUBLISHER_MAP_VERSION}.",
    }
    (RELEASE / ".zenodo.json").write_text(json.dumps(zen, indent=1, ensure_ascii=False), encoding="utf-8")
    print("docs written to", DOCS, "| data dictionary rows:", len(DICT))


if __name__ == "__main__":
    main()
