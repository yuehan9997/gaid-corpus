"""Annotation codebook (as an LLM prompt), request building and response normalisation.

The same codebook text is released with the dataset as the human annotation guideline (docs/codebook.md).
"""
from __future__ import annotations

import json
import re

CODEBOOK_VERSION = "1.1"

PURPOSES = ["LANG", "TRANS", "DRAFT", "LIT", "IDEA", "CODE", "DATA", "VIS", "FORMAT", "OTHER", "UNSPEC"]
POLARITIES = ["use", "no_use", "na", "unclear"]

SYSTEM_PROMPT = """You are annotating passages extracted from scientific articles for a research dataset on how authors declare the use of generative AI and AI-assisted tools.

Each input item has: "id"; "loc" (where the passage was found: "ai_section" = under a heading that announces an AI-use declaration; "decl_zone" = acknowledgements, declarations, footnotes or other back matter; "body" = main text); "heading" (the section heading, may be empty); and "text" (the verbatim passage, which may also contain unrelated sentences).

For each item, output one JSON object with the fields below.

"decl": 1 if the passage contains an AI-USE DECLARATION, otherwise 0.
An AI-use declaration is a statement by the article's own authors disclosing whether or how generative AI or AI-assisted tools were used AS AN AID in preparing the article or in carrying out the work behind it (writing, editing, translating, searching or summarising literature, coding, analysing data, making figures, and so on). Such tools include chatbots and large language models (e.g. ChatGPT), writing or translation aids (e.g. Grammarly, DeepL), image generators, coding assistants, AI literature tools, or "AI" named only generically. Statements of NON-use count ("No AI tools were used"), and so do minimal answers given under an AI-declaration heading ("None", "Not applicable").
Set decl = 0 when AI appears only as: the object, method, material, comparator or intervention of the research itself (e.g. "we evaluated ChatGPT's answers", "a deep-learning model was trained", "vignettes for the experiment were generated with GPT-4" in a methods description); part of the name of a funder, grant, institution, company, product, journal or person; a competing-interest or employment disclosure; a description of other people's work; general discussion of AI; an author's initials (e.g. "AI wrote the first draft" in a contribution list); a publisher's or editor's notice; or when the text is merely a heading with no statement.
If loc = "ai_section", presume the passage is a declaration unless it is clearly not an author statement about AI use.
If decl = 0, set: "pol": null, "tools": [], "purp": [], "resp": 0, "rev": 0, "nonuse": 0.

"pol": polarity of the declaration.
 "use" = the authors state that at least one AI or AI-assisted tool was used for something.
 "no_use" = they state that no AI or AI-assisted tool was used (including "None", "Not used", "Nothing to declare").
 "na" = the entire answer is "Not applicable", "N/A" or an equivalent.
 "unclear" = a declaration is present but it cannot be determined whether AI was used.
If the authors say that no generative AI was used but name an AI-assisted tool that they did use (e.g. Grammarly), choose "use".

"tools": every NAMED AI or AI-assisted product or model that the authors say THEY used, as a list of [name, version] pairs copied from the text. name = the product or model name without the vendor ("OpenAI ChatGPT" -> "ChatGPT"); version = the model, version or edition string if one is given, otherwise null. Examples: "ChatGPT (GPT-4o, OpenAI)" -> ["ChatGPT", "GPT-4o"]; "ChatGPT-4o" -> ["ChatGPT", "4o"]; "GPT-5.5" -> ["GPT-5.5", null]; "Claude (Sonnet 4.6 and Opus 4.8)" -> ["Claude", "Sonnet 4.6 and Opus 4.8"]; "Grammarly" -> ["Grammarly", null]. Do not list: generic descriptions that are not names ("AI tools", "an AI chatbot", "a large language model"); tools mentioned only as not used or only as the object of the study; a vendor alone when its product is named; non-AI software (e.g. Word, SPSS, BioRender, Canva, EndNote). Use an empty list when no tool is named.

"purp": what the AI was used for (only when pol = "use"); list every code that applies.
 "LANG" = editing text written by the authors: grammar, spelling, style, clarity, readability, fluency, concision, polishing, proofreading, rephrasing or paraphrasing, reorganising existing text.
 "TRANS" = translation between languages.
 "DRAFT" = generating new text: drafting sections, abstracts, titles, summaries, outlines, cover letters or replies to reviewers.
 "LIT" = finding, screening, summarising or synthesising literature; checking or suggesting references.
 "IDEA" = brainstorming, generating or refining ideas, feedback or critique on drafts, study-design suggestions.
 "CODE" = writing, debugging or explaining code, scripts, formulas or queries.
 "DATA" = data analysis, statistics, data extraction, annotation, processing, or interpretation of results.
 "VIS" = creating or editing figures, images, illustrations, graphical abstracts or diagrams.
 "FORMAT" = formatting the manuscript, references or tables; checking compliance with journal or reporting guidelines.
 "OTHER" = any other stated purpose (e.g. generating study materials or test items, transcription).
 "UNSPEC" = use is declared but no purpose is stated, or only vaguely ("assisted in manuscript preparation", "writing assistance", "in the writing process"). Do not add DRAFT for such vague wording; use UNSPEC only when no specific purpose is given.
For any other polarity use an empty list.

"resp": 1 only if the authors explicitly assert responsibility or accountability for the content or work ("take full responsibility", "are responsible for", "remain accountable"); otherwise 0. Saying that the authors performed, reviewed or approved the work is not enough.
"rev": 1 if the authors state that they reviewed, edited, checked, verified, validated, approved or supervised the AI-assisted output, or the content after AI was used; otherwise 0.
"nonuse": 1 if, in a declaration with pol = "use", the authors also explicitly NEGATE AI involvement in some task or content ("AI was not used for data analysis", "no content was generated by AI", "the tools had no role in interpretation"); otherwise 0. Wording such as "solely for language editing" or "all analysis was done by the authors" does not count by itself.

Judge only what the text says; do not guess. Texts may be in any language.

EXAMPLES
{"id":1,"loc":"ai_section","heading":"Declaration of generative AI and AI-assisted technologies in the writing process","text":"During the preparation of this work the authors used ChatGPT (GPT-4o, OpenAI) and DeepL in order to improve readability and translate parts of the text. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication."}
-> {"id":1,"decl":1,"pol":"use","tools":[["ChatGPT","GPT-4o"],["DeepL",null]],"purp":["LANG","TRANS"],"resp":1,"rev":1,"nonuse":0}
{"id":2,"loc":"ai_section","heading":"Generative AI statement","text":"The author(s) declare that no Generative AI was used in the creation of this manuscript."}
-> {"id":2,"decl":1,"pol":"no_use","tools":[],"purp":[],"resp":0,"rev":0,"nonuse":0}
{"id":3,"loc":"ai_section","heading":"Use of AI tools","text":"Not applicable."}
-> {"id":3,"decl":1,"pol":"na","tools":[],"purp":[],"resp":0,"rev":0,"nonuse":0}
{"id":4,"loc":"decl_zone","heading":"Acknowledgments","text":"We thank J. Smith for technical help. Generative AI tools were used solely for language polishing; they were not used to generate scientific content or to analyse data. Figure 1 was created with Midjourney v6."}
-> {"id":4,"decl":1,"pol":"use","tools":[["Midjourney","v6"]],"purp":["LANG","VIS"],"resp":0,"rev":0,"nonuse":1}
{"id":5,"loc":"decl_zone","heading":"Funding","text":"This work was supported by the National Artificial Intelligence Research Program (grant 2023-0042)."}
-> {"id":5,"decl":0,"pol":null,"tools":[],"purp":[],"resp":0,"rev":0,"nonuse":0}
{"id":6,"loc":"body","heading":"Methods","text":"Responses to the 30 patient questions were generated with ChatGPT-4 and Gemini and rated by two physicians."}
-> {"id":6,"decl":0,"pol":null,"tools":[],"purp":[],"resp":0,"rev":0,"nonuse":0}
{"id":7,"loc":"body","heading":"Statistical analysis","text":"Analyses were run in R 4.3. GitHub Copilot assisted in writing the analysis scripts, and the manuscript was proofread with Grammarly; all outputs were checked by the authors."}
-> {"id":7,"decl":1,"pol":"use","tools":[["GitHub Copilot",null],["Grammarly",null]],"purp":["LANG","CODE"],"resp":0,"rev":1,"nonuse":0}
{"id":8,"loc":"decl_zone","heading":"Author contributions","text":"AI and JS designed the study. AI wrote the first draft. All authors approved the final manuscript."}
-> {"id":8,"decl":0,"pol":null,"tools":[],"purp":[],"resp":0,"rev":0,"nonuse":0}

Return a single minified JSON object (no line breaks or extra spaces) of the form {"results":[{"id":...,"decl":...,"pol":...,"tools":[...],"purp":[...],"resp":...,"rev":...,"nonuse":...},...]} with exactly one result per input item, in the input order, and no other text."""


def build_user(items: list[dict]) -> str:
    """items: [{'id':..., 'loc':..., 'heading':..., 'text':...}]"""
    lines = [json.dumps({"id": it["id"], "loc": it["loc"], "heading": it.get("heading") or "", "text": it["text"]},
                        ensure_ascii=False) for it in items]
    return "Annotate these %d items.\n" % len(items) + "\n".join(lines)


_JSON_RE = re.compile(r"\{.*\}", re.S)


def _b(v) -> int:
    return 1 if v in (1, True, "1", "true", "True", "yes") else 0


def normalise(rec: dict) -> dict | None:
    """Coerce one model result to the canonical form; None if unusable."""
    if not isinstance(rec, dict) or "id" not in rec:
        return None
    decl = _b(rec.get("decl"))
    out = {"id": rec["id"], "decl": decl, "pol": None, "tools": [], "purp": [], "resp": 0, "rev": 0, "nonuse": 0}
    if not decl:
        return out
    pol = rec.get("pol")
    pol = str(pol).strip().lower().replace("-", "_").replace(" ", "_") if pol is not None else "unclear"
    pol = {"nouse": "no_use", "none": "no_use", "not_applicable": "na", "n/a": "na", "null": "unclear"}.get(pol, pol)
    out["pol"] = pol if pol in POLARITIES else "unclear"
    tools = []
    for t in rec.get("tools") or []:
        if isinstance(t, (list, tuple)) and t:
            name = str(t[0]).strip() if t[0] is not None else ""
            ver = str(t[1]).strip() if len(t) > 1 and t[1] not in (None, "", "null") else None
        elif isinstance(t, dict):
            name = str(t.get("name") or t.get("n") or "").strip()
            ver = t.get("version") or t.get("v")
            ver = str(ver).strip() if ver not in (None, "", "null") else None
        else:
            name, ver = str(t).strip(), None
        if name:
            tools.append([name, ver])
    out["tools"] = tools if out["pol"] == "use" else []
    purp = [str(p).strip().upper() for p in (rec.get("purp") or []) if str(p).strip().upper() in PURPOSES]
    purp = sorted(set(purp), key=PURPOSES.index)
    if out["pol"] == "use":
        if len(purp) > 1 and "UNSPEC" in purp:
            purp.remove("UNSPEC")
        out["purp"] = purp or ["UNSPEC"]
    out["resp"], out["rev"] = _b(rec.get("resp")), _b(rec.get("rev"))
    out["nonuse"] = _b(rec.get("nonuse")) if out["pol"] == "use" else 0
    return out


def parse_response(content: str | None, expected_ids: list) -> dict:
    """Return {id: normalised_result} for the ids that could be parsed."""
    if not content:
        return {}
    txt = content.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    try:
        d = json.loads(txt)
    except Exception:  # noqa: BLE001
        m = _JSON_RE.search(txt)
        if not m:
            return {}
        try:
            d = json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return {}
    res = d.get("results") if isinstance(d, dict) else d
    if isinstance(d, dict) and res is None and "id" in d:
        res = [d]
    out = {}
    exp = {str(i): i for i in expected_ids}
    for r in res or []:
        n = normalise(r)
        if n is not None and str(n["id"]) in exp:
            n["id"] = exp[str(n["id"])]
            out[n["id"]] = n
    return out
