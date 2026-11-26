"""JATS parsing: article metadata + a high-recall superset of text blocks that may hold an AI-use declaration.

Stage A (this module, run while downloading) keeps every block that could conceivably matter, so that the
precise passage rules (passage_rules.py) can be tuned and audited offline without re-downloading full texts.
"""
from __future__ import annotations

import re

from lxml import etree

XML_NS = "{http://www.w3.org/XML/1998/namespace}"
XLINK = "{http://www.w3.org/1999/xlink}"
PARSER_KW = dict(resolve_entities=False, no_network=True, recover=True, huge_tree=True, remove_comments=True)

# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------
# case-insensitive product / concept terms
_CI_TERMS = (
    r"artificial[\s-]+intelligence|generative|gen[\s-]?AI\b|chat[\s-]?bots?|chat[\s-]?GPT|GPT[\s-]?\d[\w.]*|\bGPTs?\b|open[\s-]?AI\b|"
    r"(?:large[\s-]+)?language[\s-]+models?|\bLLMs?\b|grammarly|quill[\s-]?bot|deepl\b|paperpal|writefull|deepseek|"
    r"co-?pilot|anthropic|\bclaude\b|\bgemini\b|\bbard\b|perplexity|notebook[\s-]?lm|scispace|wordtune|trinka|midjourney|"
    r"dall[\s·-]?e\b|ernie[\s-]?bot|\bkimi\b|doubao|\bqwen\b|\bllama\b|\bmistral\b|\belicit\b|\bscite\b|\bjenni\b|"
    r"\bcurie\b|prowritingaid|languagetool|wordvice|instatext|scholarcy|humata|stable[\s-]+diffusion|\bgrok\b|"
    r"machine[\s-]+translat\w*|automated[\s-]+(?:writing|translation|editing)|inteligencia[\s-]+artificial|"
    r"intelig[êe]ncia[\s-]+artificial|intelligence[\s-]+artificielle|k[üu]nstliche[rn]?[\s-]+intelligenz|intelligenza[\s-]+artificiale|"
    r"人工智能|生成式|大语言模型|人工知能|생성형"
)
# case-sensitive acronyms (author initials such as "AI" are resolved downstream)
_CS_TERMS = r"\bA\.?I\.?(?![A-Za-z])|\bIA\b|\bKI\b|\bGAI\b"
BROAD_RE = re.compile(f"(?i:{_CI_TERMS})|{_CS_TERMS}")

# writing / declaration context required for body paragraphs to enter the superset
BODY_CONTEXT_RE = re.compile(
    r"(?i)manuscript|writing|written|\bwrote\b|draft|edit|proof-?read|gramma|spell|readab|\blanguage\b|polish|paraphras|"
    r"rephras|translat|wording|clarity|declar|disclos|statement|acknowledg|prepar|authors?\b|responsib|"
    r"english|correct|improv|refin|assist|figures?\b|images?\b|illustrat|graphical|"
    r"redac|escrit|manuscrit|r[ée]dig")
# body paragraphs that read like a first-person disclosure are never dropped by the per-article cap
STRICT_BODY_RE = re.compile(
    r"(?i)(?:preparation|writing|drafting|editing|revision|proof-?reading|development) of (?:this|the|our) (?:manuscript|paper|article|work|study|review|report)|"
    r"during the (?:preparation|writing|drafting)|(?:this|our) manuscript|"
    r"authors? (?:have |has )?(?:used|utili[sz]ed|employed|did not use|declare|acknowledge|take|confirm)|\bwe (?:used|utili[sz]ed|employed|acknowledge|declare|did not use)\b|"
    r"(?:was|were) (?:used|utili[sz]ed|employed) (?:solely |only |exclusively )?(?:to|for) (?:improv|edit|check|polish|correct|refin|translat|proof|enhanc|assist)|"
    r"language (?:editing|polishing|improvement|refinement|correction|enhancement)|gramma|proof-?read|"
    r"readability of (?:the|this) (?:manuscript|text|paper|article)")
# titles of declaration-type sections that some publishers place inside <body>
BACKLIKE_TITLE_RE = re.compile(
    r"(?i)acknowledge?ment|declaration|disclosure|statement|funding|conflict|competing|contribut|ethic|additional information|"
    r"\bnotes?\b|availability|agradec|remerciement|danksagung|financ|author information|article information|disclaimer|transparen")

# a (section) title that announces an AI-use declaration
AI_TITLE_RE = re.compile(
    r"(?i:artificial[\s-]+intelligence|generative|gen[\s-]?AI\b|chat[\s-]?GPT|chat[\s-]?bots?|language[\s-]+models?|\bLLMs?\b|"
    r"AI[\s-]+(?:assist|generat|tool|technolog|use|usage|statement|declar|disclos|writing|contribut|acknowledg|support)|"
    r"(?:use|usage|utili[sz]ation|disclosure|declaration|statement|acknowledge?ment|role)\s+(?:of|on|regarding|about|for)\s+(?:the\s+use\s+of\s+)?(?:generative\s+)?AI\b|"
    r"AI[\s-]+and[\s-]+AI|inteligencia[\s-]+artificial|intelig[êe]ncia[\s-]+artificial|intelligence[\s-]+artificielle|k[üu]nstliche[rn]?[\s-]+intelligenz)"
    r"|\bAI\b|\bGAI\b|\bIA\b")
TITLE_DECL_CUE_RE = re.compile(
    r"(?i)declar|statement|disclos|disclaim|acknowledg|\buse\b|\busage\b|\busing\b|utili[sz]|assist|transparen|writing|"
    r"\bnote\b|complian|tools?\b|technolog|contribut|support|\buso\b|utilisation|nutzung|einsatz")

SKIP_ANCESTORS = {"abstract", "trans-abstract", "ref-list", "table-wrap", "fig", "kwd-group", "supplementary-material",
                  "disp-formula", "sub-article", "response", "front-stub", "table-wrap-foot", "fig-group", "media",
                  "inline-formula", "tex-math", "math", "glossary", "def-list", "bio", "ref", "mixed-citation",
                  "element-citation", "funding-group", "contrib-group", "aff", "history", "permissions"}
TITLED = {"sec", "ack", "notes", "app", "fn-group", "boxed-text", "fn", "author-notes", "app-group"}

MAX_BLOCK_CHARS = 3000
MAX_BODY_BLOCKS = 40
MAX_BACK_BLOCKS = 80


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def lname(el) -> str:
    t = el.tag
    return etree.QName(el).localname if isinstance(t, str) else ""


def el_text(el) -> str:
    return norm_ws("".join(el.itertext()))


def direct_title(el) -> str:
    for ch in el:
        if lname(ch) in ("title", "label"):
            t = el_text(ch)
            if t:
                return t
    return ""


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------
_CC_RE = re.compile(r"creativecommons\.org/(licenses|publicdomain)/([a-z\-]+)/?(\d\.\d)?", re.I)


def license_from(art):
    perm = art.find(".//article-meta/permissions")
    if perm is None:
        return None, None
    urls = []
    for lic in perm.iter():
        n = lname(lic)
        if n == "license":
            h = lic.get(XLINK + "href")
            if h:
                urls.append(h)
        elif n == "license_ref":
            if lic.text:
                urls.append(lic.text.strip())
        elif n == "ext-link":
            h = lic.get(XLINK + "href")
            if h:
                urls.append(h)
    txt = " ".join(urls) or el_text(perm)
    m = _CC_RE.search(txt)
    if not m:
        return (urls[0] if urls else None), None
    kind, code, ver = m.group(1).lower(), m.group(2).lower(), m.group(3)
    if kind == "publicdomain":
        return m.group(0), "CC0"
    return m.group(0), "CC " + code.upper() + (f" {ver}" if ver else "")


def pub_dates(art):
    out = {}
    for pd in art.iterfind(".//article-meta/pub-date"):
        typ = pd.get("pub-type") or pd.get("date-type") or pd.get("publication-format") or "unknown"
        y = pd.findtext("year")
        if not y or not y.strip().isdigit():
            continue
        m = (pd.findtext("month") or "").strip()
        d = (pd.findtext("day") or "").strip()
        s = y.strip()
        if m.isdigit():
            s += f"-{int(m):02d}"
            if d.isdigit():
                s += f"-{int(d):02d}"
        out.setdefault(typ, s)
    return out


def article_meta(art) -> dict:
    meta = {"article_type": art.get("article-type"), "lang": art.get(XML_NS + "lang")}
    jm = art.find(".//journal-meta")
    if jm is not None:
        meta["journal"] = norm_ws(jm.findtext(".//journal-title") or "") or None
        for jid in jm.iterfind("journal-id"):
            if jid.get("journal-id-type") == "nlm-ta":
                meta["journal_nlm_ta"] = norm_ws(jid.text or "")
        for issn in jm.iterfind("issn"):
            key = "issn_e" if (issn.get("pub-type") or issn.get("publication-format") or "").startswith("e") else "issn_p"
            meta.setdefault(key, norm_ws(issn.text or ""))
        meta["publisher"] = norm_ws(jm.findtext(".//publisher-name") or "") or None
    am = art.find(".//article-meta")
    if am is not None:
        for aid in am.iterfind("article-id"):
            t = aid.get("pub-id-type")
            if t in ("pmcid", "pmc", "pmid", "doi", "pmcaid"):
                meta.setdefault({"pmc": "pmcid"}.get(t, t), norm_ws(aid.text or ""))
        tt = am.find(".//title-group/article-title")
        if tt is not None:
            meta["title"] = el_text(tt)[:500]
        subj = [el_text(s) for s in am.iterfind(".//article-categories/subj-group/subject")]
        if subj:
            meta["subjects"] = [s for s in subj if s][:6]
        meta["pub_dates"] = pub_dates(art)
        url, code = license_from(art)
        meta["license_url"], meta["license"] = url, code
        meta["n_authors"] = sum(1 for c in am.iterfind(".//contrib-group/contrib") if (c.get("contrib-type") or "author") == "author")
        countries = []
        for c in am.iter("country"):
            v = (c.get("country") or el_text(c)).strip()
            if v and v not in countries:
                countries.append(v)
        if countries:
            meta["aff_countries"] = countries[:20]
    return meta


# ---------------------------------------------------------------------------
# block superset
# ---------------------------------------------------------------------------
def _context(el):
    """Return (skip, part, zone_container, titles(nearest first), sec_type, titled_ancestor_el)."""
    part, container, titles, sec_type, titled_el = None, None, [], None, None
    cur = el.getparent()
    while cur is not None:
        n = lname(cur)
        if n in SKIP_ANCESTORS:
            return True, None, None, [], None, None
        if n in TITLED:
            if container is None:
                container = n
            t = direct_title(cur)
            if t:
                if titled_el is None:
                    titled_el = cur
                if len(titles) < 3:
                    titles.append(t[:200])
            if sec_type is None:
                sec_type = cur.get("sec-type") or cur.get("notes-type") or cur.get("fn-type")
        if n in ("front", "body", "back", "floats-group") and part is None:
            part = n
        cur = cur.getparent()
    # (the walk continues to the root so that text inside <sub-article>/<response>, e.g. peer-review files, is skipped)
    return False, part, container, titles, sec_type, titled_el


def _clip(txt: str) -> tuple[str, bool]:
    """Long blocks: keep windows around AI-term matches instead of a blind prefix."""
    if len(txt) <= MAX_BLOCK_CHARS:
        return txt, False
    spans = []
    for m in BROAD_RE.finditer(txt):
        a, b = max(0, m.start() - 700), min(len(txt), m.end() + 900)
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
        if sum(e - s for s, e in spans) > MAX_BLOCK_CHARS:
            break
    if not spans:
        return txt[:MAX_BLOCK_CHARS], True
    return " […] ".join(txt[s:e] for s, e in spans)[:MAX_BLOCK_CHARS + 400], True


def is_ai_title(title: str, in_body_main: bool) -> bool:
    if not title or len(title) > 220 or not AI_TITLE_RE.search(title):
        return False
    if in_body_main:
        return bool(TITLE_DECL_CUE_RE.search(title))
    return True


def extract(xml_bytes: bytes) -> dict | None:
    """Parse one JATS document. Returns {'meta':..., 'blocks':[...], 'back_titles':[...]} or None if unparseable."""
    try:
        root = etree.fromstring(xml_bytes, etree.XMLParser(**PARSER_KW))
    except Exception:  # noqa: BLE001
        return None
    if root is None:
        return None
    art = root if lname(root) == "article" else root.find(".//article")
    if art is None:
        return None
    meta = article_meta(art)
    blocks, back_titles = [], []
    sec_ids: dict[int, int] = {}
    n_body = n_back = n_body_p = n_body_ai_p = 0
    prev_heading_parent = None  # parent element of an inline AI heading paragraph

    for back in art.iterfind("back"):
        for el in back.iter("sec", "ack", "notes", "fn-group", "app"):
            t = direct_title(el)
            if t and len(back_titles) < 60:
                back_titles.append(t[:120])

    for el in art.iter("p", "custom-meta"):
        n = lname(el)
        skip, part, container, titles, sec_type, titled_el = _context(el)
        if skip or part is None or part == "floats-group":
            continue
        if n == "p":
            par = el.getparent()
            # nested paragraphs: only keep the outermost
            anc_p = False
            cur = par
            while cur is not None and lname(cur) not in ("front", "body", "back"):
                if lname(cur) == "p":
                    anc_p = True
                    break
                cur = cur.getparent()
            if anc_p:
                continue
            txt = el_text(el)
        else:
            # custom-meta: keep only when the meta-name itself announces an AI declaration
            mname = norm_ws(el.findtext("meta-name") or "")
            mval = el.find("meta-value")
            if mval is None or not is_ai_title(mname, False):
                continue
            txt = norm_ws(mname + ": " + el_text(mval))
            titles = [mname[:200]]
            par = el.getparent()
        if not txt:
            continue

        if part == "front":
            zone = "front"
        elif part == "back":
            zone = "ack" if container == "ack" else ("appendix" if container in ("app", "app-group") else "back")
        else:
            zone = "body"
            n_body_p += 1
            # declaration-like blocks that publishers place at the end of <body>
            if container in ("fn", "fn-group", "notes", "author-notes", "ack"):
                zone = "back"
            elif titles and len(titles[0]) <= 120 and BACKLIKE_TITLE_RE.search(titles[0]):
                zone = "back"
        body_main = zone in ("body", "appendix")

        ai_sec = any(is_ai_title(t, body_main) for t in titles[:2])
        follows_heading = prev_heading_parent is not None and par is prev_heading_parent
        prev_heading_parent = None
        broad = bool(BROAD_RE.search(txt))
        # an inline heading paragraph such as "<p><bold>Declaration of AI use</bold></p>"
        if broad and len(txt) <= 160 and not txt.rstrip().endswith((".", "。")) and is_ai_title(txt, body_main):
            prev_heading_parent = par

        if broad and zone == "body":
            n_body_ai_p += 1
        keep = ai_sec or follows_heading
        strict = False
        if not keep and broad:
            if body_main:
                strict = bool(STRICT_BODY_RE.search(txt))
                keep = strict or bool(BODY_CONTEXT_RE.search(txt))
            else:
                keep = True
        if not keep:
            continue
        if body_main and not (ai_sec or follows_heading):
            n_body += 1
            if n_body > MAX_BODY_BLOCKS and not strict:
                continue
        else:
            n_back += 1
            if n_back > MAX_BACK_BLOCKS:
                continue
        sid = None
        if titled_el is not None:
            sid = sec_ids.setdefault(id(titled_el), len(sec_ids))
        clipped, was_clipped = _clip(txt)
        blk = {"z": zone, "t": titles[:2], "sid": sid, "txt": clipped}
        if sec_type:
            blk["st"] = sec_type
        if ai_sec:
            blk["ai_sec"] = 1
        if follows_heading:
            blk["fh"] = 1
        if was_clipped:
            blk["trunc"] = 1
        blocks.append(blk)
    meta["n_body_p"], meta["n_body_ai_p"] = n_body_p, n_body_ai_p
    return {"meta": meta, "blocks": blocks, "back_titles": back_titles}
