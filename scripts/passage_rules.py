"""Stage B rules: turn the stored block superset of one article into candidate declaration passages.

Three capture routes (recorded in `route`):
  ai_section : a section / note / footnote whose heading announces an AI-use declaration (whole section text)
  decl_zone  : acknowledgements, back matter, front notes, declaration-type sections -> sentences naming AI/tools
  body       : main text -> sentences that name AI/tools AND read like a first-person disclosure about the manuscript
The rules are deliberately permissive (recall-oriented); an LLM later decides whether each passage is a genuine
declaration, and a separate audit measures what the rules miss.
"""
from __future__ import annotations

import hashlib
import re

from extract_lib import AI_TITLE_RE, is_ai_title, norm_ws

RULES_VERSION = "1.4"

# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------
UNAMBIG_TOOLS = (
    r"chat[\s-]?GPT\w*|GPT[\s-]?\d[\w.]*|\bGPTs?\b|open[\s-]?AI\b|grammarly|quill[\s-]?bot|\bdeepl\b|paper[\s-]?pal|writefull|deep[\s-]?seek|"
    r"co-?pilot|\banthropic\b|notebook[\s-]?lm|scispace|wordtune|trinka|midjourney|dall[\s·-]?e\b|ernie[\s-]?bot|doubao|\bqwen\b|"
    r"prowritingaid|languagetool|wordvice|instatext|scholarcy|humata|stable[\s-]+diffusion|\bmanus\b(?=\s*(?:AI|\d|\())|"
    r"\bgrok\b|\bsora\b(?=\s*(?:AI|\d|\())|napkin(?:\s+AI|\.ai)|gamma\.app|consensus(?:\s+AI|\.app)|scite(?:\.ai|\s+AI)|"
    r"elicit(?:\.com|\.org|\s+AI)|jenni(?:\.ai|\s+AI)|perplexity(?:\.ai|\s+AI)|\bcurie\b(?=\s*(?:AI|\(|by|from|,\s*Springer))|"
    r"research\s*rabbit|\bkimi\b(?=\s*(?:AI|\d|\(|chat))|\bllama[\s-]?\d|mistral(?:\s+AI|\s+large|\s*\d)|"
    r"gemini[\s-]?(?:\d|pro\b|flash|ultra|advanced|AI\b)|claude[\s-]?(?:\d|opus|sonnet|haiku|AI\b|code\b)|\bbard\b(?=\s*(?:AI|\(|,?\s*google))|"
    r"google\s+(?:gemini|bard|translate)|microsoft\s+bing|bing\s+(?:chat|AI)|meta\s+AI|you\.com|poe\.com|"
    r"wenxin|yiyan|tongyi|qianwen|hunyuan|spark\s*desk|zhipu|chatglm|\bGLM[\s-]?\d|yuanbao|豆包|文心一言|通义千问"
)
TOOL_RE = re.compile(f"(?i:{UNAMBIG_TOOLS})")
AMBIG_NAME_RE = re.compile(r"(?i:\bclaude\b|\bgemini\b|\bbard\b|\bperplexity\b|\bkimi\b|\bllama\b|\bmistral\b|\belicit\b|\bscite\b|"
                           r"\bjenni\b|\bcurie\b|\bconsensus\b|\bmanus\b|\bsora\b)")
AMBIG_CUE_RE = re.compile(r"(?i:\banthropic\b|google|\bxai\b|moonshot|artificial[\s-]+intelligence|language[\s-]+models?|chat[\s-]?bots?|"
                          r"generative|AI[\s-]?assistant)|\bAI\b|\bLLMs?\b")
SPELLED_GENERIC = (
    r"\bgenerative\b|artificial[\s-]+intelligence|gen[\s-]?AI\b|chat[\s-]?bots?|(?:large[\s-]+)?language[\s-]+models?|"
    r"machine[\s-]+translat\w*|automated[\s-]+(?:writing|translation|editing)|inteligencia[\s-]+artificial|"
    r"(?:grandes[\s-]+)?modelos?[\s-]+(?:de[\s-]+)?(?:lenguaje|ling[üu][íi]sticos?|linguagem)|"
    r"intelig[êe]ncia[\s-]+artificial|intelligence[\s-]+artificielle|k[üu]nstliche[rn]?[\s-]+intelligenz|"
    r"intelligenza[\s-]+artificiale|人工智能|生成式|大语言模型|人工知能|생성형")
SPELLED_GENERIC_RE = re.compile(f"(?i:{SPELLED_GENERIC})")
GENERIC_RE = re.compile(
    f"(?i:{SPELLED_GENERIC})" + r"|\bLLMs?\b|\bGAI\b"
    r"|\bAI[\s-](?:assist\w*|generat\w*|based|powered|driven|enabled|aided|supported|enhanced|tools?|technolog\w*|software|"
    r"models?|systems?|platforms?|assistance|writing|chatbots?|language|services?|programs?|applications?|apps\b|editing|"
    r"translation|content|usage|use\b|agents?|assistants?|support\b|help\b|free\b)"
    r"|\bIA\s+(?i:generativa|générative)|(?i:de|por|con|sin|da|com|sem|d'|l')\s*IA\b|\bKI[\s-](?i:gest|basier|tool|werkzeug)")
# unambiguous collocations of "AI" (safe even in contribution lists, where bare "AI" / "LLM" may be author initials)
AI_COLLOC_RE = re.compile(
    r"\bAI[\s-](?:assist\w*|generat\w*|based|powered|driven|enabled|aided|supported|enhanced|tools?|technolog\w*|software|"
    r"models?|systems?|platforms?|assistance|writing|chatbots?|language|services?|programs?|applications?|apps\b|editing|"
    r"translation|content|usage|use\b|agents?|assistants?|support\b|help\b|free\b)"
    r"|(?i:use|usage|using|used|utili[sz]ation|utili[sz]ing|utili[sz]ed|employing|employed|help|aid|assistance|support|involvement)"
    r"\s+(?i:of|from)\s+(?i:any\s+|the\s+)?(?i:generative\s+)?AI\b"
    r"|(?i:generated|created|written|produced|assisted|edited|translated|polished|reviewed|developed)\s+(?i:by|with|using)\s+(?i:generative\s+)?AI\b"
    r"|(?i:\bno|\bwithout|\bany|\bnon[\s-])\s*(?i:form\s+of\s+)?AI\b(?!\s*(?:,|and\s+[A-Z]{2}\b))"
    r"|\bAI\s+(?i:was|were|is|has|had|did|played)\s+(?i:not\s+|never\s+|no\s+)?(?i:been\s+)?(?i:used|utili[sz]ed|employed|involved|applied|use\b|role|part)"
    r"|\bAIGC\b|\bIA\s+(?i:generativa|générative)|(?i:de|por|con|sin|da|com|sem|d'|l')\s*IA\b|\bKI[\s-](?i:gest|basier|tool|werkzeug)")
ACRONYM_RE = re.compile(r"\bLLMs?\b|\bGAI\b")
BARE_AI_RE = re.compile(r"\bAI\b(?![-.]?[A-Z])|\bA\.I\.(?!\s+[A-Z][a-z])")
# "AI" as a person's initials followed by a verb typical of contribution / disclosure lists
_PERSON_VERBS = (r"declares?|reports?|receives?|received|has received|is an?\b|serves|holds|owns|acknowledges|thanks|"
                 r"was supported|is supported|is funded|was funded|contributed|wrote|designed|conceived|performed|analy[sz]ed|"
                 r"collected|drafted|reviewed|supervised|curated|led\b|conducted|prepared|edited|revised|approved|verified|"
                 r"accessed|had\b|did\b|provided|developed|interpreted|recruited|carried|obtained|participated")
PERSON_AI_RE = re.compile(r"\bA\.?I\.?,?\s+(?:" + _PERSON_VERBS + r"|and [A-Z]{1,3}\b|,\s*[A-Z]{1,3}\b)|[A-Z]{2,3},\s*AI\b|"
                          r"\band AI\b(?=\s+(?:" + _PERSON_VERBS + r"))")
# a block that enumerates people by initials (contribution lists, CRediT statements) even without a heading
INITIALS_LIST_RE = re.compile(r"\b(?:[A-Z]\.){2,4}(?=[,;:\s])|\b[A-Z]{2,4}(?=(?:,| and [A-Z]{2,4}\b|:| contributed| wrote| designed| conceived| performed| reviewed))")
FUNDING_SENT_RE = re.compile(r"(?i)\b(?:supported|funded|financed|sponsored|financiado|apoiado)\b|\bgrants?\b|\bfunding\b|\bfellowship|"
                             r"\bscholarship|\baward(?:ed|s)?\b|\bproject (?:no|number)|\bprogram(?:me)? (?:of|for|no)")
STRONG_AUTHORING_RE = re.compile(r"(?i)manuscript|\bwriting|written|edit|proof-?read|gramma|\blanguage|translat|polish|figures?\b|images?\b|"
                                 r"english|readab|\btext\b|draft")
# stricter writing context, required under headings such as Funding, Ethics or Data availability
HARD_AUTHORING_RE = re.compile(r"(?i)manuscript|\bwriting|written|\bedit|proof-?read|gramma|\blanguage (?!model)|translat|polish|"
                               r"english|readab|\btext\b|\bdraft|graphical abstract|this (?:article|paper)|redac|escrit|r[ée]dig")
NEG_USE_RE = re.compile(r"(?i)\b(?:no|not|without|never|neither|none|did not|didn't|sin|sem|sans|ohne|kein\w*|ning[úu]n\w*|nenhum\w*|aucun\w*)\b"
                        r".{0,80}\b(?:use[sd]?|using|utili[sz]\w*|employ\w*|appl(?:y|ied)|involv\w*|generat\w*|declar\w*|utiliz\w*|us[oó]\b)"
                        r"|\b(?:use[sd]?|utili[sz]ed|employed)\b.{0,40}\b(?:no|not|none|never)\b")

# cues
USE_CUE_RE = re.compile(
    r"(?i)\buse[sd]?\b|\busing\b|\busage\b|utili[sz]|employ|appli(?:ed|cation)|assist|help|\baid(?:ed)?\b|support|generat|creat|"
    r"produc|prepar|declar|acknowledg|disclos|edit|writ|draft|translat|polish|improv|proof|check|refin|correct|revis|review|"
    r"enhanc|rephras|paraphras|summari[sz]|consult|rel(?:y|ied)|leverag|\bno\b|\bnot\b|without|neither|\bnone\b|n/a|applicable|"
    r"thank|statement|\bvia\b|through|\buso\b|utiliz|emple|nutz|verwend|utilis|a[iy]uda|ajuda|aide|appreciat|gratitude|grateful|"
    r"\brole\b|contribut|ningun|nenhum|aucun")
AUTHORING_RE = re.compile(
    r"(?i)manuscript|\bwriting|written|\bwrote\b|\bdraft(?:ing|ed|s)?\b|edit|proof-?read|gramma|spell|readab|\blanguage|polish|"
    r"paraphras|rephras|translat|wording|clarity|english|\btext\b|prepar\w+ (?:of|this|the)|"
    r"figures?\b|images?\b|illustrat|graphical|\bcode\b|scripts?\b|literature|content|redac|escrit|r[ée]dig|idioma|texto|sprach|"
    r"manuscrit|artigo|art[íi]culo")
NONUSE_RE = re.compile(
    r"(?i)\b(?:no|not|without|neither|never|none|nor|did not|didn't|n/a|not applicable|sin|sem|sans|ohne|kein\w*|ning[úu]n\w*|nenhum\w*|aucun\w*)\b")
DECL_RE = re.compile(r"(?i)\bdeclar|\bdisclos|statement|acknowledge the use|\bconfirm|erkl[äa]r")

# section titles
CONTRIB_TITLE_RE = re.compile(r"(?i)contribut|credit\b|authorship|author'?s?'? (?:roles?|statement)|author information|contributors")
CONTRIB_TEXT_RE = re.compile(r"(?i)^\W{0,3}(?:authors?'?s?'? ?contributions?|contributors?|credit|author statement)\b")
HARD_LOWPRI_TITLE_RE = re.compile(
    r"(?i)funding|financ|grant|availability|data sharing|ethic|consent|review board|supplement|abbreviation|registration|"
    r"associated data|additional file|orcid|provenance|license|copyright|open access|reprint")
SOFT_LOWPRI_TITLE_RE = re.compile(r"(?i)conflict|competing|interest|publisher|peer review|correspond|biograph|patient|footnote")
NUMERIC_LABEL_RE = re.compile(r"^\W*\d{1,3}\W*$")
HARD_LOWPRI_TEXT_RE = re.compile(r"(?i)^\W{0,3}(?:funding|financial support|grant support|data availability|ethic\w* (?:approval|statement)|"
                                 r"supported by|this (?:work|study|research|project|publication) (?:was|is|has been) (?:supported|funded|financed))")
# titles under which back-matter text is treated as a declaration zone; other titled back-matter (appendices,
# endnotes, supplementary methods ...) is treated like running text
DECL_TITLE_RE = re.compile(
    r"(?i)acknowledge?ment|declar|disclos|statement|\bnotes?\b|conflict|competing|interest|contribut|credit\b|author|ethic|"
    r"funding|financ|information|footnote|transparen|disclaimer|agradec|remerciement|danksagung|erkl[äa]rung|"
    r"assistance|editorial|artificial|\bAI\b|generative|language model|\bLLM")
# titles that, on their own, announce an AI-use declaration when found in the main text
_PAREN = r"(?:\s*\((?:AI|GenAI|GAI|LLMs?|IA)\))?"
STRICT_AI_DECL_TITLE_RE = re.compile(
    r"(?i)declar|statement|disclos|disclaim|acknowledg|transparen|\bnote\b|"
    r"^\W*(?:\d+(?:\.\d+)*\.?\s*)?(?:the\s+)?(?:use|usage|utili[sz]ation|role)\s+of\s+(?:generative\s+)?(?:AI|artificial intelligence|"
    r"large language models?|LLMs?|chatgpt|AI[\s-]assisted|AI[\s-]based)" + _PAREN +
    r"(?:\s+tools?|\s+technolog\w+|\s+and\s+AI[\s-]assisted\s+technolog\w+|,\s*AI\s+and\s+machine\s+learning\s+tools)?" + _PAREN +
    r"(?:\s+(?:in|for|during)\s+(?:the\s+)?(?:writing|preparation|manuscript|this|scientific|drafting|research process).*)?\W*$|"
    r"^\W*(?:\d+(?:\.\d+)*\.?\s*)?(?:generative\s+)?(?:AI|artificial intelligence)" + _PAREN +
    r"[\s-]+(?:use|usage|assistance|tools?|contribution|support|utili[sz]ation)\W*$|"
    r"^\W*(?:\d+(?:\.\d+)*\.?\s*)?uso\s+de\s+.{0,40}(?:inteligencia|intelig[êe]ncia)\s+artificial.{0,40}$|"
    r"^\W*(?:\d+(?:\.\d+)*\.?\s*)?(?:using|use\s+of|usage\s+of)?\s*(?:generative\s+)?(?:AI|artificial\s+intelligence)" + _PAREN +
    r"\s*[-–—]?\s*(?:assisted|generated|based)?\s*(?:tools?|technolog\w+|use|usage|assistance)?\W*$")
EXCLUDE_TEXT_RE = re.compile(
    r"(?i)^\W{0,3}(?:cite this article as|how to cite|to cite this|this article was submitted to|open access:|list of abbreviations|"
    r"abbreviations?:|citation:|copyright|publisher'?s note|edited by|reviewed by|specialty section|peer review under|©|"
    r"this article is part of|handling editor|academic editor|section-at-acceptance)|text and data mining, AI training|"
    r"a section of the journal")

# body route
BODY_WRITE_RE = re.compile(
    r"(?i)manuscript|proof-?read|gramma|spelling|readab|phrasing|fluency|academic tone|"
    r"language (?:edit|polish|improv|refin|correct|enhanc|check|quality|revis|support|assist)|english|wording|paraphras|rephras|"
    r"(?:writing|drafting|preparation|preparing|revision|editing|polishing) (?:of )?(?:this|the present|the|our) "
    r"(?:paper|article|report|review|commentary|editorial|letter|essay|text|work|publication|study)|"
    r"writing (?:process|assist|support)|copy-?edit|edit(?:ing|ed) (?:of |the |this )?(?:text|paper|draft)|(?:first|initial|early) draft")
BODY_CODE_RE = re.compile(
    r"(?i)(?:\bcode|scripts?|programming|software)\b.{0,90}(?:written|generated|developed|debugged|assisted|produced|created|optimi[sz]ed)\b"
    r".{0,50}(?:by|with|using|via|through)\b|(?:assist\w*|help\w*) (?:with|in) (?:the )?(?:coding|programming|debugging|writing (?:the |of )?code)|"
    r"\bcoding\b.{0,40}\bassist|\bcode\b.{0,60}\b(?:adjusted|refined|troubleshoot\w*|debug\w*|re-?factor\w*)|\bprogrammed (?:in|with|using)\b|"
    r"(?:troubleshoot\w*|debug\w*|re-?factor\w*)\b.{0,60}\b(?:code|scripts?|errors?|functions?)|code development")
BODY_SELF_RE = re.compile(
    r"(?i)\bwe\b|\bauthors?\b|\bour\b|(?:was|were|been|is|are) (?:not )?(?:used|utili[sz]ed|employed|applied|generated|created|"
    r"edited|polished|translated|proofread|checked|prepared|written|drafted|refined|improved|corrected|revised|provided|"
    r"conducted|performed|carried out|supported|done)|"
    r"with the (?:help|aid|assistance|support) of|assisted by|supported by|assistance|\bno\b")
# negated use of AI in general (used only together with a reference to the authors' own work)
BODY_ANY_AI_RE = re.compile(r"(?i:artificial[\s-]+intelligence)|\bAI\b")
# main-text statements of non-use or explicit declarations that refer to the authors' own work
BODY_GENAI_RE = re.compile(
    r"(?i:\bgenerative\b|gen[\s-]?AI\b|chat[\s-]?GPT|chat[\s-]?bots?|large[\s-]+language[\s-]+models?|artificial[\s-]+intelligence(?:[\s-]+\(AI\))?"
    r"[\s-]+(?:tools?|technolog\w+|generated|software|assist\w+)|inteligencia[\s-]+artificial|intelig[êe]ncia[\s-]+artificial)"
    r"|\bAIGC\b|\bLLMs?\b|\bAI[\s-](?:tools?|assist\w*|generat\w*|technolog\w*|based|software)")
BODY_OWN_RE = re.compile(
    r"(?i)\b(?:this|the|our|present) (?:study|work|manuscript|article|paper|research|review|report|publication|case)\b|\bwe\b|"
    r"\bauthors?\b|(?:at|in|during) any (?:stage|part|phase|point)|\bmanuscript\b|research or writing|\bnuestro\b|"
    r"\beste (?:estudio|trabajo|art[íi]culo|manuscrito)\b")
BODY_DECL_RE = re.compile(r"(?i)\b(?:declare|declares|declared|disclose[sd]?|attest|confirm(?:s|ed)?|acknowledge[sd]?|state[sd]?)\b")
BODY_FIG_RE = re.compile(
    r"(?i)(?:figures?|fig\.|images?|illustrations?|graphical abstract|schematics?|artwork|cover|diagrams?|icons?)\b.{0,90}"
    r"(?:generated|created|produced|drawn|designed|made|prepared|rendered)\b.{0,50}(?:by|with|using|via|through)\b")

# publisher-inserted notices that are not author declarations
PUBLISHER_NOTICE_RES = [
    ("frontiers_alt_text", re.compile(
        r"Any alternative text \(alt text\) provided alongside figures in this article has been generated by Frontiers "
        r"with the support of artificial intelligence.*?(?:please contact us\.?|$)", re.I | re.S)),
]

MAX_AI_SECTION_CHARS = 2500
MAX_BODY_AI_SECTION_CHARS = 1500
MAX_PASSAGE_CHARS = 1800
MAX_PASSAGES_PER_ARTICLE = 8

_SPLIT_RE = re.compile(r"(?<=[.!?。！？;])[\"”’)\]]*\s+(?=[A-Z0-9\"“‘(\[ÁÉÍÓÚÀÂÄÖÜÇ¿¡])")
_NOSPLIT_END_RE = re.compile(r"(?:\b(?:e\.g|i\.e|et al|vs|Figs?|Dr|Prof|Inc|Ltd|Corp|Co|No|Nos|v|ver|approx|St|Mr|Mrs|Ms|cf|Eq|Ref|Suppl)\.|\b[A-Z]\.)$")


def split_sentences(text: str) -> list[str]:
    parts = _SPLIT_RE.split(text)
    out: list[str] = []
    for p in parts:
        if out and _NOSPLIT_END_RE.search(out[-1]):
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return [s for s in out if s.strip()]


def strip_notices(text: str) -> tuple[str, list[str]]:
    found = []
    for name, rx in PUBLISHER_NOTICE_RES:
        new = rx.sub(" ", text)
        if new != text:
            found.append(name)
            text = new
    return norm_ws(text), found


def _has_tool(s: str) -> bool:
    return bool(TOOL_RE.search(s)) or bool(AMBIG_NAME_RE.search(s) and AMBIG_CUE_RE.search(s))


def _has_generic(s: str, contrib: bool) -> bool:
    """Generic AI wording. In contribution lists bare 'AI' / 'LLM' may be author initials, so only spelled-out
    expressions and unambiguous collocations ('AI tools', 'No AI', 'AI was not used') count there."""
    if SPELLED_GENERIC_RE.search(s) or AI_COLLOC_RE.search(s):
        return True
    if contrib:
        return False
    if ACRONYM_RE.search(s) or GENERIC_RE.search(s):
        return True
    return bool(BARE_AI_RE.search(s)) and not PERSON_AI_RE.search(s)


def decl_trigger(s: str, contrib: bool, hard_lowpri: bool) -> bool:
    """Does this sentence (in a declaration zone) plausibly state AI use / non-use?"""
    tool = bool(TOOL_RE.search(s))
    if not tool and AMBIG_NAME_RE.search(s):
        # 'Gemini', 'Claude' ... count when a vendor / AI cue is present, or when they are used for writing tasks
        tool = bool(AMBIG_CUE_RE.search(s) or (STRONG_AUTHORING_RE.search(s) and USE_CUE_RE.search(s)))
    if tool:
        return bool(USE_CUE_RE.search(s)) and (not hard_lowpri or bool(AUTHORING_RE.search(s)))
    if _has_generic(s, contrib):
        if hard_lowpri:
            # under Funding / Ethics / Data-availability headings: only clear statements about writing or explicit non-use
            if FUNDING_SENT_RE.search(s) and not HARD_AUTHORING_RE.search(s):
                return False
            return bool(HARD_AUTHORING_RE.search(s) or NEG_USE_RE.search(s))
        if FUNDING_SENT_RE.search(s) and not STRONG_AUTHORING_RE.search(s):
            return False
        return bool(AUTHORING_RE.search(s) or NONUSE_RE.search(s) or DECL_RE.search(s))
    return False


def body_trigger(s: str) -> bool:
    tool = _has_tool(s)
    if not (tool or _has_generic(s, False)):
        return False
    if BODY_WRITE_RE.search(s) and BODY_SELF_RE.search(s):
        return True
    if tool and (BODY_FIG_RE.search(s) or BODY_CODE_RE.search(s)):
        return True
    # short statements of non-use, or explicit declarations, about the authors' own work
    if len(s) <= 400 and BODY_OWN_RE.search(s):
        if BODY_GENAI_RE.search(s) and (NEG_USE_RE.search(s) or BODY_DECL_RE.search(s)):
            return True
        if len(s) <= 220 and BODY_ANY_AI_RE.search(s) and NEG_USE_RE.search(s):
            return True
    return False


def is_heading_like(text: str) -> bool:
    """A paragraph that is only a heading (no statement). 'Declaration of AI use: none' is NOT heading-only."""
    t = text.strip()
    if len(t) > 170 or t.endswith((".", "。")) or not AI_TITLE_RE.search(t):
        return False
    return not re.search(r":\s*\S{2,}", t)


def norm_key(text: str) -> str:
    t = re.sub(r"[\W_]+", " ", text.lower(), flags=re.UNICODE).strip()
    return hashlib.sha1(t.encode("utf-8")).hexdigest()[:20]


def _window(sentences: list[str], idx: list[int], after: int, limit: int) -> str:
    lo, hi = min(idx), min(len(sentences) - 1, max(idx) + after)
    txt = " ".join(sentences[lo:hi + 1])
    if len(txt) > limit:
        keep = sorted({j for i in idx for j in (i, i + 1) if j < len(sentences)})
        txt = " ".join(sentences[j] for j in keep)
    return txt[:limit]


def passages_from_article(rec: dict) -> tuple[list[dict], list[str]]:
    """rec: one line of the block superset. Returns (passages, publisher_notices_found)."""
    blocks = rec.get("blocks") or []
    passages: list[dict] = []
    notices: set[str] = set()
    used = set()

    # ---- route 1a: sections whose own (nearest) heading announces an AI-use declaration ----
    groups: dict = {}
    for i, b in enumerate(blocks):
        if not b.get("ai_sec") or b.get("sid") is None:
            continue
        title = (b.get("t") or [""])[0]
        main_text = b["z"] in ("body", "appendix")
        if not is_ai_title(title, main_text):
            continue  # only an ancestor heading mentions AI -> ordinary block
        if main_text and not STRICT_AI_DECL_TITLE_RE.search(title):
            continue
        groups.setdefault(b["sid"], []).append(i)
    for key, idxs in groups.items():
        zone = blocks[idxs[0]]["z"]
        title = (blocks[idxs[0]].get("t") or [""])[0]
        text = " ".join(blocks[i]["txt"] for i in idxs)
        text, found = strip_notices(text)
        notices.update(found)
        if zone in ("body", "appendix") and len(text) > MAX_BODY_AI_SECTION_CHARS:
            continue  # a research section about AI, not a declaration -> leave its blocks to the body route
        used.update(idxs)
        if not text:
            continue
        passages.append({"route": "ai_section", "zone": zone, "title": title, "text": text[:MAX_AI_SECTION_CHARS],
                         "trunc": int(len(text) > MAX_AI_SECTION_CHARS)})

    # ---- route 1b: an inline heading paragraph followed by the statement paragraph ----
    for i, b in enumerate(blocks):
        if i in used or not b.get("fh") or i == 0:
            continue
        head = blocks[i - 1]
        if (i - 1) in used or not is_heading_like(head["txt"]):
            continue
        text, found = strip_notices(b["txt"])
        notices.update(found)
        used.update((i - 1, i))
        if text:
            passages.append({"route": "ai_section", "zone": b["z"], "title": head["txt"].strip()[:200],
                             "text": text[:MAX_AI_SECTION_CHARS], "trunc": int(len(text) > MAX_AI_SECTION_CHARS)})

    # ---- routes 2 and 3 -----------------------------------------------------
    for i, b in enumerate(blocks):
        if i in used:
            continue
        zone = b["z"]
        titles = b.get("t") or []
        title = titles[0] if titles else ""
        text, found = strip_notices(b["txt"])
        notices.update(found)
        if not text or EXCLUDE_TEXT_RE.search(text[:400]) or is_heading_like(text):
            continue
        sents = split_sentences(text)
        # running text: main body, appendices, numbered endnotes, and titled back-matter that is not a declaration section
        as_body = (zone in ("body", "appendix") or bool(NUMERIC_LABEL_RE.match(title))
                   or (bool(title) and not DECL_TITLE_RE.search(title)))
        if as_body:
            idx = [j for j, s in enumerate(sents) if body_trigger(s)]
            if not idx:
                continue
            ptxt = _window(sents, idx, after=1, limit=1200)
            passages.append({"route": "body", "zone": zone, "title": title, "text": ptxt, "trunc": int(bool(b.get("trunc")))})
        else:
            contrib = (bool(CONTRIB_TITLE_RE.search(title)) or bool(CONTRIB_TEXT_RE.search(text[:60]))
                       or len(INITIALS_LIST_RE.findall(text)) >= 3)
            hard = bool(HARD_LOWPRI_TITLE_RE.search(title))
            idx = [j for j, s in enumerate(sents) if decl_trigger(s, contrib, hard)]
            if not idx:
                continue
            if SOFT_LOWPRI_TITLE_RE.search(title) and not any(
                    AUTHORING_RE.search(sents[j]) or (NONUSE_RE.search(sents[j]) and GENERIC_RE.search(sents[j])) for j in idx):
                continue
            ptxt = text if len(text) <= 900 else _window(sents, idx, after=2, limit=MAX_PASSAGE_CHARS)
            passages.append({"route": "decl_zone", "zone": zone, "title": title, "text": ptxt[:MAX_PASSAGE_CHARS],
                             "trunc": int(bool(b.get("trunc")) or len(ptxt) > MAX_PASSAGE_CHARS)})

    # ---- de-duplicate within article, rank, cap -----------------------------
    order = {"ai_section": 0, "decl_zone": 1, "body": 2}
    seen, out = set(), []
    for p in sorted(passages, key=lambda p: order[p["route"]]):
        k = norm_key(p["text"])
        if k in seen:
            continue
        seen.add(k)
        p["key"] = k
        out.append(p)
    return out[:MAX_PASSAGES_PER_ARTICLE], sorted(notices)
