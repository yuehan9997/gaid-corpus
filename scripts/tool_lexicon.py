"""Deterministic normalisation of tool mentions (name, version) extracted by the annotator.

canonicalise(name, version) -> dict(tool, developer, category, model_version)
The ordered rule table is exported with the dataset (tool_lexicon.csv) so that users can audit or change it.
"""
from __future__ import annotations

import re

LEXICON_VERSION = "1.0"

CAT_LLM = "general-purpose chatbot/LLM"
CAT_WRITE = "writing/language assistant"
CAT_TRANS = "machine translation"
CAT_IMG = "image/figure generator"
CAT_CODE = "coding assistant"
CAT_RES = "research/literature assistant"
CAT_OTHER = "other/unclassified"

# (pattern on the lower-cased, hyphen-normalised name, canonical tool, developer, category) - first match wins
RULES = [
    (r"github\s*co-?pilot", "GitHub Copilot", "GitHub (Microsoft)", CAT_CODE),
    (r"\bcodex\b", "OpenAI Codex", "OpenAI", CAT_CODE),
    (r"claude\s*code", "Claude Code", "Anthropic", CAT_CODE),
    (r"\bcursor\b", "Cursor", "Anysphere", CAT_CODE),
    (r"dall[\s·.\-]*e\b|dalle", "DALL-E", "OpenAI", CAT_IMG),
    (r"\bsora\b", "Sora", "OpenAI", CAT_IMG),
    (r"whisper", "Whisper", "OpenAI", CAT_OTHER),
    (r"chat\s*gpt\s*images?|gpt[\s-]*image", "ChatGPT image generation", "OpenAI", CAT_IMG),
    (r"chat\s*g[pt]{2}|chatgtp|\bgpt[\s-]*(?:\d|o\d|4o)|^gpt$|^gpts$|open\s*ai\b(?!\s*codex)|\bo[134][\s-]*(?:mini|pro|preview)?$",
     "ChatGPT/GPT", "OpenAI", CAT_LLM),
    (r"claude|anthropic", "Claude", "Anthropic", CAT_LLM),
    (r"notebook\s*lm", "NotebookLM", "Google", CAT_RES),
    (r"nano\s*banana|imagen\b", "Gemini image generation", "Google", CAT_IMG),
    (r"google\s*translat", "Google Translate", "Google", CAT_TRANS),
    (r"gemini|\bbard\b|google\s*ai(?:\s*studio)?|^ai studio$", "Gemini", "Google", CAT_LLM),
    (r"co-?\s*pilot|bing\s*(?:chat|ai)", "Microsoft Copilot", "Microsoft", CAT_LLM),
    (r"deep\s*seek", "DeepSeek", "DeepSeek", CAT_LLM),
    (r"deep\s*l\b|deepl", "DeepL", "DeepL SE", CAT_TRANS),
    (r"gramm?[ae]r\w*ly|grammarly", "Grammarly", "Grammarly Inc.", CAT_WRITE),
    (r"quill\s*bot", "QuillBot", "QuillBot (Learneo)", CAT_WRITE),
    (r"paper\s*pal", "Paperpal", "Cactus Communications", CAT_WRITE),
    (r"writefull", "Writefull", "Writefull", CAT_WRITE),
    (r"trini?ka", "Trinka", "Crimson AI", CAT_WRITE),
    (r"wordtune", "Wordtune", "AI21 Labs", CAT_WRITE),
    (r"insta\s*text", "InstaText", "InstaText", CAT_WRITE),
    (r"wordvice", "Wordvice AI", "Wordvice", CAT_WRITE),
    (r"\bcurie\b", "Curie", "Springer Nature", CAT_WRITE),
    (r"rubriq", "Rubriq", "American Journal Experts", CAT_WRITE),
    (r"language\s*tool", "LanguageTool", "LanguageTooler", CAT_WRITE),
    (r"prowritingaid", "ProWritingAid", "ProWritingAid", CAT_WRITE),
    (r"\bginger\b", "Ginger", "Ginger Software", CAT_WRITE),
    (r"smodin", "Smodin", "Smodin", CAT_WRITE),
    (r"jenni", "Jenni AI", "Jenni AI", CAT_WRITE),
    (r"hemingway", "Hemingway Editor", "Hemingway", CAT_WRITE),
    (r"youdao|有道", "Youdao Translate", "NetEase", CAT_TRANS),
    (r"perplexity", "Perplexity", "Perplexity AI", CAT_RES),
    (r"\belicit\b", "Elicit", "Elicit", CAT_RES),
    (r"consensus", "Consensus", "Consensus", CAT_RES),
    (r"sci\s*space", "SciSpace", "SciSpace (Typeset)", CAT_RES),
    (r"\bscite\b", "Scite", "Scite (Research Solutions)", CAT_RES),
    (r"open\s*evidence", "OpenEvidence", "OpenEvidence", CAT_RES),
    (r"research\s*rabbit", "ResearchRabbit", "ResearchRabbit", CAT_RES),
    (r"rayyan", "Rayyan", "Rayyan Systems", CAT_RES),
    (r"scopus\s*ai", "Scopus AI", "Elsevier", CAT_RES),
    (r"scholar\s*gpt|scholarcy|humata|answerthis|undermind|semantic scholar|litmaps|connected papers|paperguide|stork\b|citexs",
     "Other literature assistant", None, CAT_RES),
    (r"julius", "Julius AI", "Julius AI", CAT_RES),
    (r"\bgrok\b", "Grok", "xAI", CAT_LLM),
    (r"\bkimi\b|moonshot", "Kimi", "Moonshot AI", CAT_LLM),
    (r"doubao|豆包", "Doubao", "ByteDance", CAT_LLM),
    (r"\bqwen\b|tongyi|qianwen|通义", "Qwen/Tongyi", "Alibaba", CAT_LLM),
    (r"ernie|wenxin|yiyan|文心", "ERNIE Bot", "Baidu", CAT_LLM),
    (r"zhipu|chatglm|\bglm\b|qingyan", "GLM/Zhipu Qingyan", "Zhipu AI", CAT_LLM),
    (r"hunyuan|yuanbao", "Hunyuan/Yuanbao", "Tencent", CAT_LLM),
    (r"\bllama\b|llama\s*\d|meta\s*ai", "Llama/Meta AI", "Meta", CAT_LLM),
    (r"mistral|le\s*chat", "Mistral/Le Chat", "Mistral AI", CAT_LLM),
    (r"\bmanus\b", "Manus", "Manus", CAT_LLM),
    (r"genspark", "Genspark", "Genspark", CAT_LLM),
    (r"\bsider\b|\bpoe\b|you\.com|monica", "LLM aggregator (Sider/Poe/You.com/Monica)", None, CAT_LLM),
    (r"midjourney", "Midjourney", "Midjourney", CAT_IMG),
    (r"stable\s*diffusion", "Stable Diffusion", "Stability AI", CAT_IMG),
    (r"firefly", "Adobe Firefly", "Adobe", CAT_IMG),
    (r"canva", "Canva AI", "Canva", CAT_IMG),
    (r"napkin", "Napkin AI", "Napkin AI", CAT_IMG),
    (r"figure\s*labs", "FigureLabs", "FigureLabs", CAT_IMG),
    (r"biorender", "BioRender AI", "BioRender", CAT_IMG),
    (r"gamma", "Gamma", "Gamma Tech", CAT_IMG),
    (r"\w+gpt\b|\bgpt\s+\w+$", "Institutional or third-party GPT deployment", None, CAT_LLM),
]
_COMPILED = [(re.compile(p, re.I), t, d, c) for p, t, d, c in RULES]

_GPT_VER = re.compile(
    r"(?:gpt|chatgpt|chat gpt|version|v\.?|model)[\s\-]*((?:3\.5|4o|4\.1|4\.5|5\.\d|4|5|o1|o3|o4)(?:[\s-]*(?:mini|turbo|pro|thinking|instant|sol|luna|terra|nano))?)\b", re.I)
_GPT_VER_BARE = re.compile(r"^\W*((?:3\.5|4o|4\.1|4\.5|5\.\d|4\.0|4|5|o1|o3|o4)(?:[\s-]*(?:mini|turbo|pro|thinking|instant|sol|luna|terra|nano))?)\W*$", re.I)


def _norm(s: str) -> str:
    return re.sub(r"[‐-―−]", "-", s or "").strip()


def canonicalise(name: str, version: str | None = None) -> dict:
    n = _norm(name)
    v = _norm(version) if version else None
    tool, dev, cat = None, None, CAT_OTHER
    for rx, t, d, c in _COMPILED:
        if rx.search(n):
            tool, dev, cat = t, d, c
            break
    if tool is None and v:  # e.g. name="Generative AI tools", version="GPT-5.6 Sol"
        for rx, t, d, c in _COMPILED:
            if rx.search(v):
                tool, dev, cat = t, d, c
                break
    mv = None
    if tool == "ChatGPT/GPT":
        s = f"{n} {v or ''}"
        m = _GPT_VER.search(s) or (_GPT_VER_BARE.match(v) if v else None)
        if m:
            mv = "GPT-" + re.sub(r"\s+", " ", m.group(1).lower().replace("4.0", "4")).replace("- ", "-")
            mv = re.sub(r"^GPT-(o\d)", r"\1", mv)
    return {"tool": tool or n, "developer": dev, "category": cat, "mapped": int(tool is not None), "model_version": mv}


if __name__ == "__main__":
    for name, ver in [("ChatGPT", "GPT-4o"), ("ChatGPT", "4o"), ("GPT-5.5", None), ("Chat GPT-5.5", None), ("ChatGTP", None),
                      ("Microsoft Copilot", None), ("GitHub Copilot", None), ("Claude", "Sonnet 4.6"), ("DeepL Write", None),
                      ("Grammaryly", None), ("Google Gemini", "2.5 Pro"), ("DukeGPT", None), ("OpenAI", None), ("Nano Banana Pro", None),
                      ("Generative AI tools", "GPT-5.6 Sol"), ("ChatGPT", "version 3.5"), ("Curie", None), ("FooBar", None)]:
        print(f"{name!r:28} {ver!r:14} -> {canonicalise(name, ver)}")
