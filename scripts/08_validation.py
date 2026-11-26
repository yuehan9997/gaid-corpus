"""Step 8. Label-quality validation.

  sample        stratified random sample of unique candidate passages (by capture route) -> validation_sample.jsonl
  sample_extra  additional stratum: passages added by rules v1.3
  human_tool    self-contained HTML annotation tool for the human subset (blind to model labels)
  check_excel   check the expert workbooks (12_export_excel.py) placed in data/validation/human/
  refs M...     annotate the sample with strong reference models (needs API key)
  second        independent second annotator on a random subsample (needs API key)
  retest        re-annotate a random subsample of 5,000 texts with the primary model (needs API key)
  context       re-label texts that occur under more than one capture route with each other route (needs API key)
  report        agreement statistics -> validation_results.json
"""
from __future__ import annotations

import asyncio
import collections
import json
import math
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annot_schema import CODEBOOK_VERSION, POLARITIES, PURPOSES, SYSTEM_PROMPT, build_user, parse_response  # noqa: E402
from common import INTERIM, ROOT, VALID, read_jsonl  # noqa: E402
from tool_lexicon import canonicalise  # noqa: E402

SAMPLE = VALID / "validation_sample.jsonl"
N_PER_ROUTE = {"ai_section": 200, "decl_zone": 250, "body": 150}
HUMAN_PER_ROUTE = {"ai_section": 100, "decl_zone": 125, "body": 75}
from models import MAX_TOKENS, PRIMARY, REF_MODELS, SECOND, extra_for  # noqa: E402


def cmd_sample(seed=20261114):
    rng = random.Random(seed)
    by_route = collections.defaultdict(list)
    for u in read_jsonl(INTERIM / "unique_texts.jsonl.gz"):
        by_route[u["loc"]].append(u)
    out = []
    for route, n in N_PER_ROUTE.items():
        pool = sorted(by_route[route], key=lambda u: u["key"])
        pick = rng.sample(pool, n)
        hum = set(u["key"] for u in rng.sample(pick, HUMAN_PER_ROUTE[route]))
        for u in pick:
            out.append({"key": u["key"], "loc": u["loc"], "heading": u["heading"], "text": u["text"], "n": u["n"],
                        "human_subset": int(u["key"] in hum), "stratum_size": len(pool)})
        print(f"{route}: population {len(pool)} unique texts, sampled {n} ({HUMAN_PER_ROUTE[route]} for human annotation)")
    rng.shuffle(out)
    for i, r in enumerate(out, 1):
        r["id"] = i
    with open(SAMPLE, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", SAMPLE, len(out))


def cmd_sample_extra(n=100, n_human=50, seed=20261115):
    """Rules v1.3 added passages after the first sample was drawn. Add a random sample of those additions
    (distinct texts not present under rules v1.2) as a fourth stratum. Must run BEFORE the additions are annotated."""
    rng = random.Random(seed)
    have = {s["key"] for s in load_sample()}
    if any(s.get("stratum") == "v13_additions" for s in load_sample()):
        print("extra stratum already present")
        return
    annotated = set()
    for l in open(INTERIM / "annot_primary.jsonl", encoding="utf-8"):
        annotated.add(json.loads(l)["key"])
    pool = sorted((u for u in read_jsonl(INTERIM / "unique_texts.jsonl.gz") if u["key"] not in annotated and u["key"] not in have),
                  key=lambda u: u["key"])
    pick = rng.sample(pool, n)
    hum = set(u["key"] for u in rng.sample(pick, n_human))
    rows = load_sample()
    for u in pick:
        rows.append({"key": u["key"], "loc": u["loc"], "heading": u["heading"], "text": u["text"], "n": u["n"],
                     "human_subset": int(u["key"] in hum), "stratum_size": len(pool), "stratum": "v13_additions", "id": len(rows) + 1})
    with open(SAMPLE, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"added {n} passages ({n_human} for human annotation) from {len(pool)} texts new under rules v1.3; sample now {len(rows)}")


def load_sample():
    return [json.loads(l) for l in open(SAMPLE, encoding="utf-8")]


# ------------------------------------------------------------------ model runs
async def run_model_on(items, model, outp, tag, batch=8, conc=6, max_tokens=8000, extra=None):
    from llm_lib import LLM
    done = {}
    if outp.exists():
        for l in open(outp, encoding="utf-8"):
            r = json.loads(l)
            done[r["key"]] = r
    todo = [it for it in items if it["key"] not in done]
    if not todo:
        print(model, "complete")
        return
    batches = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    fh = open(outp, "a", encoding="utf-8")
    extra = extra if extra is not None else extra_for(model)
    max_tokens = MAX_TOKENS.get(model, max_tokens)
    print(f"{model}: {len(todo)} items, batch {batch}, conc {conc}, max_tokens {max_tokens}, params {extra}", flush=True)
    async with LLM(model, tag, conc=conc, max_tokens=max_tokens, extra=extra, timeout=600) as llm:
        async def do(b):
            its = [{"id": i + 1, "loc": r["loc"], "heading": (r.get("heading") or "")[:200], "text": r["text"][:2600]} for i, r in enumerate(b)]
            for _ in range(3):
                content, _u = await llm.chat(SYSTEM_PROMPT, build_user(its))
                got = parse_response(content, [x["id"] for x in its])
                if len(got) == len(its):
                    break
            for i, r in enumerate(b):
                if (i + 1) in got:
                    g = got[i + 1]
                    g.pop("id", None)
                    fh.write(json.dumps({"key": r["key"], **g, "model": model, "cb": CODEBOOK_VERSION}, ensure_ascii=False) + "\n")
            fh.flush()
        await asyncio.gather(*[do(b) for b in batches])
        print(f"{model}: calls={llm.acc['calls']} usd={llm.acc['usd']:.3f} errors={llm.acc['errors']}")
    fh.close()


def cmd_refs(models):
    items = load_sample()
    for m in models or REF_MODELS:
        asyncio.run(run_model_on(items, m, VALID / f"ref_{m}.jsonl", "valid_ref", batch=8, conc=8))


def cmd_second(n=10000, seed=7):
    rng = random.Random(seed)
    rows = sorted(read_jsonl(INTERIM / "unique_texts.jsonl.gz"), key=lambda u: u["key"])
    pick = rng.sample(rows, n)
    (VALID / "second_keys.txt").write_text("\n".join(u["key"] for u in pick))
    asyncio.run(run_model_on(pick, SECOND, VALID / f"second_{SECOND}.jsonl", "valid_second", batch=16, conc=12))


def context_items():
    """Each distinct text is labelled once, with the capture route and heading most frequent among its passages. For texts
    that also occur under another capture route, one passage per other route is labelled again with its own route and
    heading (the most frequent heading of that route)."""
    uni = {u["key"]: u for u in read_jsonl(INTERIM / "unique_texts.jsonl.gz")}
    by = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for p in read_jsonl(INTERIM / "passages.jsonl.gz"):
        by[p["key"]][p["route"]][p["title"] or ""] += 1
    items = []
    for k, routes in by.items():
        if len(routes) < 2 or k not in uni:
            continue
        for r, heads in routes.items():
            if r == uni[k]["loc"]:
                continue
            items.append({"key": f"{k}|{r}", "base": k, "loc": r, "heading": heads.most_common(1)[0][0],
                          "text": uni[k]["text"], "n": sum(heads.values())})
    return sorted(items, key=lambda x: x["key"])


def cmd_context():
    items = context_items()
    print(f"{len(items)} (text, other route) pairs covering {sum(i['n'] for i in items)} passages")
    asyncio.run(run_model_on(items, PRIMARY, VALID / f"context_{PRIMARY}.jsonl", "valid_context", batch=16, conc=12))


def context_report(primary: dict) -> dict | None:
    f = VALID / f"context_{PRIMARY}.jsonl"
    if not f.exists():
        return None
    lab = load_labels(f)
    items = [i for i in context_items() if i["key"] in lab and i["base"] in primary]
    if not items:
        return None
    d_same = [bool(lab[i["key"]]["decl"]) == bool(primary[i["base"]]["decl"]) for i in items]
    both = [i for i in items if lab[i["key"]]["decl"] and primary[i["base"]]["decl"]]
    p_same = [lab[i["key"]].get("pol") == primary[i["base"]].get("pol") for i in both]
    return {"pairs": len(items), "distinct_texts": len({i["base"] for i in items}), "passages": sum(i["n"] for i in items),
            "decl_agree": sum(d_same) / len(items), "decl_differ_pairs": len(items) - sum(d_same),
            "decl_differ_passages": sum(i["n"] for i, s in zip(items, d_same) if not s),
            "pol_agree": (sum(p_same) / len(p_same)) if p_same else None, "pol_n": len(p_same)}


def weighted_decl(sample, A: dict, B: dict) -> dict:
    """Precision and recall of A against B for the declaration decision, with each stratum weighted by its size
    (number of distinct texts) instead of its sample size."""
    cell = collections.defaultdict(collections.Counter)
    size = {}
    for s in sample:
        k = s["key"]
        if k not in A or k not in B:
            continue
        st = s.get("stratum") or s["loc"]
        size[st] = s["stratum_size"]
        cell[st][(bool(A[k]["decl"]), bool(B[k]["decl"]))] += 1
    tp = fp = fn = 0.0
    for st, c in cell.items():
        w = size[st] / sum(c.values())
        tp += w * c[(True, True)]
        fp += w * c[(True, False)]
        fn += w * c[(False, True)]
    return {"precision": tp / (tp + fp), "recall": tp / (tp + fn), "strata": {st: size[st] for st in cell}}


def cmd_retest(n=5000, seed=8):
    rng = random.Random(seed)
    rows = sorted(read_jsonl(INTERIM / "unique_texts.jsonl.gz"), key=lambda u: u["key"])
    pick = rng.sample(rows, n)
    (VALID / "retest_keys.txt").write_text("\n".join(u["key"] for u in pick))
    asyncio.run(run_model_on(pick, PRIMARY, VALID / f"retest_{PRIMARY}.jsonl", "valid_retest", batch=16, conc=12))


# ------------------------------------------------------------------ agreement statistics
def tool_set(r):
    return frozenset(canonicalise(t[0], t[1])["tool"] for t in (r.get("tools") or []))


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def compare(A: dict, B: dict, keys=None) -> dict:
    """Agreement between two labelings {key: label-dict}. Field metrics other than `decl` use items both call declarations."""
    ks = [k for k in (keys or A) if k in A and k in B]
    out = {"n": len(ks)}
    if not ks:
        return out
    da, db = [A[k]["decl"] for k in ks], [B[k]["decl"] for k in ks]
    tp = sum(x == 1 and y == 1 for x, y in zip(da, db))
    out["decl"] = {"agree": sum(x == y for x, y in zip(da, db)) / len(ks), "kappa": kappa(da, db),
                   "precision_A_vs_B": tp / max(1, sum(da)), "recall_A_vs_B": tp / max(1, sum(db))}
    pos = [k for k in ks if A[k]["decl"] == 1 and B[k]["decl"] == 1]
    out["n_both_decl"] = len(pos)
    if not pos:
        return out
    pa_, pb_ = [A[k]["pol"] for k in pos], [B[k]["pol"] for k in pos]
    out["pol"] = {"agree": sum(x == y for x, y in zip(pa_, pb_)) / len(pos), "kappa": kappa(pa_, pb_)}
    for f in ("resp", "rev", "nonuse"):
        fa, fb = [A[k][f] for k in pos], [B[k][f] for k in pos]
        out[f] = {"agree": sum(x == y for x, y in zip(fa, fb)) / len(pos), "kappa": kappa(fa, fb)}
    use = [k for k in pos if A[k]["pol"] == "use" and B[k]["pol"] == "use"]
    out["n_both_use"] = len(use)
    if use:
        per = {}
        TP = FP = FN = 0
        for p in PURPOSES:
            a = [int(p in A[k]["purp"]) for k in use]
            b = [int(p in B[k]["purp"]) for k in use]
            tp = sum(x and y for x, y in zip(a, b))
            fp = sum(x and not y for x, y in zip(a, b))
            fn = sum((not x) and y for x, y in zip(a, b))
            TP, FP, FN = TP + tp, FP + fp, FN + fn
            per[p] = {"f1": 2 * tp / max(1, 2 * tp + fp + fn), "kappa": kappa(a, b), "n_B": sum(b)}
        out["purp"] = {"exact": sum(A[k]["purp"] == B[k]["purp"] for k in use) / len(use),
                       "micro_f1": 2 * TP / max(1, 2 * TP + FP + FN), "per_label": per}
        ta, tb = [tool_set(A[k]) for k in use], [tool_set(B[k]) for k in use]
        tp = sum(len(x & y) for x, y in zip(ta, tb))
        fp = sum(len(x - y) for x, y in zip(ta, tb))
        fn = sum(len(y - x) for x, y in zip(ta, tb))
        out["tools"] = {"exact": sum(x == y for x, y in zip(ta, tb)) / len(use), "micro_f1": 2 * tp / max(1, 2 * tp + fp + fn)}
    return out


CI_FIELDS = {"decl": "decl.kappa", "pol": "pol.kappa", "resp": "resp.kappa", "rev": "rev.kappa", "nonuse": "nonuse.kappa",
             "purp": "purp.micro_f1", "tools": "tools.micro_f1"}


def _path(d, path):
    for part in path.split("."):
        d = d.get(part) if isinstance(d, dict) else None
    return d


def boot_ci(A: dict, B: dict, keys, reps=1000, seed=20261122) -> dict:
    """Percentile 95% bootstrap intervals (resampling passages) for the agreement statistics reported by compare()."""
    ks = [k for k in keys if k in A and k in B]
    if not ks:
        return {}
    rng = random.Random(seed)
    vals = {f: [] for f in CI_FIELDS}
    for _ in range(reps):
        c = compare(A, B, [ks[rng.randrange(len(ks))] for _ in ks])
        for f, path in CI_FIELDS.items():
            v = _path(c, path)
            if isinstance(v, float) and not math.isnan(v):
                vals[f].append(v)
    out = {}
    for f, v in vals.items():
        if len(v) >= 0.9 * reps:
            v.sort()
            out[f] = [v[int(0.025 * (len(v) - 1))], v[int(round(0.975 * (len(v) - 1)))]]
    out["reps"] = reps
    return out


def majority(labelings: list[dict], keys) -> dict:
    cons = {}
    for k in keys:
        rs = [L[k] for L in labelings if k in L]
        if len(rs) < 2:
            continue
        c = {f: collections.Counter(r[f] for r in rs).most_common(1)[0][0] for f in ("decl", "pol", "resp", "rev", "nonuse")}
        c["purp"] = [p for p in PURPOSES if sum(p in r["purp"] for r in rs) * 2 > len(rs)]
        ts = collections.Counter(tool_set(r) for r in rs).most_common(1)[0][0]
        c["tools"] = [[t, None] for t in sorted(ts)]
        if c["decl"] == 0:
            c.update({"pol": None, "purp": [], "tools": [], "resp": 0, "rev": 0, "nonuse": 0})
        cons[k] = c
    return cons


def load_labels(path) -> dict:
    out = {}
    if Path(path).exists():
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            out[r["key"]] = r
    return out


def load_human(path) -> dict:
    """Export of the HTML tool: {annotator, items:[{key, decl, pol, tools:'a; b', purp:[...], resp, rev, nonuse}]}"""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for it in d.get("items", []):
        if it.get("decl") is None:
            continue
        tools = [[t.strip(), None] for t in (it.get("tools") or "").replace("\n", ";").split(";") if t.strip()]
        decl = int(it["decl"])
        out[it["key"]] = {"decl": decl, "pol": it.get("pol") if decl else None, "tools": tools if decl else [],
                          "purp": sorted(it.get("purp") or [], key=PURPOSES.index) if decl else [],
                          "resp": int(bool(it.get("resp"))) if decl else 0, "rev": int(bool(it.get("rev"))) if decl else 0,
                          "nonuse": int(bool(it.get("nonuse"))) if decl else 0}
    return out


_TRUE = {"1", "1.0", "y", "yes", "true", "x", "√", "✓", "是"}
_FALSE = {"0", "0.0", "n", "no", "false", "否"}


def _cell_bool(v):
    if v is None or str(v).strip() == "":
        return None
    s = str(v).strip().lower()
    return 1 if s in _TRUE else 0 if s in _FALSE else "invalid"


def load_human_xlsx(path, verbose=False) -> tuple[dict, list]:
    """Workbook exported by 12_export_excel.py and filled in by an expert. Returns (labels by key, problems)."""
    from openpyxl import load_workbook
    idmap = json.loads((VALID / "human_excel_map.json").read_text(encoding="utf-8"))
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(wb.worksheets[0].iter_rows(values_only=True))
    finally:
        wb.close()
    rows = iter(rows)
    header = [str(h or "").split("\n")[0].strip() for h in next(rows)]
    ix = {h: i for i, h in enumerate(header)}
    need = ["编号", "decl", "pol", "tools", "resp", "rev", "nonuse"] + PURPOSES
    missing_cols = [c for c in need if c not in ix]
    if missing_cols:
        raise ValueError(f"{path}: missing columns {missing_cols}")
    out, problems, seen, empty = {}, [], set(), []
    for row in rows:
        if row is None or row[ix["编号"]] in (None, ""):
            continue
        no = str(int(float(row[ix["编号"]])))
        key = idmap.get(no)
        if key is None:
            problems.append(f"#{no}: unknown item number")
            continue
        seen.add(no)
        decl = _cell_bool(row[ix["decl"]])
        if decl is None:
            empty.append(no)
            continue
        if decl == "invalid":
            problems.append(f"#{no}: decl invalid ({row[ix['decl']]!r})")
            continue
        if decl == 0:
            out[key] = {"decl": 0, "pol": None, "tools": [], "purp": [], "resp": 0, "rev": 0, "nonuse": 0}
            continue
        pol = str(row[ix["pol"]] or "").strip().lower().replace("-", "_").replace(" ", "_")
        if pol not in POLARITIES:
            problems.append(f"#{no}: decl=1 but pol {'empty' if not pol else 'invalid (' + pol + ')'}")
            continue
        purp, flags = [], {}
        for p in PURPOSES:
            b = _cell_bool(row[ix[p]])
            if b == "invalid":
                problems.append(f"#{no}: {p} invalid ({row[ix[p]]!r})")
            if b == 1:
                purp.append(p)
        for f in ("resp", "rev", "nonuse"):
            b = _cell_bool(row[ix[f]])
            if b == "invalid":
                problems.append(f"#{no}: {f} invalid ({row[ix[f]]!r})")
            flags[f] = 1 if b == 1 else 0
        tools_txt = str(row[ix["tools"]] or "")
        tools = [[t.strip(), None] for t in tools_txt.replace("；", ";").replace("\n", ";").split(";") if t.strip()]
        if pol == "use" and not purp:
            problems.append(f"#{no}: pol=use but no purpose marked")
        if pol != "use":
            if purp or tools or flags["nonuse"]:
                problems.append(f"#{no}: pol={pol} but purposes/tools/nonuse filled (ignored)")
            purp, tools, flags["nonuse"] = [], [], 0
        if len(purp) > 1 and "UNSPEC" in purp:
            problems.append(f"#{no}: UNSPEC together with other purposes (UNSPEC dropped)")
            purp.remove("UNSPEC")
        out[key] = {"decl": 1, "pol": pol, "tools": tools, "purp": purp, **flags}
    if empty:
        problems.insert(0, f"{len(empty)} items not yet labelled (decl empty), e.g. #{', #'.join(empty[:10])}")
    not_done = sorted((set(idmap) - seen), key=int)
    if not_done:
        problems.append(f"{len(not_done)} item numbers missing from the sheet (first: {not_done[:10]})")
    if verbose:
        print(f"{Path(path).name}: {len(out)} of {len(idmap)} items usable; {len(problems)} problems")
        for pr in problems[:200]:
            print("   ", pr)
    return out, problems


def human_files():
    hdir = VALID / "human"
    if not hdir.exists():
        return []
    return sorted(list(hdir.glob("*.json")) + [p for p in hdir.glob("*.xlsx") if not p.name.startswith("~$")])


def cmd_check_excel(paths):
    for p in paths or [str(f) for f in human_files() if f.suffix == ".xlsx"]:
        load_human_xlsx(p, verbose=True)


def cmd_report():
    sample = load_sample()
    keys = [s["key"] for s in sample]
    route = {s["key"]: s["loc"] for s in sample}
    primary = load_labels(INTERIM / "annot_primary.jsonl")
    res = {"sample_n": len(sample), "codebook": CODEBOOK_VERSION}
    refs = {m: load_labels(VALID / f"ref_{m}.jsonl") for m in REF_MODELS}
    refs = {m: L for m, L in refs.items() if L}
    if len(refs) >= 2:
        cons = majority(list(refs.values()), keys)
        res["reference_models"] = list(refs)
        res["primary_vs_reference"] = compare(primary, cons, keys)
        res["primary_vs_reference_by_route"] = {r: compare(primary, cons, [k for k in keys if route[k] == r]) for r in N_PER_ROUTE}
        res["primary_vs_reference_weighted"] = weighted_decl(sample, primary, cons)
        ms = list(refs)
        res["reference_pairwise"] = {f"{ms[i]}|{ms[j]}": compare(refs[ms[i]], refs[ms[j]], keys)
                                     for i in range(len(ms)) for j in range(i + 1, len(ms))}
    res["models"] = {"primary": PRIMARY, "second": SECOND, "reference": REF_MODELS}
    current = [u["key"] for u in read_jsonl(INTERIM / "unique_texts.jsonl.gz")]   # distinct passages of the final rule version
    res["n_distinct_passages"] = len(current)
    second = load_labels(INTERIM / "annot_second.jsonl") or load_labels(VALID / f"second_{SECOND}.jsonl")
    if second:
        res["primary_vs_second"] = compare(primary, second, current)
        res["second_vs_reference"] = compare(second, cons, keys) if len(refs) >= 2 else None
    retest = load_labels(VALID / f"retest_{PRIMARY}.jsonl")
    if retest:
        res["primary_test_retest"] = compare(primary, retest, current)
    ctx = context_report(primary)
    if ctx:
        res["context_check"] = ctx
    humans = human_files()
    H = [load_human_xlsx(p)[0] if p.suffix == ".xlsx" else load_human(p) for p in humans]
    if H:
        names = [chr(65 + i) for i in range(len(humans))]          # annotators are reported as A, B, ...
        res["human_annotators"] = {nm: p.name for nm, p in zip(names, humans)}
        hk = [s["key"] for s in sample if s["human_subset"]]
        res["human_items_labelled"] = {nm: sum(1 for k in hk if k in h) for nm, h in zip(names, H)}
        if len(H) >= 2:
            res["human_inter_annotator"] = compare(H[0], H[1], hk)
            gold = {k: H[0][k] for k in hk if k in H[0] and k in H[1] and H[0][k] == H[1][k]}
            res["human_full_agreement_items"] = len(gold)
        res["primary_vs_human"] = {nm: compare(primary, h, hk) for nm, h in zip(names, H)}
        if len(refs) >= 2:
            res["reference_vs_human"] = {nm: compare(cons, h, hk) for nm, h in zip(names, H)}
    # 95% bootstrap intervals (fewer resamples for the large comparisons, whose intervals are very narrow)
    ci = {}
    if len(refs) >= 2:
        ci["primary_vs_reference"] = boot_ci(primary, cons, keys)
    if H:
        for nm, h in zip(names, H):
            ci[f"primary_vs_human.{nm}"] = boot_ci(primary, h, hk)
        if len(H) >= 2:
            ci["human_inter_annotator"] = boot_ci(H[0], H[1], hk)
    if retest:
        ci["primary_test_retest"] = boot_ci(primary, retest, [k for k in current if k in retest], reps=300)
    if second:
        ci["primary_vs_second"] = boot_ci(primary, second, current, reps=100)
    res["ci95"] = ci
    (VALID / "validation_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1)[:6000])


def cmd_human_tool():
    import html
    items = [{"key": s["key"], "loc": s["loc"], "heading": s["heading"], "text": s["text"]} for s in load_sample() if s["human_subset"]]
    random.Random(99).shuffle(items)
    guide = SYSTEM_PROMPT.split("Return a single minified JSON object")[0]
    guide = guide.replace("You are annotating passages", "You will annotate passages").replace(
        "For each item, output one JSON object with the fields below.", "For each item, fill in the fields below.")
    tpl = (Path(__file__).parent / "human_tool_template.html").read_text(encoding="utf-8")
    page = (tpl.replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
            .replace("__CODEBOOK__", html.escape(guide.strip())).replace("__CBVER__", CODEBOOK_VERSION))
    out = ROOT / "annotation" / "annotate_gaid.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(page, encoding="utf-8")
    (VALID / "human").mkdir(exist_ok=True)
    print(f"wrote {out} with {len(items)} items; put exported files into {VALID / 'human'}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check_excel":
        cmd_check_excel(sys.argv[2:])
    elif cmd == "human_tool":
        cmd_human_tool()
    elif cmd == "sample_extra":
        cmd_sample_extra()
    elif cmd == "sample":
        cmd_sample()
    elif cmd == "refs":
        cmd_refs(sys.argv[2:])
    elif cmd == "second":
        cmd_second()
    elif cmd == "retest":
        cmd_retest()
    elif cmd == "context":
        cmd_context()
    elif cmd == "report":
        cmd_report()
