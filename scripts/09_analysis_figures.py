"""Step 9. Descriptive statistics (paper/stats.json) and figures (paper/figures/*.png, *.pdf) from the release tables.

  python -I 09_analysis_figures.py                  # figures 1-4
  python -I 09_analysis_figures.py --all-figures    # also the publisher and quarterly-trend figures
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annot_schema import PURPOSES  # noqa: E402
from common import RELEASE, ROOT, VALID, read_jsonl  # noqa: E402

PAPER = ROOT / "paper"
FIG = PAPER / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---- figure style ---------------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
plt.rcParams.update({
    "font.family": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.labelsize": 8.5, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8, "text.color": INK,
    "axes.titlelocation": "left", "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "axes.axisbelow": True,
})


def year_axis(ax):
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(4, 7, 10)))
    ax.tick_params(axis="x", which="minor", length=2, color=AXIS)
PURPOSE_LABEL = {"LANG": "Language editing", "TRANS": "Translation", "DRAFT": "Text generation", "LIT": "Literature search / synthesis",
                 "IDEA": "Ideation / feedback", "CODE": "Code", "DATA": "Data analysis", "VIS": "Figures / images",
                 "FORMAT": "Formatting / references", "OTHER": "Other stated purpose", "UNSPEC": "Purpose not specified"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("  figure", name)


def pct(x, pos=None):
    return f"{x:.0f}%" if x >= 10 or x == 0 else f"{x:.1f}%"


def main():
    summ = json.loads((RELEASE / "build_summary.json").read_text())
    st = pd.read_parquet(RELEASE / "gaid_statements.parquet")
    ar = pd.read_parquet(RELEASE / "gaid_articles.parquet")
    tl = pd.read_parquet(RELEASE / "gaid_statement_tools.parquet")
    dp = pd.read_parquet(RELEASE / "gaid_denominators_publisher_month.parquet")
    dj = pd.read_parquet(RELEASE / "gaid_denominators_journal_month.parquet")
    sc = pd.read_parquet(RELEASE / "gaid_screened_articles.parquet")
    partial = summ["annotation_coverage_passages"] < 0.999 or summ["index_months_done"] < 46
    S = {"partial_data": bool(partial), "build": summ}

    # ---------------- headline numbers --------------------------------------------------------
    use = st[st.polarity == "use"]
    S["n_statements"] = int(len(st))
    S["n_articles"] = int(len(ar))
    S["statement_polarity"] = st.polarity.value_counts().to_dict()
    S["article_polarity"] = ar.polarity.value_counts().to_dict()
    S["n_journals"] = int(ar.journal.nunique())
    S["n_publisher_brands"] = int(ar.publisher_brand.nunique())
    S["languages"] = ar.language.fillna("unknown").value_counts().head(8).to_dict()
    S["capture_route"] = st.capture_route.value_counts().to_dict()
    S["location"] = st.location.value_counts().to_dict()
    S["has_dedicated_section_share"] = float(ar.has_dedicated_section.mean())
    S["polarity_conflict_articles"] = int(ar.polarity_conflict.sum())
    S["corrections_retractions"] = int(ar.is_correction_or_retraction.sum())
    S["license_statements"] = st.source_license.fillna("none").value_counts().head(8).to_dict()
    S["text_withheld"] = int(st.text_withheld.sum())
    S["unique_statement_texts"] = int(st.text_key.nunique())
    top_tpl = st.groupby("text_key").size().sort_values(ascending=False)
    S["top5_template_share"] = float(top_tpl.head(5).sum() / len(st))
    S["singleton_text_share"] = float((top_tpl == 1).sum() / len(st))
    S["statement_chars_median_use"] = float(use.statement_text.str.len().median()) if len(use) else None
    S["use_statements"] = int(len(use))
    S["use_named_tool_share"] = float((use.n_tools > 0).mean()) if len(use) else None
    S["use_version_share"] = float((use.gpt_model_versions.fillna("") != "").mean()) if len(use) else None
    S["use_resp_share"] = float(use.responsibility.mean()) if len(use) else None
    S["use_rev_share"] = float(use.human_review.mean()) if len(use) else None
    S["use_nonuse_share"] = float(use.nonuse_scope.mean()) if len(use) else None
    S["purpose_share_use_statements"] = {p: float(use["purpose_" + p].mean()) for p in PURPOSES} if len(use) else {}
    S["mean_purposes_per_use_statement"] = float(use[["purpose_" + p for p in PURPOSES]].sum(axis=1).mean()) if len(use) else None
    S["article_types"] = ar.article_type.fillna("unknown").value_counts().head(10).to_dict()
    if "annotators_agree" in st.columns and st.annotators_agree.notna().any():
        S["annotators_agree_share"] = float(st.annotators_agree.dropna().mean())
        S["annotators_agree_n"] = int(st.annotators_agree.notna().sum())

    # tools (article level, among articles declaring use)
    ar_use = ar[ar.polarity == "use"]
    tool_counts = collections.Counter()
    for t in ar_use.tools.fillna(""):
        for x in filter(None, t.split("|")):
            tool_counts[x] += 1
    S["n_distinct_tools"] = len(tool_counts)
    S["top_tools_use_articles"] = [(k, int(v), v / max(1, len(ar_use))) for k, v in tool_counts.most_common(20)]
    S["tool_mentions"] = int(len(tl))
    S["tool_lexicon_mapped_share"] = float(tl.lexicon_mapped.mean()) if len(tl) else None
    S["tool_name_verbatim_share"] = float(tl.name_verbatim_in_text.mean()) if len(tl) else None
    S["tool_category_mentions"] = tl.category.value_counts().to_dict() if len(tl) else {}

    # ---------------- prevalence over time (needs denominators) ---------------------------------
    m = dp.groupby("month")[["n_oa_articles", "n_candidates", "n_declaring", "n_use", "n_no_use"]].sum().sort_index()
    m = m[m.index.str.len() == 7]
    # a handful of records are returned by the date-restricted query with a first-publication date outside the window
    in_window = (m.index >= "2023-01") & (m.index <= "2026-10")
    # (07 drops them from the denominator tables and counts them in the build summary)
    S["n_index_records_outside_window"] = int(summ.get("n_index_records_outside_window", m[~in_window].n_oa_articles.sum()))
    m = m[in_window]
    S["months_with_denominator"] = int(len(m))
    S["n_oa_articles"] = int(m.n_oa_articles.sum())
    S["prevalence_any_overall"] = float(m.n_declaring.sum() / max(1, m.n_oa_articles.sum()))
    S["prevalence_use_overall"] = float(m.n_use.sum() / max(1, m.n_oa_articles.sum()))
    yr = m.groupby(m.index.str[:4]).sum()
    S["by_year"] = {y: {"n_oa": int(r.n_oa_articles), "declaring": int(r.n_declaring), "use": int(r.n_use), "no_use": int(r.n_no_use),
                        "prev_any": float(r.n_declaring / max(1, r.n_oa_articles)), "prev_use": float(r.n_use / max(1, r.n_oa_articles))}
                    for y, r in yr.iterrows()}
    # months whose deposit in PMC is still incomplete: fewer articles than 90% of the median of the preceding 12 months
    med12 = m.n_oa_articles.shift(1).rolling(12, min_periods=12).median()
    inc = (m.n_oa_articles < 0.9 * med12) & (m.index >= m.index[-1][:4] + "-01")
    # only a trailing run of such months counts (an isolated low month earlier is not a deposit lag)
    trail = []
    for mo in reversed(m.index):
        if not inc.get(mo, False):
            break
        trail.append(mo)
    S["incomplete_months"] = sorted(trail)
    S["incomplete_months_share_of_median"] = {mo: float(m.n_oa_articles[mo] / med12[mo]) for mo in trail}
    if trail:
        full = m[(m.index >= trail[0][:4] + "-01") & (m.index < min(trail))]
        S["prev_any_last_year_complete_months"] = {"from": full.index[0], "to": full.index[-1],
                                                   "prev_any": float(full.n_declaring.sum() / full.n_oa_articles.sum())}
    if len(m):
        last = m.iloc[-1]
        S["last_month"] = {"month": m.index[-1], "prev_any": float(last.n_declaring / max(1, last.n_oa_articles)),
                           "prev_use": float(last.n_use / max(1, last.n_oa_articles))}
    pb = dp.groupby("publisher_brand")[["n_oa_articles", "n_declaring", "n_use", "n_no_use"]].sum()
    pb["prev_any"] = pb.n_declaring / pb.n_oa_articles
    S["top_publishers"] = {k: {"n_oa": int(r.n_oa_articles), "declaring": int(r.n_declaring), "use": int(r.n_use),
                               "prev_any": float(r.prev_any)} for k, r in pb.sort_values("n_oa_articles", ascending=False).head(15).iterrows()}
    S["n_journals_in_denominator"] = int(dj.journal.nunique())
    S["n_journals_with_declaration"] = int(dj[dj.n_declaring > 0].journal.nunique())
    # journals are counted by their Europe PMC title, as in the denominator tables (JATS titles vary in spelling and case
    # between journals' own versions of an article)
    S["n_journals_jats_titles"] = S["n_journals"]
    S["n_journals"] = S["n_journals_with_declaration"]
    rh = ROOT / "data" / "interim" / "rules_history.json"
    if rh.exists():
        hv = json.loads(rh.read_text())["versions"]
        S["rules_history"] = hv
        S["rules_growth_articles_share"] = hv["1.4"]["articles_with_passage"] / hv["1.2"]["articles_with_passage"] - 1
        S["rules_v14_passage_share"] = (hv["1.4"]["passages"] - hv["1.3"]["passages"]) / hv["1.4"]["passages"]
    done_files = list((ROOT / "data" / "raw" / "epmc_index_daily").glob("*.done"))
    S["index_shortfall_total"] = sum(int(f.read_text().split()[2]) for f in done_files) if done_files else None
    S["n_rows_denominators_journal"] = int(len(dj))
    S["n_rows_denominators_publisher"] = int(len(dp))
    S["candidate_share"] = (summ["n_candidates_screened"] / S["n_oa_articles"]) if summ["index_months_done"] == 46 else None
    ff = ROOT / "data" / "interim" / "fetch_failures.txt"
    S["fetch_failures"] = len(ff.read_text().split()) if ff.exists() else None
    S["capture_route_share"] = (st.capture_route.value_counts(normalize=True)).to_dict()
    # deterministic check: a publisher's standard sentence of non-use must be labelled no_use
    neg = st.statement_text.fillna("").str.match(
        r"(?i)^\W*the author(?:\(s\)|s)? (?:declare[sd]?|verif\w+|confirm\w*) that (?:no (?:generative|gen) ?ai was used|"
        r"(?:generative|gen) ?ai was not used) in the creation of this manuscript\W*$")
    S["template_negative_n"] = int(neg.sum())
    S["template_negative_consistency"] = float((st[neg].polarity == "no_use").mean()) if neg.sum() else None

    # ---------------- further numbers reported in the text ----------------------------------------
    lic = st.source_license.fillna("none")
    S["license_mix_statements"] = {
        "cc_by": float(lic.str.match(r"^CC BY \d").mean()),
        "cc_by_nc_any": float(lic.str.contains("-NC").mean()),
        "cc_by_nd_or_sa_only": float(lic.str.match(r"^CC BY-(ND|SA) ").mean()),
        "cc0": float((lic == "CC0").mean()), "none": float((lic == "none").mean())}
    S["statements_text_truncated"] = int(st.text_truncated.sum())
    S["use_articles_no_tool_share"] = float((ar_use.tools.fillna("") == "").mean()) if len(ar_use) else None
    S["corpus_not_in_index"] = int((ar.in_epmc_index == 0).sum())
    S["articles_all_passages_rejected"] = int(((sc.n_candidate_passages > 0) & (sc.has_declaration == 0)).sum())
    S["articles_with_passages"] = int((sc.n_candidate_passages > 0).sum())
    S["articles_no_passage"] = int((sc.n_candidate_passages == 0).sum())
    pbr = dp.groupby("publisher_brand").n_oa_articles.sum()
    S["publisher_unmapped_share"] = float(pbr.get("Other/unknown", 0) / pbr.sum())
    S["english_share_corpus"] = float((ar.language == "en").mean())
    trunc_p = cap = n_p = 0
    per_art = collections.Counter()
    for p in read_jsonl(ROOT / "data" / "interim" / "passages.jsonl.gz"):
        n_p += 1
        trunc_p += int(bool(p.get("trunc")))
        per_art[p["pmcid"]] += 1
    S["passages_truncated"] = trunc_p
    S["passages_truncated_share"] = trunc_p / max(1, n_p)
    S["articles_at_passage_cap"] = sum(1 for v in per_art.values() if v >= 8)
    vf = ROOT / "data" / "interim" / "pmc_versions.jsonl"
    if vf.exists():
        nv = [len(json.loads(l)["versions"]) for l in open(vf, encoding="utf-8") if l.strip()]
        S["candidates_listed_versions"] = len(nv)
        S["candidates_multi_version"] = sum(1 for v in nv if v > 1)
    lf = ROOT / "data" / "interim" / "blocks_v2" / "shard_latest.jsonl.gz"
    S["articles_from_latest_version"] = sum(1 for r in read_jsonl(lf) if r.get("status") == "ok") if lf.exists() else 0
    # incomplete months: how much of their higher prevalence comes from a different mix of journals
    if trail:
        djm = dj.assign(mo=dj.month.astype(str).str[:7])
        full_from, full_to = S["prev_any_last_year_complete_months"]["from"], S["prev_any_last_year_complete_months"]["to"]
        e = djm[(djm.mo >= full_from) & (djm.mo <= full_to)].groupby("journal")[["n_oa_articles", "n_declaring"]].sum()
        l = djm[djm.mo.isin(trail)].groupby("journal")[["n_oa_articles", "n_declaring"]].sum()
        both = e.index.intersection(l.index)
        e, l = e.loc[both], l.loc[both]
        w = e.n_oa_articles
        S["incomplete_months_mix"] = {
            "raw": float(djm[djm.mo.isin(trail)].n_declaring.sum() / djm[djm.mo.isin(trail)].n_oa_articles.sum()),
            "same_journals_complete": float(e.n_declaring.sum() / e.n_oa_articles.sum()),
            "same_journals_incomplete_reweighted": float(((l.n_declaring / l.n_oa_articles) * w).sum() / w.sum()),
            "n_journals": int(len(both))}

    (PAPER / "stats.json").write_text(json.dumps(S, indent=1, default=str), encoding="utf-8")
    print("stats ->", PAPER / "stats.json", "(PARTIAL DATA)" if partial else "")

    tag = "  [partial data - pipeline test]" if partial else ""

    # ---------------- Fig 2: monthly prevalence ---------------------------------------------------
    if len(m) >= 2:
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        x = pd.to_datetime(m.index + "-01")
        series = [("Any AI-use declaration", 100 * m.n_declaring / m.n_oa_articles, SERIES[0]),
                  ("Declares AI use", 100 * m.n_use / m.n_oa_articles, SERIES[1]),
                  ("Declares no AI use", 100 * m.n_no_use / m.n_oa_articles, SERIES[2])]
        ends = sorted(((y.iloc[-1], i) for i, (_, y, _) in enumerate(series)))
        offs = {}
        for rank, (val, i) in enumerate(ends):  # nudge end labels apart when two series finish close together
            prev = ends[rank - 1][0] if rank else None
            offs[i] = 7 if (prev is not None and val - prev < 0.05 * max(e[0] for e in ends)) else 0
        n_inc = len(S.get("incomplete_months") or [])
        if n_inc:
            # months with incomplete deposit: shaded, and the series drawn dashed from the last complete month on
            x0 = x[-n_inc] - pd.Timedelta(days=15)
            ax.axvspan(x0, x[-1] + pd.Timedelta(days=20), color=GRID, alpha=0.55, lw=0, zorder=0)
            ax.text(x0 + pd.Timedelta(days=4), 0.02, "Deposit\nincomplete", transform=ax.get_xaxis_transform(), ha="left",
                    va="bottom", fontsize=7, color=MUTED, linespacing=1.15)
        for i, (label, y, c) in enumerate(series):
            if n_inc:
                ax.plot(x[:len(x) - n_inc], y.iloc[:len(x) - n_inc], color=c, lw=1.8, label=label, solid_capstyle="round")
                ax.plot(x[-n_inc - 1:], y.iloc[-n_inc - 1:], color=c, lw=1.5, ls=(0, (2.5, 1.5)))
            else:
                ax.plot(x, y, color=c, lw=1.8, label=label, solid_capstyle="round")
            if not n_inc:   # end values of incomplete months would be read as the current level, so they are not labelled
                ax.annotate(pct(y.iloc[-1]), (x[-1], y.iloc[-1]), xytext=(5, offs[i] - (4 if any(offs.values()) and offs[i] == 0 and i != ends[-1][1] else 0)),
                            textcoords="offset points", va="center", fontsize=8, color=INK)
        ax.set_ylabel("Share of open-access articles")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}%"))
        ax.set_ylim(bottom=0)
        if tag:
            ax.set_title(tag.strip())
        ax.legend(loc="upper left", ncol=1)
        ax.grid(axis="x", visible=False)
        ax.margins(x=0.02)
        save(fig, "fig2_monthly_prevalence")

    # ---------------- publishers (optional, --all-figures) ------------------------------------------
    top = pb[pb.index != "Other/unknown"].sort_values("n_oa_articles", ascending=False).head(14)
    if len(top) and "--all-figures" in sys.argv:
        top = top.assign(u=100 * top.n_use / top.n_oa_articles, n=100 * top.n_no_use / top.n_oa_articles)
        top = top.sort_values("prev_any")
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        yy = np.arange(len(top))
        ax.barh(yy, top.u, height=0.62, color=SERIES[1], label="Declares AI use", edgecolor=SURFACE, linewidth=1.2)
        ax.barh(yy, top.n, left=top.u, height=0.62, color=SERIES[2], label="Declares no AI use", edgecolor=SURFACE, linewidth=1.2)
        for i, (u, n) in enumerate(zip(top.u, top.n)):
            ax.text(u + n + 0.6, i, pct(u + n), va="center", fontsize=8, color=INK)
        ax.set_yticks(yy, [f"{k}  (n = {int(v):,})" for k, v in zip(top.index, top.n_oa_articles)])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(pct))
        ax.set_xlabel("Share of the publisher's open-access articles in the corpus window")
        ax.set_title("AI-use declarations by publisher (14 largest by open-access output)" + tag)
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right")
        ax.set_xlim(0, max(5, (top.u + top.n).max() * 1.12))
        save(fig, "extra_publishers")

    # ---------------- Fig 3: tools and purposes (Data Overview) ---------------------------------------
    if len(ar_use):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"wspace": 0.9})
        ax = axes[0]
        tt = tool_counts.most_common(12)[::-1]
        ax.barh([k for k, _ in tt], [100 * v / len(ar_use) for _, v in tt], height=0.62, color=SERIES[0])
        for i, (_, v) in enumerate(tt):
            ax.text(100 * v / len(ar_use) + 0.6, i, pct(100 * v / len(ar_use)), va="center", fontsize=7.5, color=INK)
        ax.set_title("a", loc="left", fontweight="bold")
        ax.set_title("Tools named", loc="center")
        ax.set_xlabel("Share of articles declaring AI use")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(pct))
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 100 * tt[-1][1] / len(ar_use) * 1.18)

        ax = axes[1]
        ps = sorted(PURPOSES, key=lambda p: use["purpose_" + p].mean())
        vals = [100 * use["purpose_" + p].mean() for p in ps]
        ax.barh([PURPOSE_LABEL[p] for p in ps], vals, height=0.62, color=SERIES[0])
        for i, v in enumerate(vals):
            ax.text(v + 0.8, i, pct(v), va="center", fontsize=7.5, color=INK)
        ax.set_title("b", loc="left", fontweight="bold")
        ax.set_title("Declared purposes", loc="center")
        ax.set_xlabel("Share of statements declaring AI use")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(pct))
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(vals) * 1.18)
        if partial:
            fig.suptitle("partial data - pipeline test", fontsize=8, color=MUTED, x=0.99, ha="right")
        save(fig, "fig3_tools_purposes")

    # quarterly trends (optional, --all-figures)
    if len(ar_use) and "--all-figures" in sys.argv:
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), gridspec_kw={"hspace": 0.5, "wspace": 0.75})
        # quarterly series need publication dates
        au = ar_use.dropna(subset=["pub_month"]).copy()
        au["q"] = pd.PeriodIndex(pd.to_datetime(au.pub_month + "-01"), freq="Q")
        ax = axes[1, 0]
        fam = [k for k, _ in tool_counts.most_common(5)]
        qs = sorted(au.q.unique())
        qs = [q for q in qs if (au.q == q).sum() >= 50]
        for i, f in enumerate(fam):
            y = [100 * au[au.q == q].tools.fillna("").str.split("|").apply(lambda L: f in L).mean() for q in qs]
            ax.plot([q.to_timestamp() for q in qs], y, color=SERIES[i], lw=1.8, label=f, marker="o", ms=3)
        ax.set_title("c  Leading tools over time (quarterly)")
        ax.set_ylabel("Share of articles declaring AI use")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(pct))
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, columnspacing=1.2, handlelength=1.6)
        year_axis(ax)

        ax = axes[1, 1]
        su = use.merge(ar[["pmcid", "pub_month"]], on="pmcid", how="left").dropna(subset=["pub_month"])
        su["q"] = pd.PeriodIndex(pd.to_datetime(su.pub_month + "-01"), freq="Q")
        for i, (col, lab) in enumerate([("human_review", "Human review of AI output"), ("responsibility", "Responsibility statement"),
                                        ("nonuse_scope", "States what AI was not used for")]):
            y = [100 * su[su.q == q][col].mean() for q in qs]
            ax.plot([q.to_timestamp() for q in qs], y, color=SERIES[i], lw=1.8, label=lab, marker="o", ms=3)
        ax.set_title("d  Accountability elements (quarterly)")
        ax.set_ylabel("Share of statements declaring AI use")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(pct))
        ax.set_ylim(0, 100)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, handlelength=1.6)
        year_axis(ax)
        axes[0, 0].axis("off")
        axes[0, 1].axis("off")
        save(fig, "extra_quarterly_trends")

    # ---------------- Fig 1: workflow ------------------------------------------------------------------
    # Flow of articles and passages, with the excluded sets (grey) and the audits and validation attached to them.
    b = summ
    from common import DATE_END, DATE_START, GENERIC_TERMS, TOOL_TERMS
    vA = json.loads((VALID / "audit" / "verified_A3.json").read_text()) if (VALID / "audit" / "verified_A3.json").exists() else None
    vB = json.loads((VALID / "audit" / "verified_B.json").read_text()) if (VALID / "audit" / "verified_B.json").exists() else None
    vC = json.loads((VALID / "audit" / "verified_C.json").read_text()) if (VALID / "audit" / "verified_C.json").exists() else None
    sc = pd.read_parquet(RELEASE / "gaid_screened_articles.parquet", columns=["n_candidate_passages"])
    n_with_pass = int((sc.n_candidate_passages > 0).sum())
    n_oa, n_cand = S["n_oa_articles"], b["n_candidates_in_index"]
    vr0 = json.loads((VALID / "validation_results.json").read_text()) if (VALID / "validation_results.json").exists() else {}
    n_hum = len(vr0.get("human_annotators", {}))
    main_boxes = [
        ("Sampling frame", f"Open-access articles in Europe PMC and PMC,\nfirst published {DATE_START} to {DATE_END}\nn = {n_oa:,} articles"),
        ("Candidate articles", f"Full-text query ({len(TOOL_TERMS)} product names, {len(GENERIC_TERMS)} generic terms);\n"
                               f"JATS XML parsed for n = {b['n_candidates_screened']:,} articles"),
        ("Candidate passages", f"Section-aware rules: {b['n_candidate_passages']:,} passages\n"
                               f"({b['n_unique_passage_texts']:,} distinct texts) in n = {n_with_pass:,} articles"),
        ("LLM annotation", "Distinct texts labelled with the codebook\nby a primary and an independent second annotator"),
        ("GAID corpus", f"{S['n_statements']:,} declaration statements in\nn = {S['n_articles']:,} articles from {S['n_journals']:,} journals"),
    ]

    def audit_line(v, label):
        if not v:
            return f"Audit {label}: not run"
        r = 100 * v["rate"]
        return (f"Audit {label}: missed declarations in {v['confirmed']} of\n"
                f"{v['screened']:,} screened articles ({r:.1f}%)" if r >= 0.1 or r == 0 else
                f"Audit {label}: missed declarations in {v['confirmed']} of\n{v['screened']:,} screened articles ({r:.2f}%)")
    n_not_retrieved = vB["population"] if vB else n_oa - n_cand     # the audited stratum, as reported in the text
    side = {  # gap index (between main box i and i+1) or box index -> (title, body, kind)
        ("gap", 0): ("Not retrieved", f"n = {n_not_retrieved:,} articles\n" + audit_line(vB, "B"), "excl"),
        ("gap", 1): ("No candidate passage", f"n = {b['n_candidates_screened'] - n_with_pass:,} articles\n" + audit_line(vA, "A"), "excl"),
        ("box", 3): ("Label validation", f"{n_hum} human annotators (350 passages),\nthree-model reference (700), second\n"
                                         f"annotator ({vr0.get('primary_vs_second', {}).get('n', 0):,}), test-retest "
                                         f"({vr0.get('primary_test_retest', {}).get('n', 0):,})", "qc"),
        ("gap", 3): ("No declaration", f"n = {S['articles_all_passages_rejected']:,} articles; their {b['n_nondeclaration_passages']:,}\n"
                                       "passages are released as hard negatives\n" + audit_line(vC, "C"), "excl"),
    }
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.axis("off")
    ax.set_xlim(0, 10)
    mx, mw, mh, gap = 0.1, 4.9, 2.0, 0.9         # main column
    sx, sw, sh = 5.75, 4.15, 2.0                 # side column
    ax.set_ylim(13.5 - 5 * mh - 4 * gap - 0.15, 13.65)
    tops = [13.5 - i * (mh + gap) for i in range(len(main_boxes))]
    MAIN_FC, EXCL_FC = "#eef4fc", "#f6f6f3"

    def box(x0, y_top, w, h, title, body, fc, ec, ls="-"):
        ax.add_patch(FancyBboxPatch((x0, y_top - h), w, h, boxstyle="round,pad=0.0,rounding_size=0.12", fc=fc, ec=ec, lw=0.9,
                                    ls=ls))
        ax.text(x0 + 0.18, y_top - 0.2, title, fontsize=8.5, fontweight="bold", va="top", color=INK)
        ax.text(x0 + 0.18, y_top - 0.62, body, fontsize=7.5, va="top", color=INK2, linespacing=1.35)

    for (title, body), t in zip(main_boxes, tops):
        box(mx, t, mw, mh, title, body, MAIN_FC, SERIES[0])
    arrow = dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=0, shrinkB=1, mutation_scale=9)
    cx = mx + mw / 2
    for i in range(len(tops) - 1):
        ax.annotate("", xy=(cx, tops[i + 1]), xytext=(cx, tops[i] - mh), arrowprops=arrow)
    for (kind, i), (title, body, style) in side.items():
        if kind == "gap":
            yc = tops[i] - mh - gap / 2
            ax.annotate("", xy=(sx, yc), xytext=(cx, yc), arrowprops=arrow)
        else:
            yc = tops[i] - mh / 2 + 0.8
            ax.plot([mx + mw, sx], [yc, yc], color=INK2, lw=0.9)
        h = sh + 0.35 * max(0, body.count("\n") - 2)       # taller box for four-line texts, extending downwards
        box(sx, yc + sh / 2, sw, h, title, body, EXCL_FC if style == "excl" else SURFACE, AXIS if style == "excl" else SERIES[0],
            "-" if style == "excl" else "--")
    if partial:
        ax.text(0.1, tops[-1] - mh - 0.1, "* partial data - pipeline test", fontsize=7.5, color=MUTED)
    save(fig, "fig1_workflow")

    # ---------------- Fig 4: validation (when results exist) -----------------------------------------
    vr = VALID / "validation_results.json"
    if vr.exists():
        V = json.loads(vr.read_text())
        comps = [(k, lab) for k, lab in [("primary_vs_reference", "Released labels vs three-model reference"),
                                         ("primary_vs_second", "Released labels vs second annotator"),
                                         ("primary_test_retest", "Released labels, test-retest")] if k in V]
        ci_key = {"primary_vs_reference": "primary_vs_reference", "primary_vs_second": "primary_vs_second",
                  "primary_test_retest": "primary_test_retest"}
        if "primary_vs_human" in V:
            human = []
            for name, d in sorted(V["primary_vs_human"].items()):
                V["h_" + name] = d
                ci_key["h_" + name] = f"primary_vs_human.{name}"
                human.append(("h_" + name, f"Released labels vs human annotator {name}"))
            comps = human + comps
        CI = V.get("ci95", {})
        fields = [("decl", "Declaration"), ("pol", "Polarity"), ("resp", "Responsibility"), ("rev", "Human review"),
                  ("nonuse", "Non-use scope"), ("purp", "Purposes\n(micro-F1)"), ("tools", "Tools\n(micro-F1)")]
        if comps:
            fig, ax = plt.subplots(figsize=(7.0, 3.6))
            comps = comps[:8]
            allv = []
            for i, (k, lab) in enumerate(comps):
                ys = [V[k].get(f, {}).get("micro_f1" if f in ("purp", "tools") else "kappa", np.nan) for f, _ in fields]
                ci = CI.get(ci_key.get(k, ""), {})
                lo = [ci.get(f, [y, y])[0] for (f, _), y in zip(fields, ys)]
                hi = [ci.get(f, [y, y])[1] for (f, _), y in zip(fields, ys)]
                allv += [v for v in lo + ys if v == v]
                dodge = (i - (len(comps) - 1) / 2) * 0.13      # side-by-side markers instead of overplotting
                xs = [x + dodge for x in range(len(ys))]
                n_items = V[k].get("n")
                ax.errorbar(xs, ys, yerr=[np.subtract(ys, lo), np.subtract(hi, ys)], fmt="o", ms=5, color=SERIES[i],
                            mec=SURFACE, mew=0.8, elinewidth=1.1, capsize=0,
                            label=f"{lab} (n = {n_items:,})" if n_items else lab)
            for xb in range(1, len(fields)):
                ax.axvline(xb - 0.5, color=GRID, lw=0.6)
            ax.set_xticks(range(len(fields)), [l for _, l in fields])
            ax.set_ylabel("Cohen's kappa\n(micro-F1 for multi-label fields)")
            ax.set_ylim(np.floor(min(allv) * 20) / 20 - 0.02, 1.005)
            ax.set_xlim(-0.5, len(fields) - 0.5)
            ax.grid(axis="x", visible=False)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, handletextpad=0.3, columnspacing=1.4)
            save(fig, "fig4_validation")


if __name__ == "__main__":
    main()
