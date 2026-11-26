"""Print the validation results as a compact table."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import VALID  # noqa: E402

V = json.loads((VALID / "validation_results.json").read_text(encoding="utf-8"))


def f(x):
    return "  -  " if x is None else f"{x:.3f}"


def row(name, d):
    if not d or "decl" not in d:
        print(f"{name:36s} (missing)")
        return
    g = lambda a, b: (d.get(a) or {}).get(b)  # noqa: E731
    print(f"{name:36s} n={d['n']:6d} decl_k={f(g('decl', 'kappa'))} decl_agree={f(g('decl', 'agree'))} pol_k={f(g('pol', 'kappa'))} "
          f"purpF1={f(g('purp', 'micro_f1'))} purpEx={f(g('purp', 'exact'))} toolsF1={f(g('tools', 'micro_f1'))} "
          f"resp_k={f(g('resp', 'kappa'))} rev_k={f(g('rev', 'kappa'))} nonuse_k={f(g('nonuse', 'kappa'))}")


print("models", V.get("models"))
row("primary vs reference", V.get("primary_vs_reference"))
for r, d in (V.get("primary_vs_reference_by_route") or {}).items():
    row("  route " + r, d)
row("second vs reference", V.get("second_vs_reference"))
row("primary vs second (all)", V.get("primary_vs_second"))
row("primary test-retest", V.get("primary_test_retest"))
for k2, d in (V.get("reference_pairwise") or {}).items():
    row("ref pair " + k2[:27], d)
for name, d in (V.get("primary_vs_human") or {}).items():
    row("primary vs human " + name, d)
if V.get("human_inter_annotator"):
    row("human vs human", V["human_inter_annotator"])
pr = (V.get("primary_vs_reference") or {}).get("decl")
if pr:
    print("primary decl precision / recall vs reference:", round(pr["precision_A_vs_B"], 3), round(pr["recall_A_vs_B"], 3))
    print("per-label purposes F1 (n in reference):", {k: (round(v["f1"], 2), v["n_B"]) for k, v in V["primary_vs_reference"]["purp"]["per_label"].items()})
