"""Export the human-validation items as one Excel workbook per expert.

The workbook holds only the items and the columns to fill in; the instructions are in annotation/标注说明.md.
The item number -> passage key mapping is written to data/validation/human_excel_map.json (not into the workbook).

  python -I scripts/12_export_excel.py            # writes annotation/GAID_annotation_expert1.xlsx and ..._expert2.xlsx
  python -I scripts/12_export_excel.py --force    # overwrite existing workbooks
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, VALID  # noqa: E402

OUT_DIR = ROOT / "annotation"
MAP_FILE = VALID / "human_excel_map.json"
N_EXPERTS = 2

LOC_LABEL = {"ai_section": "AI声明专节", "decl_zone": "致谢/声明/脚注", "body": "正文"}
PURPOSES = ["LANG", "TRANS", "DRAFT", "LIT", "IDEA", "CODE", "DATA", "VIS", "FORMAT", "OTHER", "UNSPEC"]

# (header, width, kind)  kind: info = given data, input = to be filled in
COLUMNS = ([("编号", 7, "info"), ("位置", 13, "info"), ("章节标题", 22, "info"), ("待标注文本", 72, "info"),
            ("decl\n是否为声明", 10, "input"), ("pol\n极性", 11, "input"), ("tools\n工具名称", 20, "input")]
           + [(p, 7.5, "input") for p in PURPOSES]
           + [("resp\n责任声明", 9, "input"), ("rev\n人工审阅", 9, "input"), ("nonuse\n未使用范围", 10, "input"),
              ("备注", 24, "input")])

FONT = "Arial"
HEAD_INFO = PatternFill("solid", fgColor="D9E1F2")
HEAD_INPUT = PatternFill("solid", fgColor="FFE699")
CELL_INPUT = PatternFill("solid", fgColor="FFF8E1")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def load_items():
    rows = [json.loads(l) for l in open(VALID / "validation_sample.jsonl", encoding="utf-8")]
    items = [r for r in rows if r["human_subset"]]
    random.Random(99).shuffle(items)          # same order as the earlier HTML tool
    return items


def row_height(text: str, heading: str) -> float:
    lines_text = sum(max(1, math.ceil(len(par) / 82)) for par in text.split("\n"))
    lines_head = max(1, math.ceil(len(heading) / 24)) if heading else 1
    return min(409, max(30, 13.5 * max(lines_text, lines_head) + 6))


def build(items, expert_no: int, path: Path):
    wb = Workbook()
    ws = wb.active
    ws.title = "标注"
    # header
    for c, (head, width, kind) in enumerate(COLUMNS, 1):
        cell = ws.cell(row=1, column=c, value=head)
        cell.font = Font(name=FONT, bold=True, size=10)
        cell.fill = HEAD_INPUT if kind == "input" else HEAD_INFO
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER
        ws.column_dimensions[cell.column_letter].width = width
    ws.row_dimensions[1].height = 32
    # items
    for r, it in enumerate(items, 2):
        values = [r - 1, LOC_LABEL[it["loc"]], it["heading"] or "", it["text"]]
        for c in range(1, len(COLUMNS) + 1):
            cell = ws.cell(row=r, column=c, value=values[c - 1] if c <= len(values) else None)
            kind = COLUMNS[c - 1][2]
            cell.font = Font(name=FONT, size=10)
            cell.border = BORDER
            if kind == "input":
                cell.fill = CELL_INPUT
                cell.alignment = Alignment(horizontal="center", vertical="top", wrap_text=True)
            else:
                cell.alignment = Alignment(horizontal="left" if c > 1 else "center", vertical="top", wrap_text=True)
        ws.row_dimensions[r].height = row_height(it["text"], it["heading"] or "")
    last = len(items) + 1
    col = {head.split("\n")[0]: ws.cell(row=1, column=i).column_letter for i, (head, _, _) in enumerate(COLUMNS, 1)}
    for c in ("tools", "备注"):
        for r in range(2, last + 1):
            ws[f"{col[c]}{r}"].alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
    # drop-down lists
    def add_list(formula, cols, prompt):
        dv = DataValidation(type="list", formula1=formula, allow_blank=True, showErrorMessage=True,
                            errorTitle="无效输入", error=prompt)
        ws.add_data_validation(dv)
        for c in cols:
            dv.add(f"{col[c]}2:{col[c]}{last}")
    add_list('"1,0"', ["decl"], "请填 1（是声明）或 0（不是声明）。")
    add_list('"use,no_use,na,unclear"', ["pol"], "请从下拉列表选择：use / no_use / na / unclear。")
    add_list('"1,0"', PURPOSES + ["resp", "rev", "nonuse"], "请填 1 或 0（也可留空，留空视为 0）。")
    ws.freeze_panes = "E2"
    ws.auto_filter.ref = f"A1:{ws.cell(row=1, column=len(COLUMNS)).column_letter}{last}"
    # document properties: neutral
    p = wb.properties
    p.creator = ""
    p.lastModifiedBy = ""
    p.title = f"人工标注表 - 专家{expert_no}"
    p.subject = p.description = p.keywords = p.category = None
    wb.save(path)
    neutral_app_properties(path)


def neutral_app_properties(path: Path):
    """Replace the writer's library name in docProps/app.xml with a generic application name."""
    import re
    import shutil
    import zipfile
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "docProps/app.xml":
                xml = data.decode("utf-8")
                xml = re.sub(r"<Application>.*?</Application>", "<Application>Microsoft Excel</Application>", xml)
                xml = re.sub(r"<AppVersion>.*?</AppVersion>", "<AppVersion>16.0300</AppVersion>", xml)
                data = xml.encode("utf-8")
            zout.writestr(item, data)
    shutil.move(tmp, path)


def main():
    force = "--force" in sys.argv
    items = load_items()
    OUT_DIR.mkdir(exist_ok=True)
    targets = [OUT_DIR / f"GAID_annotation_expert{i}.xlsx" for i in range(1, N_EXPERTS + 1)]
    if not force and any(t.exists() for t in targets):
        print("workbooks already exist; use --force to overwrite:", [str(t) for t in targets if t.exists()])
        return
    MAP_FILE.write_text(json.dumps({str(i): it["key"] for i, it in enumerate(items, 1)}, indent=0), encoding="utf-8")
    for i, t in enumerate(targets, 1):
        build(items, i, t)
        print("wrote", t)
    print(f"{len(items)} items; mapping -> {MAP_FILE}")


if __name__ == "__main__":
    main()
