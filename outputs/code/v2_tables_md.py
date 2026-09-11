# -*- coding: utf-8 -*-
"""由 v2 结果文件生成 Markdown 表格（表1-6）"""
import os, numpy as np, openpyxl, json
RD = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "result_files"))


def read(path, sheet=0):
    wb = openpyxl.load_workbook(os.path.join(RD, path), read_only=True)
    ws = wb.worksheets[sheet]
    rows = list(ws.iter_rows(values_only=True))
    hdr = [x for x in rows[0][1:]]
    return hdr, rows[1:]


def row_at(rows, t):
    for r in rows:
        if abs(float(r[0]) - t) < 1e-6:
            return r
    raise KeyError(t)


def md(cols, rows, times, tfmt="%g", fmt="%.4f"):
    out = ["| 时间 | " + " | ".join(str(c) for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for t in times:
        r = row_at(rows, t)
        out.append("| " + (tfmt % t) + " | " + " | ".join(fmt % v for v in r[1:]) + " |")
    return "\n".join(out)


hdr, T1 = read("result1.xlsx", 0); _, C1 = read("result1.xlsx", 1)
hdr2, T2 = read("result2.xlsx", 0); _, C2 = read("result2.xlsx", 1)
_, C3 = read("result3.xlsx", 0)
_, C4 = read("result4.xlsx", 0)
_, C4L = read("result4_local.xlsx", 0)
IDX = [0, 5, 10, 15, 20]
cols = ["0 cm", "0.5 cm", "1 cm", "1.5 cm", "2 cm"]
L = []
L.append("### 表 1 温度 (°C)\n" + md(cols, [[r[0]] + [r[1 + i] for i in IDX] for r in T1],
                                   [100, 300, 600, 900, 1200, 1500, 1800], tfmt="%d"))
L.append("### 表 2 水分浓度 (kg/kg)\n" + md(cols, [[r[0]] + [r[1 + i] for i in IDX] for r in C1],
                                       [100, 300, 600, 900, 1200, 1500, 1800], tfmt="%d"))
L.append("### 表 3 温度 (°C)\n" + md(cols, [[r[0]] + [r[1 + i] for i in IDX] for r in T2],
                                   [1800, 3600, 5400, 7200, 9000, 10800], tfmt="%g"))
L.append("### 表 4 水分浓度 (kg/kg)\n" + md(cols, [[r[0]] + [r[1 + i] for i in IDX] for r in C2],
                                       [1800, 3600, 5400, 7200, 9000, 10800], tfmt="%g"))
t3 = [6 * 3600.0 * k for k in range(1, 10) if 6 * 3600.0 * k < float(C3[-1][0])] + [C3[-1][0]]
L.append("### 表 5 水分浓度 (kg/kg)\n" + md(cols, [[r[0]] + [r[1 + i] for i in IDX] for r in C3],
                                       t3, tfmt="%.2f" if True else "%g"))
I4 = [0, 5, 10, 13]
c4 = ["0 cm", "0.5 cm", "1.0 cm", "药材表面"]
def rows6(rows):
    t_end = float(rows[-1][0])
    tt = [6 * 3600.0 * k for k in range(1, 20) if 6 * 3600.0 * k < t_end]
    return tt + [t_end]
L.append("### 表 6 基线（仿射收缩）\n" + md(c4, [[r[0]] + [r[1 + i] for i in I4] for r in C4], rows6(C4), tfmt="%.2f"))
L.append("### 表 6v2 非均匀收缩模型\n" + md(c4, [[r[0]] + [r[1 + i] for i in I4] for r in C4L], rows6(C4L), tfmt="%.2f"))
# 时间换算为小时展示
def to_h(md_text):
    lines = []
    for ln in md_text.split("\n"):
        if ln.startswith("| ") and not ln.startswith("| 时间"):
            parts = ln.split("|")
            try:
                v = float(parts[1])
                parts[1] = " %.2f h " % (v / 3600)
                lines.append("|".join(parts))
                continue
            except Exception:
                pass
        lines.append(ln)
    return "\n".join(lines)
L = [to_h(x) for x in L]
open(os.path.join(os.path.dirname(RD), "v2_tables.md"), "w").write("\n\n".join(L))
print("\n\n".join(L[:3]))
print("...")
print(L[-2]); print(L[-1])
