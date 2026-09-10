# -*- coding: utf-8 -*-
"""正式计算: 问题 1-4 求解、结果文件生成、表 1-6 生成。

输出:
  work/out/p1.npz, p2.npz, p3.npz, p4.npz   (原始结果缓存)
  outputs/result1.xlsx ... result4.xlsx      (竞赛要求的结果文件)
  work/out/tables.json                       (表 1-6 数值)
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import openpyxl
from openpyxl.utils import get_column_letter
import herb_model as hm

OUT = hm.WORK_OUT
DELIV = os.environ.get("DELIV") or hm.RESULT_DIR
os.makedirs(OUT, exist_ok=True)
os.makedirs(DELIV, exist_ok=True)

N_PROD = int(os.environ.get("NPROD", "400"))
R0 = 0.02


def build_env(t, T, C):
    return hm.Environment(t, T, C)


def solve_case(mode, t_end, schedule, t_eval, N=N_PROD, moving=False, radius=None):
    t, T, C = hm.load_oven()
    env = hm.Environment(t, T, C)
    props = hm.Props(mode)
    if radius is None:
        radius = hm.RadiusLaw(*hm.load_radius()) if moving else hm.ConstRadius(R0)
    solver = hm.DryingSolver(props, radius, env, N=N, theta=0.5)
    res = solver.run(t_end, schedule, t_eval=t_eval)
    res["radius_t"] = np.array([radius.R_of(tt) for tt in res["t"]])
    res["envT"] = np.array([env.T_inf(tt) for tt in res["t"]])
    res["envC"] = np.array([env.C_inf(tt) for tt in res["t"]])
    return res, env, radius, solver


def write_result_xlsx(path, sheets, times, radii, values_by_sheet, header_last=None):
    """sheets: [(sheetname, values matrix (nt, nr))]; radii in cm"""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, mat in zip(sheets, values_by_sheet):
        ws = wb.create_sheet(name)
        hdr = ["时间\\到药材中心的距离"] + [round(float(r), 4) for r in radii]
        if header_last is not None:
            hdr[-1] = header_last
        ws.append(hdr)
        for i, tt in enumerate(times):
            row = [float(tt)] + [round(float(v), 4) for v in mat[i]]
            ws.append(row)
    wb.save(path)


def drying_time(times, C_center, thr=0.15):
    """线性插值求首次 max_r C < thr 的时刻 (s)"""
    idx = np.where(C_center < thr)[0]
    if idx.size == 0:
        return None
    i = idx[0]
    if i == 0:
        return float(times[0])
    t1, t2 = times[i - 1], times[i]
    c1, c2 = C_center[i - 1], C_center[i]
    return float(t1 + (thr - c1) * (t2 - t1) / (c2 - c1))


def main():
    ttab = {}
    t0 = time.time()

    # ---------------- 问题 1 ----------------
    print(">>> 问题 1 ...", flush=True)
    te = np.arange(0, 1800 + 1, 1.0)
    res1, env1, rad1, _ = solve_case("p1", 1800.0, hm.schedule_p1, te)
    np.savez_compressed(os.path.join(OUT, "p1.npz"),
                        t=res1["t"], T=res1["T"], C=res1["C"], xi=res1["xi"])
    radii_cm = np.round(np.arange(0, 2.0001, 0.1), 4)
    r_m = radii_cm / 100.0
    T1 = hm.radial_records(res1, rad1, r_m, res1["t"], which="T")
    C1 = hm.radial_records(res1, rad1, r_m, res1["t"], which="C")
    write_result_xlsx(os.path.join(DELIV, "result1.xlsx"),
                      ["温度", "水分浓度"], res1["t"], radii_cm, [T1, C1])
    # 表 1 / 表 2
    tab1_t = [100, 300, 600, 900, 1200, 1500, 1800]
    tab1_r = [0, 0.5, 1, 1.5, 2]
    ttab["表1"] = [[float(T1[i, int(np.argmin(np.abs(r_m - x / 100.0)))]) for x in tab1_r] for i in
                   [int(np.argmin(np.abs(res1["t"] - x))) for x in tab1_t]]
    ttab["表2"] = [[float(C1[i, int(np.argmin(np.abs(r_m - x / 100.0)))]) for x in tab1_r] for i in
                   [int(np.argmin(np.abs(res1["t"] - x))) for x in tab1_t]]
    ttab["表1_t"] = tab1_t; ttab["表1_r"] = tab1_r
    print("    完成 (%.0f s), C_surf(1800)=%.4f" % (time.time() - t0, res1["C"][-1, -1]))

    # ---------------- 问题 2 ----------------
    print(">>> 问题 2 ...", flush=True)
    te = np.arange(0, 10800 + 1, 1.0)
    res2, env2, rad2, _ = solve_case("p3", 10800.0, hm.schedule_p2, te)
    np.savez_compressed(os.path.join(OUT, "p2.npz"),
                        t=res2["t"], T=res2["T"], C=res2["C"], xi=res2["xi"])
    T2 = hm.radial_records(res2, rad2, r_m, res2["t"], which="T")
    C2 = hm.radial_records(res2, rad2, r_m, res2["t"], which="C")
    write_result_xlsx(os.path.join(DELIV, "result2.xlsx"),
                      ["温度", "水分浓度"], res2["t"], radii_cm, [T2, C2])
    tab2_t = [1800, 3600, 5400, 7200, 9000, 10800]
    ttab["表3"] = [[float(T2[i, int(np.argmin(np.abs(r_m - x / 100.0)))]) for x in tab1_r] for i in
                   [int(np.argmin(np.abs(res2["t"] - x))) for x in tab2_t]]
    ttab["表4"] = [[float(C2[i, int(np.argmin(np.abs(r_m - x / 100.0)))]) for x in tab1_r] for i in
                   [int(np.argmin(np.abs(res2["t"] - x))) for x in tab2_t]]
    ttab["表3_t"] = tab2_t
    print("    完成 (%.0f s)" % (time.time() - t0,), flush=True)

    # ---------------- 问题 3 ----------------
    print(">>> 问题 3 (长时程) ...", flush=True)
    T_END3 = 120.0 * 3600.0
    te = np.arange(0, T_END3 + 1, 60.0)
    res3, env3, rad3, _ = solve_case("p3", T_END3, hm.schedule_long, te)
    np.savez_compressed(os.path.join(OUT, "p3.npz"),
                        t=res3["t"], T=res3["T"], C=res3["C"], xi=res3["xi"])
    maxC3 = res3["C"].max(axis=1)
    print("    check: 全域最大浓度与中心值之差 = %.2e" % np.max(np.abs(maxC3 - res3["C"][:, 0])))
    tdry3 = drying_time(res3["t"], maxC3, 0.15)
    print("    干燥时间 = %.4f h (%.1f s)" % (tdry3 / 3600.0, tdry3), flush=True)
    ttab["问题3_干燥时长_h"] = tdry3 / 3600.0
    # 截取到干燥结束
    m = res3["t"] <= tdry3 + 1e-9
    C3 = hm.radial_records(res3, rad3, r_m, res3["t"][m], which="C")
    write_result_xlsx(os.path.join(DELIV, "result3.xlsx"), ["水分浓度"],
                      res3["t"][m], radii_cm, [C3])
    # 表 5: 每 6 h (至干燥结束)
    tt5 = list(np.arange(6 * 3600, tdry3, 6 * 3600)) + [tdry3]
    C3_all = hm.radial_records(res3, rad3, r_m, res3["t"], which="C")
    idx5 = [int(np.argmin(np.abs(r_m - rr / 100.0))) for rr in tab1_r]
    tab5 = []
    for x in tt5:
        i = int(np.searchsorted(res3["t"], x))
        if abs(res3["t"][i] - x) < 1e-9:
            row = [float(C3_all[i, k]) for k in idx5]
        else:   # 干燥结束点插值
            t1, t2 = res3["t"][i - 1], res3["t"][i]
            w = (x - t1) / (t2 - t1)
            Cx = (1 - w) * res3["C"][i - 1] + w * res3["C"][i]
            cs = PchipInterpolatorAt(res3["xi"], Cx)
            row = [float(cs(rr / 100.0 / rad3.R_of(x))) for rr in tab1_r]
        tab5.append(row)
    ttab["表5"] = tab5
    ttab["表5_t_h"] = [x / 3600.0 for x in tt5]
    print("    完成 (%.0f s)" % (time.time() - t0,), flush=True)

    # ---------------- 问题 4 (收缩域) ----------------
    print(">>> 问题 4 (收缩域) ...", flush=True)
    T_END4 = 120.0 * 3600.0
    te = np.arange(0, T_END4 + 1, 60.0)
    res4, env4, rad4, _ = solve_case("p4", T_END4, hm.schedule_long, te, moving=True)
    np.savez_compressed(os.path.join(OUT, "p4.npz"),
                        t=res4["t"], T=res4["T"], C=res4["C"], xi=res4["xi"],
                        R=res4["radius_t"])
    maxC4 = res4["C"].max(axis=1)
    print("    check: 全域最大浓度与中心值之差 = %.2e" % np.max(np.abs(maxC4 - res4["C"][:, 0])))
    tdry4 = drying_time(res4["t"], maxC4, 0.15)
    print("    干燥时间 = %.4f h (%.1f s)" % (tdry4 / 3600.0, tdry4), flush=True)
    ttab["问题4_干燥时长_h"] = tdry4 / 3600.0
    m = res4["t"] <= tdry4 + 1e-9
    radii4_cm = np.round(np.arange(0, 1.2001, 0.1), 4)
    r4_m = radii4_cm / 100.0
    C4 = np.full((m.sum(), len(r4_m) + 1), np.nan)
    for i, (tt, R) in enumerate(zip(res4["t"][m], res4["radius_t"][m])):
        Y = res4["C"][i]
        C4[i, :-1] = hm.radial_records({"t": [tt], "C": [Y], "xi": res4["xi"]},
                                       rad4, r4_m, [tt], which="C")[0]
        C4[i, -1] = Y[-1]
    radii4_hdr = list(radii4_cm) + ["药材表面"]
    wb = openpyxl.Workbook(); wb.remove(wb.active)
    ws = wb.create_sheet("水分浓度")
    ws.append(["时间\\到药材中心的距离"] + radii4_hdr)
    for i, tt in enumerate(res4["t"][m]):
        ws.append([float(tt)] + [round(float(v), 4) for v in C4[i]])
    wb.save(os.path.join(DELIV, "result4.xlsx"))
    # 表 6
    tt6 = list(np.arange(6 * 3600, tdry4, 6 * 3600)) + [tdry4]
    tab6 = []
    cols_r = [0.0, 0.5, 1.0]
    for x in tt6:
        i = int(np.searchsorted(res4["t"], x))
        if abs(res4["t"][i] - x) < 1e-9:
            Y = res4["C"][i]; R = res4["radius_t"][i]
        else:
            t1, t2 = res4["t"][i - 1], res4["t"][i]
            w = (x - t1) / (t2 - t1)
            Y = (1 - w) * res4["C"][i - 1] + w * res4["C"][i]
            R = (1 - w) * res4["radius_t"][i - 1] + w * res4["radius_t"][i]
        fY = PchipInterpolatorAt(res4["xi"], Y)
        row = [float(fY(rr / 100.0 / R)) for rr in cols_r] + [float(Y[-1])]
        tab6.append(row)
    ttab["表6"] = tab6
    ttab["表6_t_h"] = [x / 3600.0 for x in tt6]
    ttab["表6_cols"] = ["0", "0.5", "1.0", "药材表面"]
    print("    完成 (%.0f s)" % (time.time() - t0,), flush=True)

    with open(os.path.join(OUT, "tables.json"), "w") as fh:
        json.dump(ttab, fh, ensure_ascii=False, indent=1)
    print("全部完成, 总耗时 %.0f s" % (time.time() - t0))


def PchipInterpolatorAt(xi, Y):
    from scipy.interpolate import PchipInterpolator
    return PchipInterpolator(xi, Y, extrapolate=False)


if __name__ == "__main__":
    main()
