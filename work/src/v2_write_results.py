# -*- coding: utf-8 -*-
"""v2 结果文件生成（流式写出，支持全烘干过程 1 s 分辨率）。

  result1.xlsx        问题 1：0—1800 s 每 1 s × r=0—2.0 cm 每 0.1 cm（温度/水分浓度）
  result2.xlsx        问题 2：0—t_dry 每 1 s（**全烘干过程**）× 同上
  result2_3h.xlsx     问题 2：0—3 h 每 1 s（表 3/表 4 对应区间，便于对照）
  result3.xlsx        问题 3：0—t_dry 每 60 s（末行为烘干结束时刻）× 同上
  result4.xlsx        问题 4：基线（仿射收缩）0—t_dry 每 60 s
  result4_local.xlsx  问题 4：v2 非均匀收缩自洽模型
"""
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import openpyxl
import herb_v2 as v2

RESULT_DIR = os.environ.get("DELIV") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "result_files"))
WORK_OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out"))
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(WORK_OUT, exist_ok=True)
R_OUT = np.round(np.arange(0, 2.0001, 0.1), 4) / 100.0
R_OUT4 = np.round(np.arange(0, 1.2001, 0.1), 4) / 100.0


def write_xlsx(path, times, radii, mats, sheets, last_col_label=None):
    wb = openpyxl.Workbook(write_only=True)
    for name, mat in zip(sheets, mats):
        ws = wb.create_sheet(name)
        hdr = ["时间\\到药材中心的距离"] + [round(float(x) * 100.0, 4) for x in radii]  # 表头单位 cm
        if last_col_label:
            hdr[-1] = last_col_label
        ws.append(hdr)
        for i, tt in enumerate(times):
            row = [float(tt)]
            for v in mat[i]:
                row.append(None if (v is None or not np.isfinite(v)) else round(float(v), 4))
            ws.append(row)
    wb.save(path)


def collect(solver, res, radii, extra_surface=False):
    n = len(res["t"])
    nr = len(radii) + (1 if extra_surface else 0)
    T = np.full((n, nr), np.nan)
    C = np.full((n, nr), np.nan)
    for i, tt in enumerate(res["t"]):
        To, Co = solver.profiles_at_radii(tt, res["T"][i], res["C"][i], radii)
        T[i, :len(radii)] = To
        C[i, :len(radii)] = Co
        if extra_surface:
            T[i, -1] = res["T"][i, -1]
            C[i, -1] = res["C"][i, -1]
    return T, C


def dry_time(t, C, thr=0.15):
    Cmax = C.max(axis=1)
    idx = np.where(Cmax < thr)[0]
    if idx.size == 0:
        return None
    i = idx[0]
    return float(t[i - 1] + (thr - Cmax[i - 1]) * (t[i] - t[i - 1]) / (Cmax[i] - Cmax[i - 1]))


def main():
    t0 = time.time()
    t_oven, T_inf, C_inf = v2.hm.load_oven()
    env = v2.hm.Environment(t_oven, T_inf, C_inf)
    t_rad, R_rad = v2.hm.load_radius()
    out = {}

    # ---------------- 问题 1 ----------------
    te1 = np.arange(0, 1801, 1.0)
    s = v2.LagSolver(v2.PropsV2("p1"), v2.hm.ConstRadius(0.02), env, N=400,
                     mapping="geometric", cap_mode="fixed_vol")
    r1 = s.run(1800.0, v2.schedule_prod, t_eval=te1)
    T1, C1 = collect(s, r1, R_OUT)
    write_xlsx(os.path.join(RESULT_DIR, "result1.xlsx"), r1["t"], R_OUT, [T1, C1],
               ["温度", "水分浓度"])
    print("问题1 完成 (%.0f s)" % (time.time() - t0), flush=True)

    # ---------------- 问题 2/3 共用模型（附录 3、定尺寸）----------------
    props3 = v2.PropsV2("p3")
    solver2 = v2.LagSolver(props3, v2.hm.ConstRadius(0.02), env, N=400,
                           mapping="geometric", cap_mode="fixed_vol")
    te60 = np.arange(0, 120 * 3600 + 1, 60.0)
    r3 = solver2.run(120 * 3600.0, v2.schedule_prod, t_eval=te60)
    t_dry = dry_time(r3["t"], r3["C"])
    print("问题2/3 烘干时长 = %.4f h (%.1f s)" % (t_dry / 3600, t_dry), flush=True)
    out["问题2_烘干时长_h"] = t_dry / 3600
    out["问题3_烘干时长_h"] = t_dry / 3600

    n_out = int(np.floor(t_dry)) + 1
    Tfull = np.full((n_out, len(R_OUT)), np.nan, dtype=np.float32)
    Cfull = np.full((n_out, len(R_OUT)), np.nan, dtype=np.float32)
    state = {"i": 0}

    def save_fn(tt, T, C):
        k = state["i"]
        if k >= n_out:
            return
        if abs(tt - k) < 1e-9:
            To, Co = solver2.profiles_at_radii(tt, T, C, R_OUT)
            Tfull[k] = To
            Cfull[k] = Co
            state["i"] = k + 1

    t1 = time.time()
    solver2.run(t_dry, v2.schedule_1s, t_eval=np.arange(0, n_out, 1.0), save_fn=save_fn)
    m = state["i"]
    print("  全流程 1 s 求解完成: %d 行 (%.0f s)" % (m, time.time() - t1), flush=True)
    write_xlsx(os.path.join(RESULT_DIR, "result2.xlsx"), np.arange(m), R_OUT,
               [Tfull[:m], Cfull[:m]], ["温度", "水分浓度"])
    write_xlsx(os.path.join(RESULT_DIR, "result2_3h.xlsx"), np.arange(10801), R_OUT,
               [Tfull[:10801], Cfull[:10801]], ["温度", "水分浓度"])
    np.savez_compressed(os.path.join(WORK_OUT, "result2_full.npz"),
                        t=np.arange(m), T=Tfull[:m], C=Cfull[:m], radii=R_OUT)
    print("问题2 result2.xlsx (全流程 1 s) 完成", flush=True)

    # ---------------- 问题 3：每 60 s + 终点行 ----------------
    n3 = int(np.floor(t_dry / 60.0)) + 1
    te3 = np.concatenate([np.arange(0, n3 * 60.0, 60.0), [t_dry]])
    s3 = v2.LagSolver(props3, v2.hm.ConstRadius(0.02), env, N=400,
                      mapping="geometric", cap_mode="fixed_vol")
    r3b = s3.run(t_dry, v2.schedule_prod, t_eval=te3)
    T3b, C3b = collect(s3, r3b, R_OUT)
    write_xlsx(os.path.join(RESULT_DIR, "result3.xlsx"), r3b["t"], R_OUT, [C3b], ["水分浓度"])
    print("问题3 result3.xlsx (%d 行, 含终点行) 完成" % len(r3b["t"]), flush=True)

    # ---------------- 问题 4：基线（固定物理距离列 0—2.0 cm，材料外留空 + 表面位置表）----------------
    R_OUT4FULL = np.round(np.arange(0, 2.0001, 0.1), 4) / 100.0     # 0—2.0 cm（21 列）
    rad4 = v2.hm.RadiusLaw(t_rad, R_rad)

    def emit_p4(tag, solver, t_dry_x, te_x):
        rb = solver.run(t_dry_x, v2.schedule_prod, t_eval=te_x)
        _, Cm = collect(solver, rb, R_OUT4FULL, extra_surface=True)
        wb = openpyxl.Workbook(write_only=True)
        ws = wb.create_sheet("水分浓度")
        ws.append(["时间\\到药材中心的距离"] + [round(float(x) * 100.0, 4) for x in R_OUT4FULL] + ["药材表面"])
        for i, tt in enumerate(rb["t"]):
            ws.append([float(tt)] + [None if (v is None or not np.isfinite(v)) else round(float(v), 4)
                                     for v in Cm[i]])
        ws2 = wb.create_sheet("表面位置")
        ws2.append(["时间\\实际半径", "实际半径/cm"])
        for tt in rb["t"]:
            ws2.append([float(tt), round(float(rad4.R_of(tt)) * 100, 4)])
        wb.save(os.path.join(RESULT_DIR, tag))
        return rb

    s4 = v2.LagSolver(v2.PropsV2("p4"), rad4, env, N=400, mapping="geometric", cap_mode="mass")
    r4 = s4.run(120 * 3600.0, v2.schedule_prod, t_eval=te60)
    t_dry4 = out.get("问题4_基线_烘干时长_h", None) or dry_time(r4["t"], r4["C"]) / 3600.0
    n4 = int(np.floor(t_dry4 * 3600 / 60.0)) + 1
    te4 = np.concatenate([np.arange(0, n4 * 60.0, 60.0), [t_dry4 * 3600]])
    rb4 = emit_p4("result4.xlsx", s4, t_dry4 * 3600, te4)
    print("问题4 result4.xlsx (基线 %.4f h, 含表面位置表) 完成" % t_dry4, flush=True)

    s5 = v2.LagSolver(v2.PropsV2("p4"), rad4, env, N=400, mapping="local", cap_mode="mass")
    r5 = s5.run(120 * 3600.0, v2.schedule_prod, t_eval=te60)
    t_dry5 = dry_time(r5["t"], r5["C"]) / 3600.0
    out["问题4_v2非均匀收缩_烘干时长_h"] = t_dry5
    n5 = int(np.floor(t_dry5 * 3600 / 60.0)) + 1
    te5 = np.concatenate([np.arange(0, n5 * 60.0, 60.0), [t_dry5 * 3600]])
    rb5 = emit_p4("result4_local.xlsx", s5, t_dry5 * 3600, te5)
    print("问题4 result4_local.xlsx (v2 非均匀收缩 %.4f h) 完成" % t_dry5, flush=True)

    # 严格整秒停止时刻
    for tag, rr in [("问题3", r3), ("问题4基线", r4 if 'r4' in dir() else rb4)]:
        Cmax = rr["C"].max(axis=1)
        i = np.where(Cmax < 0.15)[0][0]
        out["%s_整秒停止时刻_s" % tag] = float(rr["t"][i])
        out["%s_停止时刻Cmax" % tag] = float(Cmax[i])
        print("   %s 严格达标首秒 = %.0f s, Cmax = %.10f" % (tag, rr["t"][i], Cmax[i]), flush=True)

    np.savez_compressed(os.path.join(WORK_OUT, "p3_prod.npz"),
                        t=r3["t"], C=r3["C"], T=r3["T"], s=solver2.s)
    np.savez_compressed(os.path.join(WORK_OUT, "p4_prod.npz"),
                        t=r4["t"], C=r4["C"], T=r4["T"], s=s4.s,
                        t5=r5["t"], C5=r5["C"], T5=r5["T"], s5=s5.s)
    json.dump(out, open(os.path.join(WORK_OUT, "v2_times.json"), "w"),
              ensure_ascii=False, indent=1)
    print("总用时 %.0f s" % (time.time() - t0))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
