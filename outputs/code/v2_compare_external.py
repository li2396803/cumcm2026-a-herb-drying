# -*- coding: utf-8 -*-
"""与外部独立实现（朋友模型）的差异归因：
  A. 问题 4：同一模型(仿射收缩)在不同数值设置下的烘干时长散布（网格/时间步/环境数据处理）
  B. 问题 3：平均含水率判据 vs 中心判据（复核朋友的 36.00 h / 0.1859 / 21.47 h）
  C. 严格整秒停止时刻与 C_max
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.signal import savgol_filter
import herb_v2 as v2

OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out"))
T_OVEN, T_INF, C_INF = v2.hm.load_oven()
T_RAD, R_RAD = v2.hm.load_radius()
RES = {}


def make_env(kind):
    if kind == "pchip":
        return v2.hm.Environment(T_OVEN, T_INF, C_INF)
    if kind == "linear":
        class E(v2.hm.Environment):
            def T_inf(self, t):
                return np.interp(np.clip(np.asarray(t), 0, self.t_hold), T_OVEN, T_INF)
            def C_inf(self, t):
                return np.interp(np.clip(np.asarray(t), 0, self.t_hold), T_OVEN, C_INF)
        return E(T_OVEN, T_INF, C_INF)
    if kind == "savgol":
        Ts = savgol_filter(T_INF, 21, 2); Cs = savgol_filter(C_INF, 21, 2)
        class E(v2.hm.Environment):
            def T_inf(self, t):
                return np.interp(np.clip(np.asarray(t), 0, self.t_hold), T_OVEN, Ts)
            def C_inf(self, t):
                return np.interp(np.clip(np.asarray(t), 0, self.t_hold), T_OVEN, Cs)
        return E(T_OVEN, T_INF, C_INF)
    if kind == "plateau_mean":     # 恒温段取平台均值 49.95 ℃
        class E(v2.hm.Environment):
            def T_inf(self, t):
                t = np.asarray(t)
                v = super().T_inf(np.minimum(t, self.t_hold))
                return np.where(t <= self.t_hold, v, 49.95)
        return E(T_OVEN, T_INF, C_INF)
    raise ValueError(kind)


def dry(t, C, thr=0.15):
    Cmax = C.max(axis=1); i = np.where(Cmax < thr)[0][0]
    return float(t[i - 1] + (thr - Cmax[i - 1]) * (t[i] - t[i - 1]) / (Cmax[i] - Cmax[i - 1]))


def stop_second(t, C, thr=0.15):
    Cmax = C.max(axis=1)
    idx = np.where(Cmax < thr)[0][0]
    t0 = t[idx]                          # 首个严格达标输出时刻
    return float(t0), float(Cmax[idx])


def run(mode, moving, mapping, cap, N, envkind, dt_scale=1.0, t_max=100 * 3600.0):
    env = make_env(envkind)
    rad = v2.hm.RadiusLaw(T_RAD, R_RAD) if moving else v2.hm.ConstRadius(0.02)
    s = v2.LagSolver(v2.PropsV2(mode), rad, env, N=N, mapping=mapping, cap_mode=cap)
    sch = (lambda t: v2.schedule_prod(t) * dt_scale)
    te = np.arange(0, t_max + 1, 60.0)
    r = s.run(t_max, sch, t_eval=te)
    return s, r


print("A. 问题4 M0(仿射) 在不同数值设置下的烘干时长:", flush=True)
rows = []
for N in [100, 200, 400, 800]:
    _, r = run("p4", True, "geometric", "mass", N, "pchip")
    td = dry(r["t"], r["C"])
    rows.append(("N=%d" % N, td / 3600))
    print("   N=%4d            -> %.4f h" % (N, td / 3600), flush=True)
for dts in [2.0, 4.0]:
    _, r = run("p4", True, "geometric", "mass", 400, "pchip", dt_scale=dts)
    td = dry(r["t"], r["C"])
    rows.append(("dt x%.0f" % dts, td / 3600))
    print("   N=400 dt x%.0f      -> %.4f h" % (dts, td / 3600), flush=True)
for k in ["linear", "savgol", "plateau_mean"]:
    _, r = run("p4", True, "geometric", "mass", 400, k)
    td = dry(r["t"], r["C"])
    rows.append(("env:" + k, td / 3600))
    print("   env=%-12s -> %.4f h" % (k, td / 3600), flush=True)
RES["A_p4_spread"] = rows
vals = [v for _, v in rows]
print("   => 散布 %.4f ~ %.4f h (极差 %.3f h, %.2f%%)"
      % (min(vals), max(vals), max(vals) - min(vals),
         100 * (max(vals) - min(vals)) / np.mean(vals)), flush=True)

print("\nB. 问题3 判据对照 (N=400):", flush=True)
s3, r3 = run("p3", False, "geometric", "fixed_vol", 400, "pchip")
t, C = r3["t"], r3["C"]
Vc = 0.5 * (s3.s_hi ** 2 - s3.s_lo ** 2)
Cbar = np.array([float(np.sum(C[i] * Vc)) for i in range(len(t))])
i_avg = np.where(Cbar < 0.15)[0][0]
t_avg = t[i_avg]
Ccenter_at_avg = float(C[i_avg, 0])
td_center = dry(t, C)
print("   平均含水率达 0.15: t = %.2f h, 此刻中心 = %.4f (朋友: 36.00 h / 0.1859)"
      % (t_avg / 3600, Ccenter_at_avg), flush=True)
print("   中心判据烘干时长 = %.4f h; 用平均值判定将低估 %.2f h (朋友: 21.47 h)"
      % (td_center / 3600, (td_center - t_avg) / 3600), flush=True)
RES["B_avg_vs_center"] = {"t_avg_h": t_avg / 3600, "Ccenter_at_avg": Ccenter_at_avg,
                          "t_center_h": td_center / 3600,
                          "underestimate_h": (td_center - t_avg) / 3600}

print("\nC. 严格整秒停止时刻:", flush=True)
for tag, rr in [("问题3", r3)]:
    ts, c0 = stop_second(rr["t"], rr["C"])
    print("   %s: t_stop = %.0f s, C_max = %.10f < 0.15" % (tag, ts, c0), flush=True)
    RES["C_stop_%s" % tag] = {"t_stop_s": ts, "Cmax": c0}
s4, r4 = run("p4", True, "geometric", "mass", 400, "pchip")
ts4, c4 = stop_second(r4["t"], r4["C"])
print("   问题4(基线): t_stop = %.0f s, C_max = %.10f (朋友: 183914 s / 0.1499997264)"
      % (ts4, c4), flush=True)
RES["C_stop_问题4基线"] = {"t_stop_s": ts4, "Cmax": c4}
s5, r5 = run("p4", True, "local", "mass", 400, "pchip")
ts5, c5 = stop_second(r5["t"], r5["C"])
print("   问题4(非均匀): t_stop = %.0f s, C_max = %.10f" % (ts5, c5), flush=True)
RES["C_stop_问题4非均匀"] = {"t_stop_s": ts5, "Cmax": c5}

json.dump(RES, open(os.path.join(OUT, "v2_compare.json"), "w"), ensure_ascii=False, indent=1)
print("\nsaved -> out/v2_compare.json")
