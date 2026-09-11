# -*- coding: utf-8 -*-
"""v2 补充检验：
  V1 非均匀收缩模型网格收敛性
  V2 二维轴对称（含 12.5 cm 端面）下的烘干时长
  V3 附件 1 环境数据"平滑 vs 插值"对结果的影响
  V4 能量审计（表面供热 vs 内能变化）
  V5 潜热情景对问题 1 表格的影响
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "..", "..", "outputs", "code")))
import numpy as np
import herb_v2 as v2

OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out"))
T_OVEN, T_INF, C_INF = v2.hm.load_oven()
T_RAD, R_RAD = v2.hm.load_radius()
RES = {}


def dry_time(t, C, thr=0.15):
    Cmax = C.max(axis=1)
    idx = np.where(Cmax < thr)[0]
    if idx.size == 0:
        return None
    i = idx[0]
    return float(t[i - 1] + (thr - Cmax[i - 1]) * (t[i] - t[i - 1]) / (Cmax[i] - Cmax[i - 1]))


def p4_local(N, te):
    env = v2.hm.Environment(T_OVEN, T_INF, C_INF)
    rad = v2.hm.RadiusLaw(T_RAD, R_RAD)
    s = v2.LagSolver(v2.PropsV2("p4"), rad, env, N=N, mapping="local", cap_mode="mass")
    r = s.run(120 * 3600.0, v2.schedule_prod, t_eval=te)
    return dry_time(r["t"], r["C"]), r


def main():
    t0 = time.time()
    te = np.arange(0, 120 * 3600 + 1, 60.0)

    # ---- V1 网格收敛 ----
    print("V1 非均匀收缩模型网格收敛:", flush=True)
    for N in [200, 400, 800]:
        td, _ = p4_local(N, te)
        print("   N=%4d -> t_dry = %.4f h" % (N, td / 3600), flush=True)
        RES.setdefault("V1_grid", []).append({"N": N, "t_dry_h": td / 3600})

    # ---- V3 环境数据平滑 ----
    print("V3 附件1 数据平滑影响:", flush=True)
    from scipy.signal import savgol_filter
    T_s = savgol_filter(T_INF, 21, 2)
    C_s = savgol_filter(C_INF, 21, 2)

    class EnvSmooth(v2.hm.Environment):
        def __init__(self, t, T, C):
            super().__init__(t, T, C)
            self._T2 = None

        def T_inf(self, tt):
            return np.interp(np.clip(np.asarray(tt), 0, self.t_hold), T_OVEN, T_s)

        def C_inf(self, tt):
            return np.interp(np.clip(np.asarray(tt), 0, self.t_hold), T_OVEN, C_s)

    for tag, env in [("原始PCHIP", v2.hm.Environment(T_OVEN, T_INF, C_INF)),
                     ("Savitzky-Golay平滑", EnvSmooth(T_OVEN, T_INF, C_INF))]:
        sol = v2.LagSolver(v2.PropsV2("p3"), v2.hm.ConstRadius(0.02), env, N=200,
                           mapping="geometric", cap_mode="fixed_vol")
        r = sol.run(120 * 3600.0, v2.schedule_prod, t_eval=te)
        td = dry_time(r["t"], r["C"])
        print("   %-18s t_dry = %.4f h" % (tag, td / 3600), flush=True)
        RES.setdefault("V3_smooth", []).append({"case": tag, "t_dry_h": td / 3600})

    # ---- V4 能量审计（基线：问题 3） ----
    print("V4 能量审计 (问题 3, N=400):", flush=True)
    env = v2.hm.Environment(T_OVEN, T_INF, C_INF)
    sol = v2.LagSolver(v2.PropsV2("p3"), v2.hm.ConstRadius(0.02), env, N=400,
                       mapping="geometric", cap_mode="fixed_vol")
    r = sol.run(60 * 3600.0, v2.schedule_prod, t_eval=np.arange(0, 60 * 3600 + 1, 60.0))
    Res = np.array([0.0]) if False else None
    # 内能：Σ ρcp V T（按控制体体积）
    E = []
    for i, tt in enumerate(r["t"]):
        Tv, Cv = r["T"][i], r["C"][i]
        rho_cp = v2.PropsV2("p3").rho(Cv) * v2.PropsV2("p3").cp(Cv)
        Vcell = 0.5 * (sol.s_hi ** 2 - sol.s_lo ** 2) * 0.02 ** 2   # pi*H 归一
        E.append(float(np.sum(rho_cp * Vcell * Tv)))
    E = np.array(E)
    dE = np.gradient(E, r["t"])
    Tinf_t = np.array([float(env.T_inf(tt)) for tt in r["t"]])
    q_in = 2 * np.pi * 0.02 * 25.0 * (Tinf_t - r["T"][:, -1])
    # 蒸发带走的焓（按表面水分通量 × L）
    prop = v2.PropsV2("p3")
    Cinf_t = np.array([float(env.C_inf(tt)) for tt in r["t"]])
    q_lat = 2 * np.pi * 0.02 * (1.0 / prop.v(r["C"][:, -1])) * 8e-7 * (r["C"][:, -1] - Cinf_t) * 2.4e6
    resid = (dE - q_in)
    scale = np.maximum(np.abs(q_in), 1e-6)
    print("   平均 |dE/dt - q_in|/q_in = %.2e（不含蒸发潜热项，故残差主要由蒸发与 ρcp 变化造成）"
          % np.mean(np.abs(resid[1:-1]) / scale[1:-1]), flush=True)
    print("   蒸发潜热通量量级: 平均 %.0f W/m, 表面换热通量量级: 平均 %.0f W/m"
          % (np.mean(q_lat), np.mean(np.abs(q_in))), flush=True)
    RES["V4_energy"] = {"mean_resid": float(np.mean(np.abs(resid[1:-1]) / scale[1:-1])),
                        "mean_q_latent": float(np.mean(q_lat)),
                        "mean_q_conv": float(np.mean(np.abs(q_in)))}

    # ---- V5 潜热对问题 1 表格的影响 ----
    print("V5 潜热情景 (问题 1):", flush=True)
    te1 = np.array([600., 1800.])
    for lat in [False, True]:
        s1 = v2.LagSolver(v2.PropsV2("p1"), v2.hm.ConstRadius(0.02), env, N=400,
                          mapping="geometric", cap_mode="fixed_vol", latent=lat)
        r1 = s1.run(1800.0, v2.schedule_prod, t_eval=te1)
        To, Co = s1.profiles_at_radii(1800.0, r1["T"][-1], r1["C"][-1],
                                      np.array([0.0, 0.01, 0.02]))
        print("   潜热=%-5s : 1800 s 表面温度 %.4f °C, 表面含水率 %.4f, 中心温度 %.4f"
              % (lat, To[-1], Co[-1], To[0]), flush=True)
        RES.setdefault("V5_latent_p1", []).append(
            {"latent": lat, "T_surf": float(To[-1]), "C_surf": float(Co[-1]),
             "T_center": float(To[0])})

    # ---- V2 二维含端面（问题 3）----
    print("V2 二维轴对称（含 12.5 cm 端面）问题 3:", flush=True)
    import verify_03_2d_axisym as v3d
    env2 = v2.hm.Environment(T_OVEN, T_INF, C_INF)
    te2d = np.array([6 * 3600., 24 * 3600., 48 * 3600., 60 * 3600.])
    t2, T2, C2, r_c, z_c, ns = v3d.solve_2d(v2.hm.Props("p3"), env2, te2d,
                                            nr=40, nz=60, dt_big=120.0)
    for i, tt in enumerate(t2):
        print("   t=%5.1f h: 2D 中心 C=%.5f  2D 端面内侧(z=12.4cm) C=%.5f"
              % (tt / 3600, C2[i, 0, 0], C2[i, 0, -1]), flush=True)
    print("   2D 最干点(端面)在 60 h 已达 %.5f，中心为 %.5f；一维模型中心 57.147 h 达标"
          % (C2[-1, 0, -1], C2[-1, 0, 0]), flush=True)
    RES["V2_2d"] = [{"t_h": float(tt / 3600), "C_center": float(C2[i, 0, 0]),
                     "C_endface": float(C2[i, 0, -1])} for i, tt in enumerate(t2)]

    json.dump(RES, open(os.path.join(OUT, "v2_verify.json"), "w"), ensure_ascii=False, indent=1)
    print("总用时 %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
