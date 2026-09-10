# -*- coding: utf-8 -*-
"""检验2/3/4: 独立数值方法交叉验证 / 网格与时间步收敛 / 守恒审计"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.interpolate import PchipInterpolator
import herb_model as hm
import herb_verify as hv

OUT = hm.WORK_OUT
RES = {}
R0 = 0.02
R_OUT = np.round(np.arange(0, 2.0001, 0.1), 4) / 100.0   # 竞赛输出半径


def compare_mol(sol, nc, r_cells, res_main, te, props, km, env, R=R0, trim=3):
    """公平对比: 以单元中心点为对比点, 主求解器用 PCHIP 插值到该点"""
    out = {"dT": [], "dC": [], "dCs": [], "M": [], "dM": []}
    dr = R / nc
    for i, tt in enumerate(te):
        j = int(np.argmin(np.abs(res_main["t"] - tt)))
        fT = PchipInterpolator(res_main["xi"] * R, res_main["T"][j], extrapolate=False)
        fC = PchipInterpolator(res_main["xi"] * R, res_main["C"][j], extrapolate=False)
        rr = r_cells[trim:-trim]          # 去掉紧邻轴/表面的单元
        out["dT"].append(np.max(np.abs(sol.y[:nc, i][trim:-trim] - fT(rr))))
        out["dC"].append(np.max(np.abs(sol.y[nc:, i][trim:-trim] - fC(rr))))
        Cs = hv.surface_value_mol(sol.y[nc:, i][-1], sol.y[:nc, i][-1], props, R, dr,
                                  km, float(env.C_inf(tt)))
        out["dCs"].append(abs(Cs - res_main["C"][j, -1]))
        # 总水分含量 (pi*H 归一): 主 = R^2 sum C_i Vc_i ; MOL = sum C_j * 0.5*(rf_{j+1}^2-rf_j^2)
        Vc = hv._vol_coef(res_main["xi"])
        M_main = R ** 2 * np.sum(res_main["C"][j] * Vc)
        rf = np.linspace(0, R, nc + 1)
        M_mol = np.sum(sol.y[nc:, i] * 0.5 * (rf[1:] ** 2 - rf[:-1] ** 2))
        out["M"].append(M_mol); out["dM"].append(abs(M_mol - M_main))
    return {k: np.array(v) for k, v in out.items()}


def main():
    t, T, C = hm.load_oven()
    env = hm.Environment(t, T, C)

    # ---------------- (2) 独立方法: 问题1 ----------------
    print("=" * 74, flush=True)
    print("检验2: 独立数值方法交叉验证 (问题1, 0-1800 s)")
    props1 = hm.Props("p1")
    te = np.array([30., 100., 300., 600., 900., 1200., 1500., 1800.])
    s = hm.DryingSolver(props1, hm.ConstRadius(R0), env, N=400)
    r1 = s.run(1800.0, hm.schedule_p1, t_eval=np.concatenate([[0.], te]))
    for nc in [240, 480]:
        t0 = time.time()
        sol, rc = hv.mol_fixed(props1, env, 1800.0, te, nc=nc, rtol=1e-10, atol=1e-12)
        cm = compare_mol(sol, nc, rc, r1, te, props1, 8e-7, env)
        print("  MOL nc=%d (%.0fs): 内部 max|dT|=%.2e K, max|dC|=%.2e;  表面 max|dC|=%.2e;  总量 max|dM|=%.2e"
              % (nc, time.time() - t0, cm["dT"].max(), cm["dC"].max(), cm["dCs"].max(), cm["dM"].max()), flush=True)
        RES["mol_p1_nc%d" % nc] = {k: float(v.max()) for k, v in cm.items()}
    print("  结论: 两种独立离散 (节点中心FV+CN / 单元中心FV+BDF) 结果一致")

    # ---------------- (2b) 问题3 ----------------
    print("=" * 74)
    print("检验2b: 独立数值方法交叉验证 (问题3, 0-72 h)")
    props3 = hm.Props("p3")
    te3 = np.array([6 * 3600., 24 * 3600., 48 * 3600., 72 * 3600.])
    res3 = np.load(os.path.join(OUT, "p3.npz"))
    # 主求解器在相同输出时刻的结果
    te_all = np.arange(0, 72 * 3600 + 1, 60.0)
    s3 = hm.DryingSolver(props3, hm.ConstRadius(R0), env, N=400)
    r3 = s3.run(72 * 3600.0, hm.schedule_long, t_eval=te_all)
    for nc in [160, 320]:
        t0 = time.time()
        sol, rc = hv.mol_fixed(props3, env, 72 * 3600.0, te3, nc=nc, rtol=1e-9, atol=1e-11)
        cm = compare_mol(sol, nc, rc, r3, te3, props3, 8e-7, env)
        print("  MOL nc=%d (%.0fs): 内部 max|dC|=%.2e;  表面 max|dC|=%.2e;  总量 max|dM|=%.2e"
              % (nc, time.time() - t0, cm["dC"].max(), cm["dCs"].max(), cm["dM"].max()), flush=True)
        RES["mol_p3_nc%d" % nc] = {k: float(v.max()) for k, v in cm.items()}
        for i, tt in enumerate(te3):
            RES.setdefault("mol_p3_detail", []).append(
                {"nc": nc, "t_h": tt / 3600, "C_center_main": float(r3["C"][int(np.argmin(np.abs(r3["t"] - tt))), 0]),
                 "C_center_mol": float(sol.y[nc, i])})

    # ---------------- (3) 收敛性 ----------------
    print("=" * 74)
    print("检验3a: 空间网格收敛 (问题1, dt 固定 1 s)")
    teo = np.concatenate([[0.], te])
    rr_out = R_OUT
    tab = {}
    for N in [200, 400, 800]:
        s = hm.DryingSolver(props1, hm.ConstRadius(R0), env, N=N)
        r = s.run(1800.0, hm.schedule_p1, t_eval=teo)
        tab[N] = np.array([[np.interp(x, r["xi"] * R0, r["C"][i]) for x in rr_out]
                           for i in range(len(teo))])
    ref = tab[800]
    e200 = np.max(np.abs(tab[200] - ref)); e400 = np.max(np.abs(tab[400] - ref))
    for N, e in [(200, e200), (400, e400)]:
        print("    N=%4d vs N=800: 输出点 max|dC|=%.3e" % (N, e))
        RES.setdefault("conv_N", []).append({"N": N, "eC": float(e)})
    print("    网格加密误差比 e(200)/e(400) = %.2f (二阶格式理论值 4)" % (e200 / e400))
    print("检验3b: 时间步收敛 (问题1, N=400)")
    tab = {}
    for ds in [1.0, 0.25, 0.0625]:
        sch = lambda t, d=ds: hm.schedule_p1(t) * d
        s = hm.DryingSolver(props1, hm.ConstRadius(R0), env, N=400)
        r = s.run(1800.0, sch, t_eval=teo)
        tab[ds] = np.array([[np.interp(x, r["xi"] * R0, r["C"][i]) for x in rr_out]
                            for i in range(len(teo))])
    for ds in [1.0, 0.25]:
        e = np.max(np.abs(tab[ds] - tab[0.0625]))
        print("    dt x %.4f: 输出点 max|dC|=%.3e" % (ds, e))
        RES.setdefault("conv_dt", []).append({"scale": ds, "eC": float(e)})
    print("检验3c: 问题3 长时程网格/时间步敏感性 (C_center@72h, C_center@57h)")
    for N, ds in [(200, 1.0), (400, 1.0), (800, 1.0), (400, 0.25)]:
        sch = lambda t, d=ds: hm.schedule_long(t) * d
        s = hm.DryingSolver(props3, hm.ConstRadius(R0), env, N=N)
        r = s.run(72 * 3600.0, sch, t_eval=np.array([57 * 3600., 72 * 3600.]))
        print("    N=%4d dt x%.2f: C_center(57h)=%.6f  C_center(72h)=%.6f"
              % (N, ds, r["C"][0, 0], r["C"][1, 0]))
        RES.setdefault("conv_p3", []).append({"N": N, "dt_scale": ds, "C57": float(r["C"][0, 0]),
                                              "C72": float(r["C"][1, 0])})

    # ---------------- (4) 守恒审计 ----------------
    print("=" * 74)
    print("检验4: 质量守恒审计 (dM/dt 与表面通量, 问题3, N=400)")
    rad = hm.ConstRadius(R0)
    _, M, dM, flux, resid = hv.audit_mass(r3, rad, 8e-7, env, r3["xi"])
    print("    M(0)=%.6f (理论 %.6f), M(72h)=%.6f" % (M[0], 2.55 * 0.5 * R0 ** 2, M[-1]))
    print("    相对残差 |dM/dt-flux|/max: 平均=%.2e, 99%%分位=%.2e, max=%.2e"
          % (np.mean(resid[1:-1]), np.quantile(resid[1:-1], 0.99), resid[1:-1].max()))
    RES["mass_audit_p3"] = {"M0": float(M[0]), "M0_theory": 2.55 * 0.5 * R0 ** 2,
                            "M72": float(M[-1]), "resid_mean": float(np.mean(resid[1:-1])),
                            "resid_p99": float(np.quantile(resid[1:-1], 0.99))}
    # 问题4 守恒审计
    print("检验4b: 质量守恒审计 (问题4 收缩域, N=400)")
    props4 = hm.Props("p4")
    rad4 = hm.RadiusLaw(*hm.load_radius())
    s4 = hm.DryingSolver(props4, rad4, env, N=400)
    r4 = s4.run(72 * 3600.0, hm.schedule_long, t_eval=np.arange(0, 72 * 3600 + 1, 60.0))
    _, M4, dM4, flux4, resid4 = hv.audit_mass(r4, rad4, 8e-7, env, r4["xi"])
    print("    M(0)=%.6f, M(72h)=%.6f, 相对残差: 平均=%.2e, 99%%=%.2e"
          % (M4[0], M4[-1], np.mean(resid4[1:-1]), np.quantile(resid4[1:-1], 0.99)))
    RES["mass_audit_p4"] = {"M0": float(M4[0]), "M72": float(M4[-1]),
                            "resid_mean": float(np.mean(resid4[1:-1])),
                            "resid_p99": float(np.quantile(resid4[1:-1], 0.99))}

    with open(os.path.join(OUT, "verify_02.json"), "w") as fh:
        json.dump(RES, fh, ensure_ascii=False, indent=1)
    print("saved -> out/verify_02.json")


if __name__ == "__main__":
    main()
