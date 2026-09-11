# -*- coding: utf-8 -*-
"""v2 能量审计（积分口径）：Q_conv = ΔE + 蒸发携带焓（潜热+显热）"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import herb_v2 as v2

OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out"))
T_OVEN, T_INF, C_INF = v2.hm.load_oven()
L_VAP = 2.4e6


def audit(mode, moving, mapping, cap_mode, t_end, N=400, label=""):
    env = v2.hm.Environment(T_OVEN, T_INF, C_INF)
    rad = v2.hm.RadiusLaw(*v2.hm.load_radius()) if moving else v2.hm.ConstRadius(0.02)
    props = v2.PropsV2(mode)
    solver = v2.LagSolver(props, rad, env, N=N, mapping=mapping, cap_mode=cap_mode)
    te = np.arange(0, t_end + 1, 60.0)
    r = solver.run(t_end, v2.schedule_prod, t_eval=te)
    t = r["t"]
    R_t = np.array([float(rad.R_of(x)) for x in t])
    Tinf_t = np.array([float(env.T_inf(x)) for x in t])
    Ts = r["T"][:, -1]
    q_conv = 2 * np.pi * R_t * 25.0 * (Tinf_t - Ts)
    E = np.zeros(len(t))
    for i in range(len(t)):
        if cap_mode == "fixed_vol":
            capT = props.rho(r["C"][i]) * props.cp(r["C"][i]) * solver.v0
        else:
            capT = (1.0 + r["C"][i]) * props.cp(r["C"][i])
        E[i] = np.sum(solver.dsm * capT * r["T"][i])
    Q_conv = float(np.trapezoid(q_conv, t))
    dE = float(E[-1] - E[0])
    m_evap = float(np.sum(solver.dsm * (r["C"][0] - r["C"][-1])))
    E_evap = m_evap * L_VAP
    print("【%s】 表面换热 Q_conv=%.4e J  内能变化 ΔE=%.4e J  蒸发水量=%.4f kg  潜热=%.4e J"
          % (label, Q_conv, dE, m_evap, E_evap))
    print("    Q_conv-ΔE = %.4e J  (与潜热之比 %.3f)" % (Q_conv - dE, (Q_conv - dE) / E_evap))
    return {"label": label, "Q_conv": Q_conv, "dE": dE, "m_evap": m_evap,
            "E_evap": E_evap, "ratio": (Q_conv - dE) / E_evap}


if __name__ == "__main__":
    out = [audit("p1", False, "geometric", "fixed_vol", 1800.0, label="问题1 (0-1800 s)"),
           audit("p3", False, "geometric", "fixed_vol", 60 * 3600.0, label="问题3 (0-60 h)"),
           audit("p4", True, "geometric", "mass", 60 * 3600.0, label="问题4 基线 (0-60 h)")]
    json.dump(out, open(os.path.join(OUT, "v2_energy.json"), "w"), ensure_ascii=False, indent=1)
