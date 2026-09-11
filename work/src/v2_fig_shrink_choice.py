# -*- coding: utf-8 -*-
"""说明文档插图：问题4 收缩建模两种合规做法（+非合规自洽做法）的对照"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import herb_v2 as v2

rcParams["font.sans-serif"] = ["Songti SC", "Heiti SC", "PingFang SC"]
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 10.5
rcParams["figure.dpi"] = 130
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "out"))
FIG = os.path.abspath(os.path.join(HERE, "..", "..", "outputs", "figures"))
T_OVEN, T_INF, C_INF = v2.hm.load_oven()
T_RAD, R_RAD = v2.hm.load_radius()
radlaw = v2.hm.RadiusLaw(T_RAD, R_RAD)


def run(mapping, N=400):
    env = v2.hm.Environment(T_OVEN, T_INF, C_INF)
    s = v2.LagSolver(v2.PropsV2("p4"), radlaw, env, N=N, mapping=mapping, cap_mode="mass")
    te = np.arange(0, 80 * 3600 + 1, 300.0)
    r = s.run(80 * 3600.0, v2.schedule_prod, t_eval=te)
    return s, r


def dry(t, C, thr=0.15):
    Cmax = C.max(axis=1); i = np.where(Cmax < thr)[0][0]
    return float(t[i - 1] + (thr - Cmax[i - 1]) * (t[i] - t[i - 1]) / (Cmax[i] - Cmax[i - 1]))


sA, rA = run("geometric")     # 做法A
sB, rB = run("local")         # 做法B
SCd = np.load(os.path.join(OUT, "selfconsistent.npz"))
tdA, tdB = dry(rA["t"], rA["C"]), dry(rB["t"], rB["C"])
tdC = json.load(open(os.path.join(OUT, "v2_selfconsistent.json")))["t_dry_h"] * 3600

fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.9))

# (a) 半径轨迹：附件2 vs 自洽预测
t2 = np.array([0, 1800, 3600, 7200, 14400, 21600, 43200, 64800, 86400, 129600, 172800])
ax[0].plot(t2 / 3600, [radlaw.R_of(x) * 100 for x in t2], "o-", color="#1f4e79", lw=2.0,
           ms=5, label="附件 2 实测 $R(t)$")
i_sc = SCd["t"] <= 80 * 3600
ax[0].plot(SCd["t"][i_sc] / 3600, SCd["R_pred"][i_sc], "-", color="#c62828", lw=2.0,
           label="附录 4 密度式自洽预测")
ax[0].set_xlabel("时间 / h"); ax[0].set_ylabel("半径 / cm")
ax[0].set_title("(a) 两个数据源不一致：密度式预测的收缩明显偏慢")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3); ax[0].set_xlim(0, 60)

# (b) 中心干燥曲线
ax[1].plot(rA["t"] / 3600, rA["C"][:, 0], "-", color="#1f4e79", lw=2.0, label="做法A 仿射（%.2f h）" % (tdA / 3600))
ax[1].plot(rB["t"] / 3600, rB["C"][:, 0], "-", color="#a52a2a", lw=2.0, label="做法B 局部（%.2f h）" % (tdB / 3600))
ax[1].plot(SCd["t"] / 3600, SCd["Ccenter"], "--", color="#7b1fa2", lw=1.8,
           label="仅用密度式（%.2f h，不合规）" % (tdC / 3600))
ax[1].axhline(0.15, color="k", ls=":", lw=1.3)
ax[1].set_yscale("log"); ax[1].set_ylim(0.05, 3)
ax[1].set_xlabel("时间 / h"); ax[1].set_ylabel("中心水分浓度 / (kg/kg)")
ax[1].set_title("(b) 三种做法的中心干燥曲线")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which="both")

# (c) 与"均匀映射"的偏差（终态）
for tag, s, r, c in [("A 仿射", sA, rA, "#1f4e79"), ("B 局部", sB, rB, "#a52a2a")]:
    i = -1
    Rnow = float(radlaw.R_of(r["t"][i]))
    xi = s.radii_of(r["t"][i], r["C"][i]) / Rnow
    ax[2].plot(s.s, (xi - np.sqrt(s.s)) * 100, "-", color=c, lw=2.0,
               label=tag + "（终态）")
ax[2].axhline(0, color="k", ls=":", lw=1.2)
ax[2].set_xlabel("材料坐标 s（干物质质量分数）")
ax[2].set_ylabel("(r/R − √s) × 100　/%")
ax[2].set_title("(c) 相对均匀映射的偏差：仅 0.5 % 的几何差异\n却使烘干时长改变 6.3 %（尾段极敏感）")
ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "v3_fig_shrink_choice.png"), bbox_inches="tight")
print("->", os.path.join(FIG, "v3_fig_shrink_choice.png"))
print("做法A %.4f h | 做法B %.4f h | 自洽 %.4f h" % (tdA / 3600, tdB / 3600, tdC / 3600))
