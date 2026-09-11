# -*- coding: utf-8 -*-
"""v2 插图：非均匀收缩 / 潜热非物理性 / 能量审计 / 工艺优化 Pareto"""
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
os.makedirs(FIG, exist_ok=True)
T_OVEN, T_INF, C_INF = v2.hm.load_oven()
T_RAD, R_RAD = v2.hm.load_radius()


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("  ->", name)


def get(mapping, cap, latent=False, N=400, te=None, mode="p4", moving=True):
    env = v2.hm.Environment(T_OVEN, T_INF, C_INF)
    rad = v2.hm.RadiusLaw(T_RAD, R_RAD) if moving else v2.hm.ConstRadius(0.02)
    s = v2.LagSolver(v2.PropsV2(mode), rad, env, N=N, mapping=mapping,
                     cap_mode=cap, latent=latent)
    r = s.run(120 * 3600.0, v2.schedule_prod, t_eval=te)
    return s, r, env, rad


def dry(t, C):
    Cmax = C.max(axis=1); i = np.where(Cmax < 0.15)[0][0]
    return float(t[i - 1] + (0.15 - Cmax[i - 1]) * (t[i] - t[i - 1]) / (Cmax[i] - Cmax[i - 1]))


def fig_shrink():
    te = np.arange(0, 60 * 3600 + 1, 300.0)
    s0, r0, _, rad = get("geometric", "mass", te=te)
    s1, r1, _, _ = get("local", "mass", te=te)
    td0, td1 = dry(r0["t"], r0["C"]), dry(r1["t"], r1["C"])
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.6))
    ax[0, 0].plot(r0["t"] / 3600, r0["C"][:, 0], "-", color="#1f4e79", lw=1.9, label="基线(仿射收缩)")
    ax[0, 0].plot(r1["t"] / 3600, r1["C"][:, 0], "-", color="#a52a2a", lw=1.9, label="v2 非均匀收缩")
    ax[0, 0].axhline(0.15, color="k", ls=":", lw=1.2)
    ax[0, 0].axvline(td0 / 3600, color="#1f4e79", ls="--", lw=1.1)
    ax[0, 0].axvline(td1 / 3600, color="#a52a2a", ls="--", lw=1.1)
    ax[0, 0].set_yscale("log"); ax[0, 0].set_xlabel("时间 / h"); ax[0, 0].set_ylabel("中心水分浓度 / (kg/kg)")
    ax[0, 0].set_title("(a) 中心干燥曲线：%.2f h vs %.2f h" % (td0 / 3600, td1 / 3600))
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3, which="both")
    for tag, r, c in [("基线", r1, "#1f4e79"), ("v2", r1, "#a52a2a")]:
        pass
    # 材料坐标映射对比（终态）
    for tag, r, c in [("仿射 (r/R=√s)", r0, "#1f4e79"), ("非均匀", r1, "#a52a2a")]:
        i = -1
        Rnow = float(rad.R_of(r["t"][i]))
        if tag.startswith("仿射"):
            xi = np.sqrt(s0.s)
        else:
            xi = s1.radii_of(r["t"][i], r["C"][i]) / Rnow
        ax[0, 1].plot(s0.s, xi, "-", color=c, lw=1.9, label=tag)
    ax[0, 1].set_xlabel("材料坐标 s（干物质质量分数）"); ax[0, 1].set_ylabel("r / R(t)")
    ax[0, 1].set_title("(b) 材料坐标→物理半径映射（终态）")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=.3)
    # 剖面
    for tt, c in [(24 * 3600, "#1565c0"), (48 * 3600, "#c62828")]:
        i0 = int(np.argmin(np.abs(r0["t"] - tt))); i1 = int(np.argmin(np.abs(r1["t"] - tt)))
        R0now = float(rad.R_of(tt))
        ax[1, 0].plot(s0.s * 0 + s0.s, r0["C"][i0], "-", color=c, lw=1.7, label="基线 %.0f h" % (tt / 3600))
        ax[1, 0].plot(s1.s, r1["C"][i1], "--", color=c, lw=1.7, label="v2 %.0f h" % (tt / 3600))
    ax[1, 0].set_xlabel("材料坐标 s"); ax[1, 0].set_ylabel("水分浓度 / (kg/kg)")
    ax[1, 0].set_title("(c) 材料坐标下的水分剖面")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=.3)
    v = v2.PropsV2("p4").v(np.linspace(0.01, 2.55, 200))
    ax[1, 1].plot(np.linspace(0.01, 2.55, 200), v * 1e3, "-", color="#2e7d32", lw=2.0)
    ax[1, 1].set_xlabel("水分浓度 C / (kg/kg)"); ax[1, 1].set_ylabel("局部比容 v = (1+C)/ρ(C) / (10⁻³ m³/kg)")
    ax[1, 1].set_title("(d) 附录 4 密度式导出的局部比容（非均匀收缩的驱动）")
    ax[1, 1].grid(alpha=.3)
    save(fig, "v2_fig1_shrinkage.png")


def fig_latent():
    te = np.arange(0, 4 * 3600 + 1, 20.0)
    s0, r0, env, rad = get("geometric", "mass", te=te, mode="p4", moving=True, latent=False)
    s1, r1, _, _ = get("geometric", "mass", te=te, mode="p4", moving=True, latent=True)
    Tinf = np.array([float(env.T_inf(x)) for x in r0["t"]])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    ax[0].plot(r0["t"] / 3600, r0["T"][:, -1], "-", color="#1f4e79", lw=1.9, label="基线（不含潜热）")
    ax[0].plot(r1["t"] / 3600, r1["T"][:, -1], "-", color="#c62828", lw=1.9, label="含蒸发潜热")
    ax[0].plot(r0["t"] / 3600, Tinf, ":", color="k", lw=1.4, label="烘房温度 T∞")
    ax[0].set_xlabel("时间 / h"); ax[0].set_ylabel("药材表面温度 / °C")
    ax[0].set_title("(a) 表面温度：含潜热时被冷却到环境以下")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(r0["t"] / 3600, r0["T"][:, 0], "-", color="#1f4e79", lw=1.9, label="基线 中心")
    ax[1].plot(r1["t"] / 3600, r1["T"][:, 0], "-", color="#c62828", lw=1.9, label="含潜热 中心")
    ax[1].axhline(25, color="gray", ls="--", lw=1.1, label="25 °C 参考")
    ax[1].set_xlabel("时间 / h"); ax[1].set_ylabel("中心温度 / °C")
    ax[1].set_title("(b) 中心温度：潜热项使整体降温")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    save(fig, "v2_fig2_latent.png")


def fig_energy():
    d = json.load(open(os.path.join(OUT, "v2_energy.json")))
    labels = [x["label"] for x in d]
    Q = [x["Q_conv"] for x in d]; Ee = [x["E_evap"] for x in d]; dE = [x["dE"] for x in d]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    x = np.arange(len(labels)); w = 0.27
    ax.bar(x - w, Q, w, label="表面对流供热 $Q_{conv}$", color="#1f4e79")
    ax.bar(x, dE, w, label="内能变化 $\\Delta E$", color="#6a1b9a")
    ax.bar(x + w, Ee, w, label="蒸发所需潜热 $m_{evap}L$", color="#c62828")
    ax.set_yscale("symlog", linthresh=1e4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("能量 / (J per kg dry)")
    ax.set_title("能量审计：题给换热系数所能提供的热量远小于蒸发需求（相差约 70 倍）")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    save(fig, "v2_fig3_energy.png")


def fig_optimize():
    d = json.load(open(os.path.join(OUT, "optimize.json")))
    par = d["pareto"]; base_h = d["t_dry_base_h"]
    base_e = (50.165 - 25) * base_h * 3600
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    xs = [p["pde_t_dry_h"] for p in par]; ys = [p["energy_proxy"] / 1e6 for p in par]
    ax[0].plot(xs, ys, "o-", color="#1f4e79", lw=1.9, ms=6, label="MILP 最优调度（PDE 复核）")
    ax[0].plot([base_h], [base_e / 1e6], "s", color="#c62828", ms=9, label="基准：恒温 50.17 °C")
    ts = [v / 3600 for v in d["t_dry_levels_h"].values()]
    es = [(float(k) - 25) * (v / 3600) * 3600 / 1e6 for k, v in d["t_dry_levels_h"].items()]
    ax[0].plot(ts, es, "^--", color="#9e9e9e", lw=1.5, ms=6, label="恒温策略 40—60 °C")
    for k, v in d["t_dry_levels_h"].items():
        ax[0].annotate("%d°C" % int(float(k)), (v / 3600, (float(k) - 25) * (v) / 1e6), fontsize=8,
                       xytext=(3, 4), textcoords="offset points")
    ax[0].set_xlabel("烘干时长（PDE 复核）/ h"); ax[0].set_ylabel("能耗代理 / (10⁶ K·s)")
    ax[0].set_title("(a) 能耗—时长 Pareto 前沿")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    p50 = [p for p in par if abs(p["budget_h"] - 50.0) < 1e-6][0]
    sched = p50["sched"][:p50["n_active"]]
    tt = 4 + 2 * np.arange(len(sched))
    ax[1].step(tt, sched, where="post", color="#c62828", lw=2.0, label="MILP 最优调度（预算 50 h）")
    ax[1].axhline(50.165, color="#1f4e79", ls="--", lw=1.6, label="基准恒温 50.17 °C")
    ax[1].axvspan(0, 16, color="gray", alpha=.15)
    ax[1].text(8, 41.5, "前 12 h 限温 ≤50 °C\n（保品质）", ha="center", fontsize=8)
    ax[1].set_xlabel("时间 / h"); ax[1].set_ylabel("烘房设定温度 / °C")
    ax[1].set_title("(b) 最优升温调度（PDE 复核 %.1f h）" % p50["pde_t_dry_h"])
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3); ax[1].set_ylim(38, 62)
    save(fig, "v2_fig4_optimize.png")


if __name__ == "__main__":
    print("生成 v2 插图:")
    fig_shrink(); fig_latent(); fig_energy(); fig_optimize()
    print("已保存到", FIG)
