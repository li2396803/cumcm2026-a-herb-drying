# -*- coding: utf-8 -*-
"""v4 新增插图：模型示意、数据图、时空场图（问题1/2/3/4）、干燥速率场、
   收缩域场图、验证图、敏感性热图。"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle, FancyArrow, Circle, Wedge
import herb_v2 as v2

rcParams["font.sans-serif"] = ["Songti SC", "Heiti SC", "PingFang SC"]
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 10
rcParams["figure.dpi"] = 130
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "out"))
FIG = os.path.abspath(os.path.join(HERE, "..", "..", "outputs", "figures"))
os.makedirs(FIG, exist_ok=True)
T_OVEN, T_INF, C_INF = v2.hm.load_oven()
T_RAD, R_RAD = v2.hm.load_radius()
RADLAW = v2.hm.RadiusLaw(T_RAD, R_RAD)


def save(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("  ->", name)


# ---------------------------------------------------------------- 模型示意
def fig_schematic():
    fig = plt.figure(figsize=(13.2, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.05], wspace=0.30)

    # (a) 几何与一维简化
    ax = fig.add_subplot(gs[0, 0])
    ax.add_patch(Rectangle((0.0, -1.45), 3.0, 2.9, color="#dfe9f5", ec="#1f4e79", lw=1.6))
    ax.plot([-0.05, 0.0], [-1.0, -1.0], "k-", lw=1.6)
    ax.plot([-0.05, 0.0], [1.0, 1.0], "k-", lw=1.6)
    ax.annotate("", xy=(0.05, 1.62), xytext=(2.95, 1.62), arrowprops=dict(arrowstyle="<->", lw=1.1))
    ax.text(1.5, 1.70, "长 25 cm", ha="center", fontsize=9)
    ax.annotate("", xy=(3.16, -1.0), xytext=(3.16, 1.0), arrowprops=dict(arrowstyle="<->", lw=1.1))
    ax.text(3.28, 0.0, "半径\n2 cm", va="center", fontsize=9)
    ax.plot([0.0, 3.0], [0, 0], "k--", lw=1.0)
    ax.text(1.5, 0.12, "中心轴 r = 0", ha="center", fontsize=8)
    for y in np.linspace(-1.0, 1.0, 7):
        ax.annotate("", xy=(3.0, y), xytext=(0.0, y), arrowprops=dict(arrowstyle="->", lw=0.7, color="#c62828"))
    ax.text(1.5, -1.72, "热风对流边界：$h=25$ W/(m²·K)，$k_m=8{\\times}10^{-7}$ m/s\n"
                        "$T_\\infty(t),\\ C_\\infty(t)$ 取自附件 1", ha="center", fontsize=8.5)
    ax.text(1.5, 1.98, "一维轴对称：长径比 25/4 = 6.25，忽略轴向", ha="center", fontsize=8.5, color="#1f4e79")
    ax.set_xlim(-0.5, 4.3); ax.set_ylim(-2.1, 2.3); ax.axis("off")
    ax.set_title("(a) 几何、简化与边界条件", fontsize=10.5)

    # (b) 材料坐标映射
    ax = fig.add_subplot(gs[0, 1])
    x0a, x0b = 0.0, 2.4
    Ra, Rb = 1.0, 0.62
    ax.add_patch(Circle((x0a, 0.35), Ra, fill=False, ec="#1f4e79", lw=1.8))
    for xi in np.linspace(0.2, 1.0, 5):
        ax.add_patch(Circle((x0a, 0.35), xi * Ra, fill=False, ec="#1f4e79", lw=0.6, alpha=0.7))
    ax.plot([x0a, x0a + Ra], [0.35, 0.35], "--", color="#1f4e79", lw=0.8)
    ax.text(x0a, -0.95, "初始 $R_0=2$ cm", ha="center", fontsize=8.5, color="#1f4e79")
    ax.add_patch(Circle((x0b, 0.35), Rb, fill=False, ec="#c62828", lw=1.8))
    for xi in np.linspace(0.2, 1.0, 5):
        ax.add_patch(Circle((x0b, 0.35), xi * Rb, fill=False, ec="#c62828", lw=0.6, alpha=0.7))
    ax.plot([x0b, x0b + Rb], [0.35, 0.35], "--", color="#c62828", lw=0.8)
    ax.text(x0b, -0.95, "当前 $R(t)=1.20$ cm", ha="center", fontsize=8.5, color="#c62828")
    ax.annotate("", xy=(x0a + Ra * 0.99, 0.35 + Ra * 0.42), xytext=(x0b + Rb * 0.99, 0.35 + Rb * 0.42),
                arrowprops=dict(arrowstyle="<->", lw=1.1, color="#2e7d32"))
    ax.text(1.5, 1.28, "同一材料层（$\\xi$ 或 $s$ 固定）", ha="center", fontsize=8.5, color="#2e7d32")
    ax.text(1.2, -1.55, "仿射：$r=\\xi R(t)$ ｜ 非均匀：$r=R(t)\\sqrt{J(s)}$，$J$ 由 $v(C)$ 决定",
            ha="center", fontsize=8.5)
    ax.set_xlim(-1.25, 3.4); ax.set_ylim(-1.9, 1.75); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(b) 收缩域与材料坐标", fontsize=10.5)

    # (c) 两阶段与边界数据
    ax = fig.add_subplot(gs[0, 2])
    env = v2.hm.Environment(T_OVEN, T_INF, C_INF)
    tt = np.linspace(0, 60, 600)
    ax.plot(tt, [float(env.T_inf(x * 3600)) for x in tt], "-", color="#1f4e79", lw=2.0,
            label="$T_\\infty$ 烘房温度")
    ax.axvspan(0, 4, color="orange", alpha=0.18)
    ax.text(1.0, 28.6, "预热\n平衡", fontsize=8.5, ha="center", va="bottom")
    ax.text(32, 28.6, "恒温干燥段（$T_\\infty$ 恒为 50.165 °C）", fontsize=8.5, ha="center", va="bottom")
    ax.set_xlabel("时间 / h"); ax.set_ylabel("温度 / °C"); ax.set_ylim(27, 53)
    ax2 = ax.twinx()
    ax2.plot(tt, [float(env.C_inf(x * 3600)) * 1000 for x in tt], "--", color="#a52a2a", lw=1.6,
             label="$C_\\infty$ 烘房湿度")
    ax2.set_ylabel("$C_\\infty$ / (10⁻³ kg/kg)", color="#a52a2a"); ax2.set_ylim(15, 55)
    ax2.tick_params(axis="y", colors="#a52a2a")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
    ax.grid(alpha=.3)
    ax.set_title("(c) 两阶段与边界条件", fontsize=10.5)
    save(fig, "v4_fig1_schematic.png")


# ---------------------------------------------------------------- 数据图
def fig_data():
    from scipy.signal import savgol_filter
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.5))
    ax[0].plot(T_OVEN / 3600, T_INF, ".", ms=2.5, color="#9e9e9e", label="附件 1 原始点")
    tk = np.linspace(0, T_OVEN[-1], 500)
    env = v2.hm.Environment(T_OVEN, T_INF, C_INF)
    ax[0].plot(tk / 3600, [float(env.T_inf(x)) for x in tk], "-", color="#1f4e79", lw=1.8, label="PCHIP 插值（采用）")
    ax[0].plot(T_OVEN / 3600, savgol_filter(T_INF, 21, 2), "--", color="#c62828", lw=1.5,
               label="Savitzky–Golay 平滑")
    ax[0].set_xlim(0, 4); ax[0].set_xlabel("时间 / h"); ax[0].set_ylabel("烘房温度 / °C")
    ax[0].set_title("(a) 附件 1：温度（平台段有 ±0.2 °C 波动）")
    ax[0].legend(fontsize=7.5); ax[0].grid(alpha=.3)
    ax[1].plot(T_OVEN / 3600, C_INF * 1000, ".", ms=2.5, color="#9e9e9e")
    ax[1].plot(tk / 3600, [float(env.C_inf(x)) * 1000 for x in tk], "-", color="#a52a2a", lw=1.8)
    ax[1].set_xlim(0, 4); ax[1].set_xlabel("时间 / h"); ax[1].set_ylabel("$C_\\infty$ / (10⁻³ kg/kg)")
    ax[1].set_title("(b) 附件 1：烘房水分浓度"); ax[1].grid(alpha=.3)
    dR = np.gradient(R_RAD, T_RAD)
    ax[2].plot(T_RAD / 3600, np.abs(dR) * 1e6, "-", color="#2e7d32", lw=1.8, label="$|\\dot R|$（差商）")
    ax[2].plot(T_RAD / 3600, np.abs([RADLAW.dR_of(x) for x in T_RAD]) * 1e6, "--", color="#1f4e79",
               lw=1.5, label="$|\\dot R|$（PCHIP 导数）")
    ax[2].set_yscale("log"); ax[2].set_xlabel("时间 / h"); ax[2].set_ylabel("$|\\dot R|$ / (10⁻⁶ m/s)")
    ax[2].set_title("(c) 附件 2：收缩速率（前 10 h 最大）"); ax[2].grid(alpha=.3, which="both")
    ax[2].legend(fontsize=8)
    save(fig, "v4_fig2_data.png")


# ---------------------------------------------------------------- 问题 1 场图
def fig_p1_field():
    import openpyxl
    cand = [os.path.join(HERE, "..", "..", "outputs", "result_files", "result1.xlsx"),
            os.path.join(HERE, "..", "..", "outputs", "result1.xlsx"),
            os.path.join(HERE, "..", "out", "result_files", "result1.xlsx")]
    path = next((c for c in cand if os.path.exists(c)), None)
    wb = openpyxl.load_workbook(path, read_only=True)
    def read(sheet):
        ws = wb[sheet]; rows = list(ws.iter_rows(values_only=True))
        return np.array(rows[0][1:], dtype=float) / 100.0, np.array([r[1:] for r in rows[1:]], dtype=float), \
            np.array([r[0] for r in rows[1:]], dtype=float)
    r, T, t = read("温度")
    _, C, _ = read("水分浓度")
    fig, ax = plt.subplots(1, 3, figsize=(13.4, 3.6))
    for k, (M, name, unit, cmap) in enumerate([(T, "温度", "°C", "inferno"), (C, "水分浓度", "kg/kg", "viridis_r")]):
        im = ax[k].pcolormesh(r * 100, t / 60, M, shading="auto", cmap=cmap)
        cb = fig.colorbar(im, ax=ax[k]); cb.set_label("%s / %s" % (name, unit))
        ax[k].set_xlabel("到药材中心距离 r / cm"); ax[k].set_ylabel("时间 / min")
        ax[k].set_title("(%s) %s时空场" % ("ab"[k], name))
    for tt, c in [(10, "w"), (20, "w"), (30, "w")]:
        i = int(np.argmin(np.abs(t - tt * 60)))
        ax[0].plot(r * 100, np.full_like(r, tt), color=c, lw=0.7, ls=":")
    ax[2].plot(t / 60, T[:, 0], "-", color="#c62828", lw=1.8, label="中心 $r=0$")
    ax[2].plot(t / 60, T[:, -1], "--", color="#1565c0", lw=1.8, label="表面 $r=2$ cm")
    ax[2].set_xlabel("时间 / min"); ax[2].set_ylabel("温度 / °C")
    ax[2].set_title("(c) 中心与表面温度"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    save(fig, "v4_fig3_p1_field.png")


# ---------------------------------------------------------------- 问题 2/3 场图
def fig_p23_field():
    d = np.load(os.path.join(OUT, "result2_full.npz"))
    t, T, C, radii = d["t"], d["T"], d["C"], d["radii"]
    m = t <= 60 * 3600
    fig, ax = plt.subplots(2, 2, figsize=(11.2, 6.6))
    m2 = t <= 4 * 3600
    for i, (M, name, cmap) in enumerate([(T, "温度 / °C", "inferno"), (C, "水分浓度 / (kg/kg)", "viridis_r")]):
        im = ax[0, i].pcolormesh(radii * 100, t[m2] / 3600, M[m2], shading="auto", cmap=cmap)
        fig.colorbar(im, ax=ax[0, i]).set_label(name)
        ax[0, i].set_xlabel("r / cm"); ax[0, i].set_ylabel("时间 / h")
        ax[0, i].set_title("(%s) 预热平衡段（0—4 h）%s" % ("ab"[i], name.split(" /")[0]))
    cs = ax[1, 1].pcolormesh(radii * 100, t[m] / 3600, C[m], shading="auto", cmap="viridis_r")
    fig.colorbar(cs, ax=ax[1, 1]).set_label("水分浓度 / (kg/kg)")
    cl = ax[1, 1].contour(radii * 100, t[m] / 3600, C[m], levels=[0.15, 0.5, 1.0, 1.5, 2.0],
                          colors="w", linewidths=0.8)
    ax[1, 1].clabel(cl, fmt="%.2f", fontsize=7)
    ax[1, 1].set_xlabel("r / cm"); ax[1, 1].set_ylabel("时间 / h")
    ax[1, 1].set_title("(d) 全过程水分场（0—60 h）与等值线（0.15 为达标阈值）")
    im = ax[1, 0].pcolormesh(radii * 100, t[m] / 3600, T[m], shading="auto", cmap="inferno")
    fig.colorbar(im, ax=ax[1, 0]).set_label("温度 / °C")
    ax[1, 0].set_xlabel("r / cm"); ax[1, 0].set_ylabel("时间 / h")
    ax[1, 0].set_title("(c) 全过程温度场（0—60 h）")
    save(fig, "v4_fig4_p23_field.png")


# ---------------------------------------------------------------- 干燥速率场
def fig_rate_field():
    d = np.load(os.path.join(OUT, "result2_full.npz"))
    t, C, radii = d["t"], d["C"], d["radii"]
    m = (t >= 60) & (t <= 60 * 3600)
    tt = t[m]
    rate = -np.gradient(C[m], tt, axis=0)
    fig, ax = plt.subplots(1, 3, figsize=(13.4, 3.6))
    im = ax[0].pcolormesh(radii * 100, tt / 3600, np.log10(np.maximum(rate, 1e-12)),
                          shading="auto", cmap="magma")
    fig.colorbar(im, ax=ax[0]).set_label("$\\log_{10}(-\\partial C/\\partial t)$ / (1/s)")
    ax[0].set_xlabel("r / cm"); ax[0].set_ylabel("时间 / h"); ax[0].set_title("(a) 干燥速率时空场")
    rate_c = -np.gradient(C[:, 0], t) * 3600
    msk = (C[:, 0] > 0.11) & (t > 120)
    ax[1].plot(C[msk, 0], rate_c[msk], "-", color="#6a1b9a", lw=2.0)
    ax[1].set_xlim(0.1, 2.6); ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("中心水分浓度 / (kg/kg)"); ax[1].set_ylabel("干燥速率 / (1/h)")
    ax[1].set_title("(b) 中心干燥特性曲线（降速干燥）"); ax[1].grid(alpha=.3, which="both")
    ax[2].plot(t / 3600, C[:, 0], "-", color="#c62828", lw=1.9, label="中心")
    ax[2].plot(t / 3600, C[:, -1], "--", color="#1565c0", lw=1.9, label="表面")
    ax[2].plot(t / 3600, (C * (0.5 * (np.concatenate([[0], 0.5 * (radii[1:] + radii[:-1])]) ** 2 -
                                     np.concatenate([[0], 0.5 * (radii[1:] + radii[:-1])]) ** 2) + 0)).sum(axis=1) if False else
               (C * (radii ** 2 - np.concatenate([[0], radii[:-1] ** 2])) / (radii[-1] ** 2)).sum(axis=1),
               "-.", color="#2e7d32", lw=1.9, label="截面平均")
    ax[2].axhline(0.15, color="k", ls=":", lw=1.3)
    ax[2].set_yscale("log"); ax[2].set_xlabel("时间 / h"); ax[2].set_ylabel("水分浓度 / (kg/kg)")
    ax[2].set_title("(c) 中心 / 表面 / 平均水分（平均判据不可用于达标判定）")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3, which="both"); ax[2].set_xlim(0, 60)
    save(fig, "v4_fig5_rate_field.png")


# ---------------------------------------------------------------- 问题 4 收缩域场图
def fig_p4_field():
    d = np.load(os.path.join(OUT, "p4_prod.npz"))
    t, C, s = d["t"], d["C"], d["s"]
    t5, C5, s5 = d["t5"], d["C5"], d["s5"]
    m = t <= 60 * 3600
    fig, ax = plt.subplots(1, 3, figsize=(13.6, 4.1))
    # (a) 物理坐标（收缩域）
    R = np.array([RADLAW.R_of(x) for x in t[m]])
    r_phys = np.sqrt(s)[None, :] * R[:, None] * 100          # cm
    im = ax[0].pcolormesh(r_phys, t[m] / 3600, C[m], shading="gouraud", cmap="viridis_r")
    ax[0].plot(R * 100, t[m] / 3600, "w-", lw=2.0, label="当前表面 $R(t)$")
    fig.colorbar(im, ax=ax[0]).set_label("水分浓度 / (kg/kg)")
    ax[0].set_xlabel("到药材中心距离 r / cm"); ax[0].set_ylabel("时间 / h")
    ax[0].set_title("(a) 物理坐标下的收缩域水分场（域随时间缩小）")
    ax[0].legend(fontsize=8)
    # (b) 材料坐标
    im = ax[1].pcolormesh(s5, t5[m] / 3600, C5[m], shading="auto", cmap="viridis_r")
    fig.colorbar(im, ax=ax[1]).set_label("水分浓度 / (kg/kg)")
    ax[1].set_xlabel("材料坐标 s（干物质质量分数）"); ax[1].set_ylabel("时间 / h")
    ax[1].set_title("(b) 材料坐标下的水分场（非均匀收缩模型）")
    # (c) 两模型差异
    D = C5[m] - C[m]
    im = ax[2].pcolormesh(s, t[m] / 3600, D, shading="auto", cmap="coolwarm",
                          vmin=-0.25, vmax=0.25)
    cb = fig.colorbar(im, ax=ax[2]); cb.set_label("局部模型 − 仿射模型 的水分浓度差 / (kg/kg)")
    ax[2].set_xlabel("材料坐标 s"); ax[2].set_ylabel("时间 / h")
    ax[2].set_title("(c) 非均匀收缩 vs 仿射：前期差异最大")
    save(fig, "v4_fig6_p4_field.png")


# ---------------------------------------------------------------- 验证图
def fig_verify():
    v2j = json.load(open(os.path.join(OUT, "v2_verify.json")))
    g = v2j["V1_grid"]
    Ns = [x["N"] for x in g]; td = [x["t_dry_h"] for x in g]
    e2 = json.load(open(os.path.join(OUT, "verify_02.json")))
    conv = e2["conv_N"]
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.6))
    xs = [c["N"] for c in conv]; ys = [c["eC"] for c in conv]
    ax[0].loglog(xs, ys, "o-", color="#1f4e79", lw=1.9, ms=6, label="问题 1 输出点 max|ΔC|")
    ax[0].loglog(xs, [ys[0] * (xs[0] / x) ** 2 for x in xs], "k--", lw=1.2, label="二阶参考斜率")
    ax[0].set_xlabel("径向网格数 N"); ax[0].set_ylabel("max|ΔC| / (kg/kg)")
    ax[0].set_title("(a) 空间网格收敛（二阶）"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, which="both")
    ax[1].plot(Ns, td, "o-", color="#a52a2a", lw=1.9, ms=6)
    for x, y in zip(Ns, td):
        ax[1].annotate("%.4f h" % y, (x, y), fontsize=8, xytext=(3, 4), textcoords="offset points")
    ax[1].set_xscale("log"); ax[1].set_xlabel("网格 N（收缩域）"); ax[1].set_ylabel("烘干时长 / h")
    ax[1].set_title("(b) 非均匀收缩模型网格收敛（±0.01 h）"); ax[1].grid(alpha=.3, which="both")
    d = np.load(os.path.join(OUT, "p3_prod.npz"))
    tt, C = d["t"], d["C"]
    Vc = None
    for k, tt_h in enumerate([6, 24, 48]):
        i = int(np.argmin(np.abs(tt - tt_h * 3600)))
        ax[2].plot(np.sqrt(d["s"]) * 2.0, C[i], "-", lw=1.8,
                   label="%.0f h" % tt_h)
        ax[2].plot(np.sqrt(d["s"] * 0 + d["s"]) * 2.0, C[i], ":", lw=1.0, alpha=0.6)
    ax[2].set_xlabel("到药材中心距离 r / cm（问题 3 固定域）"); ax[2].set_ylabel("水分浓度 / (kg/kg)")
    ax[2].set_title("(c) 典型时刻剖面"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    save(fig, "v4_fig7_verify.png")


# ---------------------------------------------------------------- 敏感性热图
def fig_sens_map():
    from scipy.signal import savgol_filter
    cache = os.path.join(OUT, "sens_map.json")
    if os.path.exists(cache):
        D = json.load(open(cache))
    else:
        D = {"T": [], "C": [], "td": []}
        print("    计算敏感性网格（5×5，约 8—12 min）...", flush=True)
        t_oven, T_inf, C_inf = T_OVEN, T_INF, C_INF
        for T_after in [45.0, 47.5, 50.0, 52.5, 55.0]:
            for C_after in [0.02, 0.04, 0.06, 0.08, 0.10]:
                class E(v2.hm.Environment):
                    def __init__(self):
                        super().__init__(t_oven, T_inf, C_inf)
                    def T_inf(self, t):
                        t = np.asarray(t)
                        return np.where(t <= self.t_hold, super().T_inf(t), T_after)
                    def C_inf(self, t):
                        t = np.asarray(t)
                        return np.where(t <= self.t_hold, super().C_inf(t), C_after)
                env = E()
                sol = v2.LagSolver(v2.PropsV2("p3"), v2.hm.ConstRadius(0.02), env, N=200,
                                   mapping="geometric", cap_mode="fixed_vol")
                rr = sol.run(120 * 3600.0, v2.schedule_prod,
                             t_eval=np.arange(0, 120 * 3600 + 1, 120.0))
                Cmax = rr["C"].max(axis=1)
                idx = np.where(Cmax < 0.15)[0]
                td = float(rr["t"][idx[0]]) / 3600 if idx.size else np.nan
                D["T"].append(T_after); D["C"].append(C_after); D["td"].append(td)
        json.dump(D, open(cache, "w"))
    T = np.array(D["T"]).reshape(5, 5); Cc = np.array(D["C"]).reshape(5, 5)
    td = np.array(D["td"]).reshape(5, 5)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    cf = ax.contourf(Cc, T, td, levels=14, cmap="RdYlGn_r")
    cl = ax.contour(Cc, T, td, levels=[45, 50, 55, 60, 65, 70], colors="k", linewidths=0.7)
    ax.clabel(cl, fmt="%.0f h", fontsize=8)
    fig.colorbar(cf, ax=ax).set_label("烘干时长 / h")
    ax.plot([0.04986], [50.165], "k*", ms=16, label="基准（附件 1 末值）")
    ax.set_xlabel("恒温段烘房湿度 $C_\\infty$ / (kg/kg)")
    ax.set_ylabel("恒温段烘房温度 $T_\\infty$ / °C")
    ax.set_title("烘干时长对恒温段温度/湿度的响应面（问题 3，PDE 计算）")
    ax.legend(fontsize=9)
    save(fig, "v4_fig8_sens_map.png")


# ---------------------------------------------------------------- 二维轴对称场图
def fig_2d_field():
    import verify_03_2d_axisym as v3d
    cache = os.path.join(OUT, "field2d.npz")
    te = np.array([6 * 3600., 24 * 3600., 48 * 3600., 72 * 3600.])
    if os.path.exists(cache):
        d = np.load(cache); T2, C2, r_c, z_c = d["T"], d["C"], d["r"], d["z"]
        print("    （读自缓存 field2d.npz）")
    else:
        print("    二维轴对称计算中（约 3—4 min）...", flush=True)
        env = v2.hm.Environment(T_OVEN, T_INF, C_INF)
        t2, T2, C2, r_c, z_c, ns = v3d.solve_2d(v2.hm.Props("p3"), env, te,
                                                nr=40, nz=60, dt_big=120.0)
        np.savez_compressed(cache, T=T2, C=C2, r=r_c, z=z_c)
    R, Z = np.meshgrid(r_c * 100, z_c * 100, indexing="ij")
    fig, axes = plt.subplots(2, 4, figsize=(14.6, 5.4))
    from matplotlib.colors import LogNorm
    for j, tt in enumerate(te):
        im = axes[0, j].pcolormesh(R, Z, T2[j] - 50.165, shading="auto", cmap="coolwarm",
                                   vmin=-1.0, vmax=1.0)
        axes[0, j].set_title("%.0f h" % (tt / 3600), fontsize=10)
        if j == 3:
            cb = fig.colorbar(im, ax=axes[0, j], format="%.1f")
            cb.set_label("$T-50.165$ / °C")
        im = axes[1, j].pcolormesh(R, Z, C2[j], shading="auto", cmap="viridis_r",
                                   norm=LogNorm(vmin=0.05, vmax=2.6))
        cs = axes[1, j].contour(R, Z, C2[j], levels=[0.15, 1.0, 2.0], colors="w", linewidths=0.7)
        axes[1, j].clabel(cs, fmt="%.2f", fontsize=6)
        if j == 3:
            cb = fig.colorbar(im, ax=axes[1, j], format="%.2f")
            cb.set_label("水分浓度 / (kg/kg)")
        for i in range(2):
            axes[i, j].set_xlabel("r / cm", fontsize=9)
            axes[i, j].set_xlim(0, 2); axes[i, j].set_ylim(0, 12.5)
            axes[i, j].tick_params(labelsize=8)
            if j == 0:
                axes[i, j].set_ylabel("z / cm（半长 12.5）", fontsize=9)
            else:
                axes[i, j].set_yticklabels([])
    axes[0, 0].text(-0.62, 6.2, "温度偏差场 (T−T∞)", fontsize=10, color="#c62828",
                    rotation=90, va="center", ha="center")
    axes[1, 0].text(-0.62, 6.2, "水分场 C(r,z,t)", fontsize=10, color="#2e7d32",
                    rotation=90, va="center", ha="center")
    fig.suptitle("二维轴对称场（半径 2 cm × 半长 12.5 cm，端面暴露）："
                 "端面附近（z > 10 cm）明显更干，中心截面（z≈0）由一维径向模型描述", fontsize=11)
    fig.tight_layout(rect=[0.02, 0, 1, 0.94])
    save(fig, "v4_fig9_2d_field.png", tight=False)


if __name__ == "__main__":
    b = sys.argv[1] if len(sys.argv) > 1 else "all"
    if b in ("all", "quick"):
        fig_schematic(); fig_data(); fig_p1_field(); fig_p23_field()
        fig_rate_field(); fig_p4_field(); fig_verify()
    if b in ("all", "sens"):
        fig_sens_map()
    if b in ("all", "2d"):
        fig_2d_field()
    print("完成")
