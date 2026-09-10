# -*- coding: utf-8 -*-
"""生成说明文档所需的全部插图."""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import herb_model as hm

rcParams["font.sans-serif"] = ["Songti SC", "Heiti SC", "PingFang SC", "Arial Unicode MS"]
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 10.5
rcParams["figure.dpi"] = 130

OUT = hm.WORK_OUT
FIG = hm.FIG_DIR
os.makedirs(FIG, exist_ok=True)

t_oven, T_oven, C_oven = hm.load_oven()
t_rad, R_rad = hm.load_radius()


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  ->", name)


# ---------------------------------------------------------------- 图1 环境
def fig_env():
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.3))
    ax[0].plot(t_oven / 3600, T_oven, "-", color="#1f4e79", lw=1.8)
    ax[0].set_xlabel("时间 / h"); ax[0].set_ylabel("烘房温度 / °C")
    ax[0].set_title("(a) 附件1 烘房温度")
    ax[0].grid(alpha=.3)
    ax[1].plot(t_oven / 3600, C_oven, "-", color="#a52a2a", lw=1.8)
    ax[1].set_xlabel("时间 / h"); ax[1].set_ylabel("烘房水分浓度 / (kg/kg)")
    ax[1].set_title("(b) 附件1 烘房水分浓度")
    ax[1].grid(alpha=.3)
    save(fig, "fig1_env.png")


def fig_radius():
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(t_rad / 3600, R_rad * 100, "-o", ms=2.5, color="#2e7d32", lw=1.8)
    ax.set_xlabel("时间 / h"); ax.set_ylabel("药材半径 R(t) / cm")
    ax.set_title("附件2 药材半径随烘干时间的变化（收缩）")
    ax.grid(alpha=.3)
    save(fig, "fig2_radius.png")


# ---------------------------------------------------------------- 图2 问题1
def fig_p1():
    d = np.load(os.path.join(OUT, "p1.npz"))
    t, T, C, xi = d["t"], d["T"], d["C"], d["xi"]
    r = xi * 0.02
    ts = [100, 300, 600, 900, 1200, 1500, 1800]
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.6))
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(ts)))
    for c, tt in zip(cmap, ts):
        i = int(np.argmin(np.abs(t - tt)))
        ax[0].plot(r * 100, T[i], "-", color=c, lw=1.7, label="%d s" % tt)
        ax[1].plot(r * 100, C[i], "-", color=c, lw=1.7, label="%d s" % tt)
    ax[0].set_xlabel("到药材中心距离 r / cm"); ax[0].set_ylabel("温度 / °C")
    ax[1].set_xlabel("到药材中心距离 r / cm"); ax[1].set_ylabel("水分浓度 / (kg/kg)")
    ax[0].set_title("(a) 预热平衡阶段温度分布"); ax[1].set_title("(b) 预热平衡阶段水分浓度分布")
    ax[0].legend(fontsize=8, ncol=2); ax[1].legend(fontsize=8, ncol=2)
    ax[0].grid(alpha=.3); ax[1].grid(alpha=.3)
    save(fig, "fig3_p1.png")


# ---------------------------------------------------------------- 图3 问题2
def fig_p2():
    d = np.load(os.path.join(OUT, "p2.npz"))
    t, T, C, xi = d["t"], d["T"], d["C"], d["xi"]
    r = xi * 0.02
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.4))
    cmap = plt.cm.plasma(np.linspace(0.05, 0.85, 6))
    for c, th in zip(cmap, [0.5, 1, 1.5, 2, 2.5, 3]):
        i = int(np.argmin(np.abs(t - th * 3600)))
        ax[0, 0].plot(r * 100, T[i], "-", color=c, lw=1.7, label="%.1f h" % th)
        ax[0, 1].plot(r * 100, C[i], "-", color=c, lw=1.7, label="%.1f h" % th)
    ax[0, 0].set_title("(a) 3 h 内温度分布"); ax[0, 0].set_ylabel("温度 / °C")
    ax[0, 1].set_title("(b) 3 h 内水分浓度分布"); ax[0, 1].set_ylabel("水分浓度 / (kg/kg)")
    for a in ax[0]:
        a.set_xlabel("到药材中心距离 r / cm"); a.legend(fontsize=8); a.grid(alpha=.3)
    ax[1, 0].plot(t / 3600, T[:, 0], "-", color="#c62828", lw=1.8, label="中心 r=0")
    ax[1, 0].plot(t / 3600, T[:, -1], "--", color="#1565c0", lw=1.8, label="表面 r=2 cm")
    ax[1, 0].set_xlabel("时间 / h"); ax[1, 0].set_ylabel("温度 / °C")
    ax[1, 0].set_title("(c) 中心与表面温度变化"); ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=.3)
    ax[1, 1].plot(t / 3600, C[:, 0], "-", color="#c62828", lw=1.8, label="中心 r=0")
    ax[1, 1].plot(t / 3600, C[:, -1], "--", color="#1565c0", lw=1.8, label="表面 r=2 cm")
    ax[1, 1].set_xlabel("时间 / h"); ax[1, 1].set_ylabel("水分浓度 / (kg/kg)")
    ax[1, 1].set_title("(d) 中心与表面水分浓度变化"); ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=.3)
    save(fig, "fig4_p2.png")


# ---------------------------------------------------------------- 图4 问题3
def fig_p3():
    d = np.load(os.path.join(OUT, "p3.npz"))
    t, C, T, xi = d["t"], d["C"], d["T"], d["xi"]
    r = xi * 0.02
    tj = json.load(open(os.path.join(OUT, "tables.json")))
    td = tj["问题3_干燥时长_h"] * 3600
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.6))
    ax[0, 0].plot(t / 3600, C[:, 0], "-", color="#c62828", lw=1.8, label="中心 r=0")
    ax[0, 0].plot(t / 3600, C[:, -1], "--", color="#1565c0", lw=1.8, label="表面 r=2 cm")
    ax[0, 0].axhline(0.15, color="k", ls=":", lw=1.4, label="要求上限 0.15")
    ax[0, 0].axvline(td / 3600, color="g", ls="-.", lw=1.4, label="烘干结束 %.2f h" % (td / 3600))
    ax[0, 0].set_yscale("log"); ax[0, 0].set_xlabel("时间 / h")
    ax[0, 0].set_ylabel("水分浓度 / (kg/kg)")
    ax[0, 0].set_title("(a) 中心/表面水分浓度（对数）")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3, which="both")
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, 10))
    for c, th in zip(cmap, [6, 12, 18, 24, 30, 36, 42, 48, 54, 57.15]):
        i = int(np.argmin(np.abs(t - th * 3600)))
        ax[0, 1].plot(r * 100, C[i], "-", color=c, lw=1.6, label="%.0f h" % th)
    ax[0, 1].set_xlabel("到药材中心距离 r / cm"); ax[0, 1].set_ylabel("水分浓度 / (kg/kg)")
    ax[0, 1].set_title("(b) 水分浓度剖面演化")
    ax[0, 1].legend(fontsize=8, ncol=2); ax[0, 1].grid(alpha=.3)
    ax[1, 0].plot(t / 3600, T[:, 0], "-", color="#c62828", lw=1.8, label="中心")
    ax[1, 0].plot(t / 3600, T[:, -1], "--", color="#1565c0", lw=1.8, label="表面")
    ax[1, 0].set_xlabel("时间 / h"); ax[1, 0].set_ylabel("温度 / °C")
    ax[1, 0].set_title("(c) 温度演化"); ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=.3)
    # 干燥速率
    dt_ = np.gradient(C[:, 0], t)
    ax[1, 1].plot(t / 3600, -dt_, "-", color="#6a1b9a", lw=1.8)
    ax[1, 1].set_xlabel("时间 / h"); ax[1, 1].set_ylabel("中心干燥速率 -dC/dt / (1/h)")
    ax[1, 1].set_title("(d) 中心含水率下降速率 (转折反映机理变化)")
    ax[1, 1].set_yscale("log"); ax[1, 1].grid(alpha=.3, which="both")
    save(fig, "fig5_p3.png")


# ---------------------------------------------------------------- 图5 问题4
def fig_p4():
    d = np.load(os.path.join(OUT, "p4.npz"))
    t, C, T, xi, R = d["t"], d["C"], d["T"], d["xi"], d["R"]
    tj = json.load(open(os.path.join(OUT, "tables.json")))
    td = tj["问题4_干燥时长_h"] * 3600
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.6))
    ax[0, 0].plot(t / 3600, C[:, 0], "-", color="#c62828", lw=1.8, label="中心 (r=0)")
    ax[0, 0].plot(t / 3600, C[:, -1], "--", color="#1565c0", lw=1.8, label="药材表面")
    ax[0, 0].axhline(0.15, color="k", ls=":", lw=1.4, label="要求上限 0.15")
    ax[0, 0].axvline(td / 3600, color="g", ls="-.", lw=1.4, label="烘干结束 %.2f h" % (td / 3600))
    ax[0, 0].set_yscale("log"); ax[0, 0].set_xlabel("时间 / h")
    ax[0, 0].set_ylabel("水分浓度 / (kg/kg)")
    ax[0, 0].set_title("(a) 中心/表面水分浓度（收缩域）")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3, which="both")
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, 9))
    for c, th in zip(cmap, [6, 12, 18, 24, 30, 36, 42, 48, 50.82]):
        i = int(np.argmin(np.abs(t - th * 3600)))
        rr = xi * R[i]
        ax[0, 1].plot(rr * 100, C[i], "-", color=c, lw=1.6, label="%.0f h" % th)
    ax[0, 1].set_xlabel("到药材中心距离 r / cm"); ax[0, 1].set_ylabel("水分浓度 / (kg/kg)")
    ax[0, 1].set_title("(b) 水分浓度剖面（横轴为当前半径）")
    ax[0, 1].legend(fontsize=8, ncol=2); ax[0, 1].grid(alpha=.3)
    ax[1, 0].plot(t_rad / 3600, R_rad * 100, "-", color="#2e7d32", lw=1.8)
    ax[1, 0].set_xlabel("时间 / h"); ax[1, 0].set_ylabel("半径 / cm")
    ax[1, 0].set_title("(c) 半径收缩过程（附件2）"); ax[1, 0].grid(alpha=.3)
    # 材料坐标下的剖面 (xi)
    for c, th in zip(cmap, [6, 12, 18, 24, 30, 36, 42, 48, 50.82]):
        i = int(np.argmin(np.abs(t - th * 3600)))
        ax[1, 1].plot(xi, C[i], "-", color=c, lw=1.6)
    ax[1, 1].set_xlabel("材料坐标 ξ = r/R(t)"); ax[1, 1].set_ylabel("水分浓度 / (kg/kg)")
    ax[1, 1].set_title("(d) 材料坐标下的水分剖面"); ax[1, 1].grid(alpha=.3)
    save(fig, "fig6_p4.png")


# ---------------------------------------------------------------- 图6 检验
def fig_verify():
    v2 = json.load(open(os.path.join(OUT, "verify_02.json")))
    v3 = json.load(open(os.path.join(OUT, "verify_03.json")))
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.6))
    # (a) 收敛
    Ns = [c["N"] for c in v2["conv_N"]] + [800]
    eC = [c["eC"] for c in v2["conv_N"]] + [0.0]
    ax[0, 0].loglog(Ns[:2], eC[:2], "o-", color="#1f4e79", lw=1.8, label="数值误差 (相对 N=800)")
    x = np.array([200., 400.])
    ax[0, 0].loglog(x, eC[0] * (x / 200.) ** (-2), "k--", lw=1.2, label="二阶参考斜率")
    ax[0, 0].set_xlabel("径向网格数 N"); ax[0, 0].set_ylabel("max|ΔC| / (kg/kg)")
    ax[0, 0].set_title("(a) 空间网格收敛（二阶）")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3, which="both")
    # (b) 二维 vs 一维
    th = [x["t_h"] for x in v3]; dC = [x["dC"] for x in v3]
    ax[0, 1].plot(th, np.array(dC) * 1e3, "o-", color="#a52a2a", lw=1.8)
    ax[0, 1].axhline(0, color="k", lw=0.8)
    ax[0, 1].set_xlabel("时间 / h"); ax[0, 1].set_ylabel("ΔC(中心) / (10⁻³ kg/kg)")
    ax[0, 1].set_title("(b) 二维轴对称与一维模型之差（中心点）")
    ax[0, 1].grid(alpha=.3)
    # (c) 干燥曲线对比 (问题3 vs 问题4)
    d3 = np.load(os.path.join(OUT, "p3.npz")); d4 = np.load(os.path.join(OUT, "p4.npz"))
    ax[1, 0].plot(d3["t"] / 3600, d3["C"][:, 0], "-", color="#1f4e79", lw=1.9, label="问题3 中心（不收缩）")
    ax[1, 0].plot(d4["t"] / 3600, d4["C"][:, 0], "-", color="#a52a2a", lw=1.9, label="问题4 中心（收缩）")
    ax[1, 0].axhline(0.15, color="k", ls=":", lw=1.3)
    ax[1, 0].set_xlim(0, 75); ax[1, 0].set_ylim(0.05, 2.7); ax[1, 0].set_yscale("log")
    ax[1, 0].set_xlabel("时间 / h"); ax[1, 0].set_ylabel("中心水分浓度 / (kg/kg)")
    ax[1, 0].set_title("(c) 两种模型中心干燥曲线对比")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=.3, which="both")
    # (d) 模型形式对比 (守恒 vs 非守恒)
    try:
        v5 = json.load(open(os.path.join(OUT, "verify_05.json")))
        labels = ["问题3\n基准", "问题4\n基准", "问题4\n非守恒变体"]
        vals = [v5["基准_问题3_h"], v5["基准_问题4_h"], v5.get("问题4_非守恒变体_h") or np.nan]
        ax[1, 1].bar(labels, vals, color=["#1f4e79", "#a52a2a", "#9e9e9e"])
        for i, v in enumerate(vals):
            ax[1, 1].text(i, v + 1, "%.1f h" % v, ha="center", fontsize=9)
        ax[1, 1].set_ylabel("烘干时长 / h")
        ax[1, 1].set_title("(d) 收缩域模型形式对烘干时长的影响")
    except Exception as e:
        pass
    ax[1, 1].grid(alpha=.3, axis="y")
    save(fig, "fig7_verify.png")


# ---------------------------------------------------------------- 图7 敏感性
def fig_sens():
    v = json.load(open(os.path.join(OUT, "verify_05.json")))
    base3 = v["基准_问题3_h"]
    items = [
        ("环境湿度 C∞=0.02", v["Cinf_0.02000"]),
        ("环境湿度 C∞=0.12", v["Cinf_0.12000"]),
        ("恒温 45 °C", v["Tinf_45.00"]),
        ("恒温 55 °C", v["Tinf_55.00"]),
        ("传质系数 ×0.8", v["km_x0.8"]),
        ("传质系数 ×1.2", v["km_x1.2"]),
        ("换热系数 ×0.8", v["h_x0.8"]),
        ("换热系数 ×1.2", v["h_x1.2"]),
    ]
    names = [i[0] for i in items]
    vals = np.array([i[1] for i in items])
    dev = 100 * (vals - base3) / base3
    order = np.argsort(np.abs(dev))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    colors = ["#c62828" if d > 0 else "#1565c0" for d in dev[order]]
    ax.barh(np.array(names)[order], dev[order], color=colors)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("烘干时长相对基准的变化 / %")
    ax.set_title("问题3 烘干时长对参数与假设的敏感性（基准 %.2f h）" % base3)
    for i, d in enumerate(dev[order]):
        ax.text(d + (0.25 if d >= 0 else -0.25), i, "%+.1f%%" % d,
                va="center", ha="left" if d >= 0 else "right", fontsize=9)
    ax.grid(alpha=.3, axis="x")
    ax.set_xlim(dev.min() * 1.35, dev.max() * 1.35)
    save(fig, "fig8_sensitivity.png")


def main():
    print("生成插图:")
    fig_env(); fig_radius(); fig_p1(); fig_p2(); fig_p3(); fig_p4()
    fig_verify(); fig_sens()
    print("全部插图已保存到", FIG)


if __name__ == "__main__":
    main()
