# -*- coding: utf-8 -*-
"""检验6: 模型参数与模型形式敏感性分析 (以烘干时长为核心指标)."""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import herb_model as hm

OUT = hm.WORK_OUT


def schedule_fast(t):
    """烘干时长估计用的较粗时间步 (与生产步长对比验证过)"""
    if t < 10.0:
        return 0.05
    if t < 60.0:
        return 0.2
    if t < 600.0:
        return 1.0
    if t < 3600.0:
        return 5.0
    if t < 14400.0:
        return 15.0
    return 240.0


class EnvHold(hm.Environment):
    """t>t_hold 后环境湿度/温度改为给定值"""

    def __init__(self, t, T, C, C_after=None, T_after=None):
        super().__init__(t, T, C)
        self.C_after = C_after
        self.T_after = T_after

    def C_inf(self, t):
        v = super().C_inf(t)
        if self.C_after is None:
            return v
        return np.where(np.asarray(t) > self.t_hold, self.C_after, v)

    def T_inf(self, t):
        v = super().T_inf(t)
        if self.T_after is None:
            return v
        return np.where(np.asarray(t) > self.t_hold, self.T_after, v)


def drying_time(t, C, thr=0.15):
    idx = np.where(C < thr)[0]
    if idx.size == 0:
        return None
    i = idx[0]
    if i == 0:
        return float(t[0])
    return float(t[i - 1] + (thr - C[i - 1]) * (t[i] - t[i - 1]) / (C[i] - C[i - 1]))


def run_case(mode="p3", moving=False, N=200, C_after=None, T_after=None,
             km_scale=1.0, h_scale=1.0, adv_mode="none", heat_cap="volume",
             t_max=100 * 3600.0, out_dt=240.0):
    t, T, C = hm.load_oven()
    env = EnvHold(t, T, C, C_after=C_after, T_after=T_after)
    props = hm.Props(mode)
    rad = hm.RadiusLaw(*hm.load_radius()) if moving else hm.ConstRadius(0.02)
    s = hm.DryingSolver(props, rad, env, N=N, h=25.0 * h_scale, km_scale=km_scale,
                        adv_mode=adv_mode, heat_cap_mode=heat_cap)
    te = np.arange(0, t_max + 1, out_dt)
    r = s.run(t_max, schedule_fast, t_eval=te)
    Cmax = r["C"].max(axis=1)
    return drying_time(r["t"], Cmax), r


def main():
    RES = {}
    t0 = time.time()

    print("基准情形 (N=200, 粗时间步):", flush=True)
    td, r = run_case("p3")
    print("  问题3 基准 烘干时长 = %.4f h" % (td / 3600), flush=True)
    RES["基准_问题3_h"] = td / 3600
    td4, r4 = run_case("p4", moving=True)
    print("  问题4 基准 烘干时长 = %.4f h" % (td4 / 3600), flush=True)
    RES["基准_问题4_h"] = td4 / 3600

    td_r, _ = run_case("p3", N=400, out_dt=60.0)
    print("  参考 (N=400, dt=60 s): %.4f h -> 粗网格偏差 %.3f%%"
          % (td_r / 3600, 100 * (td - td_r) / td_r), flush=True)
    RES["基准_问题3_N400_h"] = td_r / 3600

    print("\n环境湿度 (4 h 后) 敏感性:", flush=True)
    for c_val in [0.02, 0.03, 0.04986, 0.08, 0.12]:
        td_c, _ = run_case("p3", C_after=c_val)
        print("  C_inf = %.5f -> 烘干时长 %.3f h" % (c_val, td_c / 3600), flush=True)
        RES["Cinf_%.5f" % c_val] = td_c / 3600

    print("\n恒温阶段温度敏感性:", flush=True)
    for T_val in [45.0, 50.165, 55.0]:
        td_T, _ = run_case("p3", T_after=T_val)
        print("  T_inf = %.2f C -> 烘干时长 %.3f h" % (T_val, td_T / 3600), flush=True)
        RES["Tinf_%.2f" % T_val] = td_T / 3600

    print("\n对流传质系数敏感性:", flush=True)
    for sc in [0.8, 1.2]:
        td_k, _ = run_case("p3", km_scale=sc)
        print("  km x %.1f -> 烘干时长 %.3f h" % (sc, td_k / 3600), flush=True)
        RES["km_x%.1f" % sc] = td_k / 3600

    print("\n对流换热系数敏感性:", flush=True)
    for sc in [0.8, 1.2]:
        td_h, _ = run_case("p3", h_scale=sc)
        print("  h x %.1f -> 烘干时长 %.3f h" % (sc, td_h / 3600), flush=True)
        RES["h_x%.1f" % sc] = td_h / 3600

    print("\n问题4 模型形式敏感性 (收缩域):", flush=True)
    td_e, _ = run_case("p4", moving=True, adv_mode="euler")
    if td_e is None:
        print("  非守恒变体 (Eulerian 变换含附加对流项): 100 h 内未达标", flush=True)
    else:
        print("  非守恒变体: 烘干时长 %.3f h" % (td_e / 3600), flush=True)
    RES["问题4_非守恒变体_h"] = None if td_e is None else td_e / 3600
    td_m, _ = run_case("p4", moving=True, heat_cap="material")
    print("  材料热容变体: %.3f h (基准 %.3f h)" % (td_m / 3600, td4 / 3600), flush=True)
    RES["问题4_材料热容_h"] = td_m / 3600
    td_p3, _ = run_case("p3", moving=True)
    print("  收缩域 + 附录3物性: %.3f h" % (td_p3 / 3600), flush=True)
    RES["问题4_用附录3物性_h"] = td_p3 / 3600

    with open(os.path.join(OUT, "verify_05.json"), "w") as fh:
        json.dump(RES, fh, ensure_ascii=False, indent=1)
    print("\n总耗时 %.0f s; saved -> out/verify_05.json" % (time.time() - t0))


if __name__ == "__main__":
    main()
