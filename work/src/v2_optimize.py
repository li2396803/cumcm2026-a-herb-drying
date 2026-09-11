# -*- coding: utf-8 -*-
"""v2 工艺优化（CPLEX MILP）：给定设备与品质约束，求最优烘房温度调度。

思路
----
1. 用高保真 PDE 模型标定"恒温干燥段温度 T → 烘干时长 t_dry(T)"（40—60 °C，5 档）；
2. 建立**线性进度模型**：第 k 段（时长 Δt）选用温度档 j 时，干燥进度增量为 Δt/t_dry(T_j)，
   全部进度累计到 1 即判定烘干完成（该叠加律的合理性由第 4 步的高保真复核检验）；
3. 用 CPLEX 求解 MILP：决策为各段温度档位 z_(k,j)∈{0,1}（每段恰选一档），
   目标为最小化"温度—时间"能耗代理 Σ(T_j−T_amb)Δt z_(k,j)，
   约束为：(a) 进度 ≥ 1；(b) 恒温段温度可在 40—60 °C 间选择；
   (c) 相邻段温变 |ΔT| ≤ 5 °C（避免热冲击损伤品质）；(d) 预热平衡段 0—4 h 固定按附件 1 曲线；
4. 把 MILP 最优调度回代高保真 PDE 复核，检验进度模型的自洽性；
5. 扫描"允许最长烘干时间"，给出能耗—时长 Pareto 前沿。
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cplex
import herb_v2 as v2

OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out"))
os.makedirs(OUT, exist_ok=True)
T_SWITCH = 14400.0            # 预热平衡段结束（附件 1 覆盖 0—4 h）
T_AMB = 25.0                  # 环境温度（能耗代理的参考）
LEVELS = [40.0, 45.0, 50.0, 55.0, 60.0]


class EnvSchedule:
    """预热平衡段用附件 1 实测曲线；之后按给定调度（常数或分段）"""

    def __init__(self, base, schedule=None, T_const=None, t_switch=T_SWITCH, dt_stage=7200.0):
        self.base = base
        self.schedule = schedule        # 数组，第 k 段温度
        self.T_const = T_const
        self.tc = t_switch
        self.dt_stage = dt_stage

    def T_inf(self, t):
        t = np.asarray(t, dtype=float)
        v = self.base.T_inf(np.minimum(t, self.tc))
        if self.schedule is None:
            after = np.full_like(t, self.T_const)
        else:
            k = np.clip(((t - self.tc) / self.dt_stage).astype(int), 0, len(self.schedule) - 1)
            after = np.asarray(self.schedule, dtype=float)[k]
        return np.where(t <= self.tc, v, after)

    def C_inf(self, t):
        return self.base.C_inf(t)


def run_const_T(base_env, T_const, N=200, t_max=120 * 3600.0):
    """给定恒温段温度，用 PDE 求烘干时长"""
    env = EnvSchedule(base_env, T_const=T_const)
    props = v2.PropsV2("p3")
    solver = v2.LagSolver(props, v2.hm.ConstRadius(0.02), env, N=N,
                          mapping="geometric", cap_mode="fixed_vol")
    te = np.arange(0, t_max + 1, 60.0)
    r = solver.run(t_max, v2.schedule_prod, t_eval=te)
    Cmax = r["C"].max(axis=1)
    idx = np.where(Cmax < 0.15)[0]
    if idx.size == 0:
        return None, r
    i = idx[0]
    td = float(r["t"][i - 1] + (0.15 - Cmax[i - 1]) * (r["t"][i] - r["t"][i - 1]) /
               (Cmax[i] - Cmax[i - 1]))
    return td, r


def Envsched_Tinf(base_env, schedule, t):
    return EnvSchedule(base_env, schedule=schedule).T_inf(t)


def run_schedule(base_env, schedule, N=200, t_max=120 * 3600.0, dt_stage=7200.0):
    env = EnvSchedule(base_env, schedule=schedule, dt_stage=dt_stage)
    props = v2.PropsV2("p3")
    solver = v2.LagSolver(props, v2.hm.ConstRadius(0.02), env, N=N,
                          mapping="geometric", cap_mode="fixed_vol")
    te = np.arange(0, t_max + 1, 60.0)
    r = solver.run(t_max, v2.schedule_prod, t_eval=te)
    Cmax = r["C"].max(axis=1)
    idx = np.where(Cmax < 0.15)[0]
    td = None
    if idx.size:
        i = idx[0]
        td = float(r["t"][i - 1] + (0.15 - Cmax[i - 1]) * (r["t"][i] - r["t"][i - 1]) /
                   (Cmax[i] - Cmax[i - 1]))
    return td, r


def build_milp(t_dry_levels, K, dt_stage, budget_h, dT_max=5.0, t_fixed_progress=0.0,
               n_low_stages=6, T_low_cap=50.0):
    """MILP：各段温度档位 z_(k,j)、运行指示 w_k（干完可停机）。

    目标：最小化能耗代理 Σ (T_j-T_amb)·Δt·z_(k,j)
    约束：(a) 每段 Σ_j z_(k,j) = w_k；(b) w 单调不增（连续运行、中途不停停开开）；
          (c) z_(k,j) ≤ w_k；(d) 前 K_B 段（K_B=⌊B/Δt⌋）内累计进度 ≥ 1；
          (e) 相邻运行段的温变 |ΔT| ≤ dT_max。
    """
    J = len(LEVELS)
    prog = [dt_stage / t_dry_levels[j] for j in range(J)]
    # 变量：z_(k,j) 二元；w_k 二元
    nz = K * J
    obj, lb, ub, vtype, names = [], [], [], [], []
    for k in range(K):
        for j in range(J):
            obj.append((1.0 + 1e-3) * (LEVELS[j] - T_AMB) * dt_stage)
            lb.append(0.0); ub.append(1.0); vtype.append("B")
            names.append("z_%d_%d" % (k, j))
    for k in range(K):
        obj.append(0.0); lb.append(0.0); ub.append(1.0); vtype.append("B")
        names.append("w_%d" % k)
    idx = lambda k, j: k * J + j
    iw = lambda k: nz + k
    c = cplex.Cplex()
    c.set_log_stream(None); c.set_results_stream(None); c.set_warning_stream(None)
    c.variables.add(obj=obj, lb=lb, ub=ub, types=vtype, names=names)
    c.objective.set_sense(c.objective.sense.minimize)
    # (a) Σ_j z_(k,j) = w_k
    for k in range(K):
        c.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=[idx(k, j) for j in range(J)] + [iw(k)],
                                       val=[1.0] * J + [-1.0])],
            senses=["E"], rhs=[0.0], names=["run_%d" % k])
    # (b) w 单调不增
    for k in range(K - 1):
        c.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=[iw(k), iw(k + 1)], val=[1.0, -1.0])],
            senses=["G"], rhs=[0.0], names=["mono_%d" % k])
    # (d) 预算内累计进度（预热段已提供的进度先扣除）
    KB = int(np.floor((budget_h * 3600.0 - T_SWITCH) / dt_stage))
    KB = max(1, min(KB, K))
    c.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=[idx(k, j) for k in range(KB) for j in range(J)],
                                   val=[prog[j] for k in range(KB) for j in range(J)])],
        senses=["G"], rhs=[1.0 - t_fixed_progress], names=["progress"])
    # (f) 升温段限温（前 n_low_stages 段 ≤ T_low_cap）：抑制表面结壳/开裂，保护品质
    for k in range(min(n_low_stages, K)):
        for j in range(J):
            if LEVELS[j] > T_low_cap + 1e-9:
                c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=[idx(k, j)], val=[1.0])],
                                         senses=["E"], rhs=[0.0], names=["Q_%d_%d" % (k, j)])
    # (e) 相邻运行段温变限制：禁止相邻两段温差超过 dT_max（停机段 z=0，自动满足）
    for k in range(K - 1):
        for j in range(J):
            for j2 in range(J):
                if LEVELS[j2] - LEVELS[j] > dT_max + 1e-9:
                    c.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=[idx(k, j), idx(k + 1, j2)], val=[1.0, 1.0])],
                        senses=["L"], rhs=[1.0], names=["dT_%d_%d_%d" % (k, j, j2)])
    return c, idx, iw


def main():
    t0 = time.time()
    t_oven, T_inf, C_inf = v2.hm.load_oven()
    base_env = v2.hm.Environment(t_oven, T_inf, C_inf)
    res = {}

    # ---- 1. 标定 t_dry(T) ----
    cache = os.path.join(OUT, "calib.json")
    if os.path.exists(cache):
        td = {float(k): float(v) for k, v in json.load(open(cache)).items()}
        print("标定结果读自缓存:", {k: round(v / 3600, 3) for k, v in td.items()}, flush=True)
    else:
        td = {}
        print("标定恒温段温度—烘干时长关系 (PDE, N=200):", flush=True)
        for T in LEVELS:
            v, _ = run_const_T(base_env, T)
            td[T] = v
            print("   T = %.0f °C -> t_dry = %.3f h" % (T, v / 3600.0), flush=True)
        json.dump(td, open(cache, "w"))
    res["t_dry_levels_h"] = {str(T): td[T] / 3600.0 for T in LEVELS}

    # 基准（恒温 50.17 °C，与附件 1 末值一致）
    td_base, r_base = run_const_T(base_env, 50.165)
    print("   基准 (50.165 °C) t_dry = %.3f h" % (td_base / 3600.0), flush=True)
    res["t_dry_base_h"] = td_base / 3600.0

    # ---- 2. MILP：不同时长预算下的最小能耗调度 ----
    K, dt_stage = 36, 7200.0          # 36 段 × 2 h = 72 h 决策窗
    t_first = [td[T] for T in LEVELS]
    # 预热段（0—4 h）在 MILP 中作为固定进度：按基准温度 50.165 °C 折算
    fixed_prog = T_SWITCH / td_base
    pareto = []
    print("\nMILP 求解（CPLEX）:", flush=True)
    for budget_h in [50.0, 55.0, 57.147, 60.0, 65.0, 72.0]:
        K_eff = int(np.ceil((budget_h * 3600.0 - T_SWITCH) / dt_stage))
        if K_eff <= 0:
            continue
        c, idx, iw = build_milp(t_first, K_eff, dt_stage, budget_h,
                                dT_max=5.0, t_fixed_progress=fixed_prog)
        c.solve()
        stat = c.solution.get_status_string()
        if c.solution.get_status() not in (101, 102):
            print("   预算 %.1f h: %s" % (budget_h, stat), flush=True)
            continue
        z = c.solution.get_values()
        sched, active = [], []
        for k in range(K_eff):
            on = z[iw(k)] > 0.5
            active.append(bool(on))
            if on:
                j = int(np.argmax([z[idx(k, j)] for j in range(len(LEVELS))]))
                sched.append(LEVELS[j])
            else:
                sched.append(LEVELS[0])      # 停机段（高保真复核时按最低温近似）
        sched_active = [LEVELS[int(np.argmax([z[idx(k, j)] for j in range(len(LEVELS))]))]
                        for k in range(K_eff) if z[iw(k)] > 0.5]
        energy = c.solution.get_objective_value()
        # 3. 高保真复核
        tdv, r_o = run_schedule(base_env, sched_active)
        # 供给热量 ∫ h(T_inf - T_surface) dt（单位高度，W·s/m）
        Ts = r_o["T"][:, -1]; Tinf_t = np.array([float(Envsched_Tinf(base_env, sched_active, tt)) for tt in r_o["t"]])
        heat = float(np.trapezoid(2 * np.pi * 0.02 * 25.0 * (Tinf_t - Ts), r_o["t"]))
        pareto.append({"budget_h": budget_h, "sched": sched, "active": active,
                       "n_active": int(sum(active)), "energy_proxy": energy,
                       "heat_J_per_m": heat,
                       "pde_t_dry_h": None if tdv is None else tdv / 3600.0})
        print("   预算 %.1f h -> 运行 %d 段: %s | 能耗代理 %.3e | PDE 复核 t_dry = %s h"
              % (budget_h, sum(active), [int(x) for x in sched_active], energy,
                 "未完成" if tdv is None else "%.3f" % (tdv / 3600.0)), flush=True)
    res["pareto"] = pareto
    json.dump(res, open(os.path.join(OUT, "optimize.json"), "w"), ensure_ascii=False, indent=1)
    print("\n总用时 %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
