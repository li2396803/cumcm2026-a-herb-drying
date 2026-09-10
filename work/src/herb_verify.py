# -*- coding: utf-8 -*-
"""多方法检验工具箱：解析解、守恒审计、独立数值方法。"""

import numpy as np
from scipy.special import j0, j1
from scipy.optimize import brentq


# --------------------------------------------------------------------------
# 1. 解析解: 圆柱 + 对流(Robin)边界, 常物性
# --------------------------------------------------------------------------
def robin_roots(Bi, n=80, xmax=250.0):
    """求解 lam*J1(lam) = Bi*J0(lam) 的正根 (对流边界圆柱)."""
    f = lambda x: x * j1(x) - Bi * j0(x)
    xs = np.linspace(1e-9, xmax, 2000000)
    vals = f(xs)
    idx = np.where(vals[:-1] * vals[1:] < 0)[0]
    roots = []
    for i in idx:
        roots.append(brentq(f, xs[i], xs[i + 1], xtol=1e-15, rtol=8.9e-16))
        if len(roots) >= n:
            break
    return np.array(roots)


def analytical_cylinder(r, t, R, D, km, C0, Cinf, nterm=120):
    """常扩散系数圆柱瞬态解析解 (对流边界, 均匀初值 C0).

    (C-Cinf)/(C0-Cinf) = sum_n A_n J0(lam_n r/R) exp(-lam_n^2 D t/R^2)
    lam_n : lam J1(lam) = Bi J0(lam),  Bi = km R / D
    A_n   = 2 J1(lam_n) / (lam_n (J0^2(lam_n)+J1^2(lam_n)))
    """
    Bi = km * R / D
    lam = robin_roots(Bi, nterm)
    A = 2.0 * j1(lam) / (lam * (j0(lam) ** 2 + j1(lam) ** 2))
    r = np.atleast_1d(np.asarray(r, dtype=float))
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty((t.size, r.size))
    for i, ti in enumerate(t):
        s = np.zeros_like(r)
        for ln, an in zip(lam, A):
            s = s + an * j0(ln * r / R) * np.exp(-(ln ** 2) * D * ti / R ** 2)
        out[i] = Cinf + (C0 - Cinf) * s
    return out


def analytical_temp(r, t, R, alpha, h, k, T0, Tinf, nterm=120):
    """同样公式, 用于温度场 (常数 alpha, h, k)."""
    Bi = h * R / k
    lam = robin_roots(Bi, nterm)
    A = 2.0 * j1(lam) / (lam * (j0(lam) ** 2 + j1(lam) ** 2))
    r = np.atleast_1d(np.asarray(r, dtype=float))
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty((t.size, r.size))
    for i, ti in enumerate(t):
        s = np.zeros_like(r)
        for ln, an in zip(lam, A):
            s = s + an * j0(ln * r / R) * np.exp(-(ln ** 2) * alpha * ti / R ** 2)
        out[i] = Tinf + (T0 - Tinf) * s
    return out


# --------------------------------------------------------------------------
# 2. 守恒审计
# --------------------------------------------------------------------------
def _vol_coef(xi):
    xi_f = 0.5 * (xi[1:] + xi[:-1])
    lo = np.concatenate([[0.0], xi_f])
    hi = np.concatenate([xi_f, [1.0]])
    return 0.5 * (hi ** 2 - lo ** 2)


def audit_mass(res, radius_law, km, env, xi):
    """水分守恒审计: dM/dt 应等于外表面净流出量."""
    t = res["t"]; C = res["C"]
    Vc = _vol_coef(xi)
    M = np.array([radius_law.R_of(tt) ** 2 * np.sum(C[i] * Vc)
                  for i, tt in enumerate(t)])
    dM = np.gradient(M, t)
    flux = np.array([-radius_law.R_of(tt) * km * (C[i, -1] - float(env.C_inf(tt)))
                     for i, tt in enumerate(t)])
    scale = np.maximum.reduce([np.abs(flux), np.abs(dM), np.full_like(flux, 1e-12)])
    return t, M, dM, flux, np.abs(dM - flux) / scale


def audit_energy(res, radius_law, h, env, xi, props):
    """能量审计 (残差主要来自 rho,cp 随水分变化与水汽带走的焓)."""
    t = res["t"]; T = res["T"]; C = res["C"]
    Vc = _vol_coef(xi)
    E = np.array([radius_law.R_of(tt) ** 2 *
                  np.sum(props.rho(C[i]) * props.cp(C[i]) * T[i] * Vc)
                  for i, tt in enumerate(t)])
    dE = np.gradient(E, t)
    flux = np.array([radius_law.R_of(tt) * h * (float(env.T_inf(tt)) - T[i, -1])
                     for i, tt in enumerate(t)])
    scale = np.maximum.reduce([np.abs(flux), np.abs(dE), np.full_like(flux, 1e-9)])
    return t, E, dE, flux, np.abs(dE - flux) / scale


# --------------------------------------------------------------------------
# 3. 独立数值方法 A: 单元中心 FV + scipy BDF (固定域)
# --------------------------------------------------------------------------
def mol_fixed(props, env, t_end, t_eval, R=0.02, h=25.0, km=8e-7, nc=240,
              rtol=1e-10, atol=1e-12, T0=28.0, C0=2.55, harmonic=True):
    """独立实现(1)：单元中心有限体积, 物理坐标, 调和/算术平均面系数, BDF 自适应积分.
    仅适用于固定半径问题 (1-3)。
    """
    from scipy.integrate import solve_ivp

    r_f = np.linspace(0.0, R, nc + 1)
    r_c = 0.5 * (r_f[1:] + r_f[:-1])
    dr = R / nc
    V = 0.5 * (r_f[1:] ** 2 - r_f[:-1] ** 2)
    r_face = np.concatenate([[0.0], r_c[:-1] + 0.5 * dr, [R]])

    def rhs(t, y):
        T = y[:nc]; C = y[nc:]
        Tinf = float(env.T_inf(t)); Cinf = float(env.C_inf(t))
        kk = props.kcond(C); DD = props.D(C, T)
        kf = np.empty(nc + 1); Df = np.empty(nc + 1)
        if harmonic:
            kf[1:-1] = 2 * kk[:-1] * kk[1:] / (kk[:-1] + kk[1:])
            Df[1:-1] = 2 * DD[:-1] * DD[1:] / (DD[:-1] + DD[1:])
        else:
            kf[1:-1] = 0.5 * (kk[:-1] + kk[1:])
            Df[1:-1] = 0.5 * (DD[:-1] + DD[1:])
        # 面通量 (约定: 沿 +r 方向为正)
        # 边界采用"单元中心-表面"半格热阻 + 对流热阻的串联处理 (二阶精度)
        F = np.zeros(nc + 1)
        F[1:-1] = r_face[1:-1] * kf[1:-1] * (T[:-1] - T[1:]) / dr
        F[-1] = R * (T[-1] - Tinf) / (0.5 * dr / kk[-1] + 1.0 / h)
        G = np.zeros(nc + 1)
        G[1:-1] = r_face[1:-1] * Df[1:-1] * (C[:-1] - C[1:]) / dr
        G[-1] = R * (C[-1] - Cinf) / (0.5 * dr / DD[-1] + 1.0 / km)
        dT = (F[:-1] - F[1:]) / (V * props.rho(C) * props.cp(C))
        dC = (G[:-1] - G[1:]) / V
        return np.concatenate([dT, dC])

    y0 = np.concatenate([np.full(nc, T0), np.full(nc, C0)])
    sol = solve_ivp(rhs, (0.0, t_end), y0, method="BDF", t_eval=t_eval,
                    rtol=rtol, atol=atol)
    return sol, r_c


# --------------------------------------------------------------------------
# 4. 独立数值方法 B: 移动网格 Eulerian 形式 (收缩问题)
# --------------------------------------------------------------------------
def mol_moving(props, radius_law, env, t_end, t_eval, h=25.0, km=8e-7,
               N=240, rtol=1e-9, atol=1e-11, T0=28.0, C0=2.55, stretch=1.0):
    """独立实现(2)：物理坐标移动网格, 显式对流项.

    直接离散 Eulerian 形式:
        dC/dt|_r + v_s dC/dr = (1/r) d/dr ( r D dC/dr ),  v_s = (r/R) dR/dt
        rho cp (dT/dt|_r + v_s dT/dr) = (1/r) d/dr ( r k dT/dr )
    网格随材料仿射移动 r_i(t)=xi_i R(t), 节点时间导数即材料导数。
    """
    from scipy.integrate import solve_ivp

    if stretch == 1.0:
        xi = np.linspace(0.0, 1.0, N + 1)
    else:
        s = np.linspace(0.0, 1.0, N + 1)
        xi = 1.0 - (1.0 - s) ** stretch
    dxi = np.diff(xi)

    def rhs(t, y):
        T = y[:N + 1]; C = y[N + 1:]
        R = float(radius_law.R_of(t)); Rd = float(radius_law.dR_of(t))
        r = xi * R
        dr = dxi * R
        Tinf = float(env.T_inf(t)); Cinf = float(env.C_inf(t))
        kk = props.kcond(C); DD = props.D(C, T); rho_cp = props.rho(C) * props.cp(C)
        kf = 0.5 * (kk[:-1] + kk[1:]); Df = 0.5 * (DD[:-1] + DD[1:])
        rf = 0.5 * (r[:-1] + r[1:])
        # 面通量 (沿 +r 为正)
        FT = np.empty(N + 1); FT[:N] = rf * kf * (T[:-1] - T[1:]) / dr
        FT[N] = r[-1] * h * (T[-1] - Tinf)
        FC = np.empty(N + 1); FC[:N] = rf * Df * (C[:-1] - C[1:]) / dr
        FC[N] = r[-1] * km * (C[-1] - Cinf)
        r_lo = np.concatenate([[0.0], rf]); r_hi = np.concatenate([rf, [r[-1]]])
        V = 0.5 * (r_hi ** 2 - r_lo ** 2)
        dT = (FT[:-1] - FT[1:]) / (V * rho_cp)
        dC = (FC[:-1] - FC[1:]) / V
        gT = np.gradient(T, r); gC = np.gradient(C, r)
        vs = xi * Rd
        dT = dT - vs * gT
        dC = dC - vs * gC
        return np.concatenate([dT, dC])

    y0 = np.concatenate([np.full(N + 1, T0), np.full(N + 1, C0)])
    sol = solve_ivp(rhs, (0.0, t_end), y0, method="BDF", t_eval=t_eval,
                    rtol=rtol, atol=atol)
    return sol, xi


def surface_value_mol(C_cell_last, T_cell_last, props, R, dr, km, Cinf):
    """由单元中心值重建表面浓度 (与单元中心 FV 的半格阻力边界一致)"""
    D = props.D(C_cell_last, T_cell_last)
    return C_cell_last - (0.5 * dr / D) * (C_cell_last - Cinf) / ((0.5 * dr / D) + 1.0 / km)
