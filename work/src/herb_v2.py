# -*- coding: utf-8 -*-
"""A 题 v2 求解内核：干物质（Lagrangian）坐标下的轴对称热湿耦合有限体积求解器。

坐标与几何
----------
离散坐标取**干物质质量分数** s∈[0,1]（每个 s 单元含相同干物质质量，材料点固定在 s 上）。
材料点到物理半径的映射由局部比容 v 决定：

    r(s,t)^2 = R(t)^2 * J(s,t),   J(s,t) = ∫_0^s v dξ / ∫_0^1 v dξ

两种几何模式：

* ``mapping='geometric'``：J(s)=s（材料点按 r=R(t)√s 保持相似位置，即 v≡const）。
  问题 1—3（给定半径固定）与问题 4 的 v1 基线（仿射收缩）都属此模式。
* ``mapping='local'``：v(C) = (1+C)/rho(C)，直接取自题给 rho(C) 经验式
  —— 即"局部收缩由局部含水率决定、整体半径由附件 2 标定"的**非均匀收缩模型**。

热容基准（两种都与题给 rho(C)、c_p(C) 自洽，只是对"体积如何随含水率变化"的假定不同）

* ``cap_mode='fixed_vol'``：单元体积不变，热容 = rho(C)c_p(C)*v0（v0 为初始比容）
  —— 对应问题 1—3 的"定尺寸、密度按经验式变化"。
* ``cap_mode='mass'``：单元体积随含水率变化，热容 = (1+C)c_p(C)（单位干物质焓）
  —— 等价于 rho(C)c_p(C)*V(t)，对应问题 4 的收缩情形。

控制方程（推导见 v2 说明文档 §4）与离散
----------------------------------------
    dC_i/dt = [Q_{i-1/2} - Q_{i+1/2}] / Δm
    capT_i dT_i/dt = [H_{i-1/2} - H_{i+1/2}] / Δm
    Q_{i+1/2} = -(4π I J_f D_f / v_f^2) (C_{i+1}-C_i)/Δs
    H_{i+1/2} = -(4π I J_f k_f / v_f   ) (T_{i+1}-T_i)/Δs
    外表面: Q_out = 2πR (1/v_N) k_m (C_N - C_inf)
            H_out = 2πR [ h (T_N - T_inf) + L (1/v_N) k_m (C_N - C_inf) ]
其中 Δm = M Δs、M = πR²/I、I = ∫_0^1 v dξ。
离散格式对水分**严格守恒**（面通量望远镜求和后等于表面流出量）。
"""

import os
import sys
import numpy as np
from scipy.linalg import solve_banded

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "outputs", "code"))

import herb_model as hm  # noqa: E402  （复用 v1 的数据读取、Environment、RadiusLaw、Props）


class PropsV2:
    """物性：复用 v1 定义，并补充局部比容 v(C)=(1+C)/rho(C)"""

    def __init__(self, mode):
        self._p = hm.Props(mode)
        self.mode = mode

    def rho(self, C):
        return self._p.rho(C)

    def cp(self, C):
        return self._p.cp(C)

    def kcond(self, C):
        return self._p.kcond(C)

    def D(self, C, T):
        return self._p.D(C, T)

    def v(self, C):
        C = np.asarray(C, dtype=float)
        return (1.0 + C) / self.rho(C)


class LagSolver:
    def __init__(self, props, radius, env, N=400, theta=0.5, mapping="geometric",
                 cap_mode="fixed_vol", latent=False, L=2.4e6, h=25.0, km=8e-7,
                 T0=28.0, C0=2.55, be_start=2):
        assert mapping in ("geometric", "local")
        assert cap_mode in ("fixed_vol", "mass")
        self.props = props
        self.radius = radius
        self.env = env
        self.N = N
        self.theta = theta
        self.mapping = mapping
        self.cap_mode = cap_mode
        self.latent = latent
        self.L = L
        self.h = h
        self.km = km
        self.T0 = T0
        self.C0 = C0
        self.be_start = be_start
        # 材料坐标网格：s=(i/N)^2 —— 使物理半径 r=R*sqrt(J)≈R*(i/N) 方向近似均匀
        self.s = (np.arange(N + 1) / N) ** 2
        self.s_face = 0.5 * (self.s[:-1] + self.s[1:])          # 内部面（长度 N）
        self.dsm = np.empty(N + 1)                              # 各节点控制体干物质质量分数
        s_lo = np.concatenate([[0.0], self.s_face])
        s_hi = np.concatenate([self.s_face, [1.0]])
        self.s_lo, self.s_hi = s_lo, s_hi
        self.dsm = s_hi - s_lo
        self.ds_face = np.diff(self.s)                          # 相邻节点间距（长度 N）
        # 初始比容（fixed_vol 模式的热容基准）
        self.v0 = float(self.props.v(C0))

    # -------- 几何 --------
    def mapping_arrays(self, C, R):
        N = self.N
        if self.mapping == "geometric":
            v = np.full(N + 1, self.v0)
            J = self.s.copy()
            I = v[0]
        else:
            v = self.props.v(C)
            cum = np.concatenate([[0.0], np.cumsum(0.5 * (v[:-1] + v[1:]) * self.ds_face)])
            I = cum[-1]
            J = cum / I
        r = R * np.sqrt(J)
        return r, J, I, v

    def _coeffs(self, T, C, R):
        """返回 (gC, gT, gC_out, gT_out, gL, capT)。

        离散形式（控制体 i 的干物质质量分数 dsm_i，容量含 dsm）：
            dsm_i * dY_i/dt = G_{i-1/2}(Y_{i-1}-Y_i) - G_{i+1/2}(Y_i-Y_{i+1})
            G^C_{i+1/2} = 4 I^2 J_f D_f / (R^2 v_f^2 Δs_f)
            G^T_{i+1/2} = 4 I^2 J_f k_f / (R^2 v_f   Δs_f)
            表面:  G^C_out = 2 I k_m / (R v_N),  G^T_out = 2 I h / R,
                   G^L    = 2 I L k_m / (R v_N)
        """
        N = self.N
        r, J, I, v = self.mapping_arrays(C, R)
        Jf = 0.5 * (J[:-1] + J[1:])
        vf = 0.5 * (v[:-1] + v[1:])
        kk = self.props.kcond(C); DD = self.props.D(C, T)
        kf = 0.5 * (kk[:-1] + kk[1:]); Df = 0.5 * (DD[:-1] + DD[1:])
        if self.cap_mode == "fixed_vol":
            capT = self.props.rho(C) * self.props.cp(C) * self.v0     # 单位干物质热容
        else:
            capT = (1.0 + C) * self.props.cp(C)
        coef = 4.0 * I ** 2 / R ** 2
        gC = coef * Jf * Df / (vf ** 2 * self.ds_face)
        gT = coef * Jf * kf / (vf * self.ds_face)
        gC_out = 2.0 * I * self.km / (R * v[-1])
        gT_out = 2.0 * I * self.h / R
        gL = (2.0 * I * self.L * self.km / (R * v[-1])) if self.latent else 0.0
        return gC, gT, gC_out, gT_out, gL, capT

    # -------- 单个 theta 步的三对角求解 --------
    def _solve(self, Y_n, Cap, g, g_out, Yinf_o, Yinf_n, dt, th, extra=0.0):
        N = self.N
        q = np.empty(N + 1)
        q[:N] = g * (Y_n[:-1] - Y_n[1:])
        q[N] = g_out * (Y_n[-1] - Yinf_o) + extra
        F_old = np.empty(N + 1)
        F_old[0] = -q[0]
        F_old[1:] = q[:-1] - q[1:]
        diag = Cap / dt + th * (np.concatenate([np.zeros(1), g]) +
                                np.concatenate([g, [g_out]]))
        ab = np.zeros((3, N + 1))
        ab[0, 1:] = -th * g
        ab[1, :] = diag
        ab[2, :-1] = -th * g
        rhs = Cap / dt * Y_n + (1 - th) * F_old
        rhs[-1] += th * g_out * Yinf_n
        return solve_banded((1, 1), ab, rhs)

    def step(self, T_n, C_n, t, dt, maxiter=50, tol=1e-11, theta=None):
        th = self.theta if theta is None else theta
        R = float(self.radius.R_of(t + 0.5 * dt))
        Tinf_o = float(self.env.T_inf(t)); Tinf_n = float(self.env.T_inf(t + dt))
        Cinf_o = float(self.env.C_inf(t)); Cinf_n = float(self.env.C_inf(t + dt))
        T_g, C_g = T_n.copy(), C_n.copy()
        for it in range(maxiter):
            Tm = 0.5 * (T_n + T_g); Cm = 0.5 * (C_n + C_g)
            gC, gT, gC_out, gT_out, gL, capT = self._coeffs(Tm, Cm, R)
            C_new = self._solve(C_n, self.dsm.copy(), gC, gC_out, Cinf_o, Cinf_n, dt, th)
            # 潜热项：以本轮 C 的半步值作为驱动（Picard 收敛后为时间二阶）
            extra = gL * (Cg_last := 0.5 * (C_n[-1] + C_new[-1]) - 0.5 * (Cinf_o + Cinf_n))
            T_new = self._solve(T_n, self.dsm * capT, gT, gT_out, Tinf_o, Tinf_n, dt, th, extra=extra)
            d = max(float(np.max(np.abs(T_new - T_g))), float(np.max(np.abs(C_new - C_g))))
            T_g, C_g = T_new, C_new
            if d < tol:
                break
        self.picard = it + 1
        return T_g, C_g

    # -------- 右端项（仅用于 Hermite 稠密输出） --------
    def _rhs(self, T, C, t):
        R = float(self.radius.R_of(t))
        Tinf = float(self.env.T_inf(t)); Cinf = float(self.env.C_inf(t))
        gC, gT, gC_out, gT_out, gL, capT = self._coeffs(T, C, R)
        N = self.N
        qC = np.empty(N + 1); qC[:N] = gC * (C[:-1] - C[1:]); qC[N] = gC_out * (C[-1] - Cinf)
        fC = np.empty(N + 1); fC[0] = -qC[0]; fC[1:] = qC[:-1] - qC[1:]
        qT = np.empty(N + 1); qT[:N] = gT * (T[:-1] - T[1:])
        qT[N] = gT_out * (T[-1] - Tinf) + gL * (C[-1] - Cinf)
        fT = np.empty(N + 1); fT[0] = -qT[0]; fT[1:] = qT[:-1] - qT[1:]
        return fT / (self.dsm * capT), fC / self.dsm

    # -------- 时间推进 --------
    def run(self, t_end, dt_fun, t_eval=None, save_fn=None, hermite=True):
        N = self.N
        T = np.full(N + 1, self.T0); C = np.full(N + 1, self.C0)
        t = 0.0
        te = np.asarray([] if t_eval is None else t_eval, dtype=float)
        ie = 0
        out_t, out_T, out_C = [], [], []
        if te.size and abs(te[0]) < 1e-12:
            if save_fn:
                save_fn(0.0, T, C)
            else:
                out_t.append(0.0); out_T.append(T.copy()); out_C.append(C.copy())
            ie = 1
        nstep = 0
        while t < t_end - 1e-12:
            dt = float(dt_fun(t))
            nxt = te[ie] if ie < te.size else np.inf
            if t + dt > nxt + 1e-12 and not hermite:
                dt = nxt - t
            dt = min(dt, t_end - t)
            th = 1.0 if nstep < self.be_start else self.theta
            T_new, C_new = self.step(T, C, t, dt, theta=th)
            t_new = t + dt
            while ie < te.size and te[ie] <= t_new + 1e-12:
                tt = te[ie]
                if hermite and t + 1e-12 < tt < t_new - 1e-12:
                    w = (tt - t) / dt
                    f0T, f0C = self._rhs(T, C, t)
                    f1T, f1C = self._rhs(T_new, C_new, t_new)
                    h00 = 2 * w ** 3 - 3 * w ** 2 + 1
                    h10 = w ** 3 - 2 * w ** 2 + w
                    h01 = -2 * w ** 3 + 3 * w ** 2
                    h11 = w ** 3 - w ** 2
                    To = h00 * T + h10 * dt * f0T + h01 * T_new + h11 * dt * f1T
                    Co = h00 * C + h10 * dt * f0C + h01 * C_new + h11 * dt * f1C
                else:
                    To, Co = T_new, C_new
                if save_fn:
                    save_fn(tt, To, Co)
                else:
                    out_t.append(tt); out_T.append(To.copy()); out_C.append(Co.copy())
                ie += 1
            T, C, t = T_new, C_new, t_new
            nstep += 1
        return {"t": np.array(out_t), "T": np.array(out_T), "C": np.array(out_C),
                "nstep": nstep, "s": self.s}

    # -------- 物理半径输出 --------
    def radii_of(self, t, C):
        R = float(self.radius.R_of(t))
        r, J, I, v = self.mapping_arrays(np.asarray(C, dtype=float), R)
        return r

    def profiles_at_radii(self, t, T, C, r_out):
        r = self.radii_of(t, C)
        Tout = np.interp(r_out, r, T, left=T[0], right=np.nan)
        Cout = np.interp(r_out, r, C, left=C[0], right=np.nan)
        return Tout, Cout


def schedule_1s(t):
    """全程 1 s 输出的求解步长（配合稠密输出）"""
    if t < 10.0:
        return 0.05
    if t < 60.0:
        return 0.2
    if t < 600.0:
        return 1.0
    if t < 3600.0:
        return 15.0
    return 60.0


def schedule_prod(t):
    """生产用步长（v1 口径）"""
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
    return 60.0
