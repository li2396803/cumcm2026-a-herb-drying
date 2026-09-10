# -*- coding: utf-8 -*-
"""
2026 CUMCM A 题（药材烘干）核心数值模型库
=========================================

物理模型（一维轴对称）：
  热传导:  rho(C)*cp(C) * dT/dt = (1/r) d/dr ( k(C) * r * dT/dr )      (固定域)
           在收缩材料坐标 xi=r/R(t) 下:
           rho(C)*cp(C) * dT/dt = (1/(R(t)^2 xi)) d/dxi ( xi*k(C)*dT/dxi )
  水分扩散: dC/dt = (1/r) d/dr ( D(C,T) * r * dC/dr )                  (固定域)
           在收缩材料坐标 xi=r/R(t) 下:
           dC/dt = (1/(R(t)^2 xi)) d/dxi ( xi*D(C,T)*dC/dxi )

  说明：收缩情形采用仿射（均匀）收缩假定 r = xi*R(t)。此时干基含水率 C 定义在
  干物质上，材料单元内干物质质量守恒，故水分守恒方程在 xi 坐标中退化为上述纯扩散
  形式（无附加对流项），且总水分质量严格守恒（见文档推导）。

边界条件（r=R(t), 即 xi=1）:
  -k dT/dr = h (T - T_inf(t))        ->  -(k/R) dT/dxi = h (T - T_inf)
  -D dC/dr = k_m (C - C_inf(t))      ->  -(D/R) dC/dxi = k_m (C - C_inf)
中心对称: dT/dr = dC/dr = 0

时间离散: theta 方法 (theta=0.5 为 Crank-Nicolson), 非线性系数用 Picard 迭代取
半步状态平均值; 空间离散: 节点中心有限体积法 (轴对称, 保守型).
"""

import os
import numpy as np
import openpyxl
from scipy.interpolate import PchipInterpolator
from scipy.linalg import solve_banded


# --------------------------------------------------------------------------
# 路径解析 (可移植): 环境变量优先, 其次按仓库结构定位
# --------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root():
    p = _HERE
    for _ in range(6):
        if os.path.isdir(os.path.join(p, "data")) or os.path.isdir(os.path.join(p, ".git")):
            return p
        p = os.path.dirname(p)
    return os.path.dirname(_HERE)


ROOT = _find_repo_root()


def _find_data_dir():
    """题目附件目录: CUMCM_DATA_DIR > <repo>/data/附件 > <repo>/data"""
    env = os.environ.get("CUMCM_DATA_DIR")
    if env:
        return os.path.abspath(env)
    for cand in (os.path.join(ROOT, "data", "附件"), os.path.join(ROOT, "data")):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(
        "未找到题目附件目录, 请设置环境变量 CUMCM_DATA_DIR 指向含 附件1.xlsx 的目录")


DATA_DIR = _find_data_dir()
BASE = DATA_DIR + os.sep
WORK_OUT = os.environ.get("CUMCM_OUT") or os.path.join(ROOT, "work", "out")
RESULT_DIR = os.environ.get("CUMCM_RESULT_DIR") or os.path.join(ROOT, "outputs", "result_files")
FIG_DIR = os.environ.get("CUMCM_FIG_DIR") or os.path.join(ROOT, "outputs", "figures")
DOCS_DIR = os.path.join(ROOT, "docs")
TABLES_DIR = os.path.join(ROOT, "outputs", "tables")
for _d in (WORK_OUT, RESULT_DIR, FIG_DIR, TABLES_DIR):
    os.makedirs(_d, exist_ok=True)



# --------------------------------------------------------------------------
# 数据读取
# --------------------------------------------------------------------------
def _read_rows(path, sheet=0):
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.worksheets[sheet]
    return list(ws.iter_rows(values_only=True))


def load_oven():
    """附件1: 烘房温度 T_inf(t) [C] 与水分浓度 C_inf(t) [kg/kg]"""
    rows = _read_rows(BASE + "附件1.xlsx")[1:]
    t = np.array([float(r[0]) for r in rows])
    T = np.array([float(r[1]) for r in rows])
    C = np.array([float(r[2]) for r in rows])
    return t, T, C


def load_radius():
    """附件2: 药材半径 R(t), 返回 (t[s], R[m])"""
    rows = _read_rows(BASE + "附件2.xlsx")[1:]
    t = np.array([float(r[0]) for r in rows])
    R = np.array([float(r[1]) for r in rows]) / 100.0  # cm -> m
    return t, R


class Environment:
    """烘房环境 (温度/湿度), 单调三次插值 + 常值外延.

    参数
    ----
    t, T, C : 附件1 数据
    t_hold  : 若给定, 表示 t>t_hold 之后环境保持为 t=t_hold 时刻的值
              (默认 None, 即用数据末点常值外延)
    """

    def __init__(self, t, T, C, t_hold=None):
        if t_hold is None:
            t_hold = t[-1]
        self.t_hold = float(t_hold)
        m = t <= self.t_hold
        self._tT = PchipInterpolator(t[m], T[m], extrapolate=False)
        self._tC = PchipInterpolator(t[m], C[m], extrapolate=False)
        self.T_hold = float(self._tT(self.t_hold))
        self.C_hold = float(self._tC(self.t_hold))
        self.T0 = float(T[0])
        self.C0_data = float(C[0])

    def T_inf(self, t):
        t = np.asarray(t, dtype=float)
        v = np.where(t <= self.t_hold,
                     self._tT(np.clip(t, 0.0, self.t_hold)),
                     self.T_hold)
        return v if v.ndim else float(v)

    def C_inf(self, t):
        t = np.asarray(t, dtype=float)
        v = np.where(t <= self.t_hold,
                     self._tC(np.clip(t, 0.0, self.t_hold)),
                     self.C_hold)
        return v if v.ndim else float(v)


class RadiusLaw:
    """附件2 半径 R(t) 及其变化率 (PCHIP 插值 + 端点外延)"""

    def __init__(self, t, R):
        self.t = t
        self.R = R
        self._f = PchipInterpolator(t, R, extrapolate=False)
        self._d = self._f.derivative()
        self.R0 = float(R[0])
        self.Rmin = float(R[-1])

    def R_of(self, t):
        t = np.asarray(t, dtype=float)
        v = np.where(t <= self.t[-1], self._f(np.clip(t, 0.0, self.t[-1])), self.R[-1])
        return v if v.ndim else float(v)

    def dR_of(self, t):
        t = np.asarray(t, dtype=float)
        v = np.where(t <= self.t[-1], self._d(np.clip(t, 0.0, self.t[-1])), 0.0)
        return v if v.ndim else float(v)


class ConstRadius:
    """固定半径 (问题 1-3)"""

    def __init__(self, R0=0.02):
        self.R0 = R0
        self.Rmin = R0

    def R_of(self, t):
        return self.R0 + 0.0 * np.asarray(t, dtype=float)

    def dR_of(self, t):
        return 0.0 * np.asarray(t, dtype=float)


# --------------------------------------------------------------------------
# 物性模型
# --------------------------------------------------------------------------
class Props:
    """三种物性参数组.

    mode = 'p1' : 附录2 (问题1), 常物性
    mode = 'p3' : 附录3 (问题2、3)
    mode = 'p4' : 附录4 (问题4)
    """

    def __init__(self, mode):
        assert mode in ("p1", "p3", "p4")
        self.mode = mode

    def rho(self, C):
        C = np.asarray(C, dtype=float)
        if self.mode == "p1":
            return np.full_like(C, 820.0)
        if self.mode == "p3":
            return 650.0 + 128.0 * C
        return 760.0 + 90.0 * C

    def cp(self, C):
        C = np.asarray(C, dtype=float)
        if self.mode == "p1":
            return np.full_like(C, 2600.0)
        if self.mode == "p3":
            return 1450.0 + 2736.0 * C / (C + 1.0)
        return 1850.0 + 2150.0 * C / (C + 1.0)

    def kcond(self, C):
        C = np.asarray(C, dtype=float)
        if self.mode == "p1":
            return np.full_like(C, 0.36)
        if self.mode == "p3":
            return 0.21 + 0.38 * C / (C + 1.0)
        return 0.12 + 0.20 * C / (C + 1.0)

    def D(self, C, T_C):
        """水分扩散系数 [m^2/s]; T_C 为摄氏温度"""
        C = np.asarray(C, dtype=float)
        C = np.maximum(C, 1e-8)
        if self.mode == "p1":
            return 7e-9 * np.exp(-0.89 / C)
        T = np.asarray(T_C, dtype=float) + 273.15
        if self.mode == "p3":
            return 2.4e-3 * np.exp(-0.45 / C) * np.exp(-3850.0 / T)
        return 4.2e-4 * np.exp(-0.30 / C) * np.exp(-3850.0 / T)


# --------------------------------------------------------------------------
# 求解器
# --------------------------------------------------------------------------
class DryingSolver:
    """轴对称一维有限体积 + theta 时间离散求解器.

    Parameters
    ----------
    N      : 径向单元数 (节点数 N+1)
    props  : Props 实例
    radius : ConstRadius 或 RadiusLaw
    env    : Environment
    h      : 对流换热系数 [W/(m^2 K)]
    km     : 对流传质系数 [m/s]
    theta  : 时间离散参数 (0.5 = Crank-Nicolson, 1.0 = 隐式 Euler)
    grid   : 'uniform' 等距网格；'power' 近壁加密 (stretch 次幂拉伸)
    heat_cap_mode : 'volume'  使用 rho(C)*cp(C)*V 作为热容 (默认)
                    'material' 使用干物质守恒热容 (附录物性一致性检验用)
    """

    def __init__(self, props, radius, env, h=25.0, km=8e-7, N=400, theta=0.5,
                 grid="uniform", stretch=1.0, T0=28.0, C0=2.55,
                 heat_cap_mode="volume", km_scale=1.0, be_start=2,
                 adv_mode="none"):
        self.props = props
        self.radius = radius
        self.env = env
        self.h = h
        self.km = km * km_scale
        self.N = N
        self.theta = theta
        self.T0 = T0
        self.C0 = C0
        self.grid = grid
        self.stretch = stretch
        self.heat_cap_mode = heat_cap_mode
        self.be_start = be_start
        # adv_mode: 'none'  -> 材料(Lagrangian)坐标形式 (默认, 水分质量严格守恒)
        #           'euler' -> 直接对 Eulerian 扩散方程作 xi=r/R(t) 变换所得的
        #                      变体 (含附加对流项), 用于模型形式敏感性分析
        self.adv_mode = adv_mode
        self._build_grid()

    # ---------------- 网格 ----------------
    def _build_grid(self):
        N = self.N
        if self.grid == "uniform":
            xi = np.linspace(0.0, 1.0, N + 1)
        else:
            s = np.linspace(0.0, 1.0, N + 1)
            p = self.stretch
            xi = 1.0 - (1.0 - s) ** p
        self.xi = xi
        self.xi_f = 0.5 * (xi[1:] + xi[:-1])              # 面位置, 长度 N
        self.dxi = np.diff(xi)                            # 节点区间宽度, 长度 N
        xi_lo = np.concatenate([[0.0], self.xi_f])        # 节点控制体下界
        xi_hi = np.concatenate([self.xi_f, [1.0]])        # 节点控制体上界
        self.xi_lo, self.xi_hi = xi_lo, xi_hi
        self.A_f = self.xi_f                              # 面面积系数 (再乘 R)
        self.V_c = 0.5 * (xi_hi ** 2 - xi_lo ** 2)        # 体积系数 (再乘 R^2)
        self.d_node = self.dxi.copy()     # 相邻节点间距 = 区间宽度 (xi 单位)

    # ---------------- 物性系数 ----------------
    def _coeffs(self, T, C):
        p = self.props
        return {"rho": p.rho(C), "cp": p.cp(C), "k": p.kcond(C), "D": p.D(C, T)}

    # ---------------- 三对角求解 ----------------
    def _solve(self, Y_n, Cap, g, g_out, Yinf_old, Yinf_new, dt, th, a_adv=None):
        """theta 方法三对角系统.

        Cap_i (Y_i^{n+1}-Y_i^n)/dt = th*F^{n+1} + (1-th)*F^n
        F_i = q_{i-1} - q_i,  q_i = g_i (Y_{i+1}-Y_i) 为节点 i 经上侧面流出通量,
        q_N = g_out (Y_N - Yinf) 为外表面对流通量.
        """
        N = self.N
        q = np.empty(N + 1)
        q[:N] = g * (Y_n[:-1] - Y_n[1:])   # 流出节点 i 的通量 (沿 +xi)
        q[N] = g_out * (Y_n[-1] - Yinf_old)
        F_old = np.empty(N + 1)
        F_old[0] = -q[0]
        F_old[1:] = q[:-1] - q[1:]
        if a_adv is not None:
            # 附加对流项 a_i (Y_{i+1}-Y_{i-1})  (仅模型变体 adv_mode='euler')
            adv = np.zeros(N + 1)
            adv[1:-1] = a_adv[1:-1] * (Y_n[2:] - Y_n[:-2])
            adv[0] = 0.0
            adv[-1] = a_adv[-1] * (Y_n[-1] - Y_n[-2])
            F_old = F_old + adv

        diag = Cap / dt + th * (np.concatenate([np.zeros(1), g]) +
                                np.concatenate([g, [g_out]]))
        ab = np.zeros((3, N + 1))
        ab[0, 1:] = -th * g
        ab[1, :] = diag
        ab[2, :-1] = -th * g
        if a_adv is not None:
            # 隐式部分: 内部节点 i: +th*a_i*Y_{i+1} - th*a_i*Y_{i-1}
            ab[0, 2:] += -th * a_adv[1:-1]      # A[i,i+1] = -th*a_i
            ab[2, :-1] += np.concatenate([th * a_adv[1:-1], [th * a_adv[-1]]])
            ab[1, -1] += -th * a_adv[-1]
        rhs = Cap / dt * Y_n + (1 - th) * F_old
        rhs[-1] += th * g_out * Yinf_new
        return solve_banded((1, 1), ab, rhs)

    # ---------------- 组装 ----------------
    def _assemble(self, R, T_n, C_n, cf, dt, th, Tinf_new, Cinf_new,
                  Tinf_old, Cinf_old, Rd=0.0):
        N = self.N
        A_f = R * self.A_f                    # 内部面面积 (2*pi*H 归一)
        A_out = R                             # 外表面面积 (2*pi*H 归一)
        V = R * R * self.V_c                  # 控制体体积 (pi*H 归一)

        k_f = 0.5 * (cf["k"][:-1] + cf["k"][1:])
        D_f = 0.5 * (cf["D"][:-1] + cf["D"][1:])
        rho_cp = cf["rho"] * cf["cp"]

        # --- 温度 ---
        gT = A_f * k_f / (R * self.d_node)
        gT_out = A_out * self.h
        if self.heat_cap_mode == "volume":
            CT = rho_cp * V
        else:
            # 干物质守恒热容: 材料单元的干物质质量固定 (仿射收缩),
            # 热容 = m_d0 * (1+C) * cp,   m_d0 = rho_d0 * V_0
            rho_d0 = self.props.rho(self.C0) / (1.0 + self.C0)
            V0 = self.radius.R0 ** 2 * self.V_c
            CT = rho_d0 * V0 * (1.0 + cf["C"]) * cf["cp"]
        lam = Rd / R if R > 0 else 0.0
        a_adv = None
        if self.adv_mode == "euler" and abs(lam) > 0:
            den = np.empty(N + 1)
            den[1:-1] = self.xi[2:] - self.xi[:-2]
            den[0] = self.xi[1] - self.xi[0]
            den[-1] = self.xi[-1] - self.xi[-2]
            a_adv_c = np.zeros(N + 1)
            a_adv_c[1:-1] = self.xi[1:-1] * lam / den[1:-1]
            a_adv_c[-1] = self.xi[-1] * lam / den[-1]
            a_adv = a_adv_c
        T_new = self._solve(T_n, CT, gT, gT_out, Tinf_old, Tinf_new, dt, th,
                            None if a_adv is None else a_adv_c)

        # --- 水分浓度 ---
        gC = A_f * D_f / (R * self.d_node)
        gC_out = A_out * self.km
        C_new = self._solve(C_n, V.copy(), gC, gC_out, Cinf_old, Cinf_new, dt, th,
                            None if a_adv is None else a_adv_c)
        return T_new, C_new

    # ---------------- 单步 ----------------
    def step(self, T_n, C_n, t, dt, maxiter=50, tol=1e-11, verbose=False, theta=None):
        th = self.theta if theta is None else theta
        R = float(self.radius.R_of(t + 0.5 * dt))
        Rd = float(self.radius.dR_of(t + 0.5 * dt))
        Tinf_old = float(self.env.T_inf(t))
        Tinf_new = float(self.env.T_inf(t + dt))
        Cinf_old = float(self.env.C_inf(t))
        Cinf_new = float(self.env.C_inf(t + dt))

        T_g = T_n.copy()
        C_g = C_n.copy()
        it = 0
        for it in range(maxiter):
            Tm = 0.5 * (T_n + T_g)
            Cm = 0.5 * (C_n + C_g)
            cf = self._coeffs(Tm, Cm)
            cf["C"] = Cm
            T_new, C_new = self._assemble(R, T_n, C_n, cf, dt, th,
                                          Tinf_new, Cinf_new, Tinf_old, Cinf_old, Rd)
            dT = float(np.max(np.abs(T_new - T_g)))
            dC = float(np.max(np.abs(C_new - C_g)))
            T_g, C_g = T_new, C_new
            if max(dT, dC) < tol:
                break
        self.picard_iters = it + 1
        if max(dT, dC) >= tol and verbose:
            print("  [warn] Picard 未收敛 t=%.3f dt=%.4f dT=%.2e dC=%.2e"
                  % (t, dt, dT, dC))
        return T_g, C_g

    # ---------------- 时间推进 ----------------
    def run(self, t_end, dt_fun, t_eval=None, store=False, verbose=False):
        N = self.N
        T = np.full(N + 1, self.T0)
        C = np.full(N + 1, self.C0)
        t = 0.0
        t_eval = np.asarray([] if t_eval is None else t_eval, dtype=float)
        ie = 0
        out_t, out_T, out_C = [], [], []
        all_t = [0.0]; all_T = [T.copy()]; all_C = [C.copy()]
        if t_eval.size and abs(t_eval[0]) < 1e-12:
            out_t.append(0.0); out_T.append(T.copy()); out_C.append(C.copy()); ie = 1
        nstep = 0
        while t < t_end - 1e-12:
            dt = float(dt_fun(t))
            dt = min(dt, t_end - t)
            if ie < t_eval.size and t + dt > t_eval[ie] + 1e-12:
                dt = t_eval[ie] - t
            th_use = 1.0 if nstep < self.be_start else self.theta
            T, C = self.step(T, C, t, dt, verbose=verbose, theta=th_use)
            t += dt
            nstep += 1
            if store:
                all_t.append(t); all_T.append(T.copy()); all_C.append(C.copy())
            while ie < t_eval.size and abs(t - t_eval[ie]) < 1e-9:
                out_t.append(t); out_T.append(T.copy()); out_C.append(C.copy()); ie += 1
        res = {"t": np.array(out_t), "T": np.array(out_T), "C": np.array(out_C),
               "nstep": nstep, "xi": self.xi}
        if store:
            res["t_all"] = np.array(all_t)
            res["T_all"] = np.array(all_T)
            res["C_all"] = np.array(all_C)
        return res


def schedule_p1(t):
    """问题1 (输出每 1 s)"""
    if t < 10.0:
        return 0.05
    if t < 60.0:
        return 0.2
    return 1.0


def schedule_p2(t):
    """问题2 (输出每 1 s)"""
    if t < 10.0:
        return 0.05
    if t < 60.0:
        return 0.2
    return 1.0


def schedule_long(t):
    """问题3/4 长时程 (输出每 60 s, 步长均整除 60 s)"""
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


def schedule_fine(t, level=1):
    """细化方案: level=1/2/4 分别把步长缩小 1/1/2/4 倍 (输出仍每 1 s / 60 s)"""
    base = schedule_long(t) if t > 60 else schedule_p1(t)
    return base / float(level)


# --------------------------------------------------------------------------
# 输出与守恒审计辅助
# --------------------------------------------------------------------------
def radial_records(res, radius_law, r_out, times, which="C"):
    """把 (xi 网格) 结果插值到物理半径 r_out (m) 的序列.

    返回 shape = (len(times), len(r_out)) 数组; 超出当前半径的点返回 nan.
    """
    from scipy.interpolate import PchipInterpolator
    Y = res[which]
    t_arr = res["t"]
    out = np.full((len(times), len(r_out)), np.nan)
    for i, tt in enumerate(times):
        j = int(np.argmin(np.abs(t_arr - tt)))
        if abs(t_arr[j] - tt) > 1e-6:
            raise ValueError("time %.3f not in result grid" % tt)
        R = float(radius_law.R_of(tt))
        f = PchipInterpolator(res["xi"] * R, Y[j], extrapolate=False)
        vals = f(np.asarray(r_out, dtype=float))
        out[i] = vals
    return out


def water_mass(res, radius_law, which="C"):
    """单位高度(pi 归一)的总水分质量积分量 int C * V (仅用于守恒审计)"""
    pass
