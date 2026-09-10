# -*- coding: utf-8 -*-
"""检验5: 二维轴对称有限体积, 检验"忽略轴向" (无限长圆柱) 假定.

几何: 药材长 25 cm, 半径 2 cm。由对称性取半长 12.5 cm, z=0 为中心截面。
    rho cp dT/dt = (1/r) d/dr ( r k dT/dr ) + d/dz ( k dT/dz )
    dC/dt        = (1/r) d/dr ( r D dC/dr ) + d/dz ( D dC/dz )
边界: r=R 与 z=L/2 为对流边界 (h, km); r=0 与 z=0 对称。
时间: 隐式 Euler (系数滞后) + 稀疏直接解。
内部一致性校验: axial_sym=True 时把 z=L/2 也设为对称 => 轴向无梯度, 二维解应与一维解完全一致。
"""

import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import herb_model as hm

OUT = hm.WORK_OUT


def solve_2d(props, env, t_eval, R=0.02, Zhalf=0.125, nr=40, nz=60, h=25.0, km=8e-7,
             T0=28.0, C0=2.55, axial_sym=False, dt_small=10.0, t_small=3600.0,
             dt_big=60.0):
    """二维轴对称有限体积 (单元中心), 隐式 Euler。

    离散: 单元体积 V_ij = 0.5(r_{i+1}^2-r_i^2)*dz  (2*pi 归一)
          径向面: 通量系数 = r_f*dz*k/dr  -> 除以 V_ij
          轴向面: 通量系数 = r_c*dr_i*k/dz -> 除以 V_ij = r_c*dr_i*dz
          边界 (对流): 系数 = A/(0.5*d/k + 1/h) / V_ij
    """
    r_f = np.linspace(0, R, nr + 1)
    z_f = np.linspace(0, Zhalf, nz + 1)
    r_c = 0.5 * (r_f[1:] + r_f[:-1])
    z_c = 0.5 * (z_f[1:] + z_f[:-1])
    dr_i = np.diff(r_f)                 # 各单元径向厚度
    dz = Zhalf / nz
    N = nr * nz
    idx = lambda i, j: i * nz + j
    Vcell = (0.5 * (r_f[1:] ** 2 - r_f[:-1] ** 2))[:, None] * dz    # (nr,1)

    # ---------- 几何乘子 (已除单元体积) ----------
    face_k0, face_kn, face_mult = [], [], []
    for i in range(nr):
        for j in range(nz):
            k0 = idx(i, j)
            if i > 0:
                face_k0.append(k0); face_kn.append(idx(i - 1, j))
                face_mult.append(r_f[i] * dz / (r_c[i] - r_c[i - 1]))
            if i < nr - 1:
                face_k0.append(k0); face_kn.append(idx(i + 1, j))
                face_mult.append(r_f[i + 1] * dz / (r_c[i + 1] - r_c[i]))
            if j > 0:
                face_k0.append(k0); face_kn.append(idx(i, j - 1))
                face_mult.append(r_c[i] * dr_i[i] / dz)
            if j < nz - 1:
                face_k0.append(k0); face_kn.append(idx(i, j + 1))
                face_mult.append(r_c[i] * dr_i[i] / dz)
    face_k0 = np.array(face_k0); face_kn = np.array(face_kn)
    face_mult = np.array(face_mult, dtype=float)
    nf = len(face_k0)
    ztop = np.array([idx(i, nz - 1) for i in range(nr)])
    rtop = np.array([idx(nr - 1, j) for j in range(nz)])
    n_extra = 2 * N + 2 * nf + 2 * nz + (0 if axial_sym else 2 * nr)
    rows = np.zeros(n_extra, dtype=np.int64); cols = np.zeros(n_extra, dtype=np.int64)
    rows[:2 * N] = np.concatenate([np.arange(N), np.arange(N) + N])
    cols[:2 * N] = rows[:2 * N]
    p = 2 * N
    for m in range(nf):
        rows[p] = face_k0[m]; cols[p] = face_kn[m]; p += 1
        rows[p] = face_k0[m] + N; cols[p] = face_kn[m] + N; p += 1
    if not axial_sym:
        rows[p:p + nr] = ztop; cols[p:p + nr] = ztop; p += nr
    rows[p:p + nz] = rtop; cols[p:p + nz] = rtop; p += nz
    if not axial_sym:
        rows[p:p + nr] = ztop + N; cols[p:p + nr] = ztop + N; p += nr
    rows[p:p + nz] = rtop + N; cols[p:p + nz] = rtop + N; p += nz
    assert p == n_extra

    # 边界对流乘子 (含半格热阻)
    bm_r = R / (0.5 * dr_i[nr - 1] / 1.0 + 0.0)      # 占位, 实际按下式
    T = np.full(N, T0); C = np.full(N, C0)
    t = 0.0; t_end = float(np.max(t_eval)); te = np.asarray(t_eval, float)
    out = []; ie = 0; nstep = 0
    while t < t_end - 1e-9:
        dt = dt_small if t < t_small else dt_big
        if ie < te.size and t + dt > te[ie] - 1e-9:
            dt = max(te[ie] - t, 1e-6)
        dt = min(dt, t_end - t)
        Tinf = float(env.T_inf(t)); Cinf = float(env.C_inf(t))
        rho_cp = (props.rho(C) * props.cp(C)).ravel()
        kf = props.kcond(C).ravel(); Df = props.D(C, T).ravel()
        kface = 0.5 * (kf[face_k0] + kf[face_kn])
        Dface = 0.5 * (Df[face_k0] + Df[face_kn])
        gT_f = face_mult * kface
        gD_f = face_mult * Dface
        vals = np.zeros(n_extra)
        Vv = np.repeat(Vcell[:, 0], nz)
        vals[0:N] = rho_cp * Vv / dt
        vals[N:2 * N] = Vv / dt
        b = np.zeros(2 * N)
        b[:N] = rho_cp * Vv / dt * T
        b[N:] = Vv / dt * C
        p = 2 * N
        for m in range(nf):
            vals[p] = -gT_f[m]; p += 1
            vals[p] = -gD_f[m]; p += 1
        vals[0:N] += np.bincount(face_k0, weights=gT_f, minlength=N)
        vals[N:2 * N] += np.bincount(face_k0, weights=gD_f, minlength=N)
        # 边界
        if not axial_sym:
            mz = r_c * dz / (0.5 * dz / kf[ztop] + 1.0 / h)
            mzD = r_c * dz / (0.5 * dz / Df[ztop] + 1.0 / km)
        mr = R * dz / (0.5 * dr_i[nr - 1] / kf[rtop] + 1.0 / h)
        mrD = R * dz / (0.5 * dr_i[nr - 1] / Df[rtop] + 1.0 / km)
        if not axial_sym:
            vals[p:p + nr] = mz; p += nr
        vals[p:p + nz] = mr; p += nz
        if not axial_sym:
            vals[p:p + nr] = mzD; p += nr
        vals[p:p + nz] = mrD; p += nz
        assert p == n_extra
        if not axial_sym:
            b[ztop] += mz * Tinf; b[ztop + N] += mzD * Cinf
        b[rtop] += mr * Tinf; b[rtop + N] += mrD * Cinf
        A = sp.csr_matrix((vals, (rows, cols)), shape=(2 * N, 2 * N))
        y = spla.spsolve(A, b)
        T = y[:N]; C = y[N:]
        t += dt; nstep += 1
        while ie < te.size and abs(t - te[ie]) < 1e-6:
            out.append((t, T.copy(), C.copy())); ie += 1
    TO = np.array([o[1] for o in out]).reshape(len(out), nr, nz)
    CO = np.array([o[2] for o in out]).reshape(len(out), nr, nz)
    return te[:len(out)], TO, CO, r_c, z_c, nstep


def main():
    t, T, C = hm.load_oven()
    env = hm.Environment(t, T, C)
    props = hm.Props("p3")
    # --- 内部一致性: 轴向全对称 => 应与一维解一致 ---
    te_chk = np.array([3600., 7200.])
    tc, Tc, Cc, r_c, z_c, _ = solve_2d(props, env, te_chk, nr=40, nz=8, axial_sym=True,
                                       dt_small=60., dt_big=60.)
    s1 = hm.DryingSolver(props, hm.ConstRadius(0.02), env, N=400)
    r1 = s1.run(7200.0, hm.schedule_long, t_eval=te_chk)
    print("内部一致性校验 (轴向对称, 二维应等于一维):")
    for i, tt in enumerate(te_chk):
        j = int(np.argmin(np.abs(r1["t"] - tt)))
        fC = np.interp(r_c, r1["xi"] * 0.02, r1["C"][j])
        fT = np.interp(r_c, r1["xi"] * 0.02, r1["T"][j])
        print("  t=%.0f h: max|dC|=%.2e  max|dT|=%.2e"
              % (tt / 3600, np.max(np.abs(Cc[i, :, 0] - fC)), np.max(np.abs(Tc[i, :, 0] - fT))),
              flush=True)

    # --- 真实二维: 有限长 25 cm 圆柱 ---
    te = np.array([3600., 6 * 3600, 12 * 3600, 24 * 3600, 48 * 3600, 72 * 3600])
    print("\n二维轴对称计算 (nr=40, nz=60, 半长 12.5 cm) ...", flush=True)
    t0 = time.time()
    t2, T2, C2, r_c, z_c, ns = solve_2d(props, env, te, nr=40, nz=60)
    print("  完成 %d 步, %.0f s" % (ns, time.time() - t0), flush=True)
    s1 = hm.DryingSolver(props, hm.ConstRadius(0.02), env, N=400)
    r1 = s1.run(72 * 3600.0, hm.schedule_long, t_eval=te)
    print("\n  药材中心 (z=0, r=0) 与一维模型对比:")
    print("   t/h     C_2D      C_1D      dC         T_2D      T_1D      dT")
    res = []
    for i, tt in enumerate(t2):
        dC = C2[i, 0, 0] - r1["C"][i, 0]; dT = T2[i, 0, 0] - r1["T"][i, 0]
        print("  %5.1f  %.5f   %.5f  %+.2e   %.4f   %.4f  %+.2e"
              % (tt / 3600, C2[i, 0, 0], r1["C"][i, 0], dC, T2[i, 0, 0], r1["T"][i, 0], dT))
        res.append({"t_h": tt / 3600, "C2d": float(C2[i, 0, 0]), "C1d": float(r1["C"][i, 0]),
                    "dC": float(dC), "T2d": float(T2[i, 0, 0]), "T1d": float(r1["T"][i, 0]),
                    "dT": float(dT)})
    print("\n  t=%.0f h 沿轴向 (r=0) 的水分浓度:" % (t2[-1] / 3600))
    print("   z/cm:", np.array2string(z_c * 100, precision=2, max_line_width=250))
    print("   C   :", np.array2string(C2[-1, 0, :], precision=3, max_line_width=250))
    print("  t=%.0f h 沿轴向 (r=0) 的温度:" % (t2[-1] / 3600))
    print("   T   :", np.array2string(T2[-1, 0, :], precision=3, max_line_width=250))
    print("\n  t=%.0f h 沿径向 (z=0) 的水分浓度:" % (t2[-1] / 3600))
    print("   r/cm:", np.array2string(r_c * 100, precision=3, max_line_width=250))
    print("   C   :", np.array2string(C2[-1, :, 0], precision=4, max_line_width=250))
    with open(os.path.join(OUT, "verify_03.json"), "w") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print("saved -> out/verify_03.json")


if __name__ == "__main__":
    main()
