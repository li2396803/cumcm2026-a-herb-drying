# -*- coding: utf-8 -*-
"""第三种情形：完全由附录 4 的密度律自洽预测半径（不强制标定附件 2）。
   用于量化"附件 2 的整体半径"与"附录 4 的密度律"两条数据的一致性。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import herb_v2 as v2


class SelfConsistentRadius:
    """半径由状态自洽决定：R(t)=R0*sqrt(I(t)/I0)，I=∫_0^1 v(C)dξ"""
    def __init__(self, R0, I0):
        self.R0 = R0; self.I0 = I0; self.Rmin = 0.0
        self._sol = None
    def bind(self, solver):
        self._sol = solver
    def R_of(self, t):
        if self._sol is None or self._sol._I_now is None:
            return self.R0
        I = float(self._sol._I_now)
        return self.R0 * np.sqrt(max(I, 1e-12) / self.I0)
    def dR_of(self, t):
        return 0.0


class SC(v2.LagSolver):
    """半径由状态自洽预测（不读附件 2）：R(t)=R0*sqrt(I(t)/I0)"""
    def __init__(self, *a, R0=0.02, **kw):
        super().__init__(*a, **kw)
        self.R0 = R0
        self._I_now = None
        self._I0 = None

    def _cum(self, C):
        v = self.props.v(C)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (v[:-1] + v[1:]) * self.ds_face)])
        I = cum[-1]
        if self._I0 is None:
            self._I0 = I
        self._I_now = I
        return v, cum, I

    def mapping_arrays(self, C, R_unused):
        v, cum, I = self._cum(np.asarray(C, dtype=float))
        J = cum / I
        R = self.R0 * np.sqrt(I / self._I0)
        return R * np.sqrt(J), J, I, v

    def _coeffs(self, T, C, R_unused):
        Rh = float(self.R_of(0.0))          # 先按当前状态确定 R
        return super()._coeffs(T, C, Rh)

    def R_of(self, t):
        if self._I_now is None:
            return self.R0
        return float(self.R0 * np.sqrt(self._I_now / self._I0))


def dry(t, C, thr=0.15):
    Cmax = C.max(axis=1); i = np.where(Cmax < thr)[0][0]
    return float(t[i - 1] + (thr - Cmax[i - 1]) * (t[i] - t[i - 1]) / (Cmax[i] - Cmax[i - 1]))


if __name__ == "__main__":
    t_oven, T_inf, C_inf = v2.hm.load_oven()
    env = v2.hm.Environment(t_oven, T_inf, C_inf)
    t_rad, R_rad = v2.hm.load_radius()
    te = np.arange(0, 120 * 3600 + 1, 60.0)

    s = SC(v2.PropsV2("p4"), v2.hm.ConstRadius(0.02), env, N=400,
           mapping="local", cap_mode="mass", R0=0.02)
    r = s.run(120 * 3600.0, v2.schedule_prod, t_eval=te)
    td = dry(r["t"], r["C"])
    # 末态半径
    s._cum(r["C"][-1])
    R_end = s.R_of(r["t"][-1])
    R_attach_end = float(v2.hm.RadiusLaw(t_rad, R_rad).R_of(td))
    print("自洽（不标定附件2）模型:")
    print("   烘干时长 = %.4f h" % (td / 3600))
    print("   预测末半径 = %.4f cm ; 附件2 在同刻给出 %.4f cm ; 相对差 %+.2f%%"
          % (R_end * 100, R_attach_end * 100, 100 * (R_end - R_attach_end) / R_attach_end))
    # 半径随时间的对照（几个时刻）
    radlaw = v2.hm.RadiusLaw(t_rad, R_rad)
    print("   时刻(h)  自洽预测R(cm)  附件2 R(cm)")
    for h in [0, 6, 12, 24, 48, td / 3600]:
        i = int(np.argmin(np.abs(r["t"] - h * 3600)))
        # 用该时刻状态重算 R
        sc = SC(v2.PropsV2("p4"), v2.hm.ConstRadius(0.02), env, N=400, mapping="local",
                cap_mode="mass", R0=0.02)
        # 用初始状态确定 I0，再用该时刻状态求 I
        sc._cum(r["C"][0]); sc._cum(r["C"][i])
        Rpred = sc.R_of(0.0)
        print("   %6.1f      %8.4f      %8.4f" % (h, Rpred * 100, radlaw.R_of(h * 3600) * 100))
    # 保存轨迹供作图
    Rp = []
    for i in range(len(r["t"])):
        sc = SC(v2.PropsV2("p4"), v2.hm.ConstRadius(0.02), env, N=400, mapping="local",
                cap_mode="mass", R0=0.02)
        sc._cum(r["C"][0]); sc._cum(r["C"][i])
        Rp.append(sc.R_of(0.0) * 100)
    np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "selfconsistent.npz"),
                        t=r["t"], R_pred=np.array(Rp), Ccenter=r["C"][:, 0], C=r["C"][:, ::20])
    json.dump({"t_dry_h": td / 3600, "R_end_pred_cm": R_end * 100,
               "R_end_attach_cm": R_attach_end * 100,
               "rel_diff_pct": 100 * (R_end - R_attach_end) / R_attach_end},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out",
                                "v2_selfconsistent.json"), "w"), ensure_ascii=False, indent=1)
