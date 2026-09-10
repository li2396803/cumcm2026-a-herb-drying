# -*- coding: utf-8 -*-
"""检验4b(修正): 收缩域水分守恒审计.
收缩仿射材料坐标下, 单位高度(pi 归一)的物理水量正比于 Q = int_0^1 C(xi) xi dxi
(因为 rho_d * R^2 = const, 体积因子 R^2 与干密度因子 1/R^2 相消)。
守恒要求: dQ/dt = -k_m (C_s - C_inf) / R(t)
"""
import sys, os, json
sys.path.insert(0,'work')
import numpy as np, herb_model as hm
import herb_verify as hv
OUT = hm.WORK_OUT
t,T,C=hm.load_oven(); env=hm.Environment(t,T,C)
props=hm.Props('p4'); rad=hm.RadiusLaw(*hm.load_radius())
s=hm.DryingSolver(props,rad,env,N=400)
r=s.run(72*3600.,hm.schedule_long,t_eval=np.arange(0,72*3600+1,60.))
Vc=hv._vol_coef(r['xi'])
Q=np.array([np.sum(r['C'][i]*Vc) for i in range(len(r['t']))])
dQ=np.gradient(Q,r['t'])
flux=np.array([-8e-7*(r['C'][i,-1]-float(env.C_inf(r['t'][i])))/float(rad.R_of(r['t'][i])) for i in range(len(r['t']))])
scale=np.maximum.reduce([np.abs(flux),np.abs(dQ),np.full_like(flux,1e-12)])
resid=np.abs(dQ-flux)/scale
print("问题4 收缩域守恒审计 (N=400):")
print("  Q(0)=%.6f (理论 2.55/2=1.275), Q(72h)=%.6f"%(Q[0],Q[-1]))
print("  dQ/dt 与 -km(Cs-Cinf)/R 相对残差: 平均=%.2e, 99%%=%.2e, max(去掉首末)=%.2e"%(
    resid[1:-1].mean(),np.quantile(resid[1:-1],0.99),resid[1:-1].max()))
json.dump({"Q0":float(Q[0]),"Q72":float(Q[-1]),"resid_mean":float(resid[1:-1].mean()),
           "resid_p99":float(np.quantile(resid[1:-1],0.99))},open(os.path.join(OUT,'verify_04.json'),'w'),indent=1)
# 同时给出"若错误使用固定域度量"的残差, 用于文档说明
M2=np.array([rad.R_of(r['t'][i])**2*np.sum(r['C'][i]*Vc) for i in range(len(r['t']))])
dM2=np.gradient(M2,r['t']); flux2=np.array([-rad.R_of(r['t'][i])*8e-7*(r['C'][i,-1]-float(env.C_inf(r['t'][i]))) for i in range(len(r['t']))])
sc2=np.maximum.reduce([np.abs(flux2),np.abs(dM2),np.full_like(flux2,1e-12)])
pass
