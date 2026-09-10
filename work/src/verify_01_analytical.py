# -*- coding: utf-8 -*-
"""检验 1: 线性化问题 (常扩散系数 + 常环境) 与圆柱解析解对比"""
import sys; sys.path.insert(0,'work')
import numpy as np, herb_model as hm, herb_verify as hv

R=0.02; D=5e-9; km=8e-7; C0=2.55; Cinf=0.02
RHO=820.; CP=2600.; K=0.36; H=25.; T0=28.; Tinf=50.

class PropsLin(hm.Props):
    """常数 D 的线性模型 (用于解析解对比)"""
    def __init__(self): super().__init__('p1')
    def rho(self,C): return np.full_like(np.asarray(C,float), RHO)
    def cp(self,C):  return np.full_like(np.asarray(C,float), CP)
    def kcond(self,C): return np.full_like(np.asarray(C,float), K)
    def D(self,C,T): return np.full_like(np.asarray(C,float), D)

class EnvConst:
    def __init__(self,Tinf,Cinf): self.Ti=Tinf; self.Ci=Cinf
    def T_inf(self,t): return self.Ti
    def C_inf(self,t): return self.Ci

env=EnvConst(Tinf,Cinf)
props=PropsLin()
teval=np.concatenate([[0.],np.geomspace(1,1800,40)])

print("=== 温度场对比 (解析 vs 数值) ===")
for N,dt_scale in [(200,1.0),(400,1.0),(800,1.0),(400,0.25)]:
    sch=lambda t: hm.schedule_p1(t)*dt_scale
    s=hm.DryingSolver(props,hm.ConstRadius(R),env,N=N,theta=0.5)
    res=s.run(1800.0,sch,t_eval=teval)
    tt=np.array([10.,100.,600.,1800.]); 
    num_r=np.array([0.,0.005,0.01,0.015,0.02])
    idx=[int(np.argmin(abs(res['t']-x))) for x in tt]
    num=np.array([np.interp(num_r,res['xi']*R,res['T'][i]) for i in idx])
    alpha=K/(RHO*CP)
    ana=hv.analytical_temp(num_r,tt,R,alpha,H,K,T0,Tinf)
    print(" N=%4d dt*%4.2f  max|err_T|=%.3e  max_rel=%.2e"%(N,dt_scale,np.max(np.abs(num-ana)),np.max(np.abs(num-ana))/(Tinf-T0)))

print("=== 水分浓度场对比 ===")
for N,dt_scale in [(200,1.0),(400,1.0),(800,1.0),(400,0.25)]:
    sch=lambda t: hm.schedule_p1(t)*dt_scale
    s=hm.DryingSolver(props,hm.ConstRadius(R),env,N=N,theta=0.5)
    res=s.run(1800.0,sch,t_eval=teval)
    tt=np.array([10.,100.,600.,1800.])
    num_r=np.array([0.,0.005,0.01,0.015,0.02])
    idx=[int(np.argmin(abs(res['t']-x))) for x in tt]
    num=np.array([np.interp(num_r,res['xi']*R,res['C'][i]) for i in idx])
    ana=hv.analytical_cylinder(num_r,tt,R,D,km,C0,Cinf)
    print(" N=%4d dt*%4.2f  max|err_C|=%.3e  Bi=%.3f"%(N,dt_scale,np.max(np.abs(num-ana)),km*R/D))
