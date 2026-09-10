# -*- coding: utf-8 -*-
"""最终校核: 用 N=800 高分辨率网格复核问题3/4 的烘干时长"""
import sys, os, json, time
sys.path.insert(0,'work')
import numpy as np, herb_model as hm
t,T,C=hm.load_oven(); env=hm.Environment(t,T,C)
def dry(t,Cc,thr=0.15):
    idx=np.where(Cc<thr)[0]; i=idx[0]
    return t[i-1]+(thr-Cc[i-1])*(t[i]-t[i-1])/(Cc[i]-Cc[i-1])
res={}
for mode,moving,N in [('p3',False,800),('p4',True,800)]:
    props=hm.Props(mode); rad=hm.RadiusLaw(*hm.load_radius()) if moving else hm.ConstRadius(0.02)
    s=hm.DryingSolver(props,rad,env,N=N)
    t0=time.time()
    # 输出 30 s 间隔以提高交叉点插值精度
    r=s.run(100*3600.,hm.schedule_long,t_eval=np.arange(0,100*3600+1,30.))
    Cmax=r['C'].max(axis=1)
    td=dry(r['t'],Cmax)
    print("%s N=%d: 烘干时长 = %.4f h (%.1f s), 用时 %.0f s, 中心@57h=%.6f"%(mode,N,td/3600,td,time.time()-t0,r['C'][int(np.where(abs(r['t']-57*3600)<1e-6)[0][0]),0]))
    res[mode]={"t_dry_h":td/3600,"t_dry_s":td,"N":N}
json.dump(res, open(os.path.join(hm.WORK_OUT, 'final_check.json'), 'w'), indent=1)
