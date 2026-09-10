# -*- coding: utf-8 -*-
"""用已保存的 p3 结果重算表 5 (修正索引错误) 并写回 tables.json"""
import sys, os, json, numpy as np
sys.path.insert(0, 'work')
import herb_model as hm
OUT = hm.WORK_OUT
d = json.load(open(os.path.join(OUT, 'tables.json')))
res3 = np.load(os.path.join(OUT, 'p3.npz'))
t,C,xi=res3['t'],res3['C'],res3['xi']
rad=hm.ConstRadius(0.02)
r_m=np.round(np.arange(0,2.0001,0.1),4)/100.0
C_all=hm.radial_records({'t':t,'C':C,'xi':xi}, rad, r_m, t, which='C')
tab1_r=[0,0.5,1,1.5,2]
idx=[int(np.round(rr/0.1)) for rr in tab1_r]
tdry=d['问题3_干燥时长_h']*3600
tt=list(np.arange(6*3600,tdry,6*3600))+[tdry]
from scipy.interpolate import PchipInterpolator
rows=[]
for x in tt:
    i=int(np.searchsorted(t,x))
    if abs(t[i]-x)<1e-9:
        row=[float(C_all[i,k]) for k in idx]
    else:
        t1,t2=t[i-1],t[i]; w=(x-t1)/(t2-t1)
        Cx=(1-w)*C[i-1]+w*C[i]
        f=PchipInterpolator(xi,Cx,extrapolate=False)
        row=[float(f(rr/100.0/rad.R_of(x))) for rr in tab1_r]
    rows.append(row)
d['表5']=rows; d['表5_t_h']=[x/3600 for x in tt]
json.dump(d, open(os.path.join(OUT, 'tables.json'), 'w'), ensure_ascii=False, indent=1)
print("修正后的表5:")
for x,row in zip(d['表5_t_h'],rows): print("  %6.2f h: %s"%(x," ".join("%.4f"%v for v in row)))
# 与 result3.xlsx 对比
import openpyxl
ws = openpyxl.load_workbook(os.path.join(hm.RESULT_DIR, 'result3.xlsx')).worksheets[0]
err=0
for x,row in zip(tt,rows):
    if abs(x-round(x/60)*60)>1e-9: continue
    ir=int(round(x/60))+2
    for k in range(5):
        err=max(err,abs(ws.cell(ir,2+idx[k]).value-row[k]))
print("与 result3.xlsx 最大偏差 = %.2e"%err)
