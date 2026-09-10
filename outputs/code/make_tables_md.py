# -*- coding: utf-8 -*-
"""把计算结果导出为 Markdown 表格片段, 供说明文档使用."""
import json, numpy as np, os, sys
sys.path.insert(0,'work')
import herb_model as hm
tt = json.load(open(os.path.join(hm.WORK_OUT, 'tables.json')))
def md(rows, times, cols, fmt="%.4f", tfmt="%.4g"):
    out=[]
    out.append("| 时间 | " + " | ".join(str(c) for c in cols) + " |")
    out.append("|---|" + "---|"*len(cols))
    for t,row in zip(times,rows):
        out.append("| " + (tfmt%t) + " | " + " | ".join(fmt%v for v in row) + " |")
    return "\n".join(out)
L=[]
L.append("### 表1\n"+md(tt['表1'],tt['表1_t'],["%s cm"%c for c in tt['表1_r']]))
L.append("### 表2\n"+md(tt['表2'],tt['表1_t'],["%s cm"%c for c in tt['表1_r']]))
L.append("### 表3\n"+md(tt['表3'],[x/3600. for x in tt['表3_t']],["%s cm"%c for c in tt['表1_r']],tfmt="%.1f"))
L.append("### 表4\n"+md(tt['表4'],[x/3600. for x in tt['表3_t']],["%s cm"%c for c in tt['表1_r']],tfmt="%.1f"))
L.append("### 表5\n"+md(tt['表5'],tt['表5_t_h'],["%s cm"%c for c in tt['表1_r']],tfmt="%.2f"))
L.append("### 表6\n"+md(tt['表6'],tt['表6_t_h'],tt['表6_cols'],tfmt="%.2f"))
L.append("### 干燥时长\n- 问题3: %.4f h\n- 问题4: %.4f h"%(tt['问题3_干燥时长_h'],tt['问题4_干燥时长_h']))
open(os.path.join(hm.WORK_OUT, 'tables.md'), 'w').write("\n\n".join(L))
print(open(os.path.join(hm.WORK_OUT, 'tables.md')).read())
