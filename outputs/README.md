# outputs/ 成果索引（A 题：药材的烘干问题）

| 文件/目录 | 说明 |
|---|---|
| `result_files/result1.xlsx` | 问题 1：温度 + 水分浓度，1801 个时刻（0—1800 s 每 1 s）× 21 个半径（0—2.0 cm 每 0.1 cm） |
| `result_files/result2.xlsx` | 问题 2：**整个烘干过程**，205730 个时刻（0—205729 s 每 1 s） |
| `result_files/result2_3h.xlsx` | 问题 2 的 0—3 h 对照（表 3/表 4 区间） |
| `result_files/result4_local.xlsx` | 问题 4 v2 非均匀收缩模型结果（含"表面位置"工作表） |
| `tables/v2_compare.json` 等 | 与外部独立实现的对照实验结果（散布扫描、平均判据复核、严格停止时刻） |
| `result_files/result3.xlsx` | 问题 3：水分浓度，3429 个时刻（0—205680 s 每 60 s）× 21 个半径 |
| `result_files/result4.xlsx` | 问题 4 基线：水分浓度，3051 个时刻（0—182942 s 每 60 s，含终点行）× 0—1.2 cm + "药材表面" |
| `figures/fig1_env.png` | 附件 1：烘房温度与水分浓度 |
| `figures/fig2_radius.png` | 附件 2：药材半径收缩过程 |
| `figures/fig3_p1.png` | 问题 1：预热平衡阶段温度场与水分场演化 |
| `figures/fig4_p2.png` | 问题 2：3 h 内温度/水分分布与中心—表面曲线 |
| `figures/fig5_p3.png` | 问题 3：干燥曲线、剖面演化、温度演化、干燥特性曲线 |
| `figures/fig6_p4.png` | 问题 4：收缩域干燥曲线、当前构型剖面、半径收缩、材料坐标剖面 |
| `figures/fig7_verify.png` | 检验：网格收敛、二维—一维对比、两模型干燥曲线对比、模型形式影响 |
| `figures/fig8_sensitivity.png` | 灵敏度：烘干时长对参数与假设的敏感性 |
| `figures/v2_fig1_shrinkage.png` | v2：非均匀收缩模型与基线对比（干燥曲线、材料映射、剖面、局部比容） |
| `figures/v2_fig2_latent.png` | v2：显式潜热导致的非物理冷却（反证） |
| `figures/v2_fig3_energy.png` | v2：能量审计（供热 vs 蒸发需求） |
| `figures/v2_fig4_optimize.png` | v2：工艺优化 Pareto 前沿与最优升温调度 |
| `tables/关键结果表.md` | 表 1—6 + 烘干时长 + 检验摘要 + 敏感性明细（可直接粘贴进论文） |
| `tables/tables.json` | 表 1—6 与烘干时长的机器可读结果 |
| `tables/verify_02.json` | 检验 B/C/D：网格与时间步收敛、独立方法交叉验证、守恒审计 |
| `tables/verify_03.json` | 检验 E：二维轴对称与一维模型逐时刻对比 |
| `tables/verify_04.json` | 问题 4 收缩域守恒专项审计 |
| `tables/verify_05.json` | 检验 F：全部参数与模型形式敏感性结果 |
| `tables/final_check.json` | 最终校核：$N=800$ 高分辨率复核的烘干时长 |
| `code/` | 全部脚本（与 `work/src/` 同源；复现步骤见说明文档 §10） |

## 结果一览

| 问题 | 核心结果 |
|---|---|
| 1 | 1800 s：温度 33.5758（中心）—36.7858 °C（表面）；含水率 2.5500—1.5102 kg/kg |
| 2 | 3 h：温度 49.8494—49.9666 °C；含水率 1.7662—1.0081 kg/kg |
| 3 | **烘干时长 57.1470 h（≈2.38 天）** |
| 4 | **烘干时长 50.8119 h（≈2.12 天，仿射）/ 47.6208 h（≈1.98 天，非均匀收缩）** |
| 扩展 | 工艺优化：46.75 h 且能耗代理 −6.0 %（CPLEX MILP + PDE 复核） |

结果文件均按题目要求保留四位小数；完整建模链条与全部检验见
[`../docs/药材烘干问题A题_建模与求解说明文档.pdf`](../docs/药材烘干问题A题_建模与求解说明文档.pdf)。
