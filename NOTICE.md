# 许可范围说明（NOTICE）

本仓库采用 **MIT License**（见 [`LICENSE`](LICENSE)），其覆盖范围为**作者原创内容**：

| 覆盖 | 路径 |
|---|---|
| ✅ MIT 许可 | `work/`（全部源码与中间结果）、`outputs/`（结果文件、图表、表格、代码）、`docs/`（说明文档）、根目录 `README.md` / `NOTICE.md` |

**不覆盖（第三方材料）**：

| 内容 | 路径 | 版权 |
|---|---|---|
| 竞赛题面 | `data/A题.pdf` | 2026 年高教社杯全国大学生数学建模竞赛组委会 |
| 题目附件 1—2 | `data/附件/附件1.xlsx`、`data/附件/附件2.xlsx` | 同上 |
| 结果文件模板 | `data/附件/附件3/result1—4.xlsx` | 同上 |
| 论文格式规范 | `data/format2026.doc` | 同上 |

上述第三方材料仅用于复现本仓库的计算结果，其版权归主办方所有，**不在 MIT 许可范围内**，
再分发时请遵循主办方要求；如不需要，直接删除 `data/` 目录即可（不影响代码运行，
可用环境中变量 `CUMCM_DATA_DIR` 指向其他附件目录，详见 [`data/README.md`](data/README.md)）。

---

# License Scope (English)

This repository is released under the **MIT License** (see [`LICENSE`](LICENSE)), covering the
author's original work: `work/`, `outputs/`, `docs/`, `README.md`, and `NOTICE.md`.

The files under `data/` (competition problem statement and its attachments) are **third-party
materials** whose copyright belongs to the competition organizer; they are included solely for
reproducibility and are **not** covered by the MIT License.
