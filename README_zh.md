# CoRef：基于互补视觉参考证据的开放世界目标检测

CoRef 是完整的纯视觉检测系统。参考阶段利用空间反馈细化候选框，通过共享映射和实例间的学习关系提供类别证据，并用留类训练学习拒识。实例支持路线在同一批候选上，用冻结视觉特征中的类别归一化边界提供第二种未知提名信号。两路分数经已知样本校准后统一排序，再结合框质量与同类支持核验。

[English README](README.md) · [安装](docs/INSTALL.md) · [数据](docs/DATA.md) · [权重](docs/CHECKPOINTS.md) · [复现](docs/REPRODUCING.md)

## 五场景 RWD 结果

Overall 为五场景 VOC07 AP50（%）。本地三行采用 seed 42、43、44 的均值。

| 方法 | Task 1 U | Task 1 K | Task 2 PK | Task 2 CK |
|---|---:|---:|---:|---:|
| CODE（OWL-ViT L/14，已发表对照） | 21.7 | 40.8 | 43.6 | 36.2 |
| ROI Classifier | 5.5 | 28.1 | 26.8 | 22.7 |
| CoRef (relation-only) | 17.2 | 57.4 | 54.6 | 54.3 |
| **CoRef** | **23.6** | **58.1** | **56.9** | **56.6** |
| CoRef − CODE（百分点） | +1.9 | +17.3 | +13.3 | +20.4 |

在 7,112 张图的回放中，关系路线和实例支持路线分别独有提名 659 和 1,775 个未知目标。CoRef 最终检出其中的 571 和 1,560 个。

## 代码与复现

仓库中的可移植入口对应 **CoRef (relation-only)**，包括参考库构建、推理和评估。上表的完整 CoRef 行对应论文中的双路实验。命令见 [英文 README](README.md#quick-start)，数据准备和实验入口见[复现文档](docs/REPRODUCING.md)。
