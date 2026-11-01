# VRDD：视觉参考驱动的开放世界目标检测

VRDD使用有标注的视觉实例作为类别参考，通过候选定位反馈、实例关系学习与标定拒识完成已知类别识别和未知目标检测。类别验证不需要任务文本或属性提示。

## 从哪里开始

- 环境：[docs/INSTALL.md](docs/INSTALL.md)
- 数据目录与Game纠错：[docs/DATA.md](docs/DATA.md)
- 各场景、各任务的权重：[docs/CHECKPOINTS.md](docs/CHECKPOINTS.md)
- 训练、消融与诊断复现入口：[docs/REPRODUCING.md](docs/REPRODUCING.md)

公开权重按场景和任务命名，例如 `checkpoints/Game/task1/model.pt`。每个模型配套自己的参考对象清单和标定文件，不可跨场景或任务混用。Surgical对应论文里的Surgery。

Git仓库保存代码、配置、协议、轻量结果和说明；大权重单独放在Release中。数据图像、预训练骨干和本机缓存不进入仓库。

```bash
python run_vrdd.py download-weights
python run_vrdd.py check --scene Game --task task1
python run_vrdd.py build-bank --scene Game --task task1 --data assets/rwd --device cuda:0
python run_vrdd.py infer --scene Game --task task1 --image /path/to/image.jpg --device cuda:0
```

`src/refnet`为原网络，`src/refnet_f1`为标定与最终决策；`protocols`保存实例和来源组划分，`results`保存归档摘要。`versions`是模型依赖，保留该结构以避免更改历史数值实现。新推理入口不会改写原训练和评估代码。

F1是决策版本，不是额外的一套训练权重。原始模型和新标注的来源差异在机器可读清单中保留；不把旧模型描述为已在纠错数据上重新训练。Task 2为PK与CK联合拟合。

当前完成代码整理、CPU检查及权重哈希核对；便携入口还需在目标GPU环境验证，不将其当作本轮新增的正式实验成绩。
