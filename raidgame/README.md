# RaidGame 侧改动快照（P4 CL 107127 / S4.1 验证阶段）

本目录保存 **P4 CL 107127** 中三个文件的完整内容。这些改动按项目约定
**不提交 P4**，仅在此以快照形式保存，便于云端查看与日后恢复。

## 来源

| 文件 | P4 源路径 |
|---|---|
| `RaidGame.uproject` | `//RaidStreams/Raid/RaidGame/RaidGame.uproject` (edit) |
| `Source/RaidGame/NSPFNNEVerify.cpp` | `//RaidStreams/Raid/RaidGame/Source/RaidGame/NSPFNNEVerify.cpp` (add) |
| `Source/RaidGame/RaidGame.Build.cs` | `//RaidStreams/Raid/RaidGame/Source/RaidGame/RaidGame.Build.cs` (edit) |

实际磁盘位置：`G:\theozhu_THEOZHU-PC1_4001\RaidGame\`

## 这三个文件是干什么的

S4.1 阶段的 **NNE 环境验证**，属于一次性验证代码，不是生产链路：

- `RaidGame.uproject`：启用 NNE 相关插件（`NNERuntimeRDG` / `NNERuntimeORT`）
- `NSPFNNEVerify.cpp`：注册两个控制台命令
  - `NSPF.NNEVerify` — 在渲染线程用 RDG 跑一次 NNE 推理，与 torch FP32 参考做数值对比
  - `NSPF.NNEVerifyCPU` — 用 CPU runtime（ONNX Runtime）验证模型与 ONNX 导出正确性
- `RaidGame.Build.cs`：加 `NNE` / `NNERuntimeRDG` 模块依赖

**结论**：CPU 路径验证通过（误差 2.4e-07，FP32 极限），证实 NNE 支持我们的模型、
ONNX 导出正确。GPU 侧曾卡在「CPU 数据上传不上 GPU」，后来改为
**照搬官方 NNEDenoiser 的 shader 写入路径**（见 `engine/` 与 `experiments/02`）。

## 与生产代码的关系

| 阶段 | 位置 | 状态 |
|---|---|---|
| S4.1 验证（本目录）| RaidGame 项目侧 | 一次性验证，已弃用 |
| **S4.2 生产** | `engine/`（引擎侧）| ✅ 模型已在 Lumen 内实时推理，NN 输出驱动最终画面 |

生产链路**不依赖本目录任何文件**；`NSPF.NNEVerify*` 只是调试命令。

## 版本管理说明

- P4：CL 107127 未提交，按要求已 revert
- GitHub：本目录即完整备份（本仓库 `main`）
- 若需恢复：按相对路径复制回 `RaidGame/` 即可
