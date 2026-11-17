# P4 本地完整版快照

本目录保存 **revert P4 CL 107315 之前**的 P4 本地文件完整内容，
用于保留那些**不属于本项目、但在工作区里存在的他人改动**。

## 为什么要单独存

`engine/Source/Runtime/Renderer/Private/Lumen/LumenScreenProbeFiltering.cpp`
（GitHub 主快照，782 行）与 P4 本地版（818 行）相差 36 行，差异全部是
**RaidGame 项目其他人的代码**：

```
[++Raid] Add by weilan for Fpp View Lumen
  - GLumenFppForceTracesMoving / r.Lumen.Fpp.ForceTracesMoving
  - bIsFppView 分支、FppScreenProbeGatherState 选择

[++Raid] Add by shunhuang for SecondView Render
  - SHADER_PARAMETER_RDG_UNIFORM_BUFFER(SceneTexturesStruct) ×4
```

GitHub 主快照里有人把这些删掉了（commit `034cb6e`
"Remove Fpp View related code and clean up comments"）。

**我们项目自己的 NSPF 集成（`AddNSPFFilterPass` 在 `ConvertToSH` 之前）两版都有**，
见 `experiments/02-S4引擎接入/LOG.md` 第 9 节。

## 文件

| 文件 | 说明 |
|---|---|
| `LumenScreenProbeFiltering.p4-local-full.cpp` | P4 本地 818 行完整版（含上述 Fpp View / SecondView 代码 + 我们的 NSPF 集成）|

来源：`G:\theozhu_THEOZHU-PC1_4001\UnrealEngine\Engine\Source\Runtime\Renderer\Private\Lumen\LumenScreenProbeFiltering.cpp`

## 用途

仅作**回退保险**：若 revert 后发现需要别人的 Fpp View 改动，从这里取回。
日常参考请以 `engine/Source/...` 下的主快照为准（那才是我们项目要维护的版本）。
