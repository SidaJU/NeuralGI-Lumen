# 引擎侧修改（UE 5.6 C++ / Shader）快照

本目录存放 **NeuralGI-Lumen 在 UE 引擎侧的全部修改文件**，用于在云端（GitHub）完整查看本次项目的所有改动。

> ⚠️ **这不是可独立编译的工程** —— 这些文件是从 UnrealEngine 源码树中复制出来的**参考/备份快照**，不参与本仓库的 Python 构建，也不是 UE 插件。

---

## 1. 源文件位置（P4 工作区）

本地引擎路径：

```
G:\theozhu_THEOZHU-PC1_4001\UnrealEngine\Engine\
```

本目录的 `engine/` 对应引擎的 `Engine/` 目录，其下保持**相同的相对路径**，便于对照回写。

---

## 2. 文件清单（8 个）

### S4.2 —— NNE 推理接入（P4 CL 107315，未提交）

| 文件 | 动作 | 说明 |
|---|---|---|
| `Source/Runtime/Renderer/Private/Lumen/LumenScreenProbeNSPF.h` | add | NNE 推理接入的头文件与接口 |
| `Source/Runtime/Renderer/Private/Lumen/LumenScreenProbeNSPF.cpp` | add | ★ 三段式接入（特征 CS → EnqueueRDG → 写回 CS）+ CVar + 失败自动回退 |
| `Source/Runtime/Renderer/Private/Lumen/LumenScreenProbeFiltering.cpp` | edit | ★ 接入点：**在 `ConvertToSH` 之前**用 NN 结果覆盖 `ScreenProbeRadiance` |
| `Shaders/Private/Lumen/NSPFFeatureBuild.usf` | add | 特征构建 compute shader（52 维/样本） |
| `Shaders/Private/Lumen/NSPFWriteBack.usf` | add | 写回 compute shader（expm1 还原 + DebugScale 诊断缩放） |
| `Source/Runtime/Renderer/Renderer.Build.cs` | edit | 加 `NNE` 模块依赖 |

### S1 —— 训练数据采集（改动已提交 P4，此处为当前内容备份）

| 文件 | 说明 |
|---|---|
| `Source/Runtime/Renderer/Private/Lumen/LumenScreenProbeDump.h` | 采集模块声明 |
| `Source/Runtime/Renderer/Private/Lumen/LumenScreenProbeDump.cpp` | 6 路 GPU buffer readback → 写 npy/json；含 WaitForMotion 起点对齐、FrameIndex 修复、诊断日志 |

---

## 3. 如何应用（回写到引擎）

把这些文件按相对路径覆盖回：

```
G:\theozhu_THEOZHU-PC1_4001\UnrealEngine\Engine\<相对路径>
```

例如：

```
engine\Source\Runtime\Renderer\Private\Lumen\LumenScreenProbeNSPF.cpp
  → G:\...\UnrealEngine\Engine\Source\Runtime\Renderer\Private\Lumen\LumenScreenProbeNSPF.cpp
```

随后重新编译引擎（`RaidGameEditor`），并重新编译 shader。

---

## 4. 关键改动说明（最容易踩坑的一处）

`LumenScreenProbeFiltering.cpp` 中，NN 结果的替换**必须发生在 `ConvertToIrradiance`（SH 编码）之前**：

```cpp
// 在 ConvertToIrradiance pass 之前
if (FRDGTextureRef NNRadiance = LumenScreenProbeNSPF::AddNSPFFilterPass(...))
{
    ScreenProbeRadiance = NNRadiance;   // 覆盖局部变量 → 后续编码/gather 都用 NN 结果
}
// 函数末尾
GatherParameters.ScreenProbeRadiance = ScreenProbeRadiance;   // 已是 NN 或传统
```

**原因**：下游 gather 阶段真正消费的是 `ScreenProbeRadianceSHAmbient` / `SHDirectional` / `IrradianceWithBorder`（由 `ScreenProbeRadiance` 编码而来），**不是原始 radiance**。若只在函数末尾替换原始 radiance，SH 仍由传统结果编码 → NN 输出被完全绕过（实测把 NN 输出清零后画面毫无变化）。

详见 `experiments/02-S4引擎接入/LOG.md` 第 9 节。

---

## 5. 版本管理现状

| 范围 | 位置 | 版本管理 |
|---|---|---|
| 引擎侧（本目录快照的原始文件）| `G:\...\UnrealEngine\` | **Perforce** —— CL 107315（未提交，属个人修改） |
| Python 侧 + 文档 + 模型 + 本快照 | 本仓库 | **GitHub** —— `SidaJU/NeuralGI-Lumen` |

> 说明：引擎侧文件按用户要求**不提交 P4**，因此这里以快照形式保存在 GitHub，作为云端完整备份与查阅。

---

## 6. CVar（引擎内控制）

```
r.Lumen.ScreenProbeNSPF.Enable      0/1   开启 NN 滤波（触发 GameThread 加载模型）
r.Lumen.ScreenProbeNSPF.Model              NNEModelData 资产路径
r.Lumen.ScreenProbeNSPF.Runtime            后端：NNERuntimeORTDml（≈1ms）/ NNERuntimeRDGHlsl（≈23ms）
r.Lumen.ScreenProbeNSPF.DebugScale         诊断：缩放 NN 输出（0→GI 消失，50→GI 爆亮，1→正常）
r.Lumen.ScreenProbeDump.*                  采集相关（Enable / OutputDir / MaxFrames / WaitForMotion）
```

---

## 7. 状态

**S4.2 已完成** —— NN 输出真正驱动最终画面（`DebugScale` 三个值均有明显响应）。
