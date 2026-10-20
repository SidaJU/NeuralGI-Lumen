# NeuralGI-Lumen

用神经网络替换 Lumen Screen Probe 的传统时空滤波器，在保持画质的前提下
允许大幅降低每 probe 的射线数，从而换取净性能收益。

## 一句话
<img width="675" height="286" alt="企业微信截图_17921374958561" src="https://github.com/user-attachments/assets/b3f3af9b-198f-42e8-afc9-ea3b8f68b94e" />
<img width="672" height="284" alt="企业微信截图_17921374777051" src="https://github.com/user-attachments/assets/df3d655e-472b-4be1-b5f4-77f3750411c6" />


> 让 Lumen 用 1/4 的射线，跑出一样甚至更好的 GI。

## 为什么做这个

Lumen 的 Screen Probe 滤波有两个手工启发式，从源码里可以直接看到：

1. **时域权重是二值的**（`LumenScreenProbeFiltering.usf:262`）
   ```hlsl
   float PositionWeight = exp2(-10000.0f * (RelativeDepthDifference * RelativeDepthDifference)) > .1f ? 1.0f : 0.0f;
   ```
   历史要么全接受要么全丢弃，没有过渡 —— 这是鬼影/闪烁的根源。

2. **空间滤波是三个手工参数**（`SpatialFilterMaxRadianceHitAngle`、
   `SpatialFilterPositionWeightScale`、`SpatialFilterHalfKernelSize`），
   固定 4/12 邻域，无法针对局部几何自适应。

实测：滤波后 probe 内部方向维度的平滑度仅从 **0.938 降到 0.801**（-14.6%），
噪点依然明显。这就是改进空间。

## 技术路线

```
Trace（每 probe 8×8=64 条射线）
   ↓ ScreenProbeCompositeTracesWithScatterCS
CompositedScreenProbeRadiance    ← 我们要的【输入】（噪点大）
   ↓ 时域滤波（二值权重）
   ↓ 空间滤波（手工高斯核）
ScreenProbeRadiance              ← 我们要的【答案】（较平滑）

目标：用 NN 直接学 Composited → Filtered 的映射，
      然后把射线数从 64 降到 16，用省下的 trace 时间换净收益。
```

## 引擎侧改动

所有改动用 `[theozhu++]` / `[theozhu--]` 包裹，与团队 `[++Raid]` 约定区分。

| 文件 | 说明 |
|---|---|
| `Engine/.../Lumen/LumenScreenProbeDump.h` | 新增，采集接口与文档 |
| `Engine/.../Lumen/LumenScreenProbeDump.cpp` | 新增，6 路 readback + 写盘 |
| `Engine/.../Lumen/LumenScreenProbeFiltering.cpp` | 修改，调用点（1 处） |

CVar：
```
r.Lumen.ScreenProbeDump.Enable          0|1
r.Lumen.ScreenProbeDump.OutputDir       输出目录
r.Lumen.ScreenProbeDump.MaxFrames       最多帧数，0=不限
```

详细改动说明见 `docs/`。

## 数据

引擎采集输出（每帧）：

| 文件 | shape | dtype | 说明 |
|---|---|---|---|
| `frame_%06d_in.npy` | (validH, validW, 3) | float32 | 滤波前 radiance（输入） |
| `frame_%06d_gt.npy` | (validH, validW, 3) | float32 | 滤波后 radiance（答案） |
| `frame_%06d_hitdist.npy` | (validH, validW) | uint8 | 命中距离 |
| `frame_%06d_depth.npy` | (probeH, probeW) | uint32 | 深度，需 `asfloat` 解码 |
| `frame_%06d_normal.npy` | (probeH, probeW, 2) | uint8 | 法线，八面体编码 |
| `frame_%06d_wpos.npy` | (probeH, probeW, 4) | float32 | 平移后世界坐标 |
| `frame_%06d.json` | — | — | 相机矩阵（时域对齐） |

典型尺寸（1440×832 视口）：`validH=280, validW=720, probeH=35, probeW=90`
→ 每帧 **3150 probe × 64 方向 = 201,600 个训练样本**。

## 采集流程

```bat
r.Lumen.ScreenProbeDump.OutputDir D:/ProbeDump_S2
r.Lumen.ScreenProbeDump.MaxFrames 1000
r.Lumen.ScreenProbeDump.Enable 1
```
然后开 Realtime 或 PIE 跑若干秒。readback 有 1~3 帧延迟，属正常。

验证：
```bat
python tools/verify/verify_s2.py D:/ProbeDump_S2
```

## ⚠️ 当前方案的天花板（重要）

现在用的 GT 是 **方案 A：自配对** —— 同一帧里，滤波前=输入、滤波后=答案。

**这意味着网络最多学到 Lumen 的水平，不可能超越**，因为答案本身就是 Lumen 给的。
它只用于**验证训练管道**。

要获得真正的性能收益，必须用 **方案 B：双次采集**：
- `r.Lumen.ScreenProbeGather.TracingOctahedronResolution 4`  → 采输入（16 射线）
- `r.Lumen.ScreenProbeGather.TracingOctahedronResolution 16` → 采答案（256 射线）
- 相机路径必须完全一致（Sequencer）

这样网络学的才是"从 16 射线恢复 256 射线的质量"。

### 方案 B 的落地工具（已就绪）

| 文件 | 用途 |
|---|---|
| `docs/方案B-双次采集SOP.md` | 采集操作手册：Sequencer 怎么建、CVar 怎么设、坑怎么避 |
| `tools/verify/verify_b_pairing.py` | 采完**立即**校验（5 项），防止静默失败 |

**为什么必须有校验**：两次采集若相机路径错开一帧，输入和答案就对不上，
网络学到的是纯噪声 —— 而**训练不会报错**，loss 照降、权重照存。

```bat
python tools/verify/verify_b_pairing.py D:/ProbeDump_B_low D:/ProbeDump_B_high
```

全部 `[PASS]` 才能训练：

```bat
python train/train.py --data D:/ProbeDump_B_low --gt-dir D:/ProbeDump_B_high
```

## 目录结构

```
NeuralGI-Lumen/
├─ README.md                 本文件
├─ docs/                     方案、引擎改动、数据格式
├─ tools/
│  ├─ verify/                验证与诊断脚本
│  ├─ dataset/               数据加载与预处理
│  └─ capture/               采集相关
├─ train/                    训练代码
└─ eval/                     评估
```

## 参考

- **NIV (Neural Irradiance Volume)**, Eurographics 2026 —— 小型 MLP 表示 irradiance
  field，消费级 GPU 上亚毫秒；同内存预算下质量比 probe 方案高一个数量级。
  我们借鉴其性能基准与 relative L2 损失。
- **NRC (Neural Radiance Cache)**, Müller et al. 2021 / NVIDIA RTXGI v2.0 ——
  验证 NN 可替代 probe-based irradiance caching。
- **Neural Light Grid**, Iwanicki & Sloan, SIGGRAPH 2024 —— 给每个 probe 配小网络。
