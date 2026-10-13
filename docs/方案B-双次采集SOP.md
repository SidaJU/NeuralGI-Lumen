# 方案 B：双次采集 SOP

> 目标：采集「低射线输入 / 高射线答案」配对数据，让网络学
> **「从 16 条射线恢复 256 条射线的质量」** —— 这才是真正的性能收益来源。

---

## 0. 为什么必须做 B（先想清楚再做）

| | 方案 A（已完成）| 方案 B（本文）|
|---|---|---|
| 输入 | 滤波前 radiance | **低射线**（16 条）滤波前 |
| 答案 | 滤波后（Lumen 自己的输出）| **高射线**（256 条）的结果 |
| 网络学到 | 模仿 Lumen 滤波器 | 从稀疏采样重建 |
| 射线数 | 没变（64）| **64 → 16，省 4 倍 trace** |
| 收益 | **无**（负优化）| 真正的净提速 |
| 用途 | 验证训练管道 | **真实应用** |

**A 的天花板**：答案本身就是 Lumen 给的，网络最多学到 Lumen 的水平，不可能超越。

---

## 1. 射线数的含义（先搞清楚改的是哪个）

两个不同的东西，**别改错**：

| CVar | 含义 | 我们的设置 |
|---|---|---|
| `GatherOctahedronResolution` | probe **存储**的方向分辨率 | **8**（8×8=64 格子，不变）|
| `TracingOctahedronResolution` | **trace 时的射线密度** | **这个要改** |

```
TracingOctahedronResolution 4  → 4×4  = 16 条射线   ← 低（输入）
TracingOctahedronResolution 8  → 8×8  = 64 条射线   ← 默认
TracingOctahedronResolution 16 → 16×16 = 256 条射线 ← 高（答案）
```

**关键点**：Tracing=4 而 Gather=8 时，16 条射线要填充 64 个方向格子 ——
每个格子只有 1/4 的采样，**这正是网络要"补"的东西**（和超分同构）。

---

## 2. 前置：建一条可复现的相机路径（Sequencer）

**这是 B 唯一的硬门槛。** 两遍采集的相机必须逐帧严格一致，人手不可能复现，必须用 Sequencer。

### 步骤

1. 关卡里新建 **Level Sequence**
2. 加 **Camera Cuts Track** → 绑定一个 Camera Actor
3. 给 Camera 加 Transform 关键帧，做一条运动路径：
   - 时长 **10~20 秒**
   - **必须包含平移 + 旋转**（纯平移学不到时域）
   - 路径要扫过不同光照环境（亮/暗、室内/室外、墙角/开阔）
4. Sequencer 设置：
   - **Evaluation Type = Frame Locked**（锁定帧率，不受渲染卡顿影响）
   - 关闭循环播放
5. 保存

> **为什么必须锁帧率**：Sequencer 默认用 delta time 求值，帧率一波动
> 采样点就漂移，两遍采集就对不齐。

---

## 3. 采集：跑两遍

### 必开的设置（两遍都要）

## ① 控制台（CVar）

```bat
r.AntiAliasingMethod 0
```

## ② Project Settings（★ 不是 CVar，控制台设不了）

```
Edit → Project Settings → Engine → General Settings → Framerate

    ☑  Use Fixed Frame Rate        ← 勾上
       Fixed Frame Rate  = 60      ← 填 60
```

设完**重启编辑器**生效。

| 设置 | 在哪设 | 为什么 |
|---|---|---|
| `r.AntiAliasingMethod 0` | 控制台 | UE 5.6 默认用 TSR（值 4）做时域抗锯齿/超分，带亚像素 jitter。两遍 jitter 序列不一致会让 probe 位置微动。**0 = 全关** |
| **Use Fixed Frame Rate = 60** | **Project Settings** | **这是两遍对齐的关键**。它让引擎每帧固定推进 1/60 秒，**不管实际渲染多慢**。高射线（256 条）比低射线慢很多，没有它两遍的第 N 帧就对应不同相机位置 |

> ⚠️ **两个易错点**：
> 1. 旧版 UE 用 `r.TemporalAA.Enable 0`，**UE 5.6 已移除**，统一为 `r.AntiAliasingMethod`
> 2. `t.FixedFrameRate` **不是 CVar**（它是 `UPROPERTY(config)`，见 `Engine.h:1558`），
>    在控制台敲它**会静默失败**，必须去 Project Settings 设。
>    如果跳过这步，两遍会速度不同 —— 表现为校验 `[2]` 报偏差几百到几千，
>    且 `diag_frame_offset.py` 搜不到让偏差归零的 k。

### 第 1 遍：低射线 → 输入目录

```bat
r.Lumen.ScreenProbeGather.TracingOctahedronResolution 4
r.Lumen.ScreenProbeDump.OutputDir D:/ProbeDump_B_low
r.Lumen.ScreenProbeDump.MaxFrames 200
r.Lumen.ScreenProbeDump.Enable 1
```

从头播放 Sequencer，采满 200 帧后关掉：

```bat
r.Lumen.ScreenProbeDump.Enable 0
```

### 第 2 遍：高射线 → GT 目录

```bat
r.Lumen.ScreenProbeGather.TracingOctahedronResolution 16
r.Lumen.ScreenProbeDump.OutputDir D:/ProbeDump_B_high
r.Lumen.ScreenProbeDump.MaxFrames 200
r.Lumen.ScreenProbeDump.Enable 1
```

**从头播放同一个 Sequencer**（必须完全一致，从第 0 帧开始）。

> 高射线（256 条）会很慢，可能只有几 fps —— **没关系**，采集不要求实时。
> 因为 Sequencer 用了 Frame Locked，渲染慢不影响采样点。

---

## 4. 采完立刻校验（必做）

```bat
python tools/verify/verify_b_pairing.py D:/ProbeDump_B_low D:/ProbeDump_B_high
```

五项检查：

| # | 检查 | 不通过意味着 |
|---|---|---|
| 1 | 帧数/帧名对应 | 两遍长度不同 |
| 2 | **相机矩阵逐帧比对** | 相机路径没对齐 ← 最致命 |
| 3 | 几何（depth）交叉验证 | 两遍看到的不是同一场景 |
| 4 | **射线数差异** | CVar 没生效，两遍射线数一样 |
| 5 | GT 是否更干净 | 射线数设反了 |

**全部 `[PASS]` 才能训练。** 配对错了也能训，loss 还会降，但学到的是纯噪声。

---

## 5. 训练

```bat
python train/train.py --data D:/ProbeDump_B_low --gt-dir D:/ProbeDump_B_high
```

`--gt-dir` 是关键：GT 从高射线目录取，几何/输入从低射线目录取。

---

## 6. 坑与排查

| 现象 | 原因 | 修法 |
|---|---|---|
| `[4] 射线数没变` | CVar 没生效 / 拼错 / 改完没重开 | 确认 CVar 全名，改完重新 Enable |
| `[4] 高射线没更平滑` | **射线数设反了** | low 目录应该更噪（平滑度更高）|
| `[2] 相机超差` | Sequencer 未锁帧率 | `t.FixedFrameRate 60` + Frame Locked |
| `[2] 相机超差` | TAA jitter 漂移 | `r.TemporalAA.Enable 0` |
| `[1] 帧数不一致` | 两遍采的长度不同 | 用同一个 `MaxFrames` |
| `[3] 几何偏差大` | 相机没对齐，或场景有动态物体 | 检查相机；采集时避免动态物件 |

---

## 7. 参考基准（方案 A 实测，用于对比）

| 指标 | 数值 |
|---|---|
| 输入平滑度（滤波前）| **0.940** |
| GT 平滑度（滤波后）| **0.803** |
| 模型输出平滑度（A 训练后）| 待 B 重测 |

做完 B 后看：模型输出若能把**低射线输入（应比 0.940 更噪）**压到
接近**高射线 GT** 的平滑度，就说明网络真的学会了"补射线"。

---

## 8. 做完 B 之后

B 解决「模型能不能带来收益」，但还不是真实应用。还差最后一步：

**S4 — 推理接回引擎**：把 TensorRT engine 接进 UE，替换
`LumenScreenProbeFiltering.usf` 的传统滤波 pass。

见 `docs/` 后续补充。
