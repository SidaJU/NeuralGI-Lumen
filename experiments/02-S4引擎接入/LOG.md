# 实验 02：S4.2 —— NNE 推理接入 Lumen（替换 Screen Probe 滤波）

## 1. 目标

把训练好的模型接回引擎，**实时**替换 Lumen 的 Screen Probe 传统滤波。

```
CompositedScreenProbeRadiance (GPU 纹理, 滤波前)
    ↓ ① 特征构建 compute shader
    InputBuf (52 维/样本)
    ↓ ② NNE EnqueueRDG
    OutputBuf (3 维/样本)
    ↓ ③ 写回 compute shader
    ScreenProbeRadiance (替换传统滤波结果)
```

---

## 2. 关键决策：放弃 CPU 上传，照搬官方 shader 写入路径

### 背景
S4.1 阶段用 `QueueBufferUpload` 把 CPU 数据喂进 NNE input buffer，**始终失败**
（GPU readback 全 0）。尝试过的修复均无效：

| 尝试 | 结果 |
|---|---|
| typed buffer（非 structured）| 失败 |
| 加 `EBufferUsageFlags::NNE` | 失败 |
| `CreateUploadDesc` + `AddCopyBufferPass` | 失败 |
| 两阶段 Graph（模仿 NNE 内部权重上传）| 失败 |
| 调整 `ConvertToExternalBuffer` 与上传的先后顺序 | 失败 |

### 根因（查证官方实现后确认）
官方 `NNEDenoiser` **从不使用 CPU 上传**：

- `NNEDenoiser` 里搜索 `QueueBufferUpload` → **0 结果**
- 官方用 `InputProcess.AddPasses()`（compute shader）把纹理写进 buffer
  （`NNEDenoiserGenericDenoiser.cpp:126-130`）
- NNE 内部唯一的 `QueueBufferUpload` 只用于上传**权重**
  （`NNERuntimeRDGModelHlsl.cpp:82-88`）

**结论**：CPU 上传不是 NNE RDG 支持的路径。而 shader 写入恰好就是生产形态
（我们的输入本来就是 GPU 纹理），因此直接走官方路径。

### 官方三段式（严格照搬）
```cpp
// NNEDenoiserGenericDenoiser.cpp:126-135
1. InputProcess.AddPasses(..., InputBuffers);        // shader 写 buffer
2. ModelInstance.EnqueueRDG(GraphBuilder, ...);      // 推理
3. OutputProcess.AddPasses(..., OutputBuffers, ...); // shader 读 buffer
```

### 必须遵守的 4 个官方细节
| 细节 | 出处 |
|---|---|
| typed buffer（`Gemm` 用 `PF_R32_FLOAT` 建 SRV）| `NNERuntimeRDGGemm.cpp:138-143` |
| `Desc.Usage \|= EBufferUsageFlags::NNE` | `NNEDenoiserGenericDenoiser.cpp:42-45` |
| `CreateUAV(FRDGBufferUAVDesc(Buf, PF_R32_FLOAT))` **显式 Format** | `NNEDenoiserIOProcessBase.cpp:387` |
| `ERDGPassFlags::Compute \| NeverCull` | `NNEDenoiserIOProcessBase.cpp:187` |

---

## 3. ★ 特征对齐表（shader ↔ Python）

**这是正确性核心。** 必须逐维与 `tools/dataset/probe_dataset.py::_build_features` 对齐。

| 维度 | 含义 | Python 实现 | shader 实现 | 状态 |
|---|---|---|---|---|
| `[0:27]` | 3×3 邻域 radiance | `log1p(in)` | `log(1+x)` | ✅ |
| `[27:36]` | 3×3 邻域 hitdist | 原始 | 原始 | ✅ |
| `[36:45]` | 3×3 邻域 valid | probe 级 `valid` | `depth > 0` | ⚠️ 近似 |
| `[45]` | depth | `sign(d)*log1p(abs(d))` | 同 | ✅ |
| `[46:48]` | world normal | 原始 | 原始 | ✅ |
| `[49:51]` | world position | `(w−中位数)/1000` | `w/1000` | ⚠️ 近似 |
| — | 无效 probe | `geo[~valid] = 0` | 清零 | ✅ |

### 踩过的坑（第一版全错，已修）
| 维度 | 第一版（错）| 后果 |
|---|---|---|
| `[0:27]` | 直接填线性 radiance | 量级差几个数量级，模型输出完全错误 |
| `[45]` | `log1p(d)` | 未处理符号，负深度出错 |
| `[49:51]` | probe UV（−1..1）| 与相对世界坐标**完全无关** |
| 无效 probe | 未清零 | wpos 的 −1e8 垃圾值污染特征 |

### 近似项的说明
- **valid**：Python 用 `cur["valid"]`（probe 级标记），shader 用 `depth>0` 近似。
  两者高度相关但不完全等价，待测影响。
- **wpos**：Python 用 `(w − 有效probe中位数)/1000`，shader 算中位数需额外
  reduction pass，改用 `w/1000` 近似（translated 坐标系下相机在原点附近，
  与中位数同量级）。

---

## 4. 输入维度：52 还是 56？（易错点）

| 训练方式 | `in_dim` | 说明 |
|---|---|---|
| 带历史（默认）| **56** | 含 `[52:55]` 历史 radiance + `[55]` 历史有效 |
| `--no-history` | **52** | 历史维度被**完全移除**，不是填 0 |

**当前部署模型 `runs/b_nohist` 是 52**（训练日志 `in_dim=52`）。

shader 输出 stride 与 cpp `NSPF_IN_DIM` 必须同步为 52。换模型时若维度不同，
两处都要改（已在代码注释里相互标注）。

**历史特征的价值**：Lumen 传统滤波本身就有时域累积，去掉历史可能影响质量。
第一版先跑通，后续若要加回历史，需在引擎侧保存上一帧纹理与相机矩阵做重投影。

---

## 5. 实验记录

| 编号 | 内容 | 配置 | 状态 |
|---|---|---|---|
| E1 | 生产管线搭建（三段式）| 照搬 NNEDenoiser | ✅ 编译通过 |
| E2 | 训练 `--no-history` 模型以匹配 shader | 混合 2284 帧, 15 epoch, in_dim=52 | ✅ **完成** loss=**0.1072**（15 轮全部刷新 best，无反弹），17.7min |
| E3 | **引擎端到端跑通** | model_nspf_dyn (动态batch) + GameThread 预加载修复 | ✅ **成功**：模型加载→渲染线程推理→写回→画面正常→可回退。无崩溃 |
| E4 | **换 DirectML 后端解决性能** | `NNERuntimeORTDml` 替 `NNERuntimeRDGHlsl` | ✅ **耗时 23.42ms → 1ms**（约 23 倍），无需 CUDA 互操作 |

### E3 实测数据（RTX 5060, stat gpu）

| 项 | 值 |
|---|---|
| `LumenScreenProbeFilter`（含传统滤波 + 我们的 NN 管线）| **avg 23.42ms** |
| `NNE.Operator.Hlsl.ElementWise.Unary` | 1.00ms |
| 整帧 GPU | 34.37ms |
| 画面 | 与传统滤波**肉眼无明显差别** |

### E3 结论与问题

1. **管线全通** —— 动态 batch 的 onnx 被 NNE 正常加载推理，94,656 样本/帧实时处理，写回生效且可开关。
2. **性能不达标**：NN 管线约 22ms（23.42 − 传统滤波 ~1ms），远超 16.6ms 整帧预算。
   对比 TensorRT 同模型 **0.749ms** —— NNE HLSL 的通用 Gemm 实现慢约 30 倍（无 tensor core / kernel 融合）。
   注意：当前是"传统滤波 + NN"双份开销，但瓶颈明显在 NN 侧。
3. **画面无明显差别**：有两种解释，待区分——
   a) 模型输出本就接近传统滤波（B' 的 GT 就是 Lumen 滤波结果，学得像 = 正常）；
   b) 写回未真正生效（需要验证）。

### 性能已解决（E4：换 DirectML 后端）

把后端从 `NNERuntimeRDGHlsl` 换成 **`NNERuntimeORTDml`**（ONNX Runtime + DirectML）：

| 后端 | ScreenProbe 耗时 |
|---|---|
| NNERuntimeRDGHlsl | 23.42 ms |
| **NNERuntimeORTDml** | **1 ms**（约 23 倍提升）|

**DirectML 是 D3D12 原生 GPU 后端**，与渲染共享显存，因此：
- 不需要 CUDA-D3D12 互操作（省掉数周工程量）
- 不需要改 RHI
- 已接近 TensorRT 的 0.749ms

**结论：DirectML 就是当前生产后端，TensorRT 路线暂不需要。**

### 后端可热切换

`r.Lumen.ScreenProbeNSPF.Runtime`（或 `.Model`）改变时触发：
`FlushRenderingCommands()`（等渲染线程空闲，避免悬垂指针）→ 丢弃旧实例 → 按新值重载。
**无需重启编辑器。**

---

### 待办（性能解决后的重点）

1. **画质对比确认**：Enable 0/1 截图对比，确认 NN 输出真正生效且优于/接近传统滤波
2. **特征数值对齐**：Python 算特征 vs shader readback 定量对比
3. 加回历史特征（当前 no-history，传统滤波有时域累积优势）

---

## 6. 验证计划（待执行）

```bat
REM ① 导出 onnx（NNE 兼容：静态 batch + 真 opset 13）
python tools/export_onnx.py runs/b_nohist/best.pt --fixed-batch 1024 --legacy -o model_nohist.onnx

REM ② 编辑器导入 onnx → NNEModelData 资产

REM ③ 启用
r.Lumen.ScreenProbeNSPF.Enable 1
r.Lumen.ScreenProbeNSPF.Model /Game/model_nohist.model_nohist
```

| 检查项 | 期待 |
|---|---|
| Output Log `[theozhu] NSPF: 模型已加载` | NNE 就绪 |
| 画面 GI 有变化，不崩黑/全白 | 管线在跑 |
| `Enable 0` 立刻恢复 | 回退机制正常 |

**第一版不追求画质超过传统滤波**，先验证"跑通 + 不崩 + 可回退"。

---

## 7. 风险与待办

| 项 | 说明 |
|---|---|
| 特征近似项 | valid / wpos 两处近似，需定量验证影响 |
| 无历史 | 质量可能低于传统滤波（后者有时域累积）|
| 未做数值对齐验证 | 计划：Python 算特征 vs shader readback 对比 |
| 性能未测 | 需对比 NN 管线 vs 传统滤波耗时 |

---

## 8. 文件清单

| 文件 | 作用 |
|---|---|
| `Engine/Shaders/Private/Lumen/NSPFFeatureBuild.usf` | 特征构建 shader |
| `Engine/Shaders/Private/Lumen/NSPFWriteBack.usf` | 写回 shader（expm1 还原）|
| `Engine/.../Lumen/LumenScreenProbeNSPF.h/.cpp` | 三段式集成 + CVar |
| `Engine/.../Lumen/LumenScreenProbeFiltering.cpp` | 接入点（失败自动回退）|
| `Engine/Source/Runtime/Renderer/Renderer.Build.cs` | 加 `NNE` 依赖 |

CVar：
- `r.Lumen.ScreenProbeNSPF.Enable`（0/1）
- `r.Lumen.ScreenProbeNSPF.Model`（资产路径）

P4: CL **107315**
