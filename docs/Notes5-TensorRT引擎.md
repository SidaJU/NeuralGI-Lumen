# Notes 5：TensorRT 引擎

> 通用知识点。
>
> 目标：搞懂 TensorRT 的本质、engine 是什么、它为什么快、怎么调用。
>
> 前置：Notes4（ONNX）——TensorRT 的输入就是 ONNX。

---

## 1. 一句话本质

> **TensorRT 是深度学习模型的"编译器"。**

```
C++ 源码 (.cpp)  →  gcc / MSVC  →  可执行文件 (.exe)
                                        ↑ 针对特定 CPU

ONNX (.onnx)     →  TensorRT     →  Engine (.engine)
                                        ↑ 针对特定 GPU
```

**完全同构**：

| | 源码 | 编译器 | 产物 | 绑定 |
|---|---|---|---|---|
| 传统编程 | `.cpp` | gcc | `.exe` | 绑定 CPU 架构 |
| 深度学习 | `.onnx` | **TensorRT** | `.engine` | 绑定 GPU 型号 |

**关键定位**：TensorRT **只管推理，不管训练**。

| | 训练 | 推理 |
|---|---|---|
| PyTorch | ✅ | ✅（但不是最优）|
| **TensorRT** | ❌ | ✅（极致优化）|

---

## 2. 为什么叫"引擎"（Engine）

因为它是一个**执行引擎**（execution engine）：

```
游戏引擎：输入场景数据 → 渲染 → 输出画面
推理引擎：输入张量    → 计算 → 输出张量
```

它是一个**自包含**的黑盒子：权重、kernel、执行计划都在里面，不依赖训练框架。

---

## 3. 完整工作流

```
① 训练（PyTorch 等）
   产出: 权重文件
      ↓ 导出
② ONNX（计算图 + 权重常量）        ← "源代码"
      ↓ TensorRT 构建（几秒~几分钟）
③ Engine（针对目标 GPU 的可执行程序）← "exe"
      ↓ 每次推理
④ 加载 engine → 喂显存地址 → 出结果
```

**只有 ②③ 是 TensorRT 的事。训练（①）它完全不参与。**

---

## 4. ⭐ 构建时在干什么（"编译"的 6 步）

```
输入: ONNX 计算图
   ↓
① 解析 (Parse)
   读进 TensorRT 的内部图表示

② 图优化 (Graph Optimization) ★
   - 层融合：Linear + ReLU 合并成一个 kernel
   - 消除无用层（如恒等 Cast）
   - 常量折叠（能提前算的先算好）

③ 精度校准 (Precision Calibration)
   FP32 → FP16（或 FP8 / INT8），数据体积减半或更多

④ Kernel 自动选择 (Auto-Tuning) ★★ 最耗时
   在你的 GPU 上真的跑几十种 kernel 实现
   测量每种耗时，选最快的组合

⑤ 显存规划 (Memory Planning)
   预分配缓冲区，规划复用（避免反复申请释放）

⑥ 序列化 (Serialize)
   把以上全部写成二进制 .engine 文件
```

### 第 ④ 步是核心

> TensorRT 不是"猜"哪个 kernel 快，是**在目标 GPU 上真的跑一遍来测速**。

这就是"**用构建时间换运行时间**"——构建慢几秒，之后每次推理都受益。

**对部署完全划算**（启动时构建一次，之后每帧都快）。

---

## 5. Engine 文件里有什么

以典型 engine 为例：

```
xxx.engine
├─ 权重（已转精度，如 FP16）
├─ Kernel 代码 / 配置        ← ④ 选中的最优实现
├─ 执行计划                  ← 哪些 kernel、什么顺序
├─ 显存布局信息              ← buffer 怎么分配、复用
└─ 元数据（版本、shape 范围等）
```

### 与 ONNX 对比

| | ONNX | Engine |
|---|---|---|
| 角色 | 源代码 | 可执行程序 |
| 权重精度 | 通常 FP32 | FP16 / FP8 等 |
| 有 kernel 吗 | ❌ 只有算子声明 | ✅ 有选好的实现 |
| 有执行计划吗 | ❌ | ✅ |
| 能直接跑吗 | ❌ 需要运行时解释 | ✅ 直接执行 |
| 跨平台 | ✅ | ❌ 绑 GPU |

> **注意**：engine 通常**比 ONNX 大**——因为它不只是模型，是"模型 + 执行程序"。

---

## 6. ⭐ 输入 / 输出是什么（最关键）

### 核心认知

> **输入输出是"显存地址"（指针），不是数据值！**

```python
# 准备（一次性）
d_in  = 输入 tensor（在 GPU 上）
d_out = 空 tensor（在 GPU 上）

# 执行
bindings = [int(d_in.data_ptr()), int(d_out.data_ptr())]
#                 ↑ 整数地址        ↑ 整数地址
context.execute_v2(bindings)
```

`bindings` 就是一个**指针数组**，TensorRT 拿着它直接读写你的显存。

### 为什么这样设计？→ **零拷贝**

```
数据本来就在显存里（上游刚产生）
   ↓
告诉 TensorRT："读地址 A，写地址 B"
   ↓
零拷贝，直接算
```

> **这是实时的关键**。如果接口是"传数据值"，每次都要 CPU↔GPU 拷贝几十 MB，光传输就几毫秒，再快的 kernel 也救不回来。

**踩坑**：如果每次推理都在 `__call__` 里做 `.cuda()` / `.cpu()`，测出来的时间会包含拷贝，**严重失真**。

---

## 7. 推理执行流程

```python
# 一次性
runtime  = trt.Runtime(logger)
engine   = runtime.deserialize_cuda_engine(blob)   # 反序列化
context  = engine.create_execution_context()        # 执行上下文
context.set_input_shape("input_name", (n, dim))    # 设本次形状

# 每次推理
context.execute_v2(bindings)     # 同步执行
# 或 execute_async_v3(stream)    # 异步（部署常用，可与渲染重叠）
```

**内部**：

```
execute_v2
   ↓ 按⑥序列化时定好的计划
   kernel_1（融合的 Linear+ReLU）：读输入 → 算 → 写中间 buffer
   kernel_2（融合的 Linear+ReLU）：读中间 → 算 → 写
   ...
   最后 kernel → 写输出地址
```

- 全程在 GPU 上
- 读写你给的显存地址
- 没有 Python，没有逐层 dispatch

---

## 8. 为什么比原生框架快（本质）

### 原生框架（如 PyTorch）每次执行的开销

```
Python 函数调用
   → 逐层 dispatch（Linear → CUDA kernel，ReLU → CUDA kernel…）
   → 每个算子一次 kernel 启动
   → Python 解释器开销
   → 算子间中间结果反复写回显存
```

### TensorRT 的执行

```
直接按编译好的计划跑 kernel
   → 没有 Python
   → 没有逐层 dispatch
   → kernel 数量少（层融合）
   → 中间结果少写回显存
   → 数据精度更低（FP16）
```

### 开销对比表

| 开销来源 | 原生框架 | TensorRT |
|---|---|---|
| Python 解释器 | ✅ 有 | ❌ 无 |
| 逐层 dispatch | ✅ 有 | ❌ 无 |
| kernel 数量 | 多（每层一次）| **少**（层融合后）|
| 数据精度 | 通常 FP32 | **FP16 / FP8** |
| 中间结果访存 | 多 | **少**（融合省掉）|

---

## 9. 主要优化手段（它凭什么快）

| 手段 | 做了什么 | 收益 |
|---|---|---|
| **层融合** | Linear+ReLU 等合并成一个 kernel | 减少 kernel 启动 + 减少中间访存 |
| **精度校准** | FP32 → FP16/FP8 | 数据体积减半，带宽压力↓ |
| **Kernel 自动选择** | 实测多种实现选最快 | 针对硬件最优 |
| **显存复用** | 预分配 + 复用 buffer | 减少分配开销 |
| **常量折叠** | 提前算好固定部分 | 减少运行时计算 |

> **对访存密集型网络**（激活值搬运 >> 权重），**层融合 + 低精度**贡献最大。

---

## 10. Engine 的两个限制（源于"编译"本质）

既然是编译产物，就有编译产物的特性：

| 限制 | 原因 | 后果 |
|---|---|---|
| **不跨 GPU** | 编译时针对具体 GPU 优化 | 换 GPU 要重新构建 |
| **不超出 shape 范围** | 编译时按 profile 范围优化 | 超范围要重新构建 |

对应地，ONNX 是"源码"（跨平台），engine 是"exe"（绑硬件）。

---

## 11. 精度支持（视版本与硬件）

| 精度 | 每数大小 | 说明 |
|---|---|---|
| FP32 | 4 字节 | 基准 |
| **FP16** | 2 字节 | 最常用，体积减半 |
| **FP8** | 1 字节 | 新硬件（如 Blackwell）支持 |
| BF16 | 2 字节 | 动态范围同 FP32 |
| INT8 | 1 字节 | 需校准，量化 |

**踩坑**：某些精简/非官方的 TensorRT 包可能**缺少 FP16/FP8 的构建 flag**。验证方法：

```python
import tensorrt as trt
print(hasattr(trt.BuilderFlag, "FP16"))   # 必须 True 才能启用
```

---

## 12. 常见坑速查

| 坑 | 解决 |
|---|---|
| Windows 报 DLL load failed | 显式 `os.add_dll_directory(<TensorRT>/bin)` |
| pip 版缺 FP16 flag | 用官方完整版（zip 解压后 pip 装 `python/*.whl`）|
| 构建报找不到 `.onnx.data` | 切工作目录到 ONNX 所在目录 |
| 推理时间远慢于预期 | 计时里混入了 CPU↔GPU 拷贝 |
| 换 GPU 后 engine 报错 | engine 不跨 GPU，重新构建 |
| shape 超限报错 | 构建时 profile 的 max 设大一点，或重新构建 |

---

## 13. 一句话总结

| 问题 | 答案 |
|---|---|
| **TensorRT 是什么？** | 深度学习模型的编译器 |
| **输入什么？** | ONNX（计算图 + 权重）|
| **输出什么？** | Engine（针对目标 GPU 的可执行程序）|
| **构建时干嘛？** | 优化 + 实测选 kernel + 序列化 |
| **输入/输出数据是什么形式？** | **显存地址（指针）**，零拷贝 |
| **为什么快？** | 层融合 + 低精度 + 无 Python/dispatch + kernel 最优 |
| **限制？** | 不跨 GPU、不超 shape 范围 |

---

## 附：核心速查表

```
工作流:
  训练权重 → ONNX → [TensorRT 构建] → Engine → 推理

构建 6 步:
  解析 → 图优化 → 精度校准 → kernel 自动选择 → 显存规划 → 序列化

调用:
  Runtime.deserialize_cuda_engine(blob)   # 加载
  engine.create_execution_context()        # 上下文
  context.set_input_shape(name, shape)    # 设形状
  context.execute_v2([in_ptr, out_ptr])   # 同步执行
  context.execute_async_v3(stream)        # 异步（部署推荐）

engine 内容:
  权重(低精度) + kernel + 执行计划 + 显存布局 + 元数据

优化手段:
  层融合 / 低精度 / kernel 选型 / 显存复用 / 常量折叠

限制:
  绑 GPU、绑 shape 范围
```
