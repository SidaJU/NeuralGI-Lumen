"""对比 PyTorch 与 TensorRT 的推理耗时.

用法:
    python tools/bench_trt.py --engine model_fp16.engine --data D:/ProbeDump_S2

说明:
    - 用真实数据, 测"整帧"推理 (20 万样本), 这才是实际部署的场景
    - 对比四种: PyTorch FP32 / FP16 / FP16+Graph / TensorRT
    - 每次都跑多轮取中位数, 避免单次波动

TensorRT 推理流程:
    runtime → 反序列化 engine → context → 设 shape → 拷数据到 GPU → 执行 → 拷回
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# --- TensorRT DLL 加载 (Windows 必需, 必须在 import tensorrt 之前) ---
_TRT_ROOT = os.environ.get(
    "TENSORRT_ROOT",
    r"C:\Users\theozhu\Downloads\TensorRT-10.16.1.11.Windows.amd64.cuda-12.9\TensorRT-10.16.1.11",
)
_DLL_DIR = os.path.join(_TRT_ROOT, "bin")
if os.path.isdir(_DLL_DIR):
    os.add_dll_directory(_DLL_DIR)
    os.environ["PATH"] = _DLL_DIR + os.pathsep + os.environ.get("PATH", "")
    sys.path.insert(0, _DLL_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "dataset"))
sys.path.insert(0, str(ROOT / "train"))

from probe_dataset import ProbeTexelDataset, ProbeSequence  # noqa: E402
from model import ProbeFilterMLP  # noqa: E402


def timeit(fn, warmup=5, repeat=20) -> float:
    """返回毫秒."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1000.0)


class TRTInfer:
    """TensorRT engine 的简单封装."""

    def __init__(self, engine_path: Path):
        import tensorrt as trt
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f:
            blob = f.read()
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(blob)
        self.context = self.engine.create_execution_context()
        self.in_name = self.engine.get_tensor_name(0)
        self.out_name = self.engine.get_tensor_name(1)
        print(f"  TensorRT engine 已加载: {engine_path.name}")
        print(f"    输入张量: {self.in_name}")
        print(f"    输出张量: {self.out_name}")

    def prepare(self, x_np: np.ndarray) -> None:
        """把输入数据搬到 GPU 并固定好 buffer.

        这一步在计时之外做一次. 真实部署时数据本来就在显存里
        (引擎刚渲染完), 不需要这步 —— 这正是 DLSS 能实时的关键.
        如果每次推理都 CPU<->GPU 拷贝 35MB, 光传输就几毫秒,
        再快的 kernel 也救不回来.
        """
        n, dim = x_np.shape
        self.context.set_input_shape(self.in_name, (n, dim))
        x = np.ascontiguousarray(x_np, dtype=np.float32)
        self.d_in = torch.from_numpy(x).cuda()
        self.d_out = torch.empty((n, 3), device="cuda", dtype=torch.float32)
        self.bindings = [int(self.d_in.data_ptr()), int(self.d_out.data_ptr())]

    def __call__(self) -> torch.Tensor:
        """只执行推理 (数据已在 GPU), 与 PyTorch 的测时口径一致."""
        self.context.execute_v2(self.bindings)
        return self.d_out

    def fetch(self) -> np.ndarray:
        """需要 CPU 结果时才拷回 (计时之外)."""
        return self.d_out.cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="model_fp16.engine")
    ap.add_argument("--ckpt", default="runs/v2_nohist/best.pt")
    ap.add_argument("--data", default="D:/ProbeDump_S2")
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--frame", type=int, default=1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[错误] 需要 CUDA")
        return 1

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # ---- 准备真实数据 (整帧) ----
    seq = ProbeSequence(args.data)
    g = seq.gather
    ds = ProbeTexelDataset(args.data, samples_per_frame=1,
                           use_history=False, max_frames=1)
    cur = seq.load_frame(args.frame)
    feat, target = ds._build_features(cur, None)

    h, w = feat.shape[:2]
    vmask = np.repeat(np.repeat(cur["valid"], g, axis=0), g, axis=1)
    x_np = feat.reshape(-1, feat.shape[-1])[vmask.reshape(-1)]      # 只取有效样本
    print(f"数据: 第 {args.frame} 帧, 有效样本 {x_np.shape[0]:,} 个, 特征 {x_np.shape[1]} 维")
    print()

    # ---- PyTorch 模型 ----
    ck = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    in_dim = 52 if a.get("no_history") else 56
    # 注意: model.float()/half() 是原地(in-place)操作, 会改变模型自身.
    # 若复用同一个实例, 第二次 half() 会让前面拿到的"FP32 模型"也变成 FP16,
    # 后续再用 float 输入就会报 dtype 不匹配. 所以这里为每种精度建独立实例.
    def make_model(dtype: torch.dtype) -> torch.nn.Module:
        m = ProbeFilterMLP(in_dim=in_dim, hidden=a.get("hidden", 64),
                           num_layers=a.get("layers", 4),
                           use_residual=not a.get("no_residual", False))
        m.load_state_dict(ck["model"])
        m.to(device).to(dtype)
        return m.eval()

    results: list[tuple[str, float]] = []

    # 1. PyTorch FP32
    m32 = make_model(torch.float32)
    x32 = torch.from_numpy(x_np).float().cuda()
    results.append(("PyTorch FP32", timeit(lambda: m32(x32), repeat=args.repeat)))

    # 2. PyTorch FP16
    m16 = make_model(torch.float16)
    x16 = x32.half()
    results.append(("PyTorch FP16", timeit(lambda: m16(x16), repeat=args.repeat)))

    # 3. PyTorch FP16 + CUDA Graph
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = m16(x16)
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            y_static = m16(x16)
        results.append(("PyTorch FP16+Graph",
                        timeit(lambda: gr.replay(), warmup=3, repeat=args.repeat)))
    except Exception as e:  # noqa: BLE001
        print(f"  [跳过 CUDA Graph] {e}")

    # 4. TensorRT
    try:
        trt_infer = TRTInfer(ROOT / args.engine)
        trt_infer.prepare(x_np)              # 数据搬到 GPU (计时之外)
        # 正确性校验: 与 PyTorch 输出对比
        with torch.no_grad():
            ref = m32(x32).cpu().numpy()
        got = trt_infer().cpu().numpy()
        diff = np.abs(ref - got).max()
        mean_rel = np.abs(ref - got).mean() / (np.abs(ref).mean() + 1e-6)
        print(f"    精度校验: 最大绝对差={diff:.6f}, 平均相对差={mean_rel:.2%}")
        if mean_rel > 0.05:
            print("    [警告] 与 PyTorch 输出偏差较大, 检查结果")
        # 只测执行耗时 (数据已在 GPU, 与 PyTorch 口径一致)
        results.append(("TensorRT", timeit(lambda: trt_infer(),
                                           warmup=3, repeat=args.repeat)))
    except Exception as e:  # noqa: BLE001
        print(f"  [TensorRT 失败] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # ---- 汇总 ----
    print()
    print("=" * 70)
    print("推理耗时对比 (整帧, 中位数值)")
    print("=" * 70)
    base = results[0][1]
    print(f"  {'方案':<24}{'耗时(ms)':>10}{'相对FP32':>10}")
    print("  " + "-" * 44)
    for name, ms in results:
        print(f"  {name:<24}{ms:>10.3f}{base/ms:>9.2f}x")

    best = min(r[1] for r in results)
    trt_ms = dict(results).get("TensorRT")
    print()
    print("收益参考: 射线 64→16 约省 2.5~4.5 ms")
    for name, ms in results:
        lo, hi = 2.5 - ms, 4.5 - ms
        verdict = "净收益为正" if hi > 0 else "可能亏损"
        print(f"  {name:<24} 净收益 {lo:+.2f} ~ {hi:+.2f} ms   ({verdict})")

    if trt_ms:
        py_best = min(r[1] for r in results if r[0] != "TensorRT")
        print()
        print(f"  TensorRT 相比最佳 PyTorch 方案: {py_best/trt_ms:.2f}x 加速")

    # TensorRT 版本诊断
    try:
        import tensorrt as _trt
        has_fp16 = hasattr(_trt.BuilderFlag, "FP16")
        has_fp8 = hasattr(_trt.BuilderFlag, "FP8")
        print()
        if has_fp16:
            print(f"  TensorRT {_trt.__version__} 支持 FP16={has_fp16} FP8={has_fp8}")
        else:
            print("  [提醒] 当前 TensorRT 缺 FP16 flag (pip 精简版所致).")
            print("         装 NVIDIA 官方完整版(解压 zip 后 pip 装 python/*.whl)")
            print("         并把 bin 目录加入 PATH, 即可启用 FP16/FP8.")
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
