"""用 TensorRT 把 ONNX 构建成优化后的 engine.

用法:
    python tools/build_engine.py model.onnx
    python tools/build_engine.py model.onnx --fp16 --out model_fp16.engine

说明:
    TensorRT 会做这些优化 (都不需要手写 CUDA kernel):
      - 层融合 (Linear+ReLU 合成一个 kernel)
      - 精度校准 (FP32 -> FP16/FP8)
      - kernel 自动选择 (在你的 GPU 上试多种实现, 选最快)
      - 显存复用

    构建可能耗时几十秒~几分钟, 这是"用构建时间换运行速度"。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# --- TensorRT DLL 加载 (Windows 必需) ---
# 官方 zip 版不会自动注册 DLL 路径, import tensorrt 前必须显式加载,
# 否则报 "DLL load failed". 可用环境变量 TENSORRT_ROOT 覆盖默认路径.
_TRT_ROOT = os.environ.get(
    "TENSORRT_ROOT",
    r"C:\Users\theozhu\Downloads\TensorRT-10.16.1.11.Windows.amd64.cuda-12.9\TensorRT-10.16.1.11",
)
_DLL_DIR = os.path.join(_TRT_ROOT, "bin")
if os.path.isdir(_DLL_DIR):
    os.add_dll_directory(_DLL_DIR)
    os.environ["PATH"] = _DLL_DIR + os.pathsep + os.environ.get("PATH", "")
    sys.path.insert(0, _DLL_DIR)
else:
    print(f"[警告] 未找到 TensorRT DLL 目录: {_DLL_DIR}")
    print("        请设置环境变量 TENSORRT_ROOT 指向解压后的 TensorRT 根目录")

import tensorrt as trt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def build(onnx_path: Path, engine_path: Path, fp16: bool,
          workspace_mb: int, opt_batch: int, max_batch: int) -> int:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # EXPLICIT_BATCH: 旧版(TensorRT<=9)需要显式指定; TensorRT 10+ 已是默认行为,
    # 且该枚举在新版被移除. 这里做兼容处理.
    try:
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    except AttributeError:
        flag = 0
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    # torch 2.11 的新导出器会把权重存成外部文件 model.onnx.data.
    # TensorRT 解析时按"相对路径"找它, 所以必须把工作目录切到 onnx 所在目录,
    # 否则报 "Failed to open file: model.onnx.data".
    import os
    os.chdir(onnx_path.parent)

    print(f"解析 ONNX: {onnx_path}")
    data = onnx_path.read_bytes()
    if not parser.parse(data):
        print("  [错误] ONNX 解析失败:")
        for i in range(parser.num_errors):
            print(f"    {parser.get_error(i)}")
        return 1
    print(f"  解析成功, 网络层数: {network.num_layers}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 workspace_mb * (1 << 20))
    if fp16:
        # 注1: 旧版 builder.platform_has_fast_fp16 在 TensorRT 10+ 已移除.
        # 注2: pip 上的 tensorrt 包(11.2)是精简绑定, BuilderFlag 缺少 FP16/FP8.
        #      官方完整版(从 NVIDIA 官网下载的 TensorRT)才有这些精度 flag.
        #      这里做能力检测, 缺什么就退回什么, 保证流程能跑通.
        if hasattr(trt.BuilderFlag, "FP16"):
            config.set_flag(trt.BuilderFlag.FP16)
            print("  启用 FP16")
        elif hasattr(trt.BuilderFlag, "FP8"):
            config.set_flag(trt.BuilderFlag.FP8)
            print("  启用 FP8")
        else:
            print("  [警告] 当前 TensorRT 包缺少 BuilderFlag.FP16/FP8")
            print("         (pip 精简版所致), 改用 FP32 构建.")
            print("         FP32 仍能享受层融合 / kernel 自动选择带来的加速.")

    # 动态 shape: 第 0 维(样本数)可变
    in_name = network.get_input(0).name
    in_dim = network.get_input(0).shape[-1]
    print(f"  输入: {in_name}, 特征维度={in_dim}")

    profile = builder.create_optimization_profile()
    profile.set_shape(in_name,
                      min=(1, in_dim),
                      opt=(opt_batch, in_dim),
                      max=(max_batch, in_dim))
    config.add_optimization_profile(profile)
    print(f"  动态 shape: min=(1,{in_dim}) opt=({opt_batch},{in_dim}) "
          f"max=({max_batch},{in_dim})")

    print()
    print("构建 engine (可能需要几十秒)...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("  [错误] engine 构建失败")
        return 1
    dt = time.time() - t0

    engine_path.write_bytes(serialized)
    print(f"  构建成功, 用时 {dt:.1f}s")
    print()
    print(f"Engine 已保存: {engine_path}")
    print(f"  大小: {engine_path.stat().st_size/1024:.1f} KB")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", nargs="?", default="model.onnx")
    ap.add_argument("--out", default="")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--fp32", action="store_true", help="强制 FP32")
    ap.add_argument("--workspace", type=int, default=256, help="MB")
    ap.add_argument("--opt-batch", type=int, default=8192)
    ap.add_argument("--max-batch", type=int, default=262144)
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.is_absolute():
        onnx_path = ROOT / onnx_path
    if not onnx_path.exists():
        print(f"[错误] 找不到 {onnx_path}")
        return 1

    if args.out:
        engine_path = Path(args.out)
    else:
        suffix = "fp32" if args.fp32 else "fp16"
        engine_path = ROOT / f"model_{suffix}.engine"

    print(f"TensorRT 版本: {trt.__version__}")
    print("=" * 66)
    return build(onnx_path, engine_path, fp16=not args.fp32,
                 workspace_mb=args.workspace,
                 opt_batch=args.opt_batch, max_batch=args.max_batch)


if __name__ == "__main__":
    sys.exit(main())
