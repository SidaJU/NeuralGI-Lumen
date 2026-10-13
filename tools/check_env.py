"""检查推理优化相关依赖是否可用 (TensorRT / ONNX / 等)."""
from __future__ import annotations

import importlib
import sys

import torch


def check(name: str, mod: str, attr: str = "__version__") -> bool:
    try:
        m = importlib.import_module(mod)
        v = getattr(m, attr, "?") if attr else "ok"
        print(f"  [OK]   {name:<16} {v}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [MISS] {name:<16} ({type(e).__name__})")
        return False


def main() -> int:
    print(f"python  {sys.version.split()[0]}")
    print(f"torch   {torch.__version__}")
    print(f"cuda    {torch.version.cuda}  (torch 编译时)")
    if torch.cuda.is_available():
        print(f"GPU     {torch.cuda.get_device_name(0)}")
        cc = torch.cuda.get_device_capability(0)
        print(f"算力    sm_{cc[0]}{cc[1]}")
        print(f"显存    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"当前显存占用 {torch.cuda.memory_allocated() / 1e6:.1f} MB")

    print()
    print("核心依赖:")
    check("torch", "torch")
    check("numpy", "numpy")

    print()
    print("推理优化相关:")
    has_trt = check("tensorrt", "tensorrt")
    has_onnx = check("onnx", "onnx")
    check("onnxruntime", "onnxruntime")
    check("torch-tensorrt", "torch_tensorrt")
    check("polygraphy", "polygraphy")

    print()
    if has_trt and has_onnx:
        print("-> TensorRT 链路可用, 可以导出 ONNX 并构建 engine")
    else:
        missing = []
        if not has_onnx:
            missing.append("onnx")
        if not has_trt:
            missing.append("tensorrt")
        print(f"-> 缺少: {', '.join(missing)}")
        print()
        print("安装建议 (CUDA 12.x 环境):")
        print("   pip install onnx onnxscript")
        print("   pip install tensorrt            # 或用 --extra-index-url 指定 NVIDIA 源")
        print()
        print("注意: TensorRT 对 CUDA 版本敏感. 当前 torch 用 CUDA "
              f"{torch.version.cuda},")
        print("      需安装匹配的 TensorRT 版本 (如 CUDA 12.8 -> TensorRT 10.x).")
        print()
        print("备选 (无需 TensorRT 也能提速):")
        print("   - 已验证: torch.compile(mode='reduce-overhead')  # 内部用 CUDA Graph")
        print("   - 已验证: fp16 (本次实测 2.08x)")
        print("   - 最终部署时 UE 内走 TensorRT / DirectML")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
