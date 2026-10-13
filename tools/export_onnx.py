"""把训练好的模型导出成 ONNX, 供 TensorRT 使用.

用法:
    python tools/export_onnx.py runs/v2_nohist/best.pt
    python tools/export_onnx.py runs/v2_nohist/best.pt --out model.onnx

注意点:
  1. 导出前必须 model.eval(), 否则会带上 dropout/batchnorm 的训练行为
  2. 第 0 维(样本数)设为动态 —— 推理时可能一次跑 8192 个, 也可能整帧 20 万个
  3. 用训练好的权重, 而不是随机初始化
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from model import ProbeFilterMLP, count_parameters  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--fixed-batch", type=int, default=0,
                    help="导出【固定 batch】(不加动态轴). "
                         "UE 的 NNE RDG(HLSL) 运行时要求静态形状 —— 动态 batch 会导致 CanCreateModelData 失败。")
    ap.add_argument("--legacy", action="store_true",
                    help="用旧版 TorchScript exporter (dynamo=False)。\n"
                         "新版 exporter 最低 opset 18, 会把算子版本抬到 14 等,"
                         "而 NNE RDG 的部分算子只注册到 13 (如 Add), 导致 CanCreateModelData 失败。\n"
                         "给 NNE 用必须加这个参数以导出真正的低 opset。")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {})

    # 从训练时的参数还原模型结构
    no_hist = a.get("no_history", False)
    in_dim = 52 if no_hist else 56

    model = ProbeFilterMLP(
        in_dim=in_dim,
        hidden=a.get("hidden", 64),
        num_layers=a.get("layers", 4),
        use_residual=not a.get("no_residual", False),
    )
    model.load_state_dict(ck["model"])
    model.eval()                      # 关键: 切到推理模式

    out_path = Path(args.out) if args.out else ROOT / "model.onnx"

    print(f"模型: {args.ckpt}")
    print(f"  in_dim={in_dim} hidden={a.get('hidden',64)} "
          f"layers={a.get('layers',4)} 参数={count_parameters(model):,}")
    if "loss" in ck:
        print(f"  训练最佳 loss={ck['loss']:.6f}")

    # dummy input: (样本数, in_dim)
    #   固定 batch 时形状写死 —— NNE RDG/HLSL 需要静态形状才能预计算所有张量
    batch = args.fixed_batch if args.fixed_batch > 0 else 1024
    dummy = torch.randn(batch, in_dim)

    export_kwargs = dict(
        input_names=["feat"],
        output_names=["radiance"],
        opset_version=args.opset,
        do_constant_folding=True,                # 常量折叠, 减小模型
    )
    if args.fixed_batch > 0:
        print(f"  模式: 固定 batch = {batch} (静态形状, 供 NNE RDG 使用)")
    else:
        export_kwargs["dynamic_axes"] = {
            "feat": {0: "num_samples"},          # 第 0 维动态
            "radiance": {0: "num_samples"},
        }
        print("  模式: 动态 batch (TensorRT 可用, NNE RDG 可能拒绝)")

    if args.legacy:
        export_kwargs["dynamo"] = False

    try:
        torch.onnx.export(model, dummy, str(out_path), **export_kwargs)
    except TypeError as e:
        # 旧版 torch 不认识 dynamo 参数 -> 去掉重试
        if "dynamo" not in str(e):
            raise
        export_kwargs.pop("dynamo", None)
        torch.onnx.export(model, dummy, str(out_path), **export_kwargs)

    print()
    print(f"导出成功: {out_path}")
    print(f"  大小: {out_path.stat().st_size/1024:.1f} KB")
    print(f"  opset: {args.opset}")
    print(f"  输入: feat (num_samples, {in_dim})")
    print(f"  输出: radiance (num_samples, 3)")
    print()
    print("下一步: 用 trtexec 构建 engine")
    print(f'  trtexec --onnx={out_path.name} --fp16 '
          f'--saveEngine=model_fp16.engine')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
