"""方案 B' 可视化: 输入 / 模型输出 / GT 三连对比图.

生成一张 PNG, 横向三格:
    左: 输入  (low 的 in —— 64 条射线, 未滤波, 噪点大)
    中: 模型输出 (NN 推理结果)
    右: GT   (high 的 gt —— 256 条射线, 已滤波, 干净)

为什么需要它
------------
平滑度是一个标量, 只能说明"整体多干净", 看不出:
    - 细节有没有被抹掉 (过度平滑)
    - 噪点是不是真的压下去了
    - 亮暗结构有没有保持
肉眼对比图能一次性暴露这些问题.

用法
----
python tools/eval/visualize_b.py runs/b_scheme/best.pt D:/ProbeDump_B_low \
       --gt-dir D:/ProbeDump_B_high --frame 1 --out compare.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")          # 无显示环境也能出图
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
for p in ("tools", "tools/dataset", "train"):
    if str(ROOT / p) not in sys.path:
        sys.path.insert(0, str(ROOT / p))

from probe_dataset import ProbeTexelDataset  # noqa: E402
from model import ProbeFilterMLP            # noqa: E402
# tonemap / smoothness / 中文字体统一来自 nspf_common
from nspf_common import setup_matplotlib, tonemap, smoothness  # noqa: E402

setup_matplotlib()


def main() -> int:
    ap = argparse.ArgumentParser(description="方案 B' 可视化对比")
    ap.add_argument("ckpt")
    ap.add_argument("low_dir")
    ap.add_argument("--gt-dir", required=True, help="高射线采集目录")
    ap.add_argument("--frame", type=int, default=1, help="用第几帧")
    ap.add_argument("--gt-use-in", action="store_true",
                    help="对比【未滤波】的 in —— 评估方案 B''(纯射线提升) 时用")
    ap.add_argument("--out", default="compare.png")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # ---------------- 模型 ----------------
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    hidden = a.get("hidden", 64)
    layers = a.get("layers", 4)
    use_history = not a.get("no_history", False)
    residual = not a.get("no_residual", False)
    in_dim = 56 if use_history else 52

    model = ProbeFilterMLP(in_dim=in_dim, hidden=hidden,
                           num_layers=layers, use_residual=residual)
    model.load_state_dict(ck["model"])
    model.to(args.device).eval()

    # ---------------- 数据 ----------------
    # max_frames=0: 配对表要覆盖所有帧, 否则取任意帧时可能 KeyError
    ds = ProbeTexelDataset(args.low_dir, samples_per_frame=1,
                           use_history=use_history, max_frames=0,
                           gt_dir=args.gt_dir,
                           gt_use_in=args.gt_use_in)
    seq = ds.seq
    g = seq.gather
    n = len(seq)
    if n < 2:
        print("帧数不足")
        return 1

    i = max(1, min(args.frame, n - 1))

    cur = seq.load_frame(i)
    prev = seq.load_frame(i - 1) if use_history else None

    feat, _ = ds._build_features(cur, prev)      # (H, W, C)
    h, w = feat.shape[:2]

    # ---------------- 推理 ----------------
    with torch.no_grad():
        x = torch.from_numpy(feat.reshape(-1, feat.shape[-1])).to(args.device)
        out = model(x).cpu().numpy().reshape(h, w, 3)

    in_lin = cur["in"]                                  # 64 条, 未滤波
    out_lin = np.expm1(out)                             # 反归一化
    # 走相机矩阵配对表取 GT, 避免两遍采样密度不同导致取错帧
    key = "in" if args.gt_use_in else "gt"
    if ds.gt_seq is not None:
        gi = ds.gt_index_map.get(i, i) if ds.gt_index_map else i
        gt_lin = ds.gt_seq.load_frame(gi)[key]        # 256 条
    else:
        gt_lin = cur[key]

    # 裁到有效区 (atlas 尾部未使用槽位是未初始化显存)
    vh, vw = cur["valid"].shape
    vh, vw = vh * g, vw * g
    in_lin = in_lin[:vh, :vw]
    out_lin = out_lin[:vh, :vw]
    gt_lin = gt_lin[:vh, :vw]

    s_in = smoothness(in_lin, g)
    s_out = smoothness(out_lin, g)
    s_gt = smoothness(gt_lin, g)

    # ---------------- 画图 ----------------
    titles = [
        f"输入 (64条射线, 未滤波)\n平滑度 {s_in:.3f}",
        f"模型输出\n平滑度 {s_out:.3f}",
        (f"GT (256条射线, 未滤波)\n平滑度 {s_gt:.3f}" if args.gt_use_in
         else f"GT (256条射线, 已滤波)\n平滑度 {s_gt:.3f}"),
    ]
    imgs = [tonemap(in_lin), tonemap(out_lin), tonemap(gt_lin)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, img, t in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(t, fontsize=12)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=110)
    plt.close(fig)

    # ---------------- 报告 ----------------
    print(f"帧: frame_{i:06d}")
    print(f"  输入      平滑度 {s_in:.3f}")
    print(f"  模型输出  平滑度 {s_out:.3f}")
    print(f"  GT        平滑度 {s_gt:.3f}")
    gap = s_in - s_gt
    if gap > 1e-6:
        print(f"  完成度 {(s_in - s_out) / gap * 100:.1f}%")
    print(f"\n已保存: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
