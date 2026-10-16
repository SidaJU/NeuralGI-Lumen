"""A/B 对照评估：比较两个 checkpoint 的输出误差（重点看【暗部】）。

用法
----
python tools/eval/ab_compare.py runs/b_nohist runs/ab_mse \
    --data D:/ProbeDump_B_low --gt-dir D:/ProbeDump_B_high [--frames N]

做什么
------
1. 用同一批样本（同一帧、同一特征）分别跑两个模型
2. 把 log1p 域输出用 expm1 还原到【线性域】
3. 按 GT 亮度把样本分成【暗部 / 中间 / 亮部】，分别统计 relL2
   （长尾应对的核心目标：暗部不应该被牺牲）

输出
----
每个模型的整体 MSE / relL2，以及分亮度区间的 relL2，并给出对比结论。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "dataset"))
sys.path.insert(0, os.path.join(ROOT, "train"))

from probe_dataset import ProbeTexelDataset  # noqa: E402
from model import ProbeFilterMLP  # noqa: E402


def load_model(ckpt_path: str, in_dim: int = 52) -> ProbeFilterMLP:
    """加载 checkpoint（兼容多种保存格式）。"""
    model = ProbeFilterMLP(in_dim=in_dim)
    state = torch.load(ckpt_path, map_location="cpu")

    if isinstance(state, dict):
        # 常见格式: {'model': state_dict} / {'state_dict': ...} / {'model_state': ...}
        sd = None
        for key in ("model", "state_dict", "model_state", "net"):
            if key in state and isinstance(state[key], dict):
                sd = state[key]
                break
        if sd is None:
            # 可能直接就是 state_dict
            sd = state
        model.load_state_dict(sd)
    else:
        model.load_state_dict(state)

    model.eval()
    return model


@torch.no_grad()
def run_model(model: ProbeFilterMLP, feat: np.ndarray, device: str) -> np.ndarray:
    """(H,W,C_in) -> (H,W,3) 线性域 radiance."""
    h, w, c = feat.shape
    x = torch.from_numpy(feat.reshape(-1, c)).float().to(device)
    out = model(x).cpu().numpy().reshape(h, w, 3)
    return np.expm1(out)  # log1p -> 线性


def rel_l2(pred: np.ndarray, gt: np.ndarray, eps: float = 0.1) -> float:
    return float((((pred - gt) ** 2) / (gt ** 2 + eps)).mean())


def mse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(((pred - gt) ** 2).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_a")
    ap.add_argument("ckpt_b")
    ap.add_argument("--data", required=True)
    ap.add_argument("--gt-dir", default="")
    ap.add_argument("--frames", type=int, default=5, help="评估帧数")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # samples_per_frame 设很大 -> __getitem__ 返回【全部有效样本】(扁平)，
    # 便于按亮度分区统计（不需要空间结构，target 已是方案 B 配对好的 GT）
    ds = ProbeTexelDataset(
        args.data,
        samples_per_frame=10**9,
        use_history=False,
        max_frames=0,
        gt_dir=args.gt_dir if args.gt_dir else None,
    )

    m_a = load_model(args.ckpt_a).to(args.device)
    m_b = load_model(args.ckpt_b).to(args.device)

    # 取若干帧（间隔开，避免都是同一位置）
    n = len(ds)
    step = max(1, n // max(1, args.frames))
    frame_ids = list(range(0, n, step))[: args.frames]

    stats = {
        "A": {"mse": [], "rel": [], "dark": [], "mid": [], "bright": []},
        "B": {"mse": [], "rel": [], "dark": [], "mid": [], "bright": []},
    }

    for i in frame_ids:
        sample = ds[i]
        feat = sample["feat"].numpy()          # (n, C) 全部有效样本（已展平）
        gt_log = sample["target"].numpy()      # (n, 3) log 域 GT（方案 B 已配对）
        gt_linear = np.expm1(gt_log)           # → 线性域

        # 每样本亮度（RGB 均值），用于分区
        gt_luma = gt_linear.mean(axis=1)

        # 分位：暗部 = 最暗 50%；亮部 = 最亮 10%
        p50 = np.percentile(gt_luma, 50)
        p90 = np.percentile(gt_luma, 90)
        dark_m = gt_luma < p50
        bright_m = gt_luma > p90
        mid_m = (~dark_m) & (~bright_m)

        for tag, m in (("A", m_a), ("B", m_b)):
            x = torch.from_numpy(feat).float().to(args.device)
            with torch.no_grad():
                out_log = m(x).cpu().numpy()
            pred = np.expm1(out_log)           # → 线性域

            stats[tag]["mse"].append(mse(pred, gt_linear))
            stats[tag]["rel"].append(rel_l2(pred, gt_linear))
            for name, mask in (("dark", dark_m), ("mid", mid_m), ("bright", bright_m)):
                if mask.any():
                    stats[tag][name].append(rel_l2(pred[mask], gt_linear[mask]))

    def agg(tag: str, key: str) -> float:
        v = stats[tag][key]
        return float(np.mean(v)) if v else float("nan")

    print()
    print("=" * 78)
    print("A/B 对照：损失函数对【暗部】的影响")
    print("=" * 78)
    print(f"模型 A (relative L2): {args.ckpt_a}")
    print(f"模型 B (MSE)       : {args.ckpt_b}")
    print(f"评估帧数: {len(frame_ids)}     设备: {args.device}")
    print()
    print(f"{'指标':<16}{'A: relativeL2':>16}{'B: MSE':>16}{'差异':>14}")
    print("-" * 78)

    rows = [
        ("整体 MSE", "mse"),
        ("整体 relL2", "rel"),
        ("暗部 relL2(关键)", "dark"),
        ("中间调 relL2", "mid"),
        ("亮部 relL2", "bright"),
    ]
    results = {}
    for name, key in rows:
        a, b = agg("A", key), agg("B", key)
        results[key] = (a, b)
        # 相对差异：(B - A) / A，正数表示 B 更差
        diff = (b - a) / a * 100 if a > 0 else float("nan")
        print(f"{name:<16}{a:>16.5f}{b:>16.5f}{diff:>13.1f}%")

    print()
    print("=" * 78)
    da, db = results["dark"]
    ba, bb = results["bright"]
    print("结论（数值越低越好）：")
    if da < db:
        print(f"  [OK] relative L2 的暗部误差更低：{da:.5f} vs MSE 的 {db:.5f}"
              f"（好 {(db - da) / db * 100:.1f}%）")
    else:
        print(f"  [!] MSE 的暗部误差反而更低：{db:.5f} vs {da:.5f}")
    print(f"  亮部：relativeL2 {ba:.5f} vs MSE {bb:.5f}")
    print()
    print("解读：长尾应对的目标就是【不让亮部主导、保住暗部】。")
    print("      若 A 的暗部明显优于 B，说明 log1p + relative L2 的机制确实有效。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
