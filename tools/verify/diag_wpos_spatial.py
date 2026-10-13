"""诊断: wpos 差异的空间分布 —— 是随机噪声, 还是真实相机运动?

关键判断:
  相机静止时, 两帧 wpos 应该【完全相同或极接近】(差异是浮点/追踪噪声)
  → 空间分布应该是"杂乱、低量级、无方向性"

  相机运动时, 差异应该【有方向性】
  → 空间分布应该呈现"梯度"(朝运动方向逐渐增大), 且量级较大
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "dataset"))
from probe_dataset import ProbeSequence  # noqa: E402

DATA = "D:/ProbeDump_S2"


def main() -> int:
    seq = ProbeSequence(DATA)
    g = seq.gather

    # 用相隔较远的帧放大信号
    for idx in (1, 5, 10):
        if idx >= len(seq):
            continue
        cur = seq.load_frame(idx)
        prev = seq.load_frame(idx - 1)
        cw = cur["wpos"][..., :3].astype(np.float64)
        pw = prev["wpos"][..., :3].astype(np.float64)
        d = np.abs(cw - pw).max(axis=-1)

        # 只统计两帧都有效的
        both = cur["valid"] & prev["valid"]
        dd = d[both]
        if dd.size == 0:
            continue

        print("=" * 68)
        print(f"frame {idx-1} -> {idx}  (间隔 {1} 帧)")
        print("=" * 68)
        print(f"  两帧都有效: {int(both.sum())} 个")
        print(f"  差异: mean={dd.mean():.3f}  median={np.median(dd):.3f}  max={dd.max():.3f}")
        print()

        # 逐轴看差异 (x/y/z)
        for ax, name in enumerate(("X", "Y", "Z")):
            axd = np.abs(cw[..., ax] - pw[..., ax])[both]
            print(f"    {name} 轴: mean={axd.mean():.4f}  max={axd.max():.4f}")

        print()
        # 方向性检查: 差异的【符号】是否一致?
        # 若是真实平移, 各轴差值应该同号且成比例
        for ax, name in enumerate(("X", "Y", "Z")):
            delta = (cw[..., ax] - pw[..., ax])[both]
            pos_ratio = float((delta > 0).mean())
            print(f"    {name} 轴 正号比例: {pos_ratio*100:5.1f}%   (50%=随机, 偏离50%=有方向性)")

        print()

    # 总结
    print("=" * 68)
    print("判读指南")
    print("=" * 68)
    print("  各轴差异 mean 很小(<1) 且 正号比例≈50%")
    print("    -> 浮点噪声 / 追踪抖动, 相机实质静止")
    print()
    print("  各轴差异 mean 较大 且 正号比例明显偏离 50%")
    print("    -> 真实相机运动（所有点朝同一方向移动）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
