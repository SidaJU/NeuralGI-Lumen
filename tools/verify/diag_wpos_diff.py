"""诊断: 相邻帧 wpos 为什么差 1 亿? 是历史帧的哨兵值污染吗?"""
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
    cur = seq.load_frame(1)
    prev = seq.load_frame(0)

    cw = cur["wpos"][..., :3].astype(np.float64)
    pw = prev["wpos"][..., :3].astype(np.float64)

    d = np.abs(cw - pw).max(axis=-1)      # (ph, pw) 每个 probe 的三维最大差

    ph, pw_ = d.shape
    print(f"probe 网格: {ph} x {pw_} = {ph*pw_} 个")
    print()
    print("=" * 68)
    print("1. wpos 差异的分布")
    print("=" * 68)
    for thr in (1e-3, 1.0, 1e3, 1e6):
        n = int((d > thr).sum())
        print(f"  差异 > {thr:<10.0e} 的 probe: {n:5d} 个  ({n/d.size*100:5.1f}%)")

    print()
    print("=" * 68)
    print("2. 差异大的 probe 的 valid 状态")
    print("=" * 68)
    big = d > 1e6
    print(f"  差异 > 1e6 的 probe 数: {int(big.sum())}")

    if big.sum():
        print()
        print("  这些 probe 在两帧的 valid 状态:")
        cur_valid_ratio = float(cur["valid"][big].mean())
        prev_valid_ratio = float(prev["valid"][big].mean())
        print(f"    cur  valid=True 的比例: {cur_valid_ratio*100:.1f}%")
        print(f"    prev valid=True 的比例: {prev_valid_ratio*100:.1f}%")
        print()
        # 关键: 差多少? 是否落在哨兵值 -1e8 附近?
        diff_vals = d[big]
        sentinel_hits = float((np.abs(diff_vals - 1e8) < 1e7).mean() * 100)
        print(f"  差异量级: min={diff_vals.min():.3e}  max={diff_vals.max():.3e}")
        print(f"  落在 1e8±1e7 范围的比例: {sentinel_hits:.1f}%")

        print()
        print("  结论判断:")
        if cur_valid_ratio > 0.9 and prev_valid_ratio < 0.1:
            print("   -> 当前帧有效, 但历史帧无效（哨兵 -1e8）")
            print("   -> 即: 可见性变化, 这些位置在上一帧没有几何")
            print("   -> 修复后的投影会落在【历史帧无效位置】-> 历史特征不可用")
            print("   -> 这正是 [55]『历史是否有效』标记要处理的场景")
        elif cur_valid_ratio < 0.1:
            print("   -> 当前帧就无效（哨兵）-> 采样时会被 valid mask 过滤, 不影响")
        else:
            print("   -> 两帧都有效但 wpos 差异巨大 -> 需要进一步查（是否真有运动?）")

    print()
    print("=" * 68)
    print("3. 两帧都有效的 probe 上, 差异多大?")
    print("=" * 68)
    both = cur["valid"] & prev["valid"]
    print(f"  两帧都有效的 probe: {int(both.sum())} 个")
    if both.sum():
        dd = d[both]
        print(f"  差异: mean={dd.mean():.6f}  max={dd.max():.6f}")
        if dd.max() < 1e-2:
            print("  -> 两帧都有效时, wpos 几乎相同（真正的静止部分）"
                  )
        else:
            print(f"  [注意] 两帧都有效但差异达 {dd.max():.4f} -> 可能有真实运动或数值问题"
                  )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
