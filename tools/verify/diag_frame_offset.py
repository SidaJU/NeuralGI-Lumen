"""方案 B: 帧偏移诊断.

用途
----
当 verify_b_pairing.py 的 [2] 报"相机超差"但 proj_matrix 偏差为 0 时,
通常不是相机路径不同, 而是两遍采集【整体错位了 N 帧】(例如
WaitForMotion 等待帧数不同、或手速快慢导致起点偏了几帧).

本脚本暴力搜索最佳整体偏移 k, 使得:
    low 的第 i 帧  ≈  high 的第 (i+k) 帧

若找到某个 k 让相机矩阵几乎完全吻合, 就说明只是帧号没对齐,
裁剪掉多出来的帧即可配对, 不必重采.

用法
----
python tools/verify/diag_frame_offset.py D:/ProbeDump_B_low D:/ProbeDump_B_high
python tools/verify/diag_frame_offset.py <low> <high> --search 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

SEP = "=" * 72


def load_cam(d: Path, idx: int) -> dict | None:
    p = d / f"frame_{idx:06d}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def cam_diff(a: dict, b: dict) -> float:
    """两个相机 json 的最大元素偏差 (只比 view/proj/pvt)."""
    worst = 0.0
    for key in ("view_matrix", "proj_matrix", "pre_view_translation"):
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            continue
        worst = max(worst, float(np.abs(
            np.asarray(va, dtype=np.float64)
            - np.asarray(vb, dtype=np.float64)).max()))
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(description="方案 B 帧偏移诊断")
    ap.add_argument("low_dir")
    ap.add_argument("high_dir")
    ap.add_argument("--search", type=int, default=8,
                    help="搜索的偏移范围 ±N 帧 (默认 8)")
    args = ap.parse_args()

    low, high = Path(args.low_dir), Path(args.high_dir)

    def frames(d: Path) -> list[int]:
        return sorted(int(p.name[6:12])
                      for p in d.glob("frame_*_in.npy"))

    lf, hf = frames(low), frames(high)
    if not lf or not hf:
        print("没有找到 *_in.npy")
        return 1

    print(SEP)
    print("方案 B 帧偏移诊断")
    print(f"  low : {low}  ({len(lf)} 帧, {lf[0]}..{lf[-1]})")
    print(f"  high: {high} ({len(hf)} 帧, {hf[0]}..{hf[-1]})")
    print(SEP)

    # 统一按"第 n 个文件"来索引, 而非帧号
    print(f"\n搜索整体偏移 k (low 第 i 个 ≈ high 第 i+k 个), 范围 ±{args.search}")
    print(f"\n  {'k':>5}{'可比帧数':>10}{'平均偏差':>16}{'最大偏差':>16}   判定")
    print("  " + "-" * 60)

    best_k, best_mean = None, float("inf")
    rows = []

    for k in range(-args.search, args.search + 1):
        diffs = []
        for i in range(len(lf)):
            j = i + k
            if j < 0 or j >= len(hf):
                continue
            cl = load_cam(low, lf[i])
            ch = load_cam(high, hf[j])
            if cl is None or ch is None:
                continue
            diffs.append(cam_diff(cl, ch))

        if not diffs:
            continue

        mean_d, max_d = float(np.mean(diffs)), float(np.max(diffs))
        rows.append((k, len(diffs), mean_d, max_d))
        if mean_d < best_mean:
            best_mean, best_k = mean_d, k

        verdict = "OK" if max_d < 1e-3 else ("接近" if max_d < 1.0 else "")
        print(f"  {k:>5}{len(diffs):>10}{mean_d:>16.3e}{max_d:>16.3e}   {verdict}")

    print()
    print(SEP)
    if best_k is not None:
        print(f"最佳偏移 k = {best_k}  (平均偏差 {best_mean:.3e})")
        print()
        if best_mean < 1e-6:
            print("结论: 两遍几乎完美对齐, 可以直接配对训练.")
        elif best_k != 0 and best_mean < 1e-3:
            print(f"结论: 只是整体错位 {best_k} 帧 —— 不是路径问题!")
            if best_k > 0:
                print(f"      修法: 删掉 high 的前 {best_k} 帧 (或 low 的后 {abs(best_k)} 帧)")
            else:
                print(f"      修法: 删掉 low 的前 {abs(best_k)} 帧")
        elif best_k == 0 and best_mean < 1e-3:
            print("结论: k=0 已对齐, [2] 报警可能是个别帧抖动, 可以放宽容差.")
        else:
            print(f"结论: 即使最佳偏移, 平均偏差仍有 {best_mean:.3e}")
            print("      -> 两遍相机路径/时序确实不一致, 建议重采.")
            print("      -> 检查 t.FixedFrameRate 60 与 r.AntiAliasingMethod 0")
    print(SEP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
