"""按相机矩阵配对(不依赖帧号对齐).

为什么需要这个
--------------
方案 B 要求两遍采集"第 N 帧对应同一相机位置". 但采集是按【渲染帧】
计数的, 而低射线(快)和高射线(慢)的渲染速度不同 —— 即使逐帧步进,
每个位置实际采到的帧数也会是 ~2.0 vs ~1.9 这种小数, 累积几百次后
帧号就彻底错位了 (实测 hold=1 时偏差 315).

与其和帧号死磕, 不如绕开它: 每一帧的 json 里都记着【真实相机矩阵】,
直接用矩阵做最近邻匹配, 就能找到"这一帧在另一遍里对应的那一帧".

本脚本做的事
------------
1. 读两遍所有帧的相机矩阵
2. 对 low 的每一帧, 在 high 里找矩阵最接近的帧
3. 报告匹配质量 + 索引关系 (判断是不是"乱配")

用法
----
python tools/verify/pair_by_camera.py D:/ProbeDump_B_low D:/ProbeDump_B_high
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
CAM_KEYS = ("view_matrix", "proj_matrix", "pre_view_translation")


def load_cam(d: Path, idx: int) -> dict | None:
    p = d / f"frame_{idx:06d}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def cam_vec(cam: dict) -> np.ndarray:
    """把相机信息拼成一个向量, 便于算距离."""
    parts = []
    for k in CAM_KEYS:
        v = cam.get(k)
        if v is not None:
            parts.append(np.asarray(v, dtype=np.float64))
    return np.concatenate(parts) if parts else np.zeros(1)


def main() -> int:
    ap = argparse.ArgumentParser(description="按相机矩阵配对")
    ap.add_argument("low_dir")
    ap.add_argument("high_dir")
    ap.add_argument("--max-check", type=int, default=200,
                    help="抽样检查多少帧 (默认 200)")
    args = ap.parse_args()

    low, high = Path(args.low_dir), Path(args.high_dir)

    def frames(d: Path) -> list[int]:
        return sorted(int(p.name[6:12]) for p in d.glob("frame_*_in.npy"))

    lf, hf = frames(low), frames(high)
    if not lf or not hf:
        print("没有找到 *_in.npy")
        return 1

    print(SEP)
    print("按相机矩阵配对 (绕开帧号对齐)")
    print(f"  low : {low}  ({len(lf)} 帧)")
    print(f"  high: {high} ({len(hf)} 帧)")
    print(SEP)

    # 先把 high 的所有相机向量读进来, 建一个矩阵便于批量算距离
    print("\n读取 high 的相机矩阵...")
    hv = np.stack([cam_vec(load_cam(high, i)) for i in hf])

    n_check = min(args.max_check, len(lf))
    step = max(1, len(lf) // n_check)
    check_idx = list(range(0, len(lf), step))[:n_check]

    print(f"抽样检查 {len(check_idx)} 帧 (步长 {step})\n")
    print(f"  {'low帧':>8}{'匹配到high帧':>14}{'索引比':>10}"
          f"{'矩阵距离':>14}")
    print("  " + "-" * 48)

    dists, ratios = [], []
    for li in check_idx:
        lv = cam_vec(load_cam(low, li))
        d = np.linalg.norm(hv - lv, axis=1)
        hj = int(np.argmin(d))
        dmin = float(d[hj])
        dists.append(dmin)
        if hf[hj] > 0:
            ratios.append(hf[hj] / max(lf[li], 1))

        if len(dists) <= 12:
            print(f"  {lf[li]:>8}{hf[hj]:>14}"
                  f"{hf[hj] / max(lf[li], 1):>10.3f}{dmin:>14.3e}")

    dists = np.asarray(dists)
    print()
    print(SEP)
    print(f"匹配距离: 中位 {np.median(dists):.3e}  "
          f"最大 {dists.max():.3e}  最小 {dists.min():.3e}")

    if ratios:
        r = np.asarray(ratios)
        print(f"索引比 high/low: 中位 {np.median(r):.3f}  "
              f"(范围 {r.min():.3f} ~ {r.max():.3f})")

    # 判读
    print()
    if dists.max() < 1e-3:
        print("[OK ] 所有帧都能找到几乎完全相同的相机位置")
        print("      -> 可以按矩阵配对训练, 帧号错位不再是问题")
    elif np.median(dists) < 1e-2:
        print("[尚可] 大部分帧能匹配, 但有少数匹配偏差较大")
        print("      -> 可以配对, 建议丢弃距离过大的帧")
    else:
        print("[不佳] 匹配距离普遍较大")
        print("      -> 两遍的相机轨迹可能真的不同, 矩阵配对也救不了")

    if ratios:
        r_spread = np.ptp(np.asarray(ratios))
        if r_spread < 0.05:
            print(f"\n索引比很稳定 (波动 {r_spread:.3f}) "
                  f"-> 两遍是【匀速错开】, 用固定比例即可对齐")
        else:
            print(f"\n索引比波动较大 ({r_spread:.3f}) "
                  f"-> 两遍速度不恒定, 必须用逐帧矩阵配对")
    print(SEP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
