"""验证时域投影修复 (2026-09-04).

修复内容:
  _sample_history 里原先用 prev["wpos"]（历史帧自己的 wpos）配历史帧相机投影
  -> 投影回它自己 -> pi/pj 恒等 -> 时域对齐从未生效（恒等映射）
  改为 cur["wpos"]（当前帧 wpos）配历史帧相机 -> 真正的时域重投影

验证要点:
  1. 静止数据下, cur.wpos 与 prev.wpos 应该几乎相同（相机没动）
     -> 修复前后结果一致（回归通过, 不会改坏现有流程）
  2. 投影后的索引 pi/pj 应该落在 probe 网格范围内
  3. 修复后能正常构建 56 维特征, 无 NaN

同时输出"相机若运动时应该出现偏移"的判断依据:
  静止时 |pi-i|, |pj-j| 应该 ≈ 0（因为恒等）
  未来采集运动数据后, 这个偏移应该 > 0 —— 那才证明对齐真的生效了
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "dataset"))

from probe_dataset import ProbeSequence, ProbeTexelDataset  # noqa: E402

DATA = "D:/ProbeDump_S2"


def main() -> int:
    seq = ProbeSequence(DATA)
    g = seq.gather

    print("=" * 70)
    print("验证1: 静止数据下 cur.wpos 与 prev.wpos 是否相同")
    print("=" * 70)

    cur = seq.load_frame(1)
    prev = seq.load_frame(0)

    cw = cur["wpos"][..., :3]
    pw = prev["wpos"][..., :3]

    diff = np.abs(cw - pw).max()
    print(f"  cur.wpos 与 prev.wpos 最大差: {diff:.6f}")
    if diff < 1e-3:
        print("  -> 相机静止, 两者几乎相同")
        print("  -> 修复前后 pi/pj 应一致（回归通过）")
    else:
        print("  [注意] 两者不同, 说明相机有运动, 修复会产生真实偏移")

    # ---- 相机矩阵是否相同 ----
    print()
    print("=" * 70)
    print("验证2: 相机矩阵是否相同")
    print("=" * 70)
    vm0 = np.array(cur["cam"].get("view_matrix", []))
    vm1 = np.array(prev["cam"].get("view_matrix", []))
    if vm0.size and vm1.size:
        vdiff = np.abs(vm0 - vm1).max()
        print(f"  view_matrix 最大差: {vdiff:.8f}")
        if vdiff < 1e-6:
            print("  -> 相机矩阵相同（静止）")

    # ---- 构建 56 维特征, 确认无 NaN ----
    print()
    print("=" * 70)
    print("验证3: 修复后能否正常构建 56 维特征")
    print("=" * 70)

    ds = ProbeTexelDataset(DATA, samples_per_frame=1, use_history=True,
                           history_mode="aligned", max_frames=1)
    feat, target = ds._build_features(cur, prev)
    print(f"  特征形状: {feat.shape}  (期望 (280,720,56))")
    print(f"  target:   {target.shape}")
    n_nan = int(np.isnan(feat).sum())
    print(f"  NaN 数量: {n_nan}")
    if n_nan:
        print("  [失败] 特征含 NaN")
        return 1

    # 历史部分 [52:55]
    hist = feat[..., 52:55]
    print(f"  历史特征 [52:55] 范围: [{hist.min():.4f}, {hist.max():.4f}]")

    # 静止时历史特征应该 ≈ 历史帧输入中心（因为投影恒等, 取的就是自己）
    hist_in = prev["in"]
    print(f"  历史帧 in 范围:      [{hist_in.min():.4f}, {hist_in.max():.4f}]")

    # ---- 投影索引偏移（关键指标）----
    print()
    print("=" * 70)
    print("验证4: 投影索引偏移量（判断时域是否真的对齐）")
    print("=" * 70)

    # 复算一次对齐索引
    from probe_dataset import build_view_proj  # noqa: E402
    cam_prev = prev.get("cam", {})
    vp = build_view_proj(cam_prev)
    if vp is None:
        print("  [跳过] 无有效相机矩阵")
        return 0

    H, W = prev["in"].shape[:2]
    ph, pw = H // g, W // g
    cur_trans = np.asarray(cur.get("cam", {}).get("pre_view_translation", [0, 0, 0]),
                           dtype=np.float64)
    xyz = cur["wpos"][..., :3].astype(np.float64) - cur_trans
    ones = np.ones(xyz.shape[:-1] + (1,), dtype=np.float64)
    clip = np.concatenate([xyz, ones], axis=-1) @ vp
    w = clip[..., 3:4]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    ndc = clip[..., :3] / w
    u = ndc[..., 0] * 0.5 + 0.5
    v = 0.5 - ndc[..., 1] * 0.5
    pj = np.clip(np.round(u * (pw - 1)), 0, pw - 1).astype(np.int32)
    pi = np.clip(np.round(v * (ph - 1)), 0, ph - 1).astype(np.int32)

    ii = np.arange(ph)[:, None]
    jj = np.arange(pw)[None, :]
    off_i = np.abs(pi - ii).mean()
    off_j = np.abs(pj - jj).mean()
    print(f"  平均 |pi - i| = {off_i:.4f}")
    print(f"  平均 |pj - j| = {off_j:.4f}")
    print()
    if off_i < 0.5 and off_j < 0.5:
        print("  -> 偏移≈0, 说明投影结果 = 自身索引（恒等）")
        print("     这是【静止数据】下的正确表现（相机没动, 空间点还在老位置）")
        print()
        print("  ★ 未来采集【相机运动】数据后重跑本脚本:")
        print("     如果偏移仍然 ≈0 -> 说明修复没生效或数据仍静止")
        print("     如果偏移  >0   -> 说明真正的时域对齐生效了")
    else:
        print("  -> 偏移 > 0, 时域对齐正在生效（相机有运动）")

    print()
    print("=" * 70)
    print("结论: 修复未破坏现有流程, 56 维特征可正常构建")
    print("     真正的时域价值需等【运动数据】才能验证")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
