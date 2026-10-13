"""数据加载器自检: 验证邻域布局/时域对齐/归一化是否正确.

这些错误都不会抛异常, 只会让 loss 莫名降不下去, 所以必须单独校验.

跑法:
    python tools/verify/test_dataset.py D:/ProbeDump_S2
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "dataset"))

from probe_dataset import (  # noqa: E402
    ProbeSequence,
    ProbeTexelDataset,
    log_normalize,
    octahedral_decode,
    depth_from_uint,
)

ok_all = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok_all
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}  {detail}")
        ok_all = False


def main() -> int:
    dump_dir = sys.argv[1] if len(sys.argv) > 1 else "D:/ProbeDump_S2"
    print(f"数据加载器自检: {dump_dir}")
    print("=" * 66)

    seq = ProbeSequence(dump_dir)
    g = seq.gather

    # ---------- 1. 几何解码 ----------
    print("\n[1] 几何解码")
    f = seq.load_frame(min(1, len(seq) - 1))

    n = f["normal"]
    norm_len = np.sqrt((n ** 2).sum(axis=-1))
    check("法线是单位向量", bool(np.allclose(norm_len, 1.0, atol=1e-3)),
          f"长度范围 [{norm_len.min():.4f}, {norm_len.max():.4f}]")

    d = f["depth"]
    valid_d = d[d > 0]
    check("深度正值部分量级合理 (cm)", bool(valid_d.size > 0 and valid_d.max() < 1e7),
          f"max={valid_d.max() if valid_d.size else 0:.1f}")
    check("存在 depth<=0 的哨兵值(无几何 probe)",
          bool((d <= 0).sum() > 0), f"{(d <= 0).sum()} 个")

    # ---------- 2. 邻域布局 ----------
    print("\n[2] 3x3 空间邻域布局")
    ds = ProbeTexelDataset(dump_dir, samples_per_frame=64,
                           use_history=False, max_frames=1)
    idx = ds.frame_indices[0]
    cur = seq.load_frame(idx)
    feat, target = ds._build_features(cur, None)
    print(f"    特征维度: {feat.shape[-1]}  (期望 3*9 + 1*9 + 1*9 + 7 = 52)")

    check("特征维度=52 (无历史)", feat.shape[-1] == 52, f"实际 {feat.shape[-1]}")

    # 中心 probe 是 3x3 窗口的索引 4 -> 通道 [12:15]
    center = feat[..., 12:15]
    expect = log_normalize(cur["in"])
    check("通道[12:15] == 中心 probe 的 log1p(radiance)",
          bool(np.allclose(center, expect, atol=1e-5)))

    # 抽查四个方向的邻居.
    # 邻域顺序 (行优先 3x3 窗口, padded 后 di/dj 各偏移 1):
    #   0=左上 1=上 2=右上
    #   3=左  4=中 5=右
    #   6=左下 7=下 8=右下
    pi, pj, tu, tv = 3, 3, 2, 2        # probe (3,3) 的 texel (2,2)
    y = pi * g + tu
    x = pj * g + tv

    # (通道块, 邻居相对偏移, 名字)
    cases = [
        (1, (-1, 0), "上"),
        (3, (0, -1), "左"),
        (4, (0, 0), "中心"),
        (5, (0, +1), "右"),
        (7, (+1, 0), "下"),
    ]
    for k, (dpi, dpj), name in cases:
        got = feat[y, x, k * 3:(k + 1) * 3]
        exp = expect[(pi + dpi) * g + tu, (pj + dpj) * g + tv]
        check(f"通道[{k*3}:{k*3+3}] == {name}邻居 probe 的同方向 texel",
              bool(np.allclose(got, exp, atol=1e-5)),
              f"{got} vs {exp}")

    # ---------- 3. 时域对齐 ----------
    print("\n[3] 时域对齐")
    ds_h = ProbeTexelDataset(dump_dir, samples_per_frame=64,
                             use_history=True, history_mode="aligned",
                             max_frames=1)
    item = ds_h[0]
    print(f"    特征维度: {item['feat'].shape[-1]} "
          f"(期望 52 + 3 + 1 = 56)")
    check("特征维度=56 (含历史)", item["feat"].shape[-1] == 56,
          f"实际 {item['feat'].shape[-1]}")
    check("采样数正确", item["feat"].shape[0] == 64,
          f"实际 {item['feat'].shape[0]}")
    check("无 NaN", bool(np.isfinite(item["feat"].numpy()).all()))
    check("无 Inf", bool(np.isfinite(item["target"].numpy()).all()))

    # ---------- 4. 归一化范围 ----------
    print("\n[4] 归一化")
    t = item["target"].numpy()
    print(f"    target 范围: [{t.min():.4f}, {t.max():.4f}]")
    check("target 量级 O(1)", bool(t.max() < 6.0), f"max={t.max():.4f}")
    check("target 非负", bool(t.min() >= 0.0), f"min={t.min():.4f}")

    fi = item["feat"].numpy()
    print(f"    feat 范围:   [{fi.min():.4f}, {fi.max():.4f}]")
    # 几何里的 wpos 可能较大, 但不应到 1e3 量级
    check("feat 无极端值", bool(np.abs(fi).max() < 1e3),
          f"abs max={np.abs(fi).max():.4g}")

    # ---------- 5. 输入/答案确实不同 ----------
    print("\n[5] 输入与答案")
    r_in_n = log_normalize(cur["in"])
    r_gt_n = log_normalize(cur["gt"])
    diff = np.abs(r_in_n - r_gt_n).mean()
    print(f"    |log1p(in) - log1p(gt)| 均值: {diff:.4f}")
    check("in 与 gt 有明显差异", bool(diff > 1e-3), f"diff={diff:.4g}")

    print()
    print("=" * 66)
    print("全部通过" if ok_all else "存在失败项, 请检查上面标记 [FAIL] 的行")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
