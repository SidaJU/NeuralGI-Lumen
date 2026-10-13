"""方案 B 双次采集配对校验.

用途
----
方案 B 要跑两遍采集:
    第 1 遍: 低射线 (TracingOctahedronResolution=4,  16 条) -> 输入目录
    第 2 遍: 高射线 (TracingOctahedronResolution=16, 256 条) -> GT 目录

两遍的相机路径必须【逐帧严格一致】. 一旦错位:

    输入: 这个像素在墙上
    答案: 那个像素在地板
            -> 网络学到的是纯噪声

而训练【不会报错】—— loss 照降、权重照存, 但学到的全是错的.
这是方案 B 最大的静默失败风险, 必须在训练前拦住.

检查项
------
[1] 帧数与帧名对应
[2] 相机矩阵逐帧比对 (view / proj / pre_view_translation)   ★核心
[3] 几何数据交叉验证 (depth)                                <- 间接佐证
[4] 射线数差异验证 (两遍的 in 必须不同)                      <- 确认 CVar 生效
[5] GT 质量验证 (高射线 gt 应比低射线 gt 更平滑)             <- 确认 B 有意义

用法
----
python tools/verify/verify_b_pairing.py D:/ProbeDump_B_low D:/ProbeDump_B_high
python tools/verify/verify_b_pairing.py <low> <high> --tol 1e-3 --max-frames 10

退出码: 0=全部通过, 1=有失败项
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Windows 控制台默认 GBK, 中文/符号会 UnicodeEncodeError, 强制 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "dataset"))

from probe_dataset import ProbeSequence, depth_from_uint  # noqa: E402

SEP = "=" * 72
CAM_KEYS = ("view_matrix", "proj_matrix", "pre_view_translation")


def smoothness(atlas: np.ndarray, g: int) -> float:
    """方向维度平滑度 (与 eval/eval_smoothness.py 同一定义).

    值越小越平滑 (噪点越少).
    """
    h, w = atlas.shape[:2]
    ph, pw = h // g, w // g
    p = atlas.reshape(ph, g, pw, g, -1).transpose(0, 2, 1, 3, 4)
    d = np.abs(np.diff(p, axis=2)).mean()
    m = np.abs(p).mean() + 1e-6
    return float(d / m)


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """相对 L2 误差."""
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-6))


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="方案 B 双次采集配对校验")
    ap.add_argument("low_dir", help="低射线采集目录 (输入侧)")
    ap.add_argument("high_dir", help="高射线采集目录 (GT 侧)")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="相机矩阵元素允许的最大偏差 (默认 1e-3)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="只校验前 N 帧 (0=全部)")
    args = ap.parse_args()

    low_dir = Path(args.low_dir)
    high_dir = Path(args.high_dir)

    print(SEP)
    print("NSPF 方案 B 配对校验")
    print(f"  输入 (低射线): {low_dir}")
    print(f"  GT   (高射线): {high_dir}")
    print(f"  相机容差     : {args.tol}")
    print(SEP)

    ok = True

    # ---------------------------------------------------- [1] 帧数/帧名 --
    print("\n[1] 帧数与帧名对应")
    try:
        low_seq = ProbeSequence(low_dir)
        high_seq = ProbeSequence(high_dir)
    except FileNotFoundError as e:
        print(f"  [FAIL] {e}")
        return 1

    n_low, n_high = len(low_seq), len(high_seq)
    print(f"  输入 {n_low} 帧 / GT {n_high} 帧")
    if n_low != n_high:
        print("  [FAIL] 帧数不一致! 两遍采集必须采到相同帧数")
        print("         检查 Sequencer 回放长度 / MaxFrames 设置")
        ok = False

    n = min(n_low, n_high)
    if args.max_frames > 0:
        n = min(n, args.max_frames)
    if n == 0:
        print("  [FAIL] 没有可校验的帧")
        return 1

    mism = [i for i in range(n)
            if low_seq.frames[i] != high_seq.frames[i]]
    if mism:
        print(f"  [FAIL] {len(mism)} 帧名字对不上, 例如第 {mism[0]} 帧: "
              f"{low_seq.frames[mism[0]]} vs {high_seq.frames[mism[0]]}")
        ok = False
    else:
        print(f"  [OK ] 前 {n} 帧帧名一一对应")

    g = low_seq.gather
    print(f"  gather 分辨率: {g}")

    # ------------------------------------------- [2] 相机矩阵逐帧比对 --
    print("\n[2] 相机矩阵逐帧比对  ★核心")
    print("    两次采集若相机路径一致, 矩阵应逐元素几乎相同")

    worst = {k: (0.0, -1) for k in CAM_KEYS}      # key -> (最大偏差, 帧号)
    per_frame_max = np.zeros(n, dtype=np.float64)
    missing_json = 0

    for i in range(n):
        base = low_seq.frames[i]
        jl, jh = low_dir / f"{base}.json", high_dir / f"{base}.json"
        if not (jl.exists() and jh.exists()):
            missing_json += 1
            continue
        cl = json.loads(jl.read_text(encoding="utf-8"))
        ch = json.loads(jh.read_text(encoding="utf-8"))

        fm = 0.0
        for k in CAM_KEYS:
            a, b = cl.get(k), ch.get(k)
            if a is None or b is None:
                continue
            d = float(np.abs(np.asarray(a, np.float64)
                             - np.asarray(b, np.float64)).max())
            if d > worst[k][0]:
                worst[k] = (d, i)
            fm = max(fm, d)
        per_frame_max[i] = fm

    if missing_json:
        print(f"  [FAIL] {missing_json} 帧缺相机 json, 无法比对")
        ok = False

    print(f"    {'字段':<26}{'最大偏差':>14}{'出现于帧':>10}")
    for k in CAM_KEYS:
        d, i = worst[k]
        flag = "OK " if d <= args.tol else "FAIL"
        print(f"    {k:<26}{d:>14.3e}{i:>10}   [{flag}]")
        if d > args.tol:
            ok = False

    bad = np.where(per_frame_max > args.tol)[0]
    if len(bad):
        print(f"\n  [FAIL] {len(bad)}/{n} 帧相机矩阵超差 (阈值 {args.tol})")
        show = [int(x) for x in bad[:5]]
        print(f"         超差帧: {show}{' ...' if len(bad) > 5 else ''}")
        print("         常见原因: Sequencer 未锁帧率 / TSR(TAA) jitter 未关")
        print("         修法: t.FixedFrameRate 60 + r.AntiAliasingMethod 0")
    else:
        print(f"\n  [OK ] 全部 {n} 帧相机矩阵一致 "
              f"(最大 {per_frame_max.max():.3e})")

    # ------------------------------------------- [3] 几何数据交叉验证 --
    print("\n[3] 几何数据交叉验证 (probe 级)")
    print("    相机对齐时, 绝大多数 probe 的深度应几乎相同")

    # 逐 probe 统计, 而不是逐帧算一个 L2.
    #
    # 为什么: Lumen 有"自适应 probe 放置"(FScreenProbeAdaptivePlacementSpawnCS),
    # 那部分 probe 的位置/数量会受射线数影响, 天然与低射线遍不同.
    # 实测约 5% 的 probe 差异很大, 但中位数只有 2% —— 主体是对齐的.
    # 若用"整帧 L2"或"最大值"判断, 会被这 5% 带偏而误报.
    geo = []
    for i in range(n):
        base = low_seq.frames[i]
        try:
            dl = depth_from_uint(np.load(low_dir / f"{base}_depth.npy"))
            dh = depth_from_uint(np.load(high_dir / f"{base}_depth.npy"))
        except FileNotFoundError:
            continue
        m = (dl > 0) & (dh > 0)          # 只比有效 probe
        if m.sum() == 0:
            continue
        d = np.abs(dl[m] - dh[m])
        r = d / (np.abs(dh[m]) + 1e-6)
        geo.append(r)

    if not geo:
        print("  [SKIP] 没有可比的 depth 数据")
    else:
        allr = np.concatenate(geo)
        med = float(np.median(allr))
        p90 = float(np.percentile(allr, 90))
        bad_frac = float((allr > 0.5).mean())

        print(f"    可比 probe 总数: {allr.size}")
        print(f"    depth 相对差异: 中位={med:.3f}  P90={p90:.3f}")
        print(f"    差异 >50% 的 probe 占比: {bad_frac * 100:.1f}% "
              f"(Lumen 自适应放置导致, 属正常)")

        if med > 0.05:
            print(f"  [FAIL] 中位差异 {med:.3f} 过大 "
                  f"—— 两遍看到的不是同一场景位置")
            ok = False
        else:
            print(f"  [OK ] 主体几何一致 (中位差异 {med:.3f} <= 0.05)")

    # ------------------------------------------- [4] 射线数差异验证 ----
    print("\n[4] 射线数差异验证")
    print("    两遍的滤波前 radiance 必须不同, 否则 CVar 没生效")

    in_diff, sm_in_low, sm_in_high = [], [], []
    for i in range(n):
        base = low_seq.frames[i]
        try:
            il = np.load(low_dir / f"{base}_in.npy")
            ih = np.load(high_dir / f"{base}_in.npy")
        except FileNotFoundError:
            continue
        in_diff.append(rel_l2(il, ih))
        sm_in_low.append(smoothness(il, g))
        sm_in_high.append(smoothness(ih, g))

    if not in_diff:
        print("  [SKIP] 没有可比的 in 数据")
    else:
        dm = _mean(in_diff)
        print(f"    in 相对 L2: mean={dm:.3e}")
        print(f"    平滑度: 低射线 in={_mean(sm_in_low):.3f}  "
              f"高射线 in={_mean(sm_in_high):.3f}")
        if dm < 1e-3:
            print("  [FAIL] 两遍的 in 几乎相同 —— 射线数没变!")
            print("         检查 TracingOctahedronResolution 是否真的生效")
            print("         (改完 CVar 需要重开采集, 且确认拼写)")
            ok = False
        elif _mean(sm_in_high) >= _mean(sm_in_low):
            print("  [FAIL] 高射线的 in 没有更平滑 —— 射线数可能设反了")
            print("         低射线目录应该更噪 (平滑度更高)")
            ok = False
        else:
            print(f"  [OK ] 射线数确实不同, 高射线更干净 "
                  f"({_mean(sm_in_low):.3f} -> {_mean(sm_in_high):.3f})")

    # ------------------------------------------- [5] GT 质量验证 -------
    print("\n[5] GT 质量验证")
    print("    高射线的 gt 应比低射线的 gt 更干净 (B 的意义所在)")

    sm_gt_low, sm_gt_high = [], []
    for i in range(n):
        base = low_seq.frames[i]
        try:
            gl = np.load(low_dir / f"{base}_gt.npy")
            gh = np.load(high_dir / f"{base}_gt.npy")
        except FileNotFoundError:
            continue
        sm_gt_low.append(smoothness(gl, g))
        sm_gt_high.append(smoothness(gh, g))

    if not sm_gt_low:
        print("  [SKIP] 没有可比的 gt 数据")
    else:
        sl, sh = _mean(sm_gt_low), _mean(sm_gt_high)
        print(f"    gt 平滑度: 低射线={sl:.3f}  高射线={sh:.3f}")
        if sh < sl:
            print(f"  [OK ] 高射线 GT 更干净 (-{(sl - sh) / sl * 100:.1f}%)")
            print(f"         网络要学的: 从 {_mean(sm_in_low):.3f} "
                  f"压到 {sh:.3f}")
        else:
            print("  [FAIL] 高射线 GT 没有更干净 —— B 数据无效")
            ok = False

    # ------------------------------------------- [6] meta 射线数直验 ----
    print("\n[6] 射线数直接验证 (meta.json)  ★最直接证据")
    print("    引擎在 meta.json 里记录了 tracing_octahedron_resolution")

    tr_low = low_seq.meta.get("tracing_octahedron_resolution")
    tr_high = high_seq.meta.get("tracing_octahedron_resolution")
    ga_low = low_seq.meta.get("gather_octahedron_resolution")
    ga_high = high_seq.meta.get("gather_octahedron_resolution")

    def _rays(res):
        """tracing 分辨率 -> 实际射线数 (res x res)."""
        return f"{res}²={res * res} 条" if res else "未知"

    print(f"    输入 (低): tracing={tr_low}  -> {_rays(tr_low)}")
    print(f"    GT   (高): tracing={tr_high} -> {_rays(tr_high)}")

    if not tr_low or not tr_high:
        print("  [SKIP] meta.json 缺 tracing_octahedron_resolution")
    elif tr_low == tr_high:
        print(f"  [FAIL] 两遍射线数相同 ({tr_low}) —— CVar 没生效")
        print(f"         期望: 低射线目录=4, 高射线目录=16")
        ok = False
    elif tr_low > tr_high:
        print(f"  [FAIL] 射线数设反了! 输入目录({tr_low}) 比 GT 目录({tr_high}) 还高")
        print(f"         低射线目录应该是 4, 高射线目录应该是 16")
        ok = False
    else:
        ratio = (tr_high * tr_high) / (tr_low * tr_low)
        print(f"  [OK ] 射线数差 {ratio:.0f}x "
              f"({tr_low**2} -> {tr_high**2} 条)")
        if ga_low and ga_high and ga_low != ga_high:
            print(f"  [WARN] gather 分辨率也不同 ({ga_low} vs {ga_high})")
            print(f"         gather 应该保持一致 (通常 8), 否则格子对不上")
            ok = False

    # ---------------------------------------------------------- 总结 --
    print()
    print(SEP)
    if ok:
        print("[PASS] 全部通过 —— 可以用来训练方案 B:")
        print(f"   python train/train.py --data {low_dir} --gt-dir {high_dir}")
    else:
        print("[FAIL] 存在失败项 —— 先用上面的提示修数据, 不要直接训练")
        print("   (配对错了也能训, 但训出来的是垃圾, 而且不报错)")
    print(SEP)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
