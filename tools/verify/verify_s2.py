"""验证 NSPF S2 采集的数据是否可用于训练.

跑法:
    python verify_s2.py D:/ProbeDump
    python verify_s2.py D:/ProbeDump --limit 3

S2 相比 S1 多了 GT (滤波后 radiance) 和几何引导数据.

核心验证: GT 是否真的比输入更平滑?
-----------------------------------
这是 S2 成败的判据. 我们 dump 的:
    in = 滤波前 (每条射线独立采样 -> 噪点大 -> 平滑度高)
    gt = 滤波后 (时域+空间滤波 -> 噪点小 -> 平滑度低)

如果 gt 的平滑度没有明显低于 in, 说明:
    - 要么 dump 位置错了 (两者采到同一个东西)
    - 要么滤波没生效 (比如时域历史被清空)
这种错误不会报错, 只会让训练出的模型毫无用处, 所以必须在这里拦住.

验证项
------
A. 文件完整性 + shape 自洽 (in/gt 必须完全同形)
B. 有效区无 NaN / 无异常巨值
C. 平滑度对比 (核心)
D. in/gt 差异合理 (既不能完全相同, 也不能完全无关)
E. 几何数据可解码 (depth 的 uint bitcast, normal 的八面体)
F. 相机 json 有效
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def smoothness(atlas: np.ndarray, gather: int) -> float:
    """probe 内部相邻 texel 的差异 / 均值. 越小越平滑."""
    h, w = atlas.shape[:2]
    ph, pw = h // gather, w // gather
    probes = atlas.reshape(ph, gather, pw, gather, -1).transpose(0, 2, 1, 3, 4)
    d = np.abs(np.diff(probes, axis=2)).mean()
    m = np.abs(probes).mean() + 1e-6
    return float(d / m)


def check_frame(d: Path, base: str, meta: dict, verbose: bool = True) -> dict:
    gather = int(meta.get("gather_octahedron_resolution", 8))
    res = {"frame": base, "ok": True, "problems": []}

    def load(tag: str) -> np.ndarray | None:
        p = d / f"{base}_{tag}.npy"
        return np.load(p) if p.exists() else None

    arr_in = load("in")
    arr_gt = load("gt")

    if arr_in is None or arr_gt is None:
        res["ok"] = False
        res["problems"].append("缺少 in/gt 文件")
        return res

    # ---- A. shape 必须完全一致 (配对的前提) ----
    if arr_in.shape != arr_gt.shape:
        res["ok"] = False
        res["problems"].append(
            f"in/gt shape 不一致: {arr_in.shape} vs {arr_gt.shape}")
        return res

    res["shape"] = arr_in.shape

    # ---- B. 数值健康度 ----
    for tag, arr in (("in", arr_in), ("gt", arr_gt)):
        n_nan = int(np.isnan(arr).sum())
        if n_nan:
            res["ok"] = False
            res["problems"].append(f"{tag} 含 NaN: {n_nan}")
        fin = arr[np.isfinite(arr)]
        res[f"{tag}_max"] = float(fin.max()) if fin.size else 0.0
        # 有效区不该出现 S1 里那种 5 万量级的未初始化巨值
        if fin.size and fin.max() > 1e3:
            res["problems"].append(
                f"{tag} max={fin.max():.4g} 异常偏大, 可能混入了未初始化槽位")

    # ---- C. 平滑度对比 (核心) ----
    s_in = smoothness(arr_in, gather)
    s_gt = smoothness(arr_gt, gather)
    res["smooth_in"] = s_in
    res["smooth_gt"] = s_gt
    if s_gt >= s_in:
        res["ok"] = False
        res["problems"].append(
            f"gt 平滑度({s_gt:.3f}) 未低于 in({s_in:.3f}) —— "
            "滤波未生效或 dump 位置错误!")
    else:
        res["smooth_ratio"] = s_gt / s_in if s_in > 0 else 0.0

    # ---- D. in/gt 差异合理 ----
    diff = np.abs(arr_in - arr_gt)
    rel = float(diff.mean() / (np.abs(arr_gt).mean() + 1e-6))
    res["rel_diff"] = rel
    if rel < 1e-4:
        res["ok"] = False
        res["problems"].append(
            f"in/gt 几乎相同 (相对差异 {rel:.2e}) —— 采到同一个东西了")
    if rel > 10.0:
        res["problems"].append(
            f"in/gt 差异过大 (相对差异 {rel:.2f}) —— 可能不是同一帧")

    # ---- E. 几何数据 ----
    for tag, dtype in (("hitdist", np.uint8), ("depth", np.uint32),
                       ("normal", np.uint8), ("wpos", np.float32)):
        arr = load(tag)
        if arr is None:
            res["problems"].append(f"缺少 {tag}.npy")
            continue
        res[f"{tag}_shape"] = arr.shape
        res[f"{tag}_dtype_ok"] = (arr.dtype == dtype)

    # depth 解码: Lumen 用 depth <= 0 (哨兵值, 常见 -1e8) 标记"该 probe 位置
    # 没有有效几何"(朝天空/超出视口). 这是正常约定而非编码错误 ——
    # shader 里就是靠 `if (SceneDepth > 0)` 来筛掉它们的.
    # 因此只统计正值部分, 并把无效比例当作信息而非报错.
    arr_depth = load("depth")
    if arr_depth is not None and arr_depth.dtype == np.uint32:
        depth_f = arr_depth.view(np.float32)
        finite = depth_f[np.isfinite(depth_f)]
        if finite.size:
            valid_mask = finite > 0
            n_valid = int(valid_mask.sum())
            res["depth_valid_ratio"] = n_valid / finite.size
            if n_valid:
                res["depth_range"] = (float(finite[valid_mask].min()),
                                      float(finite[valid_mask].max()))
            # 有效比例过低才可疑 (太少有效 probe 说明采集有问题)
            if res["depth_valid_ratio"] < 0.5:
                res["problems"].append(
                    f"有效 depth 仅 {res['depth_valid_ratio']:.1%}, 检查采集设置")
            # 有效深度不应该大到离谱 (UE 单位 cm, 几十万已是极限)
            if n_valid and finite[valid_mask].max() > 1e7:
                res["problems"].append("有效 depth 量级异常, 检查 asfloat 解码")

    # ---- F. 相机 json ----
    jp = d / f"{base}.json"
    if not jp.exists():
        res["problems"].append("缺少相机 json")
    else:
        try:
            js = json.loads(jp.read_text(encoding="utf-8"))
            res["prev_valid"] = js.get("prev_valid", False)
            if len(js.get("view_matrix", [])) != 16:
                res["problems"].append("view_matrix 不是 16 个数")
        except Exception as e:  # noqa: BLE001
            res["problems"].append(f"相机 json 解析失败: {e}")

    if verbose:
        flag = "OK " if res["ok"] else "FAIL"
        print(f"  [{flag}] {base}  shape={res.get('shape')}")
        print(f"         平滑度: in={s_in:.3f}  gt={s_gt:.3f}  "
              f"-> gt/in = {res.get('smooth_ratio', float('nan')):.3f}")
        print(f"         相对差异={rel:.3f}  in_max={res.get('in_max', 0):.4g}  "
              f"gt_max={res.get('gt_max', 0):.4g}")
        if res.get("depth_range"):
            lo, hi = res["depth_range"]
            print(f"         depth(asfloat) 有效值 = [{lo:.1f}, {hi:.1f}] "
                  f"有效占比={res.get('depth_valid_ratio', 0):.1%} "
                  f"(其余为无几何的哨兵值)")
        for p in res["problems"]:
            print(f"         ! {p}")

    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    d: Path = args.dump_dir
    if not d.exists():
        print(f"[error] 目录不存在: {d}")
        return 1

    meta_p = d / "meta.json"
    if not meta_p.exists():
        print(f"[error] 无 meta.json。跑引擎采集时会生成。")
        return 1
    meta = json.loads(meta_p.read_text(encoding="utf-8"))

    print("=" * 72)
    print(f"NSPF S2 数据验证: {d}")
    print(f"stage={meta.get('stage')}  gt_scheme={meta.get('gt_scheme')}")
    for k in ("atlas_size", "valid_atlas_size", "probe_view_size",
              "gather_octahedron_resolution", "tracing_octahedron_resolution"):
        if k in meta:
            print(f"  {k:32} = {meta[k]}")
    print("=" * 72)

    # 从 in 文件反推帧列表
    frames = sorted({p.name.rsplit("_in.npy", 1)[0] for p in d.glob("*_in.npy")})
    if not frames:
        print("[error] 没找到 *_in.npy。确认: 引擎已重新编译, 且 Enable=1 后跑过若干帧。")
        return 1
    if args.limit > 0:
        frames = frames[: args.limit]

    results = [check_frame(d, f, meta) for f in frames]
    ok = sum(1 for r in results if r["ok"])

    print()
    print("=" * 72)
    print(f"结果: {ok} / {len(results)} 帧通过")

    if results:
        s_in = np.mean([r.get("smooth_in", 0) for r in results])
        s_gt = np.mean([r.get("smooth_gt", 0) for r in results])
        print(f"平均平滑度: in={s_in:.3f}  gt={s_gt:.3f}")
        if s_in > 0 and s_gt < s_in:
            print(f"  ✓ GT 比输入平滑 {(1 - s_gt / s_in):.1%} "
                  f"—— 滤波生效, 数据可用于训练")
        else:
            print("  ✗ GT 未比输入更平滑 —— 采集有问题, 不要拿去训练!")

    if ok < len(results):
        print("\n未通过:")
        for r in results:
            if not r["ok"]:
                print(f"  {r['frame']}: {'; '.join(r['problems'])}")
    print("=" * 72)
    return 0 if ok == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
