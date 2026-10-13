"""验证 NSPF S1 从 Lumen dump 出来的 probe radiance 数据是否正确。

跑法
----
    python verify_probe_dump.py D:/ProbeDump              # 检查全部
    python verify_probe_dump.py D:/ProbeDump --vis        # 额外输出可视化 PNG
    python verify_probe_dump.py D:/ProbeDump --limit 3    # 只看前 3 帧

为什么需要这个脚本
------------------
引擎侧 dump 有几个环节容易悄悄出错, 肉眼在引擎里看不出来:
  1. R11G11B10 -> float32 解码错位 (pitch 单位搞错会让整张图斜掉)
  2. atlas 行列顺序颠倒
  3. probe 内部 8x8 的八面体布局理解错
  4. 全黑 / NaN / 值域异常

这个脚本逐项检查, 任何一项不过都会明确报出来。

检查项
------
  A. meta.json 与 npy 的 shape 是否自洽
  B. 数值健康度: NaN/Inf/全零/值域
  C. HDR 特征: radiance 应该有 >1 的值 (LDR 化说明解码错了)
  D. probe 结构: atlas 能否正确切成 probe 网格
  E. 球面平滑度: probe 内部相邻 texel 应该连续变化
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_meta(dump_dir: Path) -> dict | None:
    p = dump_dir / "meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] meta.json 解析失败: {e}")
        return None


def check_frame(npy_path: Path, meta: dict | None, verbose: bool = True) -> dict:
    """检查单帧, 返回结果字典。

    重要: atlas 纹理按屏幕最大尺寸分配, 槽位数通常多于实际 probe 数.
    多出来的槽位从未被 compute shader 写入 (composite shader 的外层 if
    没有 else 分支), 保留未初始化显存 —— 其中一部分位模式恰好是
    R11G11B10 的 NaN 编码 (表现为"只有 R 通道 NaN").

    这不是 bug. 有效范围由 meta 的 probe_view_size 给出, 检查只在
    有效区域内进行, 否则会产生误导性的 FAIL.
    """
    arr_raw = np.load(npy_path)
    res = {"name": npy_path.name, "ok": True, "problems": []}

    if arr_raw.ndim != 3 or arr_raw.shape[2] != 3:
        res["ok"] = False
        res["problems"].append(f"shape 异常: {arr_raw.shape}, 期望 (H, W, 3)")
        return res
    if arr_raw.dtype != np.float32:
        res["problems"].append(f"dtype={arr_raw.dtype}, 期望 float32")

    h, w, _ = arr_raw.shape
    res["shape"] = (h, w)

    # ---- 切出有效区域 ----
    gather = int(meta.get("gather_octahedron_resolution", 8)) if meta else 8
    valid_h, valid_w = h, w
    pvs = meta.get("probe_view_size") if meta else None
    if pvs:
        # probe_view_size = [宽, 高], 单位是 probe 数
        valid_w = min(w, int(pvs[0]) * gather)
        valid_h = min(h, int(pvs[1]) * gather)

    arr = arr_raw[:valid_h, :valid_w]
    res["valid_region"] = (valid_h, valid_w)
    res["slot_usage"] = (valid_h * valid_w) / (h * w) if h * w else 0.0
    res["gather_res"] = gather

    # 槽位利用率过低说明 atlas 分配策略浪费严重, 值得提醒
    if res["slot_usage"] < 0.7:
        res["problems"].append(
            f"槽位利用率仅 {res['slot_usage']:.1%} "
            f"(有效 {valid_h}x{valid_w} / 分配 {h}x{w}); "
            f"训练时务必只取有效区, 且 S2 应考虑只 dump 有效区")

    # ---- A. 与 meta 自洽 ----
    if meta:
        exp = meta.get("atlas_size")  # [W, H]
        if exp and (exp[1], exp[0]) != (h, w):
            res["ok"] = False
            res["problems"].append(
                f"与 meta 不符: npy={h}x{w}, meta atlas_size(W,H)={exp}")

    # ---- B. 数值健康度 ----
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    if n_nan:
        res["ok"] = False
        res["problems"].append(f"含 NaN: {n_nan} 个")
    if n_inf:
        res["ok"] = False
        res["problems"].append(f"含 Inf: {n_inf} 个")

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        res["ok"] = False
        res["problems"].append("没有任何有限值")
        return res

    vmin, vmax, vmean = float(finite.min()), float(finite.max()), float(finite.mean())
    res.update(vmin=vmin, vmax=vmax, vmean=vmean)

    if vmin < 0:
        res["problems"].append(f"存在负值 (min={vmin:.4g}), radiance 应非负")
    if vmax == 0 and vmin == 0:
        res["ok"] = False
        res["problems"].append("全黑, 大概率 dump 没抓到数据")

    # ---- C. HDR 特征 ----
    frac_gt1 = float((finite > 1.0).mean())
    res["frac_gt1"] = frac_gt1
    if frac_gt1 == 0.0:
        res["problems"].append(
            "没有任何 >1 的值 —— 数据可能被 LDR 化了, 检查解码路径")

    # ---- D/E. probe 结构 (只在有效区域内切分) ----
    if gather > 0 and valid_h % gather == 0 and valid_w % gather == 0:
        grid_h, grid_w = valid_h // gather, valid_w // gather
        res["probe_grid"] = (grid_h, grid_w)

        # 切成 probe: (gridH, gridW, gather, gather, 3)
        probes = arr.reshape(grid_h, gather, grid_w, gather, 3).transpose(0, 2, 1, 3, 4)

        # 球面平滑度: probe 内部相邻 texel 的平均绝对差 / 该 probe 的均值
        d = np.abs(np.diff(probes, axis=2)).mean(axis=(2, 3, 4))  # 垂直相邻
        mean_abs = np.abs(probes).mean(axis=(2, 3, 4)) + 1e-6
        smoothness = float((d / mean_abs).mean())
        res["probe_smoothness"] = smoothness
        if smoothness > 3.0:
            res["problems"].append(
                f"probe 内部相邻差异过大 ({smoothness:.2f}), "
                "可能是 pitch 错位或 atlas 布局理解错误")

        # 有效 probe 比例 (非全黑)
        probe_energy = probes.mean(axis=(2, 3, 4))
        res["valid_probe_ratio"] = float((probe_energy > 1e-6).mean())
    else:
        res["problems"].append(
            f"无法按 gather_res={gather} 切分有效区 {valid_h}x{valid_w}")

    if verbose:
        flag = "OK " if res["ok"] else "FAIL"
        print(f"  [{flag}] {res['name']}  atlas={res.get('shape')} "
              f"有效区={res.get('valid_region')} "
              f"槽位利用率={res.get('slot_usage', 0):.1%}")
        print(f"         min={vmin:.4g} max={vmax:.4g} mean={vmean:.4g} "
              f"HDR占比={frac_gt1:.1%}")
        if res.get("probe_grid"):
            print(f"         probe 网格={res['probe_grid']} "
                  f"gather={res['gather_res']} "
                  f"平滑度={res.get('probe_smoothness', 0):.3f} "
                  f"有效probe={res.get('valid_probe_ratio', 0):.1%}")
        for p in res["problems"]:
            print(f"         ! {p}")

    return res


def visualize(dump_dir: Path, arr: np.ndarray, meta: dict, out_dir: Path) -> None:
    """输出可视化, 便于人眼确认 atlas 结构和 probe 内容。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] 未装 matplotlib, 跳过可视化")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    h, w, _ = arr.shape
    gather = int(meta.get("gather_octahedron_resolution", 8))

    # 1) 整张 atlas (做 tone map 方便看)
    tone = np.clip(arr / (arr.max() + 1e-6), 0, 1) ** (1 / 2.2)
    plt.figure(figsize=(10, 6))
    plt.imshow(tone)
    plt.title(f"probe atlas {w}x{h} (tone mapped, gamma 2.2)")
    plt.colorbar(fraction=0.03)
    plt.tight_layout()
    p1 = out_dir / "atlas_overview.png"
    plt.savefig(p1, dpi=110)
    plt.close()

    # 2) 左上角 4x4 个 probe 放大, 看球面分布
    if h % gather == 0 and w % gather == 0:
        grid = arr.reshape(h // gather, gather, w // gather, gather, 3)
        grid = grid.transpose(0, 2, 1, 3, 4)  # (gridH, gridW, g, g, 3)
        n = min(4, grid.shape[0], grid.shape[1])
        fig, axes = plt.subplots(n, n, figsize=(2 * n, 2 * n))
        if n == 1:
            axes = np.array([[axes]])
        for i in range(n):
            for j in range(n):
                cell = grid[i, j]
                c = np.clip(cell / (cell.max() + 1e-6), 0, 1) ** (1 / 2.2)
                axes[i, j].imshow(c)
                axes[i, j].set_title(f"probe({i},{j})", fontsize=8)
                axes[i, j].axis("off")
        plt.suptitle(f"单个 probe 内部 {gather}x{gather} 八面体方向分布")
        plt.tight_layout()
        p2 = out_dir / "probe_cells.png"
        plt.savefig(p2, dpi=110)
        plt.close()
        print(f"  可视化 -> {p1}")
        print(f"  可视化 -> {p2}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=Path, help="采集输出目录 (含 meta.json)")
    ap.add_argument("--vis", action="store_true", help="输出可视化 PNG")
    ap.add_argument("--limit", type=int, default=0, help="只检查前 N 帧 (0=全部)")
    args = ap.parse_args()

    dump_dir: Path = args.dump_dir
    if not dump_dir.exists():
        print(f"[error] 目录不存在: {dump_dir}")
        return 1

    meta = load_meta(dump_dir)
    files = sorted(dump_dir.glob("probe_*.npy"))
    if not files:
        print(f"[error] 没找到 probe_*.npy。先确认:")
        print(f"        1) 引擎已编译且 r.Lumen.ScreenProbeDump.Enable 1")
        print(f"        2) 已跑过若干帧 (readback 有 1~3 帧延迟)")
        return 1
    if args.limit > 0:
        files = files[: args.limit]

    print("=" * 70)
    print(f"NSPF S1 数据验证: {dump_dir}")
    print(f"帧数: {len(files)}")
    print("=" * 70)

    if meta:
        print("meta.json:")
        for k in ("stage", "content", "atlas_size", "probe_view_size",
                  "gather_octahedron_resolution", "tracing_octahedron_resolution",
                  "downsample_factor", "source_format", "dtype"):
            if k in meta:
                print(f"  {k:32} = {meta[k]}")
    else:
        print("[warn] 无 meta.json, 部分检查会退化 (引擎未升级到写出 meta 的版本?)")
    print()

    results = [check_frame(f, meta) for f in files]

    ok = sum(1 for r in results if r["ok"])
    print()
    print("=" * 70)
    print(f"结果: {ok} / {len(results)} 帧通过")
    if ok < len(results):
        print("未通过的帧:")
        for r in results:
            if not r["ok"]:
                print(f"  {r['name']}: {'; '.join(r['problems'])}")
    print("=" * 70)

    if args.vis and results:
        first = np.load(files[0])
        visualize(dump_dir, first, meta or {}, dump_dir / "_verify")

    return 0 if ok == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
