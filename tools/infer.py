"""NSPF 推理: 对采集帧跑模型, 输出可视化 + 实测推理耗时.

为什么必须实测耗时
------------------
model.py 里的 0.86ms 是用 FLOPs / 假设算力"估算"的.
但小型 MLP 是访存密集型, 实际效率可能远低于峰值算力.
本脚本真机计时, 验证是否真的能省出时间.

跑法
----
    # 推理单帧 + 可视化 + 计时
    python tools/infer.py runs/v2_nohist/best.pt D:/ProbeDump_S2 --frame 1 --vis

    # 只测耗时 (跑 20 次取平均)
    python tools/infer.py runs/v2_nohist/best.pt D:/ProbeDump_S2 --bench 20

    # 批量出图
    python tools/infer.py runs/v2_nohist/best.pt D:/ProbeDump_S2 --frame 1 2 3 --vis
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "dataset"))
sys.path.insert(0, str(ROOT / "train"))

from probe_dataset import ProbeSequence, ProbeTexelDataset  # noqa: E402
from model import ProbeFilterMLP, count_parameters  # noqa: E402


def smoothness(atlas: np.ndarray, g: int) -> float:
    h, w = atlas.shape[:2]
    ph, pw = h // g, w // g
    p = atlas.reshape(ph, g, pw, g, -1).transpose(0, 2, 1, 3, 4)
    return float(np.abs(np.diff(p, axis=2)).mean() / (np.abs(p).mean() + 1e-6))


def load_model(ckpt_path: str, device: str) -> tuple[ProbeFilterMLP, dict]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    no_hist = a.get("no_history", False)
    in_dim = 52 if no_hist else 56
    m = ProbeFilterMLP(
        in_dim=in_dim,
        hidden=a.get("hidden", 64),
        num_layers=a.get("layers", 4),
        use_residual=not a.get("no_residual", False),
    )
    m.load_state_dict(ck["model"])
    m.to(device).eval()
    return m, {"in_dim": in_dim, "use_history": not no_hist, "args": a, "ck": ck}


@torch.no_grad()
def infer_frame(model, ds: ProbeTexelDataset, seq: ProbeSequence, idx: int,
                device: str) -> dict:
    cur = seq.load_frame(idx)
    prev = seq.load_frame(idx - 1) if ds.use_history else None
    feat, target = ds._build_features(cur, prev)

    h, w = feat.shape[:2]
    x = torch.from_numpy(feat.reshape(-1, feat.shape[-1])).to(device)
    out = model(x).cpu().numpy().reshape(h, w, 3)

    return {
        "base": cur["base"],
        "in": cur["in"],                      # 线性域 radiance
        "gt": cur["gt"],
        "out": np.expm1(np.clip(out, -20, 20)),
        "valid": cur["valid"],
        "feat": feat,
        "target": target,
    }


@torch.no_grad()
def bench(model, feat: np.ndarray, device: str, n: int = 20) -> dict:
    """实测推理耗时, 对比多种优化模式.

    小 MLP 是访存密集型, FLOPs 估算会严重低估:
      实测 fp32 = 2.69ms, 而按 FLOPs 估算只有 0.86ms (差 3.1x).
    所以必须实测, 并验证优化手段的实际收益.
    """
    x = torch.from_numpy(feat.reshape(-1, feat.shape[-1])).to(device)
    n_samples = x.shape[0]
    results: dict[str, float] = {}

    def timeit(fn, warmup=5, repeat=n):
        for _ in range(warmup):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        ts = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn()
            if device == "cuda":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return float(np.mean(ts) * 1000.0)

    # --- 1. fp32 baseline ---
    results["fp32"] = timeit(lambda: model(x))

    if device != "cuda":
        return {"n_samples": n_samples, "modes": results}

    # --- 2. fp16 ---
    m16 = model.half()
    x16 = x.half()
    results["fp16"] = timeit(lambda: m16(x16))

    # --- 3. fp16 + CUDA Graph ---
    # 消除 kernel launch 开销, 对小网络收益显著
    try:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = m16(x16)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            y_static = m16(x16)
        results["fp16+graph"] = timeit(lambda: g.replay(), warmup=3)
    except Exception as e:  # noqa: BLE001
        print(f"  [info] CUDA Graph 不可用: {e}")

    return {"n_samples": n_samples, "modes": results}


def probe_spatial(atlas: np.ndarray, g: int,
                  valid_probes_h: int | None = None) -> np.ndarray:
    """把 atlas 聚合到空间维度: 每个 probe 取 64 个方向的平均.

    atlas 看的是"方向" (每个 probe 内部 8x8 的球面分布), 人眼不直观.
    聚合后得到 (probeH, probeW, 3), 就是一张低分辨率的"GI 分布图",
    能直接看出哪里亮哪里暗、以及噪点有没有被压下去.
    """
    h, w = atlas.shape[:2]
    ph, pw = h // g, w // g
    p = atlas.reshape(ph, g, pw, g, 3).transpose(0, 2, 1, 3, 4)  # (ph,pw,g,g,3)
    s = p.mean(axis=(2, 3))
    if valid_probes_h is not None:
        s = s[:valid_probes_h]
    return s


def _setup_matplotlib():
    """初始化 matplotlib, 尽力启用中文字体.

    Windows 上 matplotlib 默认 DejaVu Sans 不含中文字形, 标题会渲染成方块.
    这里按常见中文字体依次尝试; 都不行就把中文标题降级为英文 (见 visualize).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for name in ("Microsoft YaHei", "Microsoft JhengHei", "SimHei",
                 "SimSun", "Noto Sans CJK SC", "Source Han Sans SC"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return plt, True
        except Exception:  # noqa: BLE001
            continue
    return plt, False


def visualize(r: dict, g: int, out_dir: Path) -> None:
    try:
        plt, has_cjk = _setup_matplotlib()
    except ImportError:
        print("  [warn] 无 matplotlib, 跳过可视化")
        return
    # 无中文字体时用英文标题, 避免渲染成方块
    T = (lambda cn, en: cn) if has_cjk else (lambda cn, en: en)

    out_dir.mkdir(parents=True, exist_ok=True)
    r_in, gt, out = r["in"], r["gt"], r["out"]

    def tone(a):
        m = np.nanmax(a)
        return np.clip(a / (m + 1e-6), 0, 1) ** (1 / 2.2)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, img, title in (
        (axes[0], r_in, f"INPUT (pre-filter)\nsmooth={smoothness(r_in, g):.3f}"),
        (axes[1], gt, f"GT (Lumen filtered)\nsmooth={smoothness(gt, g):.3f}"),
        (axes[2], out, f"MODEL OUTPUT\nsmooth={smoothness(out, g):.3f}"),
    ):
        ax.imshow(tone(img))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.suptitle(f"{r['base']}  probe atlas (tone mapped)")
    plt.tight_layout()
    p = out_dir / f"{r['base']}_compare.png"
    plt.savefig(p, dpi=110)
    plt.close()

    # 局部放大: 取中间一个 probe 区域, 看 8x8 方向分布
    ph, pw = r_in.shape[0] // g, r_in.shape[1] // g
    ci, cj = ph // 2, pw // 2
    sl = (slice(ci * g, (ci + 1) * g), slice(cj * g, (cj + 1) * g))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, img, title in ((axes[0], r_in, "input"),
                           (axes[1], gt, "GT"),
                           (axes[2], out, "output")):
        ax.imshow(tone(img[sl]))
        ax.set_title(f"probe({ci},{cj}) {title}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    p2 = out_dir / f"{r['base']}_probe.png"
    plt.savefig(p2, dpi=110)
    plt.close()

    # ---- 图3: 空间维度 GI 图 (最直观) ----
    # 每个 probe 取 64 个方向的平均, 得到低分辨率的 GI 分布.
    # 关键: 裁掉 atlas 末尾的 padding, 只看有效 probe 行
    # (不然 padding 的垃圾值会和网络输出对比, 误差图会出现伪红条)
    n_valid = r["valid"].shape[0]
    s_in = probe_spatial(r_in, g, valid_probes_h=n_valid)
    s_gt = probe_spatial(gt, g, valid_probes_h=n_valid)
    s_out = probe_spatial(out, g, valid_probes_h=n_valid)

    def tone3(a):
        m = np.nanmax(a)
        return np.clip(a / (m + 1e-6), 0, 1) ** (1 / 2.2)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, img, title in ((axes[0], s_in, "INPUT (noisy)"),
                           (axes[1], s_gt, "GT (Lumen filtered)"),
                           (axes[2], s_out, "MODEL OUTPUT")):
        ax.imshow(tone3(img))
        ax.set_title(f"{title}\nprobe grid {s_in.shape[0]}x{s_in.shape[1]}",
                     fontsize=10)
        ax.axis("off")
    plt.suptitle(
        f"{r['base']}  " +
        T("空间 GI 分布 (每 probe 取 64 方向平均)\n看: 明暗结构是否保留 + 噪点是否被压掉",
          "Spatial GI (mean over 64 directions per probe)\n"
          "check: structure preserved + noise suppressed"),
        fontsize=11)
    plt.tight_layout()
    p3 = out_dir / f"{r['base']}_spatial.png"
    plt.savefig(p3, dpi=130)
    plt.close()

    # ---- 图4: 误差对比 ----
    # |输入-GT| vs |输出-GT| : 若后者更小, 说明模型确实在向 GT 靠拢
    e_in = np.abs(s_in - s_gt).mean(axis=2)
    e_out = np.abs(s_out - s_gt).mean(axis=2)
    vmax = max(np.nanmax(e_in), np.nanmax(e_out)) + 1e-6

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    im0 = axes[0].imshow(e_in, cmap="magma", vmin=0, vmax=vmax)
    axes[0].set_title(f"|INPUT - GT|\nmean={np.nanmean(e_in):.4f}", fontsize=10)
    im1 = axes[1].imshow(e_out, cmap="magma", vmin=0, vmax=vmax)
    axes[1].set_title(f"|OUTPUT - GT|\nmean={np.nanmean(e_out):.4f}", fontsize=10)
    diff = e_in - e_out
    im2 = axes[2].imshow(diff, cmap="RdYlGn",
                         vmin=-np.nanmax(np.abs(diff)) - 1e-6,
                         vmax=np.nanmax(np.abs(diff)) + 1e-6)
    axes[2].set_title(T("改善量 (绿=变好, 红=变差)",
                        "Improvement (green=better, red=worse)"), fontsize=10)
    for ax in axes:
        ax.axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.03)
    plt.colorbar(im1, ax=axes[1], fraction=0.03)
    plt.colorbar(im2, ax=axes[2], fraction=0.03)
    plt.suptitle(f"{r['base']}  " +
                 T("误差对比 (同色标)", "Error comparison (shared scale)"),
                 fontsize=11)
    plt.tight_layout()
    p4 = out_dir / f"{r['base']}_error.png"
    plt.savefig(p4, dpi=130)
    plt.close()

    print(f"  可视化: {p.name}, {p2.name}, {p3.name}, {p4.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("data")
    ap.add_argument("--frame", type=int, nargs="*", default=[1])
    ap.add_argument("--vis", action="store_true")
    ap.add_argument("--bench", type=int, default=0, help="计时重复次数")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    model, info = load_model(args.ckpt, device)
    ck = info["ck"]

    print(f"模型: {args.ckpt}")
    print(f"  hidden={info['args'].get('hidden', 64)} "
          f"layers={info['args'].get('layers', 4)} "
          f"in_dim={info['in_dim']} 参数={count_parameters(model):,}")
    if "loss" in ck:
        print(f"  训练最佳 loss={ck['loss']:.6f}")
    print(f"设备: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    ds = ProbeTexelDataset(args.data, samples_per_frame=1,
                           use_history=info["use_history"], max_frames=1)
    seq = ds.seq
    g = seq.gather
    out_dir = Path(args.out) if args.out else ROOT / "output" / "infer"

    frames = args.frame if args.frame else [1]
    print()
    print(f"{'帧':<16}{'输入':>8}{'GT':>8}{'输出':>8}{'vsGT':>9}"
          f"{'MSE':>9}{'relL2':>8}")
    print("-" * 68)

    results = []
    for idx in frames:
        if idx >= len(seq):
            print(f"  [skip] 帧 {idx} 超出范围 (共 {len(seq)} 帧)")
            continue
        r = infer_frame(model, ds, seq, idx, device)
        results.append(r)

        m = r["valid"]
        mse = float(((r["out"][np.repeat(np.repeat(m, g, 0), g, 1)]
                      - r["gt"][np.repeat(np.repeat(m, g, 0), g, 1)]) ** 2).mean())
        rel = float((((r["out"] - r["gt"]) ** 2 /
                      (r["gt"] ** 2 + 0.1))[np.repeat(np.repeat(m, g, 0), g, 1)]).mean())
        s_in = smoothness(r["in"], g)
        s_gt = smoothness(r["gt"], g)
        s_out = smoothness(r["out"], g)
        print(f"{r['base']:<16}{s_in:>8.3f}{s_gt:>8.3f}{s_out:>8.3f}"
              f"{(s_gt - s_out) / s_gt * 100:>8.1f}%{mse:>9.4f}{rel:>8.4f}")

        if args.vis:
            visualize(r, g, out_dir)

    # ---- 实测耗时 ----
    if args.bench > 0 and results:
        print()
        print("=" * 68)
        print("推理耗时实测")
        print("=" * 68)
        r = results[0]
        b = bench(model, r["feat"], device, args.bench)
        modes = b["modes"]
        print(f"  样本数/帧: {b['n_samples']:,} "
              f"(probe {seq.probe_view_size[1]}x{seq.probe_view_size[0]} x {g}x{g})")
        print()
        print(f"  {'模式':<16}{'耗时(ms)':>10}{'相对fp32':>10}")
        print("  " + "-" * 36)
        base = modes.get("fp32", 1.0)
        for k in ("fp32", "fp16", "fp16+graph"):
            if k in modes:
                v = modes[k]
                print(f"  {k:<16}{v:>10.3f}{base / v:>9.2f}x")

        best = min(modes.values())
        print()
        print(f"  按 FLOPs 估算(model.py): 0.86 ms")
        print(f"  实测最快: {best:.3f} ms  (估算偏差 {best / 0.86:.2f}x)")
        print()
        print("  与收益对比 (射线 64->16 约省 2.5~4.5 ms trace):")
        if best < 1.0:
            print(f"  [OK] {best:.2f}ms < 1.0ms, 净收益充足")
        elif best < 2.0:
            print(f"  [~] {best:.2f}ms, 还有收益但已不多")
        else:
            print(f"  [NG] {best:.2f}ms 偏大, 需缩小网络或上 TensorRT")
            print("       建议: ONNX -> TensorRT (层融合 + FP16), 预期再快 2~4x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
