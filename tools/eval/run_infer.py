"""规范化推理入口 (枚举式调用).

设计目的
--------
之前每次推理都要临时拼一堆参数(--gt-dir / --gt-use-in / --stride / 各种目录),
产出的图命名随意(compare_f1 / full_f900 / b2_f300 ...), 过几天完全看不出
是哪个模型、哪个方案、哪个数据集推出来的.

本脚本把"方案/场景/模型"全部收进注册表, 调用时只用选枚举值:

    python tools/eval/run_infer.py --scheme A  --scene train --frames 150
    python tools/eval/run_infer.py --scheme B2 --scene new   --frames 30 300
    python tools/eval/run_infer.py --list                     # 查看所有可选项

输出规范
--------
    output/<场景>/<方案>_f<帧号>.png

每张图自带完整标注(标题里写清方案 / 模型 / 数据 / 三个平滑度 / 完成度),
单独打开任何一张图都能 self-explain.

三种方案的区别 (核心!)
----------------------
    A   输入 low.in(64条未滤波)  -> 答案 low.gt (64条已滤波)   = 纯滤波
    B'  输入 low.in(64条未滤波)  -> 答案 high.gt(256条已滤波)  = 滤波+补射线
    B'' 输入 low.in(64条未滤波)  -> 答案 high.in(256条未滤波)  = 纯补射线
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# 中文字体
for _f in ("Microsoft YaHei", "SimHei", "DengXian", "SimSun"):
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "dataset"))
sys.path.insert(0, str(ROOT / "train"))

from probe_dataset import ProbeTexelDataset           # noqa: E402
from model import ProbeFilterMLP                      # noqa: E402


# ======================================================================
#  注册表: 所有可选项都在这里, 调用时只引用 key
# ======================================================================

SCHEMES: dict[str, dict] = {
    "A": {
        "label": "A-纯滤波",
        "in_desc": "low.in (64条, 未滤波)",
        "gt_desc": "low.gt (64条, 已滤波)",
        "task": "学 Lumen 的滤波去噪",
        "needs_gt_dir": False,
        "gt_use_in": False,
    },
    "B": {
        "label": "B'-滤波+补射线",
        "in_desc": "low.in (64条, 未滤波)",
        "gt_desc": "high.gt (256条, 已滤波)",
        "task": "一步到位: 去噪 + 补到256条质量",
        "needs_gt_dir": True,
        "gt_use_in": False,
    },
    "B2": {
        "label": "B''-纯补射线",
        "in_desc": "low.in (64条, 未滤波)",
        "gt_desc": "high.in (256条, 未滤波)",
        "task": "只补射线 (两边都未滤波, 滤波被抵消)",
        "needs_gt_dir": True,
        "gt_use_in": True,
    },
}

SCENES: dict[str, dict] = {
    "train": {
        "label": "原场景 (训练集, 模型见过)",
        "low": "D:/ProbeDump_B_low",
        "high": "D:/ProbeDump_B_high",
    },
    "new": {
        "label": "新场景 (泛化验证, 模型没见过)",
        "low": "D:/ProbeDump_NEW_low",
        "high": "D:/ProbeDump_NEW_high",
    },
}

# 方案 -> 训练产物目录 (相对项目根)
MODELS: dict[str, str] = {
    "A": "runs/b_full",        # 注: 因旧 bug 它实际训的是方案 A
    "B": "runs/b_prime_real",  # 真正修复后训的 B'(已训练, loss 0.1141)
    "B2": "runs/b_rayonly",    # 纯补射线
}


# ======================================================================
#  工具
# ======================================================================

def tonemap(x: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    hi = float(np.percentile(x, 99))
    if hi <= 1e-8:
        hi = float(x.max()) if x.max() > 1e-8 else 1.0
    return np.clip(x / hi, 0.0, 1.0) ** (1.0 / 2.2)


def smoothness(atlas: np.ndarray, g: int) -> float:
    h, w = atlas.shape[:2]
    ph, pw = h // g, w // g
    p = atlas.reshape(ph, g, pw, g, -1).transpose(0, 2, 1, 3, 4)
    d = np.abs(np.diff(p, axis=2)).mean()
    m = np.abs(p).mean() + 1e-6
    return float(d / m)


def show_registry() -> None:
    print("=" * 70)
    print("可选方案 (--scheme)")
    print("=" * 70)
    for k, v in SCHEMES.items():
        model = ROOT / MODELS[k]
        exists = "已训练" if (model / "best.pt").exists() else "未训练"
        print(f"  {k:<4} {v['label']:<16} [{exists}]")
        print(f"       输入 {v['in_desc']}  ->  答案 {v['gt_desc']}")
        print(f"       任务: {v['task']}")
        print(f"       模型: {MODELS[k]}")
        print()

    print("=" * 70)
    print("可选场景 (--scene)")
    print("=" * 70)
    for k, v in SCENES.items():
        has = "OK" if Path(v["low"]).exists() else "缺失"
        print(f"  {k:<8} {v['label']:<28} [{has}]")
    print()


# ======================================================================
#  主流程
# ======================================================================

def run_one(scheme: str, scene: str, frame: int, device: str,
            model_override: str = "") -> bool:
    sc = SCHEMES[scheme]
    sn = SCENES[scene]

    model_dir = ROOT / (model_override if model_override else MODELS[scheme])
    model_tag = model_override if model_override else MODELS[scheme]
    ckpt = model_dir / "best.pt"
    if not ckpt.exists():
        print(f"[跳过] 方案 {scheme} 的模型不存在: {ckpt}")
        print(f"       先训练: python train/train.py --data {sn['low']} "
              f"--gt-dir {sn['high']} --out {MODELS[scheme]}"
              + (" --gt-use-in" if sc["gt_use_in"] else ""))
        return False

    low_dir = sn["low"]
    high_dir = sn["high"] if sc["needs_gt_dir"] else None
    if not Path(low_dir).exists():
        print(f"[跳过] 数据不存在: {low_dir}")
        return False

    # ---- 模型 ----
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    hidden = a.get("hidden", 64)
    layers = a.get("layers", 4)
    use_history = not a.get("no_history", False)
    residual = not a.get("no_residual", False)
    in_dim = 56 if use_history else 52

    model = ProbeFilterMLP(in_dim=in_dim, hidden=hidden,
                           num_layers=layers, use_residual=residual)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    # ---- 数据 ----
    ds = ProbeTexelDataset(low_dir, samples_per_frame=1,
                           use_history=use_history, max_frames=0,
                           gt_dir=high_dir, gt_use_in=sc["gt_use_in"])
    seq = ds.seq
    g = seq.gather

    i = max(1, min(frame, len(seq) - 1))
    cur = seq.load_frame(i)
    prev = seq.load_frame(i - 1) if use_history else None
    feat, _ = ds._build_features(cur, prev)

    with torch.no_grad():
        x = torch.from_numpy(feat.reshape(-1, feat.shape[-1])).to(device)
        out = model(x).cpu().numpy().reshape(feat.shape[0], feat.shape[1], 3)

    in_lin = cur["in"]
    out_lin = np.expm1(out)
    key = "in" if sc["gt_use_in"] else "gt"
    if ds.gt_seq is not None:
        gi = ds.gt_index_map.get(i, i) if ds.gt_index_map else i
        gt_lin = ds.gt_seq.load_frame(gi)[key]
    else:
        gt_lin = cur[key]

    # 裁有效区
    vh, vw = cur["valid"].shape
    vh, vw = vh * g, vw * g
    in_lin, out_lin = in_lin[:vh, :vw], out_lin[:vh, :vw]
    gt_lin = gt_lin[:vh, :vw]

    s_in = smoothness(in_lin, g)
    s_out = smoothness(out_lin, g)
    s_gt = smoothness(gt_lin, g)
    gap = s_in - s_gt
    done = (s_in - s_out) / gap * 100 if gap > 1e-6 else float("nan")

    # ---- 画图 (自解释标注) ----
    titles = [
        f"输入\n{sc['in_desc']}\n平滑度 {s_in:.3f}",
        f"模型输出\n平滑度 {s_out:.3f}",
        f"答案\n{sc['gt_desc']}\n平滑度 {s_gt:.3f}",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax, img, t in zip(axes, [in_lin, out_lin, gt_lin], titles):
        ax.imshow(tonemap(img))
        ax.set_title(t, fontsize=11)
        ax.axis("off")

    head = (f"方案 {sc['label']}   |   场景 {scene} ({sn['label']})   |   "
            f"帧 {i}\n"
            f"模型 {model_tag}   |   任务: {sc['task']}\n"
            f"平滑度  输入 {s_in:.3f} -> 输出 {s_out:.3f}   |   "
            f"答案 {s_gt:.3f}   |   完成度 {done:.1f}%")
    fig.suptitle(head, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    out_dir = ROOT / "output" / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = scheme if not model_override \
        else f"{scheme}_{Path(model_override).name}"
    png = out_dir / f"{tag}_f{i:04d}.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)

    print(f"[OK] {scheme:<3} {scene:<6} 帧{i:<5} "
          f"in={s_in:.3f} out={s_out:.3f} gt={s_gt:.3f} "
          f"完成度={done:.1f}%  ->  {png.relative_to(ROOT)}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description="规范化推理 (枚举式): 只需选 scheme / scene / frames")
    ap.add_argument("--scheme", choices=list(SCHEMES.keys()),
                    help="方案: A=纯滤波, B=滤波+补射线, B2=纯补射线")
    ap.add_argument("--scene", choices=list(SCENES.keys()),
                    help="场景: train=原场景, new=新场景")
    ap.add_argument("--frames", type=int, nargs="+", default=[150],
                    help="要推理的帧号 (可多个)")
    ap.add_argument("--model-dir", default="",
                    help="覆盖注册表的模型目录(相对项目根), 用于实验模型对比. "
                         "输出文件名会自动带上模型目录名以区分")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--list", action="store_true", help="列出所有可选项后退出")
    args = ap.parse_args()

    if args.list or not args.scheme:
        show_registry()
        return 0

    scene = args.scene or "train"
    ok = True
    for f in args.frames:
        ok &= run_one(args.scheme, scene, f, args.device, args.model_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
