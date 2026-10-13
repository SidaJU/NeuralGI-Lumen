"""评估模型输出质量: 平滑度是否优于 GT.

为什么看平滑度而不是 loss
------------------------
loss 下降只说明"拟合得更好", 不代表"结果更有用".
我们的目标是: 从噪点大的输入, 产出干净的 GI. 所以直接看:

    输入平滑度 (0.938)  >  GT平滑度 (0.801)  >  ? 模型输出平滑度

如果模型输出 < 0.801, 说明它比 Lumen 的传统滤波更能压噪 —— 这正是
我们敢降低射线数的底气.

同时看与 GT 的误差 (MSE / relative L2), 确认没有为了平滑而偏离真值.

跑法:
    python eval/eval_smoothness.py runs/fixed_hist/best.pt D:/ProbeDump_S2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "dataset"))
sys.path.insert(0, str(ROOT / "train"))

from probe_dataset import ProbeSequence, ProbeTexelDataset  # noqa: E402
from model import ProbeFilterMLP  # noqa: E402


def smoothness(atlas: np.ndarray, g: int) -> float:
    h, w = atlas.shape[:2]
    ph, pw = h // g, w // g
    p = atlas.reshape(ph, g, pw, g, -1).transpose(0, 2, 1, 3, 4)
    d = np.abs(np.diff(p, axis=2)).mean()
    m = np.abs(p).mean() + 1e-6
    return float(d / m)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("data")
    ap.add_argument("--gt-dir", type=str, default="",
                    help="GT 目录 (方案 B: 高射线采集目录). "
                         "空 = 方案 A (GT 与输入同目录)")
    ap.add_argument("--gt-use-in", action="store_true",
                    help="对比【未滤波】的 in 而不是 gt —— 评估方案 B''(纯射线提升) 时用")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1,
                    help="每隔多少帧取一个评估样本(默认 1). "
                         "采集时每个位置会重复 hold 次, stride 太小会只评估到"
                         "同一个位置的若干张重复帧, 结果不具代表性 —— "
                         "建议 stride >= hold")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    hidden = a.get("hidden", 64)
    layers = a.get("layers", 4)
    no_hist = a.get("no_history", False)
    residual = not a.get("no_residual", False)

    use_history = not no_hist
    in_dim = 56 if use_history else 52

    model = ProbeFilterMLP(in_dim=in_dim, hidden=hidden,
                           num_layers=layers, use_residual=residual)
    model.load_state_dict(ck["model"])
    model.to(args.device).eval()
    print(f"模型: {args.ckpt}")
    print(f"  hidden={hidden} layers={layers} in_dim={in_dim} 残差={residual}")
    if "loss" in ck:
        print(f"  训练时最佳 loss={ck['loss']:.6f}")

    # 方案 B: gt_dir 指向高射线采集目录, GT 从那里取
    # 注意 max_frames=0 (全部): 配对表要覆盖所有帧, 否则评估到第 N 帧时
    # 配对表里没有对应 key 会 KeyError.
    ds = ProbeTexelDataset(args.data, samples_per_frame=1,
                           use_history=use_history, max_frames=0,
                           gt_dir=args.gt_dir if args.gt_dir else None,
                           gt_use_in=args.gt_use_in)
    seq = ds.seq
    g = seq.gather

    if ds.gt_seq is not None:
        print(f"  GT 目录: {args.gt_dir}  (方案 B: 高射线)")
    else:
        print("  GT 目录: 与输入同目录 (方案 A)")

    print()
    print(f"{'帧':<14}{'输入':>9}{'GT(Lumen)':>11}{'模型输出':>11}"
          f"{'vs GT':>9}{'MSE':>10}{'relL2':>9}")
    print("-" * 74)
    print("  (MSE/relL2 为模型输出与 GT 的误差; 末尾会给出恒等映射的 baseline)")

    s_in_all, s_gt_all, s_out_all = [], [], []
    base_mse_all, base_rel_all = [], []

    with torch.no_grad():
        # 按 stride 采样: 若 stride=1 而每个位置重复 hold 次,
        # 会只评估到同一位置的若干张重复帧, 结果不具代表性.
        eval_idx = list(range(1, len(seq), max(1, args.stride)))[:args.frames]
        if not eval_idx:
            eval_idx = [1]

        for i in eval_idx:
            cur = seq.load_frame(i)
            prev = seq.load_frame(i - 1) if use_history else None
            feat, target = ds._build_features(cur, prev)

            h, w = feat.shape[:2]
            valid = np.repeat(np.repeat(cur["valid"], g, axis=0), g, axis=1)

            x = torch.from_numpy(feat.reshape(-1, feat.shape[-1])).to(args.device)
            out = model(x).cpu().numpy().reshape(h, w, 3)

            # 反归一化: expm1
            out_lin = np.expm1(out)
            in_lin = cur["in"]
            # 方案 B: 答案取高射线目录的 gt; 方案 A: 取同目录的 gt
            # 注意要走相机矩阵配对表, 否则两遍采样密度不同时会取错帧
            # 方案 B'' 对比未滤波的 in; 其他情况对比已滤波的 gt
            key = "in" if args.gt_use_in else "gt"
            if ds.gt_seq is not None:
                # .get 兜底: 万一配对表没覆盖这一帧, 退回按同帧号取
                gi = ds.gt_index_map.get(i, i) if ds.gt_index_map else i
                gt_lin = ds.gt_seq.load_frame(gi)[key]
            else:
                gt_lin = cur[key]

            s_in = smoothness(in_lin, g)
            s_gt = smoothness(gt_lin, g)
            s_out = smoothness(out_lin, g)
            s_in_all.append(s_in)
            s_gt_all.append(s_gt)
            s_out_all.append(s_out)

            # 与 GT 的误差 (只统计有效 texel)
            m = valid.astype(bool)
            mse = float(((out_lin[m] - gt_lin[m]) ** 2).mean())
            rel = float((((out_lin[m] - gt_lin[m]) ** 2) /
                         (gt_lin[m] ** 2 + 0.1)).mean())

            # baseline: 恒等映射 (直接拿未滤波的输入当输出)
            base_mse_all.append(
                float(((in_lin[m] - gt_lin[m]) ** 2).mean()))
            base_rel_all.append(
                float((((in_lin[m] - gt_lin[m]) ** 2) /
                       (gt_lin[m] ** 2 + 0.1)).mean()))

            better = (s_gt - s_out) / s_gt * 100
            print(f"{cur['base']:<14}{s_in:>9.3f}{s_gt:>11.3f}{s_out:>11.3f}"
                  f"{better:>8.1f}%{mse:>10.4f}{rel:>9.4f}")

    print("-" * 74)
    si, sg, so = np.mean(s_in_all), np.mean(s_gt_all), np.mean(s_out_all)
    print(f"{'平均':<14}{si:>9.3f}{sg:>11.3f}{so:>11.3f}"
          f"{(sg - so) / sg * 100:>8.1f}%")

    bm, br = float(np.mean(base_mse_all)), float(np.mean(base_rel_all))
    mm = float(np.mean([0]))  # 占位, 下面重算
    print()
    print("baseline 对照 (恒等映射 = 直接把未滤波输入当输出):")
    print(f"  恒等映射 MSE = {bm:.4f}   relL2 = {br:.4f}")
    print(f"  模型输出  MSE = {mse:.4f}   relL2 = {rel:.4f}")
    if bm > 0:
        print(f"  -> 误差降低 {(bm - mse) / bm * 100:.1f}% (MSE) / "
              f"{(br - rel) / br * 100:.1f}% (relL2)")
        if mse < bm:
            print("  -> 模型确实学到了东西, 不是靠'什么都不做'混过去的")
        else:
            print("  -> [警告] 模型误差未优于恒等映射! 检查训练")

    print()
    print("判读:")

    if ds.gt_seq is not None:
        # ---- 方案 B' ----
        # 目标不是"比 GT 更平滑", 而是"从输入的低射线质量, 逼近 GT 的高射线质量".
        # 所以看的是: 输入->GT 的总差距, 模型补上了多少.
        si = float(np.mean(s_in_all))
        gap_total = si - sg      # 输入到 GT 的总差距
        gap_done = si - so       # 模型已经补上的部分

        print(f"  输入(低射线)     平滑度 {si:.3f}")
        print(f"  模型输出         平滑度 {so:.3f}")
        print(f"  GT(高射线)       平滑度 {sg:.3f}")
        print(f"  ----")
        print(f"  输入->GT 总差距  {gap_total:.3f}")
        print(f"  模型已补上       {gap_done:.3f}")

        if gap_total > 1e-6:
            pct = gap_done / gap_total * 100
            print(f"  => 完成度 {pct:.1f}%")
            print()
            if so < sg - 0.02:
                print("  [注意] 输出比 GT 明显更平滑 -> 可能过度平滑(抹掉细节),")
                print("         必须结合 MSE/relL2 一起判断")
            elif pct >= 80:
                print("  [很好] 模型补上了绝大部分 '低射线 -> 高射线' 的差距")
            elif pct >= 50:
                print("  [尚可] 补上了大部分差距; 可增大模型容量或增加训练数据")
            else:
                print("  [不足] 只补上一小部分, 检查数据配对 / 训练配置")
    else:
        # ---- 方案 A ----
        if so < sg:
            print(f"  模型输出比 Lumen 的 GT 更平滑 {(sg - so) / sg * 100:.1f}%")
            print("  -> 说明网络压噪能力强于传统滤波, 有资格减少射线数")
        else:
            print(f"  模型输出平滑度 {so:.3f} 未优于 GT {sg:.3f}")
            print("  -> 网络还只是'模仿'Lumen, 需要先提升容量或训练更久")

    print()
    print("  注意: 平滑不是唯一目标. 若 MSE/relL2 很大, 说明过度平滑")
    print("        (把细节抹掉了), 需要配合误差一起看.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
