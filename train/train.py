"""NSPF 训练主循环.

跑法
----
    # 最小验证 (几十秒, 确认能跑通)
    python train/train.py --data D:/ProbeDump_S2 --epochs 1 --max-frames 10

    # 正式训练
    python train/train.py --data D:/ProbeDump_S2 --epochs 30 --hidden 64

    # 对照: 不用历史帧 / 用 MSE
    python train/train.py --data D:/ProbeDump_S2 --no-history
    python train/train.py --data D:/ProbeDump_S2 --loss mse

设计说明
--------
1. 一帧 = 一个 step (不是遍历所有 201,600 个 texel, 而是每帧随机采样 N 个).
   这样显存可控, 且每 step 采不同子集自带增广.

2. 默认 hidden=64: 推理约 0.86ms/帧, 才谈得上净收益.
   详见 model.py 末尾的权衡表.

3. loss 默认 relative_l2 (eps=0.1), HDR 数据的正确选择, 依据见 losses.py.

4. 训练很快: 网络只有 1.6 万参数, 一帧 8192 个样本, 5060 上几秒一个 epoch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "dataset"))
sys.path.insert(0, str(ROOT / "train"))

from probe_dataset import ProbeTexelDataset  # noqa: E402
from model import ProbeFilterMLP, count_parameters  # noqa: E402
from losses import build_loss, DarkPreserveLoss  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NSPF 训练")
    p.add_argument("--data", type=str, required=True, help="采集目录")
    p.add_argument("--out", type=str, default="", help="输出目录 (默认 runs/<时间戳>)")

    # 数据
    p.add_argument("--samples-per-frame", type=int, default=8192,
                   help="每帧采样多少 texel 组成一个 step")
    p.add_argument("--max-frames", type=int, default=0, help="只用前 N 帧 (0=全部)")
    p.add_argument("--gt-dir", type=str, default="",
                   help="GT 目录（方案 B：高射线采集目录；空 = 方案 A：输入和 GT 同目录）")
    p.add_argument("--gt-use-in", action="store_true",
                   help="答案取 GT 目录的 in(未滤波) 而不是 gt(已滤波). "
                        "用于方案 B''(纯射线提升): 两边都未滤波, 滤波影响被抵消. "
                        "配合 --gt-dir 使用")
    p.add_argument("--dark-preserve", action="store_true",
                   help="启用暗部保持损失(实验01 E4): 输入越暗, 越强地把输出"
                        "拉回输入, 强制黑入黑出. 治'暗部被涂成褐色'")
    p.add_argument("--dark-weight", type=float, default=0.5,
                   help="暗部保持损失的权重")
    p.add_argument("--no-history", action="store_true", help="不使用历史帧特征")
    p.add_argument("--history-mode", type=str, default="aligned",
                   choices=["aligned", "naive"],
                   help="aligned=相机重投影对齐; naive=按 probe 索引直接取")

    # 模型
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--no-residual", action="store_true",
                   help="关闭残差模式 (默认开启: 输出=输入中心+网络增量)")

    # 训练
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", type=str, default="relative_l2",
                   choices=["relative_l2", "mse"])
    p.add_argument("--eps", type=float, default=0.1, help="relative L2 的 eps")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # 运行
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=0, help="每 N epoch 存一次 (0=只存最后)")
    p.add_argument("--device", type=str, default="auto")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ---- 设备 ----
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"设备: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    torch.manual_seed(args.seed)

    # ---- 输出目录 ----
    if args.out:
        out_dir = Path(args.out)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "runs" / f"h{args.hidden}_l{args.layers}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出: {out_dir}")

    # ---- 数据 ----
    use_history = not args.no_history
    data_dirs = [d.strip() for d in args.data.split(",") if d.strip()]
    # gt-dir 与 data 一一对应(逗号分隔): 各场景的 GT 在各自的高射线目录,
    # 若共用一个 gt-dir, 另一个场景会按相机矩阵"硬配"到完全不同的轨迹上,
    # 拿到错误的 GT
    gt_dirs = [d.strip() for d in args.gt_dir.split(",") if d.strip()] \
        if args.gt_dir else []
    datasets = []
    for k, d in enumerate(data_dirs):
        gt_d = gt_dirs[k] if k < len(gt_dirs) else \
            (gt_dirs[0] if gt_dirs else "")
        datasets.append(ProbeTexelDataset(
            d,
            samples_per_frame=args.samples_per_frame,
            use_history=use_history,
            history_mode=args.history_mode,
            max_frames=args.max_frames,
            # ★ 下面两个之前漏传了 —— 导致 --gt-dir 形同虚设,
            #   一直退化成方案 A(用同目录的 gt 做答案)
            gt_dir=gt_d if gt_d else None,
            gt_use_in=args.gt_use_in,
        ))
    if len(datasets) == 1:
        ds = datasets[0]
    else:
        # 多场景混合训练: 各场景独立建 Dataset(各自做相机配对), 再拼接.
        # DataLoader batch_size=1, 每次取一帧, 不同 atlas 尺寸可共存.
        from torch.utils.data import ConcatDataset
        ds = ConcatDataset(datasets)
        print(f"  混合 {len(datasets)} 个场景: "
              + " + ".join(str(len(d)) for d in datasets))
    print(f"数据: {len(ds)} 帧可用 (源 {args.data})")
    print(f"  每帧采样 {args.samples_per_frame} texel, "
          f"历史={'关' if not use_history else args.history_mode}")

    # 明确打印当前是哪个方案, 避免再出现"以为在跑 B 实际是 A"
    if args.gt_dir:
        scheme = "B''(纯射线提升, 答案=高射线未滤波)" if args.gt_use_in \
            else "B'(滤波+射线, 答案=高射线已滤波)"
        print(f"  方案: {scheme}")
        print(f"  GT 目录: {args.gt_dir}")
    else:
        print("  方案: A(自配对, 答案=同目录已滤波)")

    dl = DataLoader(ds, batch_size=1, shuffle=True,
                    num_workers=args.num_workers, pin_memory=(device == "cuda"))

    in_dim = 56 if use_history else 52
    model = ProbeFilterMLP(
        in_dim=in_dim,
        hidden=args.hidden,
        num_layers=args.layers,
        use_residual=not args.no_residual,
    ).to(device)
    print(f"模型: hidden={args.hidden} layers={args.layers} "
          f"in_dim={in_dim} 参数={count_parameters(model):,}")

    criterion = build_loss(args.loss, eps=args.eps) if args.loss == "relative_l2" \
        else build_loss(args.loss)
    print(f"损失: {args.loss}" + (f" (eps={args.eps})" if args.loss == "relative_l2" else ""))

    dark_crit = None
    if args.dark_preserve:
        dark_crit = DarkPreserveLoss(weight=args.dark_weight)
        print(f"  + 暗部保持损失 (weight={args.dark_weight}, k=3.0) "
              f"—— 实验01 E4: 黑入黑出")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    # ---- 训练 ----
    log_path = out_dir / "train_log.jsonl"
    history: list[dict] = []

    print()
    print("=" * 66)
    print("开始训练")
    print("=" * 66)

    best = float("inf")
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_ep = time.time()

        for i, batch in enumerate(dl):
            feat = batch["feat"].squeeze(0).to(device, non_blocking=True)
            target = batch["target"].squeeze(0).to(device, non_blocking=True)

            if feat.shape[-1] != in_dim:
                raise RuntimeError(
                    f"特征维度不符: 数据 {feat.shape[-1]}, 模型期望 {in_dim}. "
                    f"检查 --no-history 是否与数据一致")

            pred = model(feat)
            loss = criterion(pred, target)
            if dark_crit is not None:
                # feat[:, 12:15] = 输入中心 texel (残差基准).
                # 暗部时把它往输出上"锚": 黑入必须黑出
                loss = loss + dark_crit(pred, feat[:, 12:15])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            epoch_loss += float(loss.detach())
            n_batches += 1

            if (i + 1) % args.log_every == 0:
                avg = epoch_loss / max(1, n_batches)
                print(f"  epoch {epoch+1}/{args.epochs} "
                      f"step {i+1}/{len(dl)}  loss={avg:.6f}", flush=True)

        sched.step()
        avg_loss = epoch_loss / max(1, n_batches)
        dt = time.time() - t_ep

        rec = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "lr": sched.get_last_lr()[0],
            "seconds": round(dt, 3),
        }
        history.append(rec)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        mark = ""
        if avg_loss < best:
            best = avg_loss
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch + 1,
                "loss": avg_loss,
            }, out_dir / "best.pt")
            mark = " *best*"

        print(f"[epoch {epoch+1:>3}/{args.epochs}] loss={avg_loss:.6f} "
              f"lr={rec['lr']:.2e} {dt:.1f}s{mark}", flush=True)

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "epoch": epoch + 1}, out_dir / f"ep{epoch+1:03d}.pt")

    torch.save({"model": model.state_dict(), "args": vars(args),
                "epoch": args.epochs}, out_dir / "last.pt")

    total = time.time() - t_start
    print()
    print("=" * 66)
    print(f"完成: {args.epochs} epochs, 用时 {total:.1f}s")
    print(f"最佳 loss: {best:.6f}")
    print(f"日志: {log_path}")
    print(f"权重: {out_dir / 'best.pt'}")
    print("=" * 66)

    # 训练完给个提示: 下一步该看什么
    print()
    print("下一步:")
    print(f"  1. 查看 loss 曲线是否下降: {log_path}")
    print(f"  2. 跑对照实验确认各部件贡献:")
    print(f"       python train/train.py --data {args.data} --no-history   # 去掉历史帧")
    print(f"       python train/train.py --data {args.data} --loss mse     # 换 MSE")
    print("  3. 评估输出平滑度是否优于 GT (见 README)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
