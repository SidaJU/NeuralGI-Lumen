"""NSPF 损失函数.

为什么不能用 MSE
----------------
probe radiance 是 HDR 线性值, 分布极度长尾:
    p50 = 0.029,  p99 = 2.6,  max = 15.75  (log1p 归一化后)
亮部样本虽然只占 1%, 但其平方误差会主导整个 loss, 导致网络
只顾拟合亮部、暗部全部学废.

relative L2 (NIV, Eurographics 2026)
-------------------------------------
    L = (pred - gt)^2 / (sg(pred)^2 + eps)

分母用 pred 的 stop-gradient. 这样:
  - 亮部: 分母大 -> 误差被缩小, 不会主导
  - 暗部: 分母小 -> 误差被放大, 网络必须认真学
等价于对所有亮度量级"一视同仁", 这正是 HDR 需要的.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RelativeL2Loss(nn.Module):
    """relative L2 (NIV 公式).

    Args:
        eps: 分母稳定项, NIV 用 0.01. 我们的值域已经 log1p 压到 O(1),
             用 0.01 依然合适.
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, eps: float = 0.1, reduction: str = "mean"):
        super().__init__()
        # eps 取值依据 (用真实 GT 统计, 非拍脑袋):
        #   log1p(gt)^2 分位数: p25=0.000, p50=0.003, p95=0.817, p99=1.813
        #   eps=0.01 (NIV 原值) -> 暗部放大 76.7x, 亮部 1.21x, 比值 63
        #   eps=0.1            -> 暗部放大  9.7x, 亮部 1.09x, 比值 8.9  <- 选这个
        #   eps=0.5            -> 暗部放大  2.0x, 亮部 0.76x, 比值 2.6
        # 另外 25% 的格子是背面(值恒为 0): eps 太小会让这些样本 loss 快速
        # 饱和到 1 导致梯度消失, 0.1 可以兼顾暗部学习与零值样本的稳定性.
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        denom = pred.detach() ** 2 + self.eps
        loss = (pred - target) ** 2 / denom
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MaskedMSELoss(nn.Module):
    """普通 MSE, 作为对照基线. 预期效果会明显差于 relative L2."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = (pred - target) ** 2
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class TemporalConsistencyLoss(nn.Module):
    """帧间一致性约束 (S2 之后启用).

    网络每帧独立推理, 如果输出在帧间抖动, 即使单帧指标好看,
    实机看起来也是闪烁 —— 这比 Lumen 原本的鬼影还糟.

    用法: 需要同一场景相邻两帧的输入, 比较网络输出的差异.
    要求两次推理的样本在空间上对应 (用 wpos 或 probe 索引配对).

    注意: 第一版训练先不开, 等单帧 loss 稳定收敛后再加,
    否则两个目标互相干扰, 很难判断问题出在哪.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, out_t: torch.Tensor, out_t_prev_warped: torch.Tensor,
                valid: torch.Tensor | None = None) -> torch.Tensor:
        diff = (out_t - out_t_prev_warped).abs()
        if valid is not None:
            diff = diff * valid
            n = valid.sum().clamp(min=1.0)
            return self.weight * diff.sum() / n
        return self.weight * diff.mean()


class DarkPreserveLoss(nn.Module):
    """暗部保持: 输入越暗, 越强地惩罚【输出偏离输入】(黑入黑出).

    为什么需要 (实验01-失真修复, H1)
    --------------------------------
    模型是残差结构: pred = center + delta (log1p 域), 其中 center 是
    输入中心 texel. 暗部 center≈0, 输出完全由 delta 决定 —— 训练数据
    单一时, 分布外(OOD)的暗部输入会被输出成【训练集的均值色】
    (实测: 新场景的黑色暗部被涂成褐色).

    本损失在暗部直接把 delta 往 0 拉, 强制"黑入黑出":
        输入接近黑 -> 输出必须接近黑, 不给模型猜颜色的空间.

    用法: 作为附加项与主损失相加, center 取 feat[:, 12:15].
    """

    def __init__(self, weight: float = 0.5, k: float = 3.0):
        super().__init__()
        self.weight = weight
        # 权重随亮度指数衰减: 亮度 0 -> 权重 1; 亮度 1 -> e^-3≈0.05;
        # 亮度 2 以上基本不管. 只治"接近纯黑"的区域, 不伤正常暗部细节.
        self.k = k

    def forward(self, pred: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        c_luma = center.mean(dim=-1, keepdim=True)          # (N, 1)
        w = torch.exp(-c_luma.clamp(min=0.0) * self.k)      # 越暗权重越大
        return self.weight * (w * (pred - center) ** 2).mean()


def build_loss(name: str = "relative_l2", **kwargs) -> nn.Module:
    """工厂函数."""
    name = name.lower()
    if name in ("relative_l2", "rrel", "relative"):
        return RelativeL2Loss(**kwargs)
    if name in ("mse", "l2"):
        return MaskedMSELoss(**kwargs)
    raise ValueError(f"未知损失: {name}")


if __name__ == "__main__":
    # 演示: 同样的误差, 在亮部和暗部被 relative L2 如何区别对待
    pred = torch.tensor([[0.05], [3.0]])
    gt = torch.tensor([[0.10], [3.2]])

    mse = MaskedMSELoss()
    rl2 = RelativeL2Loss()

    print("样本         pred    gt     |err|    MSE贡献    RelativeL2贡献")
    print("-" * 62)
    for i, tag in enumerate(("暗部", "亮部")):
        e = (pred[i] - gt[i]).abs().item()
        m = mse(pred[i:i + 1], gt[i:i + 1]).item()
        r = rl2(pred[i:i + 1], gt[i:i + 1]).item()
        print(f"  {tag:6}  {pred[i].item():6.2f} {gt[i].item():6.2f} "
              f"{e:7.3f}   {m:9.4f}   {r:9.4f}")

    print()
    # 注意: 必须用"各项贡献 / 贡献之和", 不能用"单项 / 平均值"
    # (后者分子分母量纲不一致, 会算出 >100% 的荒谬结果)
    m_d = mse(pred[0:1], gt[0:1]).item()
    m_b = mse(pred[1:2], gt[1:2]).item()
    r_d = rl2(pred[0:1], gt[0:1]).item()
    r_b = rl2(pred[1:2], gt[1:2]).item()
    print(f"MSE        暗部占比 {m_d / (m_d + m_b):5.1%}   "
          f"亮部占比 {m_b / (m_d + m_b):5.1%}")
    print(f"RelativeL2 暗部占比 {r_d / (r_d + r_b):5.1%}   "
          f"亮部占比 {r_b / (r_d + r_b):5.1%}")
    print()
    print("-> MSE 被亮部主导, relative L2 把两者拉到同一量级")
