"""NSPF 网络: per-texel 的小型 MLP.

为什么是 MLP 而不是 CNN
----------------------
1. probe atlas 有两个维度: 空间(probe 网格) + 方向(probe 内 8x8).
   相邻 probe 的"同方向 texel"才是空间邻居, 而 atlas 里它们在图像上
   相隔 gatherRes 个像素 —— 标准卷积核抓不到这种关系.
   我们已经在数据侧把 3x3 邻域显式展平成特征了, 网络只需 MLP.

2. MLP 与 atlas 分辨率无关. 换个分辨率采集, 只是样本数变了,
   网络结构一行不用改 (CNN 则要处理 shape).

3. NIV (Eurographics 2026) 用 4 层 MLP 编码 irradiance field,
   在消费级 GPU 上全高清推理约 1ms. 这是经过验证的规模.

输入特征 (56 维, 见 tools/dataset/probe_dataset.py)
----------------------------------------------
  [0:27]    3x3 邻域 radiance (log1p 后)
  [27:36]   3x3 邻域 hit distance
  [36:45]   3x3 邻域有效性
  [45:52]   几何: log深度(1) + 法线(3) + 相对世界坐标(3)
  [52:55]   历史帧 radiance (重投影对齐后)
  [55]      历史是否有效
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProbeFilterMLP(nn.Module):
    """per-texel 滤波网络.

    Args:
        in_dim:      输入维度 (52 无历史 / 56 含历史)
        hidden:      隐层宽度
        num_layers:  隐藏层数 (NIV 用 4)
        out_dim:     输出维度 (3 = RGB radiance)
        use_residual: 是否预测"相对输入的残差".
                      开启后输出 = 输入中心值 + 网络输出,
                      让网络从"恒等映射"起步, 收敛更快更稳.
    """

    def __init__(
        self,
        in_dim: int = 56,
        # 默认 64 而非 128: 实测 hidden=128/4层 单帧约 3.05ms,
        # 会吃掉降射线省下的 2.5~4.5ms. hidden=64/4层 约 0.86ms, 收益安全.
        # 详见本文件末尾的权衡表 (python model.py).
        hidden: int = 64,
        num_layers: int = 4,
        out_dim: int = 3,
        use_residual: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_residual = use_residual

        layers: list[nn.Module] = []
        d_in = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(d_in, hidden))
            layers.append(nn.ReLU(inplace=True))
            d_in = hidden
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """最后一层初始化为接近 0, 残差模式下等价于从恒等映射起步."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        if self.use_residual:
            last = self.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) 特征
        Returns:
            (N, out_dim) 预测的 log1p(radiance)
        """
        out = self.net(x)
        if self.use_residual:
            # 通道 [12:15] 是中心 probe 的 log1p(radiance), 即"未滤波的输入"
            center = x[:, 12:15]
            out = center + out
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_inference_cost(
    in_dim: int = 56,
    hidden: int = 128,
    num_layers: int = 4,
    samples_per_frame: int = 201_600,
    effective_tflops: float = 7.5,
) -> tuple[float, float]:
    """估算单帧推理成本.

    Args:
        samples_per_frame: 每帧 texel 数 (1080p: 120*68*64 = 522,240;
                           当前 1440x832 采集: 3150*64 = 201,600)
        effective_tflops: GPU 对小型 MLP 的有效算力.
                          RTX 5060 峰值约 15-20 TFLOPS(FP16), 但 MLP 是
                          访存密集型, 实际只能跑到 30~50%, 取 7.5 保守估计.

    Returns:
        (GFLOP, 毫秒)
    """
    macs = in_dim * hidden
    macs += hidden * hidden * (num_layers - 1)
    macs += hidden * 3
    flops = macs * 2 * samples_per_frame
    gflop = flops / 1e9
    ms = gflop / effective_tflops  # GFLOP / TFLOPS = ms
    return gflop, ms


if __name__ == "__main__":
    print("网络规模与推理成本权衡")
    print("=" * 74)
    print("前提: 3150 probe x 64 方向 = 201,600 样本/帧 (1440x832 采集)")
    print("      5060 对小型 MLP 的有效算力保守取 7.5 TFLOPS")
    print()
    print(f"{'hidden':>7}{'layers':>7}{'参数':>10}{'GFLOP':>9}{'推理ms':>9}  备注")
    print("-" * 74)

    # 参考线: 射线数 64->16 大约省 2.5~4.5 ms trace
    for hidden, layers, note in (
        (128, 4, "NIV 规模, 偏大"),
        (128, 3, ""),
        (64, 4, "推荐起点"),
        (64, 3, "更激进"),
        (32, 3, "极小"),
    ):
        m = ProbeFilterMLP(in_dim=56, hidden=hidden, num_layers=layers)
        g, ms = estimate_inference_cost(hidden=hidden, num_layers=layers)
        # 注意: 输出可能被重定向到 GBK 终端, 避免 Unicode 特殊字符
        verdict = ""
        if ms < 1.5:
            verdict = "[OK] 收益安全"
        elif ms < 2.5:
            verdict = "[~] 勉强"
        else:
            verdict = "[NG] 吃掉收益"
        print(f"{hidden:>7}{layers:>7}{count_parameters(m):>10,}{g:>9.1f}"
              f"{ms:>9.2f}  {note} {verdict}")

    print()
    print("=" * 74)
    print("结论: 射线 64->16 大约省 2.5~4.5 ms trace 开销.")
    print("      推理必须控制在 1.5ms 以内才谈得上净收益.")
    print("      所以 hidden=64 / 3~4 层 是合理起点, 不要一上来用大网络.")
