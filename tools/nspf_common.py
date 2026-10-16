"""NSPF 共享工具函数.

目的
----
之前这些函数在各脚本里各写一份：
    smoothness  重复 4 次 (run_infer / visualize_b / eval_smoothness / infer)
    tonemap     重复 2 次 (run_infer / visualize_b)
    load_model  重复 2 次 (ab_compare / infer)
    matplotlib 中文字体设置 重复多次

现在统一收敛到这里，各脚本 import 即可，主体逻辑保持精简。

用法
----
    from nspf_common import (
        setup_matplotlib, tonemap, smoothness,
        load_model, build_dataset, to_linear, save_compare_figure,
    )
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in ("tools/dataset", "train"):
    if str(ROOT / p) not in sys.path:
        sys.path.insert(0, str(ROOT / p))

from probe_dataset import ProbeTexelDataset  # noqa: E402
from model import ProbeFilterMLP  # noqa: E402


# ---------------------------------------------------------------- 绘图 ---
def setup_matplotlib() -> None:
    """Agg 后端 + 中文字体（找不到中文字体时静默降级为默认字体）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name in ("Microsoft YaHei", "SimHei", "DengXian", "SimSun"):
        try:
            matplotlib.font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [name]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def tonemap(x: np.ndarray) -> np.ndarray:
    """HDR -> 可显示：按 p99 截断 + gamma 2.2。"""
    x = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    hi = float(np.percentile(x, 99))
    if hi <= 1e-8:
        hi = float(x.max()) if x.max() > 1e-8 else 1.0
    return np.clip(x / hi, 0.0, 1.0) ** (1.0 / 2.2)


def save_compare_figure(images: list[np.ndarray], titles: list[str],
                        head: str, out_path: Path) -> None:
    """横向排布多张图并保存（统一风格，避免每个脚本各写一套）。"""
    setup_matplotlib()
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.6))
    if n == 1:
        axes = [axes]
    for ax, img, t in zip(axes, images, titles):
        ax.imshow(tonemap(img))
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    fig.suptitle(head, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- 指标 ---
def smoothness(atlas: np.ndarray, g: int) -> float:
    """probe 内部方向维度的平滑度（越低越干净）。

    atlas: (H, W, C)，g = gather 分辨率（每个 probe 是 g x g 个方向格子）
    """
    h, w = atlas.shape[:2]
    ph, pw = h // g, w // g
    p = atlas.reshape(ph, g, pw, g, -1).transpose(0, 2, 1, 3, 4)
    d = np.abs(np.diff(p, axis=2)).mean()
    m = np.abs(p).mean() + 1e-6
    return float(d / m)


def to_linear(x: np.ndarray) -> np.ndarray:
    """log1p 域 -> 线性域（等价于 np.expm1，单独封装便于统一）。"""
    return np.expm1(x)


def rel_l2(pred: np.ndarray, gt: np.ndarray, eps: float = 0.1) -> float:
    """relative L2（与 losses.py 保持一致的分母语义）。"""
    return float((((pred - gt) ** 2) / (gt ** 2 + eps)).mean())


def mse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(((pred - gt) ** 2).mean())


# ------------------------------------------------------------ 模型/数据 ---
def load_model(ckpt_path: str | Path, device: str = "cpu") -> ProbeFilterMLP:
    """加载 checkpoint，自动推断 in_dim（兼容 52 无历史 / 56 含历史）。

    兼容保存格式：直接 state_dict，或 {'model': ...} / {'state_dict': ...}
    """
    state = torch.load(str(ckpt_path), map_location="cpu")

    sd = state
    if isinstance(state, dict):
        for key in ("model", "state_dict", "model_state", "net"):
            if key in state and isinstance(state[key], dict):
                sd = state[key]
                break

    # 从第一层 Linear 的 weight 推断输入维度: (hidden, in_dim)
    w0 = sd.get("net.0.weight")
    in_dim = int(w0.shape[1]) if w0 is not None else 52

    model = ProbeFilterMLP(in_dim=in_dim)
    model.load_state_dict(sd)
    model.eval()
    return model.to(device)


def build_dataset(low_dir: str, gt_dir: str | None = None,
                  use_history: bool = False,
                  samples_per_frame: int = 1,
                  max_frames: int = 0) -> ProbeTexelDataset:
    """构建数据集（gt_dir 为空 = 方案 A，否则 = 方案 B）。"""
    return ProbeTexelDataset(
        low_dir,
        samples_per_frame=samples_per_frame,
        use_history=use_history,
        max_frames=max_frames,
        gt_dir=gt_dir if gt_dir else None,
    )
