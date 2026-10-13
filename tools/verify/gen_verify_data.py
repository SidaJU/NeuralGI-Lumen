"""生成 NNE 验证数据: 真实输入(56维) + torch 期望输出, 存为 raw float32.

用法:
    python tools/verify/gen_verify_data.py <ckpt> <data_dir> [num_samples]

产出:
    verify_input.f32     (N, 56) float32
    verify_expected.f32  (N, 3)  float32   <- torch(FP32) 的参考输出
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "dataset"))
sys.path.insert(0, str(ROOT / "train"))

from probe_dataset import ProbeTexelDataset  # noqa: E402
from model import ProbeFilterMLP             # noqa: E402


def main() -> int:
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "runs/b_full/best.pt"
    data_dir = sys.argv[2] if len(sys.argv) > 2 else "D:/ProbeDump_B_low"
    num = int(sys.argv[3]) if len(sys.argv) > 3 else 1024

    ds = ProbeTexelDataset(data_dir, samples_per_frame=num,
                           use_history=True, max_frames=1)
    cur = ds.seq.load_frame(1)
    prev = ds.seq.load_frame(0)
    feat, _target = ds._build_features(cur, prev)
    feat = feat.reshape(-1, feat.shape[-1]).astype(np.float32)

    # 取前 num 个有效 texel (valid 是 probe 级, 展开到 texel 级)
    g = ds.seq.gather
    valid = np.repeat(np.repeat(cur["valid"], g, axis=0), g, axis=1)
    valid = valid.reshape(-1)[:feat.shape[0]]
    idx = np.where(valid)[0][:num]
    feat = feat[idx]
    print(f"feat: {feat.shape}, 有效 texel 取 {len(idx)}")

    # torch 参考输出
    ck = torch.load(ROOT / ckpt_path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    model = ProbeFilterMLP(in_dim=56, hidden=a.get("hidden", 64),
                           num_layers=a.get("layers", 4),
                           use_residual=not a.get("no_residual", False))
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(feat)).numpy().astype(np.float32)

    feat.tofile("verify_input.f32")
    out.tofile("verify_expected.f32")
    print(f"saved verify_input.f32    ({feat.nbytes / 1024:.0f} KB)")
    print(f"saved verify_expected.f32 ({out.nbytes / 1024:.0f} KB)")
    print("sample out[0]:", out[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
