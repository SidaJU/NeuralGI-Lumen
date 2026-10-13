"""NSPF 数据加载器: 把引擎 dump 的 probe 数据组织成可训练样本.

设计要点
--------
1. 网络是 per-texel 的: 每个 probe 内部 8x8 的一个方向格子 = 一个训练样本.
   一帧有 probeH*probeW*64 个样本 (典型 3150*64 = 201,600).

2. 帧级 Dataset + 帧内随机采样:
   一次性读入一帧, 每个 step 从该帧随机采 N 个 texel 组成 batch.
   这样既省内存 (不用预先展开全部样本), 每步采不同子集又自带增广.

3. 时域对齐用相机重投影:
   当前 probe 的世界位置 -> 用历史帧的 ViewProj 投影到历史屏幕 ->
   取对应的历史 probe. 这是 Lumen 自己时域滤波的做法.

4. 几何解码放在 Python 侧 (引擎只存原始字节):
     depth : uint32 -> view(float32)   (asfloat)
     normal: uint8 x2 八面体编码 -> float32 x3
     wpos  : float32 x4 直接可用
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# ----------------------------------------------------------------- 工具函数 --

def depth_from_uint(arr: np.ndarray) -> np.ndarray:
    """PF_R32_UINT -> float32 (asfloat bitcast)."""
    return arr.view(np.float32)


def octahedral_decode(e: np.ndarray) -> np.ndarray:
    """八面体编码 (2 通道, [0,1]) -> 单位法线 (3 通道).

    参考 UE 的 OctahedralDecode (ShaderCommon.ush):
        e 是 [0,1] 的两通道, 先映射到 [-1,1] 再做八面体反映射.
    """
    e = np.asarray(e, dtype=np.float32) * 2.0 - 1.0
    n = np.empty(e.shape[:-1] + (3,), dtype=np.float32)

    x, y = e[..., 0], e[..., 1]
    z = 1.0 - np.abs(x) - np.abs(y)

    # 反映射: 把落在八面体下半球的部分翻上来
    xx = np.where(z >= 0.0, x, (1.0 - np.abs(y)) * np.sign(x + 1e-12))
    yy = np.where(z >= 0.0, y, (1.0 - np.abs(x)) * np.sign(y + 1e-12))

    n[..., 0] = xx
    n[..., 1] = yy
    n[..., 2] = z

    # 归一化 (数值稳定)
    length = np.sqrt(np.maximum((n ** 2).sum(axis=-1, keepdims=True), 1e-12))
    return n / length


def log_normalize(x: np.ndarray) -> np.ndarray:
    """HDR radiance -> O(1) 范围.

    实测有效区值域 0~15.75, p50=0.029, p99=2.6 (长尾分布).
    log1p 对暗部近似线性(不丢分辨率), 对亮部压缩.
    """
    return np.log1p(np.maximum(x, 0.0)).astype(np.float32)


# ------------------------------------------------------------- 序列加载器 ----

class ProbeSequence:
    """一个采集目录 (= 一条连续序列)."""

    def __init__(self, dump_dir: str | Path):
        self.dir = Path(dump_dir)
        if not self.dir.exists():
            raise FileNotFoundError(f"采集目录不存在: {self.dir}")

        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"缺少 meta.json: {meta_path}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))

        self.gather = int(self.meta.get("gather_octahedron_resolution", 8))
        self.probe_view_size = tuple(self.meta.get("probe_view_size", [0, 0]))

        # 帧列表: 以 *_in.npy 为准
        self.frames = sorted(
            p.name.rsplit("_in.npy", 1)[0] for p in self.dir.glob("*_in.npy"))
        if not self.frames:
            raise FileNotFoundError(f"{self.dir} 下没找到 *_in.npy")

    def __len__(self) -> int:
        return len(self.frames)

    def _load(self, base: str, tag: str) -> np.ndarray:
        p = self.dir / f"{base}_{tag}.npy"
        if not p.exists():
            raise FileNotFoundError(p)
        return np.load(p)

    def load_frame(self, idx: int) -> dict:
        """加载第 idx 帧的全部数据, 已解码成可直接使用的形式."""
        base = self.frames[idx]
        g = self.gather
        pw, ph = self.probe_view_size  # probe_view_size = [宽, 高]

        radiance_in = self._load(base, "in")      # (H, W, 3) float32
        radiance_gt = self._load(base, "gt")      # (H, W, 3) float32
        hitdist = self._load(base, "hitdist")     # (H, W) uint8

        depth_u = self._load(base, "depth")       # (ph, pw) uint32
        normal_e = self._load(base, "normal")     # (ph, pw, 2) uint8
        wpos = self._load(base, "wpos")           # (ph, pw, 4) float32

        depth = depth_from_uint(depth_u)                      # (ph, pw) float32
        normal = octahedral_decode(normal_e.astype(np.float32) / 255.0)

        # 有效 mask: Lumen 用 depth <= 0 (哨兵值 -1e8) 标记无几何的 probe
        valid_probe = depth > 0.0                             # (ph, pw) bool

        # 相机
        cam_path = self.dir / f"{base}.json"
        cam = {}
        if cam_path.exists():
            cam = json.loads(cam_path.read_text(encoding="utf-8"))

        return {
            "base": base,
            "in": radiance_in,
            "gt": radiance_gt,
            "hitdist": hitdist.astype(np.float32) / 255.0,
            "depth": depth,
            "normal": normal,
            "wpos": wpos,
            "valid": valid_probe,
            "cam": cam,
        }


# ------------------------------------------------------- 时域重投影对齐 ------

def build_view_proj(cam: dict) -> np.ndarray | None:
    """从相机 json 组出 ViewProj 矩阵 (4x4, 行主序)."""
    vm = cam.get("view_matrix")
    pm = cam.get("proj_matrix")
    if not vm or not pm or len(vm) != 16 or len(pm) != 16:
        return None
    V = np.asarray(vm, dtype=np.float64).reshape(4, 4)
    P = np.asarray(pm, dtype=np.float64).reshape(4, 4)
    return V @ P


def reproject_to_prev(wpos_prev_translated: np.ndarray,
                      prev_pre_view_translation: np.ndarray,
                      view_proj_prev: np.ndarray,
                      screen_size: tuple[float, float]) -> np.ndarray:
    """把历史帧 probe 的世界位置投影到历史帧屏幕, 得到像素坐标.

    Args:
        wpos_prev_translated: (ph, pw, 4) 历史帧的平移后世界坐标
        prev_pre_view_translation: (3,) 历史帧的 PreViewTranslation
        view_proj_prev: (4, 4) 历史帧 ViewProj
        screen_size: (宽, 高)

    Returns:
        (ph, pw, 2) 屏幕像素坐标 (x, y)
    """
    ph, pw, _ = wpos_prev_translated.shape
    xyz = wpos_prev_translated[..., :3].astype(np.float64)
    xyz = xyz + np.asarray(prev_pre_view_translation, dtype=np.float64)

    ones = np.ones(xyz.shape[:-1] + (1,), dtype=np.float64)
    homo = np.concatenate([xyz, ones], axis=-1)           # (ph, pw, 4)

    clip = homo @ view_proj_prev                          # 行向量 x 矩阵
    w = clip[..., 3:4]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    ndc = clip[..., :3] / w

    # NDC -> [0,1]，注意 UE 屏幕 Y 轴向下
    u = ndc[..., 0] * 0.5 + 0.5
    v = 0.5 - ndc[..., 1] * 0.5
    return np.stack([u * screen_size[0], v * screen_size[1]], axis=-1)


def bilinear_sample(img: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """双线性采样. img: (H, W, C), coords: (h, w, 2) 像素坐标."""
    h, w = img.shape[:2]
    x = coords[..., 0]
    y = coords[..., 1]

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1

    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]

    c00 = img[y0c, x0c]
    c10 = img[y0c, x1c]
    c01 = img[y1c, x0c]
    c11 = img[y1c, x1c]

    top = c00 * (1 - fx) + c10 * fx
    bot = c01 * (1 - fx) + c11 * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)


# ----------------------------------------------------------------- Dataset --

class ProbeTexelDataset(Dataset):
    """把序列组织成"帧"为单位, 每帧内部随机采样 texel 组成 batch.

    Args:
        dump_dir: 采集目录
        samples_per_frame: 每帧采样多少个 texel
        use_history: 是否把历史帧 radiance 作为特征
        history_mode:
            "none"    —— 不用历史
            "aligned" —— 用相机重投影对齐历史帧 (推荐)
            "naive"   —— 直接按 probe 索引取历史 (相机静止时可)
        max_frames: 只用前 N 帧 (调试)
    """

    def __init__(
        self,
        dump_dir: str | Path,
        samples_per_frame: int = 8192,
        use_history: bool = True,
        history_mode: str = "aligned",
        max_frames: int = 0,
        gt_dir: str | Path | None = None,
        auto_pair: bool = True,
        gt_use_in: bool = False,
    ):
        """...

        auto_pair: 用【相机矩阵】把输入帧匹配到 GT 帧(默认开).
                   两遍采集速度不同(低射线快/高射线慢), 同一相机位置
                   在两遍里帧号不同 —— 按帧号取 GT 会错位. 这里改用
                   矩阵最近邻匹配, 绕开帧号问题.

        gt_use_in: 答案取 GT 目录的【in】(未滤波)而不是 gt(已滤波).
                   False = 方案 B'  学 "滤波 + 射线提升" 的耦合任务
                   True  = 方案 B'' 学【纯射线提升】(两边都未滤波,
                                    滤波的影响被抵消掉)
                   这个开关就是为了让两种方案能直接对比, 从而判断
                   "完成度里有多少是真补射线, 有多少只是学滤波".
        """
        """
        Args:
            dump_dir: 输入数据目录（低射线采集）
            gt_dir:   GT 目录（高射线采集）。None = 方案 A（输入和 GT 同目录）
            samples_per_frame: 每帧采样多少个样本
            use_history: 是否使用历史帧特征
            history_mode: "aligned"（重投影）或 "naive"（同 index）
            max_frames: 只用前 N 帧

        方案 A vs 方案 B 的区别就在 gt_dir：
          方案 A: gt_dir=None，GT 用同目录的 frame_XXXXX_gt.npy（Lumen 自己的输出）
          方案 B: gt_dir=高射线目录，GT 用高射线采集的参考
        """
        self.seq = ProbeSequence(dump_dir)
        # 方案 B：GT 来自另一个目录（高射线采集）
        self.gt_seq = ProbeSequence(gt_dir) if gt_dir is not None else None

        self.samples_per_frame = samples_per_frame
        self.use_history = use_history
        self.history_mode = history_mode
        self.gather = self.seq.gather

        n = len(self.seq)
        # 用历史时第一帧没有前驱, 从第二帧开始
        self.frame_indices = list(range(1 if use_history else 0, n))
        if max_frames > 0:
            self.frame_indices = self.frame_indices[:max_frames]

        if not self.frame_indices:
            raise ValueError("可用帧数为 0, 检查采集数据或 use_history 设置")

        # 答案取 "in"(未滤波)还是 "gt"(已滤波) —— 见 __init__ 文档
        self.gt_use_in = gt_use_in

        # ★ 方案 B 配对（GT 来自高射线目录）
        self.gt_index_map = None
        if self.gt_seq is not None:
            if len(self.gt_seq) != n:
                # 两遍采样密度不同很常见(低射线快/高射线慢), 不是致命错误.
                # 只要走的是同一条轨迹, 用相机矩阵就能配对 —— 见下.
                print(f"[dataset] 注意: 输入 {n} 帧 vs GT {len(self.gt_seq)} 帧")
                print(f"          两遍采样密度不同, 将改用【相机矩阵】配对")

            if auto_pair:
                self.gt_index_map = self._build_camera_pairing()
            else:
                # 关掉自动配对时才要求帧号严格对齐
                if len(self.gt_seq) != n:
                    raise ValueError(
                        f"输入/GT 帧数不一致且未开 auto_pair: "
                        f"输入 {n} 帧 vs GT {len(self.gt_seq)} 帧")
                self._check_camera_alignment(tol=1e-3)

    # ---------------------------------------------- 方案 B 相机对齐校验 --

    @staticmethod
    def _sample_indices(n: int, k: int) -> list[int]:
        """在 [0, n) 里均匀抽 k 个索引（含首尾）."""
        if n <= k:
            return list(range(n))
        step = (n - 1) / (k - 1)
        return [round(i * step) for i in range(k)]

    def _camera_diff(self, idx: int) -> float | None:
        """第 idx 帧在输入/GT 两个序列里的相机矩阵最大偏差.

        返回 None 表示缺 json，无法比对（跳过这一帧）.
        """
        gt = self.gt_seq
        if gt is None:
            return None

        base = self.seq.frames[idx]
        jl = self.seq.dir / f"{base}.json"
        jh = gt.dir / f"{base}.json"
        if not (jl.exists() and jh.exists()):
            return None

        cl = json.loads(jl.read_text(encoding="utf-8"))
        ch = json.loads(jh.read_text(encoding="utf-8"))

        worst = 0.0
        for key in ("view_matrix", "proj_matrix", "pre_view_translation"):
            a, b = cl.get(key), ch.get(key)
            if a is None or b is None:
                continue
            d = float(np.abs(np.asarray(a, dtype=np.float64)
                             - np.asarray(b, dtype=np.float64)).max())
            worst = max(worst, d)
        return worst

    def _check_camera_alignment(self, tol: float = 1e-3) -> None:
        """方案 B: 抽查两次采集的相机是否逐帧对齐.

        为什么必须查
        ------------
        帧数一样不代表相机对齐。若两遍 Sequencer 回放错开一帧:

            输入: 这个像素在墙上
            答案: 那个像素在地板
                    -> 网络学的是纯噪声

        而且训练【不会报错】，loss 照样降、权重照样存，
        只有等到上线才发现是垃圾模型。所以这里主动抛异常。

        只抽查若干帧（首尾 + 均匀采样），全查太慢且没必要:
        相机路径错位一般是整体偏移，抽查就能抓住。
        """
        idxs = self._sample_indices(len(self.seq.frames), 5)

        bad: list[tuple[int, float]] = []
        for i in idxs:
            d = self._camera_diff(i)
            if d is not None and d > tol:
                bad.append((i, d))

        if bad:
            detail = ", ".join(f"帧{i}: 偏差{d:.2e}" for i, d in bad)
            raise ValueError(
                f"方案 B 相机未对齐! 抽查 {len(idxs)} 帧，"
                f"{len(bad)} 帧超差 (阈值 {tol}):\n  {detail}\n"
                f"  两遍采集的相机路径必须逐帧一致。常见原因:\n"
                f"    1. Sequencer 未锁帧率 -> t.FixedFrameRate 60\n"
                f"    2. TSR(TAA) jitter 未关 -> r.AntiAliasingMethod 0\n"
                f"  完整诊断: python tools/verify/verify_b_pairing.py "
                f"<low_dir> <high_dir>")

    # ------------------------------------------------ 相机矩阵配对 ------

    @staticmethod
    def _cam_vec(cam: dict | None) -> np.ndarray | None:
        """从相机 json 取描述向量 (view+proj+pre_view_translation)。"""
        if not cam:
            return None
        parts = []
        for key in ("view_matrix", "proj_matrix", "pre_view_translation"):
            v = cam.get(key)
            if v is not None:
                parts.append(np.asarray(v, dtype=np.float64))
        return np.concatenate(parts) if parts else None

    @staticmethod
    def _load_cam_json(seq, i: int) -> dict | None:
        """只读相机 json, 不加载 npy.

        配对只需要相机矩阵; 用 load_frame 会把 7 个 npy 也读进来,
        1195 帧时是 8000+ 个文件的无谓 IO.
        """
        p = seq.dir / f"{seq.frames[i]}.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def _build_camera_pairing(self) -> dict[int, int]:
        """用相机矩阵把每个输入帧映射到 GT 帧.

        为什么需要
        ----------
        采集是按【渲染帧】计数的, 而低射线(快)和高射线(慢)渲染速度不同 ——
        即使逐帧步进, 每个相机位置实际采到的帧数也是 ~2.0 vs ~1.9 这种
        小数, 累积几百次后帧号就彻底错位了 (实测偏差 315).

        但两遍走的是【同一条轨迹】—— 实测矩阵匹配距离 0.000e+00.
        所以不按帧号取 GT, 而是按相机矩阵找"位置完全相同"的那一帧.

        返回
        ----
        {输入帧索引 -> GT 帧索引}
        """
        gt_vecs = []
        for j in range(len(self.gt_seq.frames)):
            v = self._cam_vec(self._load_cam_json(self.gt_seq, j))
            gt_vecs.append(v if v is not None else np.zeros(1))
        gt_vecs = np.stack(gt_vecs)

        mapping: dict[int, int] = {}
        dists = []
        for i in self.frame_indices:
            v = self._cam_vec(self._load_cam_json(self.seq, i))
            if v is None:
                mapping[i] = i
                continue
            d = np.linalg.norm(gt_vecs - v, axis=1)
            j = int(np.argmin(d))
            mapping[i] = j
            dists.append(float(d[j]))

        if dists:
            d = np.asarray(dists)
            med, mx = float(np.median(d)), float(d.max())
            print(f"[dataset] 相机矩阵配对: {len(d)} 帧, "
                  f"距离中位 {med:.2e} 最大 {mx:.2e}")
            if mx > 1e-2:
                print(f"[dataset] [警告] 有帧匹配距离达 {mx:.2e}, "
                      f"两遍轨迹可能不同 —— 建议重采")
        return mapping

    def __len__(self) -> int:
        return len(self.frame_indices)

    @staticmethod
    def _to_probe_grid(texel_img: np.ndarray, g: int) -> np.ndarray:
        """(H, W, C) texel 级 -> (ph, pw, g, g, C) probe 级."""
        h, w = texel_img.shape[:2]
        ph, pw = h // g, w // g
        c = texel_img.shape[2] if texel_img.ndim == 3 else 1
        img = texel_img.reshape(ph, g, pw, g, c)
        return img.transpose(0, 2, 1, 3, 4)                   # (ph, pw, g, g, C)

    @staticmethod
    def _to_texel_grid(probe_img: np.ndarray) -> np.ndarray:
        """(ph, pw, g, g, C) -> (H, W, C)."""
        ph, pw, g, _, c = probe_img.shape
        img = probe_img.transpose(0, 2, 1, 3, 4)              # (ph, g, pw, g, C)
        return img.reshape(ph * g, pw * g, c)

    def _spatial_neighbors(self, probe_img: np.ndarray) -> np.ndarray:
        """3x3 probe 邻域, 同方向 texel.

        Lumen 的空间滤波就是在 probe 网格上按**相同方向 texel** 做邻域平均:
            GatherNeighborRadiance(ScreenTileCoord + Offset, ProbeTexelCoord, ...)
        ProbeTexelCoord 不变, 变的是 probe 坐标 —— 所以网络必须能看到
        "邻居 probe 的同一个方向格子", 否则学不会空间滤波.

        Args:
            probe_img: (ph, pw, g, g, C)
        Returns:
            (ph, pw, g, g, 9*C)  顺序按行优先的 3x3 窗口
        """
        ph, pw, g, _, c = probe_img.shape
        # 只在 probe 网格维度 pad, 方向维度不 pad
        padded = np.pad(probe_img, ((1, 1), (1, 1), (0, 0), (0, 0), (0, 0)),
                        mode="constant", constant_values=0.0)
        out = np.empty((ph, pw, g, g, 9 * c), dtype=np.float32)
        k = 0
        for di in range(3):
            for dj in range(3):
                out[..., k * c:(k + 1) * c] = \
                    padded[di:di + ph, dj:dj + pw]
                k += 1
        return out

    def _build_features(self, cur: dict, prev: dict | None) -> tuple[np.ndarray, np.ndarray]:
        """构建一帧的特征与标签.

        Returns:
            feat: (H, W, C) 每 texel 的特征
            target: (H, W, 3) 每 texel 的目标
        """
        g = self.gather
        h, w = cur["in"].shape[:2]
        ph, pw = h // g, w // g

        target = log_normalize(cur["gt"])                     # (H, W, 3)

        # ---- 邻域 radiance: 3x3 probe x 同方向 texel = 27 维 ----
        r_in = log_normalize(cur["in"])
        r_probe = self._to_probe_grid(r_in, g)                # (ph, pw, g, g, 3)
        r_nb = self._spatial_neighbors(r_probe)               # (ph, pw, g, g, 27)
        feat_list = [self._to_texel_grid(r_nb)]               # (H, W, 27)

        # ---- 邻域 hit distance: 9 维 ----
        hd_probe = self._to_probe_grid(cur["hitdist"][..., None], g)
        hd_nb = self._spatial_neighbors(hd_probe)             # (ph, pw, g, g, 9)
        feat_list.append(self._to_texel_grid(hd_nb))

        # ---- 邻域有效性: 9 维 (边界外的邻居是 pad 出来的, 标记为无效) ----
        valid_f = cur["valid"].astype(np.float32)[..., None]  # (ph, pw, 1)
        v_probe = np.repeat(np.repeat(valid_f, g, axis=0), g, axis=1)  # (ph,pw,g,g,1)
        v_probe = v_probe.reshape(ph, g, pw, g, 1).transpose(0, 2, 1, 3, 4)
        v_nb = self._spatial_neighbors(v_probe)               # (ph,pw,g,g,9)
        feat_list.append(self._to_texel_grid(v_nb))

        # ---- 几何: probe 级 -> texel 级, 7 维 ----
        geo = np.concatenate([
            cur["depth"][..., None],                          # (ph, pw, 1)
            cur["normal"],                                    # (ph, pw, 3)
            cur["wpos"][..., :3],                             # (ph, pw, 3)
        ], axis=-1).astype(np.float32)                        # (ph, pw, 7)

        valid = cur["valid"]

        # 关键: 无效 probe 的几何数据是垃圾值 (wpos 可能是 -1e8 量级),
        # 必须整体清零. 否则这些极端值混入特征后, 网络在对应输入上会
        # 输出爆炸值 (实测 feat 绝对值可达 1e5, 导致 6% 样本输出 >1e6).
        # 无效 probe 反正会被 valid mask 排除, 清零不影响训练,
        # 但能保证特征量级干净.
        geo[~valid] = 0.0

        # 深度: 量级可达数万(cm), 取符号对数压缩
        geo[..., 0] = np.sign(geo[..., 0]) * np.log1p(np.abs(geo[..., 0]))
        # 世界坐标: 绝对坐标可达数万(米), 与其余 O(1) 的特征量级差太大.
        # 转成"相对该帧有效 probe 中位数"的偏移, 既保留空间关系又压到 O(10).
        w = geo[..., 4:7]
        if valid.any():
            med = np.median(w[valid], axis=0)
        else:
            med = np.zeros(3, dtype=np.float32)
        geo[..., 4:7] = (w - med) / 1000.0                    # 相对偏移, 单位 m

        geo_texel = np.repeat(np.repeat(geo, g, axis=0), g, axis=1)  # (H, W, 7)
        feat_list.append(geo_texel)

        # ---- 历史帧 ----
        if self.use_history and prev is not None:
            hist = self._sample_history(cur, prev)
            feat_list.append(log_normalize(hist))             # (H, W, 3)
            # 历史是否有效 (重投影可能失败)
            valid_hist = cur["valid"].astype(np.float32)
            vh_texel = np.repeat(np.repeat(valid_hist[..., None], g, axis=0), g, axis=1)
            feat_list.append(vh_texel)

        feat = np.concatenate(feat_list, axis=-1).astype(np.float32)
        return feat, target

    def _sample_history(self, cur: dict, prev: dict) -> np.ndarray:
        """取历史帧 radiance, 与当前帧 probe 对齐."""
        g = self.gather
        hist_raw = prev["in"]

        if self.history_mode == "naive":
            return hist_raw

        if self.history_mode == "aligned":
            cam_prev = prev.get("cam", {})
            vp = build_view_proj(cam_prev)
            if vp is None:
                return hist_raw

            H, W = hist_raw.shape[:2]
            ph, pw = H // g, W // g

            # 【关键修复 2026-09-04】
            # 之前这里用的是 prev["wpos"]（历史帧自己的 wpos）配历史帧相机投影,
            # 结果是"历史帧 probe 投影回它自己" -> pi/pj 恒等于自身索引 -> 恒等映射,
            # 时域对齐从未真正生效（静态数据下与无时域等价, 掩盖了 bug）.
            #
            # 正确做法: 取【当前帧】probe 的世界坐标, 用【历史帧】的相机矩阵投影,
            # 得到"这个表面点在上一帧画面里出现在哪个 probe 位置".
            #
            # 坐标还原符号: UE 中 TranslatedWorld = World + PreViewTranslation
            #   所以 World = TranslatedWorld - PreViewTranslation（减, 不是加）
            #   依据见 tools/verify/diag_wpos.py（用加会让 NDC 缩在 [-0.32,0.22]）
            cur_trans = np.asarray(
                cur.get("cam", {}).get("pre_view_translation", [0, 0, 0]),
                dtype=np.float64)

            xyz = cur["wpos"][..., :3].astype(np.float64) - cur_trans
            ones = np.ones(xyz.shape[:-1] + (1,), dtype=np.float64)
            clip = np.concatenate([xyz, ones], axis=-1) @ vp
            w = clip[..., 3:4]
            w = np.where(np.abs(w) < 1e-12, 1e-12, w)
            ndc = clip[..., :3] / w

            u = ndc[..., 0] * 0.5 + 0.5            # [0,1]
            v = 0.5 - ndc[..., 1] * 0.5            # [0,1], Y 向下

            # -> probe 网格索引 (四舍五入取最近邻 probe)
            pj = np.clip(np.round(u * (pw - 1)), 0, pw - 1).astype(np.int32)
            pi = np.clip(np.round(v * (ph - 1)), 0, ph - 1).astype(np.int32)

            # 取对应的 g x g 方向块 (而不是只采一个点!).
            # 重投影是 probe 级的 (一个 probe 一个位置),
            # 但每个 probe 含 g*g 个方向 texel, 必须整块取回.
            ii = pi[:, :, None, None] * g + np.arange(g)[None, None, :, None]
            jj = pj[:, :, None, None] * g + np.arange(g)[None, None, None, :]
            blocks = hist_raw[ii, jj]                        # (ph, pw, g, g, 3)

            return blocks.transpose(0, 2, 1, 3, 4).reshape(H, W, 3)

        raise ValueError(f"未知 history_mode: {self.history_mode}")

    def __getitem__(self, i: int) -> dict:
        idx = self.frame_indices[i]
        cur = self.seq.load_frame(idx)

        prev = None
        if self.use_history:
            prev = self.seq.load_frame(idx - 1)

        feat, target = self._build_features(cur, prev)      # (H, W, C), (H, W, 3)

        # ★ 方案 B：GT 来自【高射线目录】，覆盖掉同目录（低射线）的 gt
        # 这是方案 A / B 的唯一数据流差异点
        if self.gt_seq is not None:
            # 用相机矩阵配对表取 GT 帧号(绕开两遍采样密度不同导致的帧号错位)
            gt_idx = self.gt_index_map.get(idx, idx) if self.gt_index_map else idx
            gt_frame = self.gt_seq.load_frame(gt_idx)
            # gt_use_in=True -> 取未滤波的 in, 即方案 B''(纯射线提升)
            key = "in" if self.gt_use_in else "gt"
            gt_target = log_normalize(gt_frame[key])

            # 空间尺寸必须一致（probe 网格相同）
            if gt_target.shape[:2] != feat.shape[:2]:
                raise ValueError(
                    f"输入/GT 空间尺寸不一致: 输入 {feat.shape[:2]} vs GT {gt_target.shape[:2]}")

            target = gt_target

        g = self.gather
        h, w = feat.shape[:2]

        # 有效 mask (texel 级)
        valid = np.repeat(np.repeat(cur["valid"], g, axis=0), g, axis=1)

        # 摊平成样本并随机采样
        feat_flat = feat.reshape(-1, feat.shape[-1])
        target_flat = target.reshape(-1, 3)
        valid_flat = valid.reshape(-1)

        valid_idx = np.flatnonzero(valid_flat)
        if valid_idx.size == 0:
            # 极端情况: 整帧无有效 probe, 退回全采样
            valid_idx = np.arange(feat_flat.shape[0])

        n = min(self.samples_per_frame, valid_idx.size)
        choice = np.random.choice(valid_idx, size=n, replace=False)

        return {
            "feat": torch.from_numpy(feat_flat[choice]),
            "target": torch.from_numpy(target_flat[choice]),
        }


def describe_dump(dump_dir: str | Path) -> None:
    """打印采集数据的概况, 便于快速确认."""
    seq = ProbeSequence(dump_dir)
    print(f"目录: {seq.dir}")
    print(f"  帧数: {len(seq)}")
    print(f"  gather: {seq.gather}")
    print(f"  probe_view_size: {seq.probe_view_size}")
    f = seq.load_frame(min(1, len(seq) - 1))
    print(f"  帧样例 [{f['base']}]:")
    for k in ("in", "gt", "hitdist", "depth", "normal", "wpos"):
        v = f[k]
        print(f"    {k:8} shape={str(v.shape):20} dtype={v.dtype}")
    v = f["valid"]
    print(f"    有效 probe: {int(v.sum())}/{v.size} ({v.mean():.1%})")
    pw, ph = seq.probe_view_size
    print(f"    每帧样本数: {ph * pw * seq.gather * seq.gather:,}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    args = ap.parse_args()
    describe_dump(args.dump_dir)
