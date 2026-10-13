"""方案 B' 一键采集: 低射线(输入) + 高射线(答案) 配对数据.

解决的问题
-----------
编辑器视口 (非 PIE) 不遵守 Project Settings 的 "Use Fixed Frame Rate",
所以实时播放 Sequencer 时:

    低射线 (64 条)  跑得快 -> 第 N 帧相机在路径 60% 处
    高射线 (256 条) 跑得慢 -> 第 N 帧相机在路径 30% 处
                            -> 两遍对不上, 数据无法配对

本脚本不"实时播放", 而是把 Sequencer 播放头【逐帧】推,
每推一帧等若干渲染帧. 相机位置完全由帧号决定, 与快慢无关 -> 两遍严格对齐.

为什么不用 time.sleep
---------------------
编辑器 Python 跑在主线程, sleep 会卡住渲染 -> 根本不采集.
必须用 register_slate_post_tick_callback 做异步推进.

用法
----
1. Window -> Developer Tools -> Output Log
2. 输入框左侧下拉切到 "Python"  (或在 Cmd 下用: py <本文件路径>)
3. 执行:

    import capture_b_pair
    capture_b_pair.run()

想改参数:

    capture_b_pair.run(num_frames=20, hold_low=3, hold_high=12)

采集完成后会提示你跑校验命令.
"""

import unreal

# ============================================================ 配置 ======

NUM_FRAMES = 20          # 推进多少个"采样点" (不是 Sequencer 帧数)
START_FRAME = 0          # Sequencer 从第几帧开始

# ★★★ 关键: 采样点之间的间隔(帧) ★★★
#
# 实际跳到的是 START_FRAME + i * FRAME_STEP.
# 若 Sequencer 路径总长 600 帧, 想覆盖整条路径:
#     NUM_FRAMES=40, FRAME_STEP=15  -> 覆盖 0..585  ✓
#     NUM_FRAMES=10, FRAME_STEP=60  -> 覆盖 0..540  ✓
#
# 之前漏了这个参数, 等于 FRAME_STEP=1, NUM_FRAMES=10 只走了路径前 10 帧
# (600 帧里的 1.7%), 相机几乎没动 —— 数据多样性严重不足.
FRAME_STEP = 15

# 输出目录前缀. 实际写到 "<前缀>_low" / "<前缀>_high".
#   默认: D:/ProbeDump_B_low, D:/ProbeDump_B_high
#   换个场景采集时改这里(如 D:/ProbeDump_NEW), 避免覆盖已有数据.
OUT_PREFIX = "D:/ProbeDump_B"

# 每推进一个 Sequencer 帧, 等待多少个渲染帧.
#
# ★★★ 关键: 两遍必须用【同一个值】! ★★★
# 采集是按"渲染帧"计数的, 不是按 Sequencer 帧. hold 期间相机静止但采集
# 仍在继续, 于是每个相机位置会被重复采 hold 次. 两遍 hold 不同 ->
# "帧号 -> 相机位置"的映射就不同 -> 必然错位 (之前就是这样踩的坑).
#
# 同时 MaxFrames 要设成 num_frames * hold, 否则采不满所有相机位置.
#
# 值的选择: 1 = 最省 (每个位置只采 1 帧), 但渲染慢可能采到旧画面.
#           渲染跟得上就用 1~3; 高射线卡就调大 (两遍一起调).
HOLD = 3

PHASES = [
    {
        "name": "low (输入, 64 条射线)",
        "trace": 8,
        "scale": 1.0,
        "outkey": "low",
        "hold": HOLD,
        "warmup": 60,      # Enable 后先空转多少 tick, 等 Lumen 就绪
    },
    {
        "name": "high (答案, 256 条射线)",
        "trace": 16,
        "scale": 0.5,
        "outkey": "high",
        "hold": HOLD,      # 必须与 low 相同!
        "warmup": 400,     # 高射线要重新编译/分配, 给足时间
    },
]

# 两遍之间的冷却 tick 数. 切换射线数会让 Lumen 重建资源,
# 立刻开始下一遍容易撞上 "[LumenRuntimeSC] Stage=Waiting" 而卡死.
COOLDOWN_TICKS = 240


# ============================================================ 工具 ======

def cmd(s: str) -> None:
    """执行控制台命令 (兼容不同版本的 WorldContext 要求)."""
    try:
        unreal.SystemLibrary.execute_console_command(None, s)
    except Exception:
        try:
            w = unreal.EditorLevelLibrary.get_editor_world()
            unreal.SystemLibrary.execute_console_command(w, s)
        except Exception as e:
            unreal.log_warning(f"[capture] 命令执行失败: {s} -> {e}")


def try_get_seq_length():
    """尽力探测当前 Sequencer 路径长度(帧). 失败返回 None.

    不同 UE 版本的 Python API 名不一致, 这里逐个尝试;
    都拿不到就返回 None, 让调用方回退到手动指定.
    """
    lib = getattr(unreal, "LevelSequenceEditorBlueprintLibrary", None)
    if lib is None:
        return None

    seq = None
    for getter in ("get_current_level_sequence", "get_focused_level_sequence"):
        fn = getattr(lib, getter, None)
        if fn is None:
            continue
        try:
            seq = fn()
            if seq:
                break
        except Exception:
            continue

    if not seq:
        return None

    for s_get, e_get in (("get_playback_start", "get_playback_end"),
                         ("get_playback_start_seconds", "get_playback_end_seconds")):
        sf, ef = getattr(seq, s_get, None), getattr(seq, e_get, None)
        if sf is None or ef is None:
            continue
        try:
            start, end = sf(), ef()
            # 秒为单位的话乘 60 换算成帧
            if "seconds" in s_get:
                start, end = start * 60.0, end * 60.0
            length = int(end - start)
            if length > 0:
                return length
        except Exception:
            continue
    return None


def set_seq_frame(frame: int) -> None:
    """把 Sequencer 播放头推到指定帧.

    UE 5.6 的签名是 SetCurrentTime(int32 NewFrame), 参数是【int 帧号】,
    不是 unreal.FrameNumber 对象 —— 传 FrameNumber 会报
    "Cannot nativize 'FrameNumber' as 'NewFrame' (IntProperty)".
    """
    lib = unreal.LevelSequenceEditorBlueprintLibrary
    lib.set_current_time(int(frame))


# ============================================================ 主体 ======

class _Capture:
    def __init__(self, num_frames: int, hold: int,
                 frame_step: int = FRAME_STEP, path_length: int = 0,
                 out_prefix: str = OUT_PREFIX):
        self.num_frames = num_frames
        self.hold = hold
        self.frame_step = frame_step
        self.out_prefix = out_prefix
        # 路径总长(帧). 用于把采样点钳制在路径范围内 —— 否则 step 算大了
        # 会让后面的采样点全挤到路径末尾, 又是重复画面.
        self.path_length = path_length

        # 两遍用同一个 hold —— 保证"帧号 -> 相机位置"的映射一致
        self.PHASES = [
            dict(PHASES[0], hold=hold),
            dict(PHASES[1], hold=hold),
        ]
        self.pi = 0            # 当前第几遍
        self.frame = 0         # 当前推进到第几个采样点
        self.hold = 0          # 当前点已等待的渲染帧数 (计数器)
        self.state = "init"    # init -> warmup -> stepping -> done
        self.warm = 0
        self.handle = None

    def _target_frame(self, i: int) -> int:
        """第 i 个采样点对应的 Sequencer 帧号.

        等间隔铺开: START_FRAME + i * frame_step
        再做钳制: 不超过 START_FRAME + path_length - 1
        (钳制是保险 —— step 估计偏大时, 避免采样点全堆到路径末尾)
        """
        f = START_FRAME + i * self.frame_step
        if self.path_length > 0:
            f = min(f, START_FRAME + self.path_length - 1)
        return f

    # ---------------------------------------------------------- 启动 --
    def start(self):
        unreal.log("[capture] ===== 方案 B' 配对采集开始 =====")
        cover = self.num_frames * self.frame_step
        unreal.log(f"[capture] 采样覆盖: {self.num_frames} 点 x step "
                   f"{self.frame_step} = Sequencer 帧 0..{cover}")
        unreal.log("[capture] ★ 请确认这个范围覆盖了你的路径长度, "
                   "否则相机会挤在开头几乎不动!")
        cmd("r.AntiAliasingMethod 0")
        self._begin_phase()

        self.handle = unreal.register_slate_post_tick_callback(self._tick)
        unreal.log("[capture] 已启动逐帧推进, 请勿操作编辑器, 等待提示完成.")

    def _begin_phase(self):
        p = self.PHASES[self.pi]
        unreal.log(f"[capture] ---- 第 {self.pi + 1}/2 遍: {p['name']} ----")

        outdir = f"{self.out_prefix}_{p['outkey']}"
        cmd(f"r.Lumen.ScreenProbeGather.TracingOctahedronResolution {p['trace']}")
        cmd(f"r.Lumen.ScreenProbeGather.GatherOctahedronResolutionScale {p['scale']}")
        cmd(f"r.Lumen.ScreenProbeDump.OutputDir {outdir}")
        # 每个 Sequencer 位置会重复采 hold 帧, 所以上限要乘上 hold,
        # 否则采不满 num_frames 个相机位置就停了.
        cmd(f"r.Lumen.ScreenProbeDump.MaxFrames {self.num_frames * self.hold}")

        # 关键: 每一遍都从 Sequencer 的同一帧开始
        set_seq_frame(START_FRAME)

        cmd("r.Lumen.ScreenProbeDump.Enable 1")

        self.frame = 0
        self.hold = 0
        self.warm = 0
        self.state = "warmup"

    # ---------------------------------------------------------- 每帧 --
    def _tick(self, delta_time):
        try:
            if self.state == "cooldown":
                # 切射线数后 Lumen 要重建资源, 直接开采会撞上
                # "[LumenRuntimeSC] Stage=Waiting" 而整个卡死.
                self.warm += 1
                if self.warm >= COOLDOWN_TICKS:
                    self._begin_phase()
                return True

            if self.state == "warmup":
                # 等 Lumen 就绪 + WaitForMotion 检测到"相机起步"
                p = self.PHASES[self.pi]
                self.warm += 1
                if self.warm >= p["warmup"]:
                    self.state = "stepping"
                    self.frame = 0
                    self.hold = 0
                    set_seq_frame(START_FRAME)
                return True

            if self.state == "stepping":
                p = self.PHASES[self.pi]

                if self.hold < p["hold"]:
                    self.hold += 1
                    return True

                # 等待够了 -> 推进一帧
                self.hold = 0
                self.frame += 1

                if self.frame >= self.num_frames:
                    self._end_phase()
                    return self.state != "done"

                # 等间隔铺开整条路径, 并钳制在路径范围内
                set_seq_frame(self._target_frame(self.frame))

                if self.frame % 5 == 0:
                    unreal.log(f"[capture]   第 {self.pi + 1} 遍: "
                               f"{self.frame}/{self.num_frames}")
                return True

            return False

        except Exception as e:
            unreal.log_error(f"[capture] tick 出错: {e}")
            cmd("r.Lumen.ScreenProbeDump.Enable 0")
            if self.handle is not None:
                unreal.unregister_slate_post_tick_callback(self.handle)
                self.handle = None
            self.state = "done"
            return False

    def _end_phase(self):
        p = self.PHASES[self.pi]
        cmd("r.Lumen.ScreenProbeDump.Enable 0")
        unreal.log(f"[capture]   第 {self.pi + 1} 遍完成 -> "
                   f"{self.out_prefix}_{p['outkey']}")

        self.pi += 1
        if self.pi >= len(self.PHASES):
            self._finish()
            return

        # 冷却: 等 Lumen 消化"射线数切换", 再开下一遍
        self.warm = 0
        self.state = "cooldown"
        unreal.log(f"[capture]   冷却 {COOLDOWN_TICKS} tick "
                   f"(等 Lumen 重建资源, 防止下一遍卡死)")

    def _finish(self):
        self.state = "done"
        if self.handle is not None:
            unreal.unregister_slate_post_tick_callback(self.handle)
            self.handle = None

        unreal.log("[capture] ===== 两遍采集完成 =====")
        unreal.log("[capture] 下一步: 在命令行跑校验")
        unreal.log("[capture]   cd /d \"c:\\Users\\theozhu\\Desktop\\拓展\\"
                   "Super Resolution\\NeuralGI-Lumen\"")
        unreal.log("[capture]   python tools\\verify\\verify_b_pairing.py "
                   "D:/ProbeDump_B_low D:/ProbeDump_B_high")
        unreal.log("[capture]   python tools\\verify\\diag_frame_offset.py "
                   "D:/ProbeDump_B_low D:/ProbeDump_B_high")


# ============================================================ 入口 ======

_G = {}


def run(num_frames: int = NUM_FRAMES, hold: int = HOLD,
        frame_step: int = 0, seq_length: int = 0,
        out_prefix: str = OUT_PREFIX):
    """一键采集两遍.

    参数
    ----
    out_prefix : 输出目录前缀, 写到 "<前缀>_low" / "<前缀>_high".
                 换场景采集时改它(如 "D:/ProbeDump_NEW"), 避免覆盖旧数据.
    num_frames : 采集多少个采样点 (不同相机位置)
    hold       : 每个位置等待的渲染帧数.
                 ★ 两遍自动使用同一个值 —— 这是对齐的关键, 不要分别设.
                 渲染慢就调大 (比如 5~8), 但两遍会一起变大.
    frame_step : 采样点间隔(帧). 0 = 自动(推荐), 见下.
    seq_length : 路径总长(帧). 0 = 自动探测. 探测失败时可手动指定(如 600).

    自动铺满整条路径
    ----------------
    默认 frame_step=0 时, 脚本会:
        1. 探测 Sequencer 路径长度 (失败则用 seq_length)
        2. frame_step = 路径长度 // num_frames
        3. 于是 num_frames 个采样点【均匀铺满整条路径】

    这样只要告诉它"要多少个采样点", 就不会再出现"只走路径开头几步"
    导致相机几乎不动、数据多样性极低的问题.

    实际落盘帧数 = num_frames * hold (每个位置重复 hold 帧).
    """
    if _G.get("running"):
        unreal.log_warning("[capture] 已有采集在跑, 忽略本次调用. (要重来请先调 stop())")
        return

    # ---- 路径长度: 用于自动铺满 + 帧号钳制 ----
    path_len = seq_length if seq_length > 0 else (try_get_seq_length() or 0)

    if frame_step <= 0:
        if path_len > 0:
            frame_step = max(1, path_len // max(1, num_frames))
            unreal.log(f"[capture] 路径长度 {path_len} 帧 / {num_frames} 个采样点 "
                       f"-> step={frame_step} (自动铺满全程)")
        else:
            frame_step = FRAME_STEP
            unreal.log_warning(
                f"[capture] 未探测到路径长度, 回退 step={frame_step}. "
                f"建议显式指定, 例如: run(num_frames=60, seq_length=600)")

    c = _Capture(num_frames, hold, frame_step, path_len, out_prefix)
    _G["inst"] = c
    _G["running"] = True
    c.start()


def stop():
    """中止采集."""
    c = _G.get("inst")
    if c is not None and c.handle is not None:
        unreal.unregister_slate_post_tick_callback(c.handle)
        c.handle = None
    cmd("r.Lumen.ScreenProbeDump.Enable 0")
    _G["running"] = False
    unreal.log("[capture] 已中止.")


if __name__ == "__main__":
    run()
