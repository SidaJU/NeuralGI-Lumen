"""非 PIE 采集: Sequencer 逐帧步进.

为什么需要这个
--------------
编辑器视口 (非 PIE) 【不遵守】 Project Settings 里的 "Use Fixed Frame Rate"
(那个设置只对 PIE / 游戏生效). 于是:

    低射线 (64 条)  跑得快 -> 第 N 帧时相机在路径 60% 处
    高射线 (256 条) 跑得慢 -> 第 N 帧时相机在路径 30% 处
                            -> 两遍相机位置对不上, 数据无法配对

表现为 verify_b_pairing.py 的 [2] 报"相机超差", 且 diag_frame_offset.py
搜不到让偏差归零的 k (最小值几百而不是 0).

本脚本的解法
------------
不"实时播放" Sequencer, 而是把播放头【逐帧】推到指定帧, 每推一帧
等渲染完成再推下一帧. 这样相机位置完全由【帧号】决定, 与真实耗时无关,
两遍严格 1:1 对齐.

用法 (UE 编辑器里执行)
----------------------
1. Window -> Developer Tools -> Python Console  打开 Python 控制台
2. 确保 Sequencer 面板已打开并载入了你的 Level Sequence
3. 执行:

    exec(open(r'D:/path/to/step_sequencer.py').read())

4. 或者直接在控制台里改参数后调用:

    import step_sequencer
    step_sequencer.run(start_frame=0, num_frames=20, settle=0.25)

参数说明
--------
start_frame: 从第几帧开始
num_frames : 采集多少帧
settle     : 每帧等待渲染的秒数 (渲染慢就调大, 比如高射线用 0.5)

注意
----
- 视口必须开 Realtime (视口左上角勾上), 否则不渲染就不会采集
- 执行前先设好 r.Lumen.ScreenProbeDump.OutputDir / MaxFrames / Enable 1
- 采完记得 Enable 0
"""
import time

import unreal


def get_seq_lib():
    """Sequencer 编辑器库, 拿不到就报错."""
    lib = getattr(unreal, "LevelSequenceEditorBlueprintLibrary", None)
    if lib is None:
        raise RuntimeError(
            "找不到 LevelSequenceEditorBlueprintLibrary.\n"
            "请先打开 Sequencer 面板 (双击 Level Sequence 资产).")
    return lib


def run(start_frame=0, num_frames=20, settle=0.25, verbose=True):
    """把 Sequencer 播放头逐帧推进, 每帧等渲染."""
    lib = get_seq_lib()

    # 用当前时间做一个"帧号 <-> 时间"的换算基准
    cur = lib.get_current_time()
    print(f"[step] 当前播放头: {cur}")

    for i in range(num_frames):
        frame = start_frame + i

        try:
            # FrameNumber 类型
            fn = unreal.FrameNumber(value=frame)
            lib.set_current_time(fn)
        except Exception as e:
            print(f"[step] set_current_time({frame}) 失败: {e}")
            print("[step] 尝试用秒数接口...")
            # 退回到秒 (假设 60fps)
            lib.set_current_time_seconds(float(frame) / 60.0)

        # 等渲染 + 采集落盘
        time.sleep(settle)

        if verbose and (i % 5 == 0 or i == num_frames - 1):
            print(f"[step] frame {frame}  done ({i + 1}/{num_frames})")

    print(f"[step] 完成: 推进 {num_frames} 帧, 每帧等待 {settle}s")


if __name__ == "__main__":
    # 默认: 从 0 帧开始, 采 20 帧, 每帧等 0.25 秒
    run(start_frame=0, num_frames=20, settle=0.25)
