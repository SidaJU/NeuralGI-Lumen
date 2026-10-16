// Copyright Epic Games, Inc. All Rights Reserved.

/*=============================================================================
	LumenScreenProbeDump.h

	[theozhu++] NSPF (Neural Screen Probe Filtering) 数据采集
	=============================================================================

	目的
	----
	把 Lumen Screen Probe 管线内部的中间 buffer 导出到磁盘, 用来训练一个
	神经网络, 替换 LumenScreenProbeFiltering.usf 里的传统时空滤波器
	(时域二值权重 + 手工空间高斯核).

	目标收益: 滤波器变强 -> 允许把每 probe 射线数从 8x8=64 降到 4x4=16
	-> trace 开销降到 1/4, 减去网络推理成本后仍有净提速, 画质持平.

	为什么必须改引擎
	----------------
	这些 buffer 活在 RDG 内部, SceneCaptureComponent2D 只能拿到最终画面,
	无法触达. 且数据是 HDR 线性值, 8-bit PNG 装不下.

	导出内容 (S2)
	-------------
	纹理级 (texel, atlas 尺寸 = probeAtlasSize * gatherRes):
	  InputRadiance  滤波前 radiance      R11G11B10   神经网络的输入
	  GTRadiance     滤波后 radiance      R11G11B10   神经网络的答案
	  HitDistance    射线命中距离         PF_R8       判断墙角/开阔地

	探针级 (probe 网格尺寸 = ScreenProbeAtlasBufferSize):
	  SceneDepth     深度                 PF_R32_UINT  需 asfloat 解码
	  WorldNormal    世界空间法线         PF_R8G8      八面体编码
	  WorldPosition  平移后世界坐标       RGBA32F      直接可用

	设计决策: 引擎侧只做原始 readback, 不做任何格式解码
	--------------------------------------------------
	几何数据的编码方式 (depth 的 uint bitcast, normal 的八面体压缩) 比较琐碎,
	放在引擎里一旦搞错就要重新编译 (20+ 分钟). 因此这里只把原始字节存盘,
	解码全部放到 Python 侧 —— 那里改几行就能重解, 无需碰引擎.

	文件布局
	--------
	  frame_%06d_in.npy        (validH, validW, 3) float32   输入
	  frame_%06d_gt.npy        (validH, validW, 3) float32   答案
	  frame_%06d_hitdist.npy   (validH, validW)    uint8     原始
	  frame_%06d_depth.npy     (probeH, probeW)    uint32    原始
	  frame_%06d_normal.npy    (probeH, probeW, 2) uint8     原始
	  frame_%06d_wpos.npy      (probeH, probeW, 4) float32   世界坐标
	  frame_%06d.json          相机矩阵 (时域对齐用)
	  meta.json                全局配置

	注意: 所有 *_in/_gt/_hitdist 只导出**有效区** (前 probeViewSize 行),
	atlas 尾部未使用的槽位从未被 compute shader 写入, 是未初始化显存.

	代码归属
	--------
	所有与本功能相关的改动均用 [theozhu++] / [theozhu--] 包裹,
	遵循团队既有的 [++Raid] 标识约定, 便于后续分离与 review.
// [theozhu--]
=============================================================================*/

#pragma once

#include "CoreMinimal.h"
#include "RenderGraphFwd.h"

class FViewInfo;
// 注意: BEGIN_SHADER_PARAMETER_STRUCT 宏展开为 class, 前向声明必须同为 class,
// 否则 MSVC 报 C4099 (类型名称以前使用 struct 现在使用 class).
class FScreenProbeParameters;

namespace LumenScreenProbeDump
{
	/** 采集开关是否打开 (CVar + 帧数上限综合判断). 渲染线程安全. */
	bool IsEnabled();

	/**
	 * 请求导出一帧的 screen probe 数据.
	 *
	 * 必须在 RDG 构建阶段 (渲染线程) 调用.
	 * 调用点: LumenScreenProbeFiltering.cpp 的 FilterScreenProbes 末尾
	 *         (此时输入输出变量都还在作用域内).
	 *
	 * @param CompositedRadiance  滤波前 radiance (输入侧)
	 * @param FilteredRadiance    滤波后 radiance (答案侧)
	 * @param HitDistance         texel 级命中距离
	 * @param SceneDepth          probe 级深度
	 * @param WorldNormal         probe 级世界法线
	 * @param WorldPosition       probe 级平移后世界坐标
	 */
	void EnqueueDump(
		FRDGBuilder& GraphBuilder,
		const FViewInfo& View,
		const FScreenProbeParameters& ScreenProbeParameters,
		FRDGTextureRef CompositedRadiance,
		FRDGTextureRef FilteredRadiance,
		FRDGTextureRef HitDistance,
		FRDGTextureRef SceneDepth,
		FRDGTextureRef WorldNormal,
		FRDGTextureRef WorldPosition);

	/**
	 * 轮询 readback 队列, 对已就绪的帧做格式解码并异步写盘.
	 * 渲染线程调用. (当前实现挂在 EnqueueDump 开头, 无需额外 Tick 点.)
	 */
	void Tick();

	/** 丢弃所有 pending readback. 模块卸载或采集结束时调用, 防止泄漏. */
	void Shutdown();
}
