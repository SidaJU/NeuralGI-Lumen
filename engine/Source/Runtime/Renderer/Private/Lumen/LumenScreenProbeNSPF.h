// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

//
// [theozhu++] NSPF S4.2: 用 UE 原生 NNE 替换 Lumen Screen Probe 的传统滤波
//
// 完全照搬官方 NNEDenoiser 的三段式 (NNEDenoiserGenericDenoiser.cpp:126-135):
//
//   1. InputProcess : compute shader 从纹理读, 写进 NNE 的 input buffer
//   2. EnqueueRDG   : 模型推理
//   3. OutputProcess: compute shader 从 output buffer 读, 写回纹理
//
// 关键点 (官方如此, 必须遵守):
//   - 官方【从不】用 CPU 上传填充 input buffer (NNEDenoiser 里搜
//     QueueBufferUpload = 0 结果). 数据一律由 shader pass 写入.
//   - buffer 必须是 typed (NNE Gemm 用 PF_R32_FLOAT 建 SRV,
//     见 NNERuntimeRDGGemm.cpp:138-143), 且 Usage 带 EBufferUsageFlags::NNE
//     (NNEDenoiserGenericDenoiser.cpp:42-45).
//   - 建 UAV 必须显式给 Format, 否则触发
//     "Format cannot be unknown for typed buffers" 断言.
//   - pass 要带 ERDGPassFlags::NeverCull, 防止被 RDG 剔除.
//
// 本文件只负责"替换滤波结果"这一件事, 训练数据的导出仍在
// LumenScreenProbeDump.{h,cpp}.
// [theozhu--]

class FViewInfo;
// 必须是 class 而不是 struct: 与 LumenScreenProbeDump.h:70 / LumenRadianceCache.h:9
// 保持一致, 否则报 C4099 "类型名称以前使用 class 现在使用 struct".
class FScreenProbeParameters;

namespace LumenScreenProbeNSPF
{
	/**
	 * 用 NNE 对 CompositedScreenProbeRadiance 做滤波, 返回滤波后的纹理.
	 *
	 * @return 成功返回 NN 输出纹理; CVar 关闭 / 模型未就绪 / 出错时返回 nullptr,
	 *         调用方应当继续沿用 Lumen 的传统滤波结果.
	 */
	FRDGTextureRef AddNSPFFilterPass(
		FRDGBuilder& GraphBuilder,
		const FViewInfo& View,
		const FScreenProbeParameters& ScreenProbeParameters,
		FRDGTextureRef CompositedScreenProbeRadiance,
		FRDGTextureRef ScreenProbeHitDistance,
		FRDGTextureRef ScreenProbeSceneDepth,
		FRDGTextureRef ScreenProbeWorldNormal,
		FRDGTextureRef ScreenProbeTranslatedWorldPosition);

	/** 释放缓存的 NNE 模型 (模块卸载时调用). */
	void Shutdown();
};
