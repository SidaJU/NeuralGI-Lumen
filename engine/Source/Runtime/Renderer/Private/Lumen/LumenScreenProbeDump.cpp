// Copyright Epic Games, Inc. All Rights Reserved.

/*=============================================================================
	LumenScreenProbeDump.cpp

	[theozhu++] NSPF (Neural Screen Probe Filtering) 数据采集 —— 实现
	详见 LumenScreenProbeDump.h 顶部说明.
// [theozhu--]
=============================================================================*/

#include "LumenScreenProbeDump.h"

// [theozhu++]
// 采集功能永不进包. 注意: 这里不能简单地把实现整体 #if 掉 ——
// FilterScreenProbes 里的调用点没做条件编译, 整体清空会导致 Shipping 链接失败.
// 因此 Shipping 构建下提供空实现, 保证符号存在且行为为 no-op.
#if !UE_BUILD_SHIPPING

#include "LumenScreenProbeGather.h"
#include "RendererPrivate.h"          // LogRenderer
#include "ScenePrivate.h"             // FViewInfo
#include "RenderGraphUtils.h"         // AddEnqueueCopyPass
#include "RHIGPUReadback.h"           // FRHIGPUTextureReadback
#include "RHISurfaceDataConversion.h" // ConvertRawR11G11B10FDataToFLinearColor
#include "RHI.h"                      // GPixelFormats
#include "HAL/IConsoleManager.h"
#include "HAL/PlatformFilemanager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Async/Async.h"

// ---------------------------------------------------------------- CVar -------
// [theozhu++]
static int32 CVarLumenScreenProbeDumpEnable = 0;
static FAutoConsoleVariableRef CVarLumenScreenProbeDumpEnableRef(
	TEXT("r.Lumen.ScreenProbeDump.Enable"),
	CVarLumenScreenProbeDumpEnable,
	TEXT("[theozhu] NSPF: 开启 Lumen screen probe 数据采集. 0=关 1=开"),
	ECVF_RenderThreadSafe);

static FString CVarLumenScreenProbeDumpOutputDir = TEXT("D:/ProbeDump");
static FAutoConsoleVariableRef CVarLumenScreenProbeDumpOutputDirRef(
	TEXT("r.Lumen.ScreenProbeDump.OutputDir"),
	CVarLumenScreenProbeDumpOutputDir,
	TEXT("[theozhu] NSPF: 采集输出目录"),
	ECVF_RenderThreadSafe);

static int32 CVarLumenScreenProbeDumpMaxFrames = 0;
static FAutoConsoleVariableRef CVarLumenScreenProbeDumpMaxFramesRef(
	TEXT("r.Lumen.ScreenProbeDump.MaxFrames"),
	CVarLumenScreenProbeDumpMaxFrames,
	TEXT("[theozhu] NSPF: 最多采集帧数, 0=不限"),
    ECVF_RenderThreadSafe);

static int32 CVarLumenScreenProbeDumpWaitForMotion = 1;
static FAutoConsoleVariableRef CVarLumenScreenProbeDumpWaitForMotionRef(
	TEXT("r.Lumen.ScreenProbeDump.WaitForMotion"),
	CVarLumenScreenProbeDumpWaitForMotion,
	TEXT("[theozhu] NSPF: 等相机开始运动后才计数/采集. 1=等待(默认) 0=立即采集. "
		 "方案 B 两遍采集靠它消除手动 Enable 与 Sequencer Play 之间的延迟差异."),
    ECVF_RenderThreadSafe);
// [theozhu--]

// ------------------------------------------------------------ 内部状态 -------
// [theozhu++]
namespace LumenScreenProbeDumpPrivate
{
	/** 要导出的各个 surface. */
	enum class EDumpSurface : uint8
	{
		InputRadiance = 0,   // 滤波前 radiance (texel 级)
		GTRadiance,          // 滤波后 radiance (texel 级)
		HitDistance,         // 命中距离     (texel 级, PF_R8)
		SceneDepth,          // 深度         (probe 级, PF_R32_UINT)
		WorldNormal,         // 世界法线     (probe 级, PF_R8G8)
		WorldPosition,       // 世界坐标     (probe 级, RGBA32F)
		Num
	};

	static constexpr uint32 NumSurfaces = static_cast<uint32>(EDumpSurface::Num);

	/** 采集元数据 (纯数据, 可拷贝). */
	struct FDumpMeta
	{
		FIntPoint AtlasSize = FIntPoint::ZeroValue;        // texel 级分配尺寸 (W,H)
		FIntPoint ValidAtlasSize = FIntPoint::ZeroValue;   // texel 级有效尺寸 (W,H)
		FIntPoint ProbeAtlasSize = FIntPoint::ZeroValue;   // probe 级槽位 (W,H)
		FIntPoint ProbeViewSize = FIntPoint::ZeroValue;    // probe 级有效 (W,H)
		uint32 GatherOctahedronResolution = 0;
		uint32 TracingOctahedronResolution = 0;
		uint32 DownsampleFactor = 0;
	};

	/**
	 * 相机信息, 时域对齐用.
	 * 用 double 精度的 FMatrix 直接存原始值, 不做 float 转换 ——
	 * 世界坐标在大地图上数值很大, 转成 float32 会丢精度.
	 */
	struct FCameraInfo
	{
		FMatrix ViewMatrix = FMatrix::Identity;
		FMatrix ProjMatrix = FMatrix::Identity;
		FVector PreViewTranslation = FVector::ZeroVector;
		FVector PrevPreViewTranslation = FVector::ZeroVector;
		bool bPrevValid = false;
	};

	struct FPendingDump
	{
		int32 FrameIndex = 0;
		FDumpMeta Meta;
		FCameraInfo Camera;
		TUniquePtr<FRHIGPUTextureReadback> Readbacks[NumSurfaces];
	};

	TArray<FPendingDump> GPendingDumps;
	int32 GFrameCounter = 0;
	int32 GWrittenCount = 0;
	bool bMetaWritten = false;

	// ---- 等待相机起动 (WaitForMotion) 的状态 ----
	// 见 EnqueueDump 中同名逻辑: 解决"手动 Enable 之后再去点 Sequencer Play"
	// 之间的延迟导致两遍采集帧错位的问题.
	bool        bArmed        = false;   // 相机是否已开始运动
	FCameraInfo ArmRefCamera;            // 起动前记录的参考相机
	int32       ArmWaitFrames = 0;       // 已等待的帧数 (0=还没记录参考帧)

	// EnqueueDump 被调用的总次数. 用于区分两种"没落盘":
	//   - 计数一直为 0   -> Lumen 渲染管线根本没跑到 FilterScreenProbes
	//                      (例如 Surface Cache 卡在 streaming)
	//   - 计数在涨但没落盘 -> 进了 EnqueueDump, 卡在 WaitForMotion 等相机
	int32 GEnqueueCalls = 0;

	// ---------------------------------------------------------- NPY 写出 ----
	/** 拼 numpy v1.0 header: magic(6) + ver(2) + len(2) + dict, 总长 64 字节倍数. */
	static TArray<uint8> BuildNPYHeader(const FString& Descr, const TArray<int32>& Shape)
	{
		FString ShapeStr;
		for (int32 i = 0; i < Shape.Num(); ++i)
		{
			ShapeStr += FString::Printf(TEXT("%d"), Shape[i]);
			if (i + 1 < Shape.Num()) { ShapeStr += TEXT(", "); }
		}
		// 单元素元组必须写成 (N,) 形式, 否则 numpy 读成标量
		if (Shape.Num() == 1) { ShapeStr += TEXT(","); }

		const FString DictStr = FString::Printf(
			TEXT("{'descr': '%s', 'fortran_order': False, 'shape': (%s), }"), *Descr, *ShapeStr);

		const int32 FixedPrefix = 10;
		const int32 CurLen = FixedPrefix + DictStr.Len() + 1;
		const int32 Pad = (64 - (CurLen % 64)) % 64;
		FString Padding;
		for (int32 i = 0; i < Pad; ++i) { Padding.AppendChar(TEXT(' ')); }
		const FString HeaderStr = DictStr + Padding + TEXT("\n");

		TArray<uint8> Out;
		const uint8 Magic[6] = { 0x93, 'N', 'U', 'M', 'P', 'Y' };
		Out.Append(Magic, 6);
		Out.Add(1); Out.Add(0);
		const uint16 HLen = static_cast<uint16>(HeaderStr.Len());
		Out.Add(static_cast<uint8>(HLen & 0xFF));
		Out.Add(static_cast<uint8>((HLen >> 8) & 0xFF));
		const FTCHARToUTF8 Utf8(*HeaderStr);
		Out.Append(reinterpret_cast<const uint8*>(Utf8.Get()), Utf8.Length());
		return Out;
	}

	/** 写 .npy. Payload 须已是小端序列化好的字节. */
	static void WriteNPY(const FString& Path, const TArray<uint8>& Payload,
		const FString& Descr, const TArray<int32>& Shape)
	{
		TArray<uint8> Bytes = BuildNPYHeader(Descr, Shape);
		Bytes.Append(Payload);

		IPlatformFile& PF = FPlatformFileManager::Get().GetPlatformFile();
		PF.CreateDirectoryTree(*FPaths::GetPath(Path));
		FFileHelper::SaveArrayToFile(Bytes, *Path);
	}

	// ------------------------------------------------------ 逐行拷贝 --------
	/**
	 * 从 readback 的 mapped 内存按行拷贝到紧凑 buffer.
	 * GPU readback 的 row pitch 通常 > 有效行宽 (对齐要求), 不能直接整块 memcpy.
	 */
	static TArray<uint8> CopyRows(const uint8* Mapped, int32 RowPitchBytes,
		int32 RowBytes, int32 NumRows)
	{
		TArray<uint8> Out;
		Out.SetNumUninitialized(RowBytes * NumRows);
		for (int32 y = 0; y < NumRows; ++y)
		{
			FMemory::Memcpy(Out.GetData() + y * RowBytes,
				Mapped + y * RowPitchBytes, RowBytes);
		}
		return Out;
	}

	// -------------------------------------------------- radiance 解码 -------
	/**
	 * 解码 R11G11B10 -> float32 RGB, 并裁出有效区.
	 * 返回 (validH, validW, 3) 的 float32 payload (小端).
	 */
	static bool DecodeRadianceRows(const uint8* Mapped, int32 RowPitchPixels,
		int32 FullW, int32 ValidH, int32 ValidW, TArray<uint8>& OutPayload)
	{
		const uint32 SrcPitchBytes = static_cast<uint32>(RowPitchPixels) * 4u;

		TArray<FLinearColor> Row;
		Row.SetNumUninitialized(FullW);

		TArray<float> RGB;
		RGB.SetNumUninitialized(ValidH * ValidW * 3);

		for (int32 y = 0; y < ValidH; ++y)
		{
			// 该行整体解码 (含超出有效宽度的部分, 反正后面不取)
			ConvertRawR11G11B10FDataToFLinearColor(
				static_cast<uint32>(FullW), 1u,
				const_cast<uint8*>(Mapped) + y * SrcPitchBytes,
				SrcPitchBytes, Row.GetData());

			for (int32 x = 0; x < ValidW; ++x)
			{
				const int32 Dst = (y * ValidW + x) * 3;
				RGB[Dst + 0] = Row[x].R;
				RGB[Dst + 1] = Row[x].G;
				RGB[Dst + 2] = Row[x].B;
			}
		}

		OutPayload.SetNumUninitialized(RGB.Num() * sizeof(float));
		FMemory::Memcpy(OutPayload.GetData(), RGB.GetData(), OutPayload.Num());
		return true;
	}

	// ------------------------------------------------------------ JSON -----
	static FString MatrixToJson(const FMatrix& M)
	{
		// 行主序展开成 16 个数
		FString S = TEXT("[");
		for (int32 r = 0; r < 4; ++r)
		{
			for (int32 c = 0; c < 4; ++c)
			{
				S += FString::Printf(TEXT("%.9g"), M.M[r][c]);
				if (!(r == 3 && c == 3)) { S += TEXT(", "); }
			}
		}
		S += TEXT("]");
		return S;
	}

	static FString VecToJson(const FVector& V)
	{
		return FString::Printf(TEXT("[%.9g, %.9g, %.9g]"), V.X, V.Y, V.Z);
	}

	static void WriteMetaJson(const FString& Dir, const FDumpMeta& Meta)
	{
		const FString Json = FString::Printf(
			TEXT("{\n")
			TEXT("  \"project\": \"NSPF\",\n")
			TEXT("  \"stage\": \"S2\",\n")
			TEXT("  \"gt_scheme\": \"A_self_paired\",\n")
			TEXT("  \"atlas_size\": [%d, %d],\n")
			TEXT("  \"valid_atlas_size\": [%d, %d],\n")
			TEXT("  \"probe_atlas_size\": [%d, %d],\n")
			TEXT("  \"probe_view_size\": [%d, %d],\n")
			TEXT("  \"gather_octahedron_resolution\": %u,\n")
			TEXT("  \"tracing_octahedron_resolution\": %u,\n")
			TEXT("  \"downsample_factor\": %u,\n")
			TEXT("  \"files\": {\n")
			TEXT("    \"in\": {\"dtype\": \"float32\", \"shape\": \"(validH, validW, 3)\", \"desc\": \"pre-filter radiance (input)\"},\n")
			TEXT("    \"gt\": {\"dtype\": \"float32\", \"shape\": \"(validH, validW, 3)\", \"desc\": \"post-filter radiance (target)\"},\n")
			TEXT("    \"hitdist\": {\"dtype\": \"uint8\", \"shape\": \"(validH, validW)\", \"desc\": \"R8 raw, decode via LumenScreenProbeCommon.ush\"},\n")
			TEXT("    \"depth\": {\"dtype\": \"uint32\", \"shape\": \"(probeH, probeW)\", \"desc\": \"R32_UINT raw, asfloat() to recover\"},\n")
			TEXT("    \"normal\": {\"dtype\": \"uint8\", \"shape\": \"(probeH, probeW, 2)\", \"desc\": \"R8G8 octahedral encoded\"},\n")
			TEXT("    \"wpos\": {\"dtype\": \"float32\", \"shape\": \"(probeH, probeW, 4)\", \"desc\": \"translated world position + w\"}\n")
			TEXT("  }\n")
			TEXT("}\n"),
			Meta.AtlasSize.X, Meta.AtlasSize.Y,
			Meta.ValidAtlasSize.X, Meta.ValidAtlasSize.Y,
			Meta.ProbeAtlasSize.X, Meta.ProbeAtlasSize.Y,
			Meta.ProbeViewSize.X, Meta.ProbeViewSize.Y,
			Meta.GatherOctahedronResolution,
			Meta.TracingOctahedronResolution,
			Meta.DownsampleFactor);

		const FString Path = FPaths::Combine(Dir, TEXT("meta.json"));
		IPlatformFile& PF = FPlatformFileManager::Get().GetPlatformFile();
		PF.CreateDirectoryTree(*FPaths::GetPath(Path));
		FFileHelper::SaveStringToFile(Json, *Path);
	}

	static void WriteFrameJson(const FString& Dir, int32 FrameIndex,
		const FDumpMeta& Meta, const FCameraInfo& Cam)
	{
		const FString Json = FString::Printf(
			TEXT("{\n")
			TEXT("  \"frame\": %d,\n")
			TEXT("  \"view_matrix\": %s,\n")
			TEXT("  \"proj_matrix\": %s,\n")
			TEXT("  \"pre_view_translation\": %s,\n")
			TEXT("  \"prev_pre_view_translation\": %s,\n")
			TEXT("  \"prev_valid\": %s,\n")
			TEXT("  \"probe_view_size\": [%d, %d]\n")
			TEXT("}\n"),
			FrameIndex,
			*MatrixToJson(Cam.ViewMatrix),
			*MatrixToJson(Cam.ProjMatrix),
			*VecToJson(Cam.PreViewTranslation),
			*VecToJson(Cam.PrevPreViewTranslation),
			Cam.bPrevValid ? TEXT("true") : TEXT("false"),
			Meta.ProbeViewSize.X, Meta.ProbeViewSize.Y);

		const FString Path = FPaths::Combine(Dir,
			FString::Printf(TEXT("frame_%06d.json"), FrameIndex));
		FFileHelper::SaveStringToFile(Json, *Path);
	}
}
// [theozhu--]

// ------------------------------------------------------------- 对外接口 ------
bool LumenScreenProbeDump::IsEnabled()
{
	return CVarLumenScreenProbeDumpEnable != 0;
}

// [theozhu++]
void LumenScreenProbeDump::EnqueueDump(
	FRDGBuilder& GraphBuilder,
	const FViewInfo& View,
	const FScreenProbeParameters& ScreenProbeParameters,
	FRDGTextureRef CompositedRadiance,
	FRDGTextureRef FilteredRadiance,
	FRDGTextureRef HitDistance,
	FRDGTextureRef SceneDepth,
	FRDGTextureRef WorldNormal,
	FRDGTextureRef WorldPosition)
{
	// 挂在渲染路径上轮询, 避免额外引入 Tick 点
	Tick();

	// 用命名空间别名而非 using namespace: 引擎全局已有 GFrameCounter 等符号,
	// using namespace 会同时引入二者造成歧义 (MSVC C2872).
	namespace Dump = LumenScreenProbeDumpPrivate;

	// [theozhu++] 新一轮采集判定 + 状态重置
	//
	// 背景: GFrameCounter / GWrittenCount / bMetaWritten 是静态全局状态,
	// 只在 Shutdown() (模块卸载) 时归零, 而 Enable 1->0 并不触发 Shutdown.
	// 于是第二轮采集 (方案 B 的高射线那一遍) 会因为 GFrameCounter 仍是
	// 上一轮的上限而立即 return —— 一帧都采不到, 输出目录也不会被创建.
	//
	// 判定"新一轮"的两个信号:
	//   1. Enable 的 0->1 上升沿
	//   2. OutputDir 发生变化
	// 任一成立即重置计数, 让第二轮从 frame_000000 开始编号.
	//
	// 注意: 必须放在 Enable==0 的 early return 之前. 否则 Enable 归零那些帧
	// 会提前返回, GLastEnable 永远停在 1, 上升沿就检测不到了.
	{
		static int32 GLastEnable = 0;
		static FString GLastOutputDir;

		const int32  CurEnable   = CVarLumenScreenProbeDumpEnable;
		const bool   bRisingEdge = (CurEnable != 0 && GLastEnable == 0);
		const bool   bDirChanged = (CVarLumenScreenProbeDumpOutputDir != GLastOutputDir);

		if (bRisingEdge || bDirChanged)
		{
			Dump::GFrameCounter = 0;
			Dump::GWrittenCount = 0;
			Dump::bMetaWritten  = false;
			// 重新开始"等相机起动", 否则第二轮会沿用上一轮已 armed 的状态,
			// 直接开采 —— 那就又回到手动对齐的老问题了.
			Dump::bArmed        = false;
			Dump::ArmWaitFrames = 0;
			// 心跳计数按"本轮"归零, 日志里的 #N 就是本轮第几次进来
			Dump::GEnqueueCalls = 0;
			UE_LOG(LogRenderer, Display,
				TEXT("[theozhu] ScreenProbeDump: new session (rising=%d dirChanged=%d)\n")
				TEXT("      OutputDir    = '%s'\n")
				TEXT("      MaxFrames    = %d (0=不限)\n")
				TEXT("      WaitForMotion= %d\n")
				TEXT("      TraceRes     = %u (%u 条射线/probe)\n")
				TEXT("      GatherRes    = %u (atlas 每 probe %ux%u)\n")
				TEXT("      ProbeGrid    = %dx%d, Downsample=%u"),
				bRisingEdge ? 1 : 0, bDirChanged ? 1 : 0,
				*CVarLumenScreenProbeDumpOutputDir,
				CVarLumenScreenProbeDumpMaxFrames,
				CVarLumenScreenProbeDumpWaitForMotion,
				ScreenProbeParameters.ScreenProbeTracingOctahedronResolution,
				ScreenProbeParameters.ScreenProbeTracingOctahedronResolution
					* ScreenProbeParameters.ScreenProbeTracingOctahedronResolution,
				ScreenProbeParameters.ScreenProbeGatherOctahedronResolution,
				ScreenProbeParameters.ScreenProbeGatherOctahedronResolution,
				ScreenProbeParameters.ScreenProbeGatherOctahedronResolution,
				ScreenProbeParameters.ScreenProbeViewSize.X,
				ScreenProbeParameters.ScreenProbeViewSize.Y,
				ScreenProbeParameters.ScreenProbeDownsampleFactor);
		}

		GLastEnable   = CurEnable;
		GLastOutputDir = CVarLumenScreenProbeDumpOutputDir;
	}

	if (CVarLumenScreenProbeDumpEnable == 0) { return; }

	// [theozhu++] 心跳: 证明 Lumen 管线确实跑到了这里 (FilterScreenProbes 内).
	// 若 Enable=1 后这行【完全不出现】, 说明 Lumen 渲染根本没走到这一步
	// (典型: Surface Cache 卡在 streaming, 见
	//  "[LumenRuntimeSC] ... Stage=Waiting" 警告), 与 WaitForMotion 无关.
	// 节流: 前 3 次全打, 之后每 60 次打一次, 避免刷屏.
	++Dump::GEnqueueCalls;
	if (Dump::GEnqueueCalls <= 3 || Dump::GEnqueueCalls % 60 == 0)
	{
		UE_LOG(LogRenderer, Display,
			TEXT("[theozhu] ScreenProbeDump: EnqueueDump #%d "
				 "(frameCounter=%d, armed=%d, written=%d, maxFrames=%d)"),
			Dump::GEnqueueCalls, Dump::GFrameCounter,
			Dump::bArmed ? 1 : 0, Dump::GWrittenCount,
			CVarLumenScreenProbeDumpMaxFrames);
	}

	if (Dump::GFrameCounter >= CVarLumenScreenProbeDumpMaxFrames
		&& CVarLumenScreenProbeDumpMaxFrames > 0)
	{
		return;
	}

	const int32 GatherRes = static_cast<int32>(ScreenProbeParameters.ScreenProbeGatherOctahedronResolution);

	Dump::FPendingDump Pending;
	// 注意: FrameIndex 在这里【不】自增! 必须等 WaitForMotion 放行后再编号.
	// 否则等待相机起动的那些帧会白白吃掉 MaxFrames 配额 —— 例如等了 83 帧,
	// 首帧就变成 frame_000083, 只剩 (MaxFrames-83) 帧可采, 且两遍编号对不齐.
	Pending.FrameIndex = -1;
	Pending.Meta.AtlasSize = ScreenProbeParameters.ScreenProbeAtlasBufferSize * GatherRes;
	Pending.Meta.ProbeAtlasSize = ScreenProbeParameters.ScreenProbeAtlasBufferSize;
	Pending.Meta.ProbeViewSize = ScreenProbeParameters.ScreenProbeViewSize;
	Pending.Meta.GatherOctahedronResolution = ScreenProbeParameters.ScreenProbeGatherOctahedronResolution;
	Pending.Meta.TracingOctahedronResolution = ScreenProbeParameters.ScreenProbeTracingOctahedronResolution;
	Pending.Meta.DownsampleFactor = ScreenProbeParameters.ScreenProbeDownsampleFactor;

	// 有效区 = probeViewSize(probe 数) * gatherRes(texel)
	Pending.Meta.ValidAtlasSize = FIntPoint(
		FMath::Min(Pending.Meta.ProbeViewSize.X * GatherRes, Pending.Meta.AtlasSize.X),
		FMath::Min(Pending.Meta.ProbeViewSize.Y * GatherRes, Pending.Meta.AtlasSize.Y));

	// 相机信息 (时域对齐)
	// 注意: 成员名是 bStatePrevViewInfoIsReadOnly (SceneRendering.h:1378),
	//       不是 bPrevViewInfoIsReadOnly. 上一帧矩阵取自 FPreviousViewInfo
	//       (定义在 SceneRendering.h, 非 SceneView.h).
	Pending.Camera.ViewMatrix = View.ViewMatrices.GetViewMatrix();
	Pending.Camera.ProjMatrix = View.ViewMatrices.GetProjectionMatrix();
	Pending.Camera.PreViewTranslation = View.ViewMatrices.GetPreViewTranslation();
	Pending.Camera.PrevPreViewTranslation = View.PrevViewInfo.ViewMatrices.GetPreViewTranslation();
	Pending.Camera.bPrevValid = (View.ViewState != nullptr) && !View.bStatePrevViewInfoIsReadOnly;

	// [theozhu++] 等待相机开始运动, 消除"手动 Enable -> 手动点 Play"的延迟差
	//
	// 问题: 方案 B 要求两遍采集逐帧对齐. 但实操是"先 Enable, 再伸手去点
	// Sequencer 的 Play", 中间必然有人的反应延迟 (几十到几百毫秒), 且两遍
	// 的延迟不可能一样. 不处理的话, 两遍的 frame_000000 对应的相机位置不同,
	// 输入与答案错位 —— 网络学到的是噪声, 而训练不会报错.
	//
	// 解法: Enable 后不立即计数, 等到相机【开始运动】的那一帧才起算.
	// 这样无论人手慢还是快, 两遍的 frame_000000 都对应"相机刚起步".
	if (CVarLumenScreenProbeDumpWaitForMotion != 0 && !Dump::bArmed)
	{
		const Dump::FCameraInfo& Cam = Pending.Camera;

		// 进入等待的第一帧: 只记录参考相机, 不采集
		if (Dump::ArmWaitFrames == 0)
		{
			Dump::ArmRefCamera  = Cam;
			Dump::ArmWaitFrames = 1;
			return;
		}

		// 位移: PreViewTranslation 的差 (UE 单位 cm), 比矩阵直观
		const double PosDiff = (Cam.PreViewTranslation
			- Dump::ArmRefCamera.PreViewTranslation).Size();

		// 朝向: 直接比 ViewMatrix 元素的最大变化
		double MatDiff = 0.0;
		for (int32 r = 0; r < 4; ++r)
		{
			for (int32 c = 0; c < 4; ++c)
			{
				MatDiff = FMath::Max(MatDiff,
					FMath::Abs(Cam.ViewMatrix.M[r][c]
						- Dump::ArmRefCamera.ViewMatrix.M[r][c]));
			}
		}

		++Dump::ArmWaitFrames;

		const bool bMoved   = (PosDiff > 1.0) || (MatDiff > 1e-4);
		const bool bTimeout = (Dump::ArmWaitFrames > 600);   // 兜底, 避免永久等待

		if (!bMoved && !bTimeout)
		{
			// 节流: 每 30 帧报一次"还在等", 并带上当前差异量.
			// 这样用户能判断是"相机真没动"还是"动了但没过阈值".
			if (Dump::ArmWaitFrames % 30 == 0)
			{
				UE_LOG(LogRenderer, Display,
					TEXT("[theozhu] ScreenProbeDump: waiting for camera motion... %d frames "
						 "(posDiff=%.4f / need >1.0, matDiff=%.3e / need >1e-4)"),
					Dump::ArmWaitFrames, PosDiff, MatDiff);
			}
			return;   // 相机还没动: 不计数, 不落盘
		}

		Dump::bArmed = true;
		UE_LOG(LogRenderer, Display,
			TEXT("[theozhu] ScreenProbeDump: camera started after %d waiting frames "
				 "(posDiff=%.2f, matDiff=%.3e%s) - capturing from frame_000000 -> '%s'"),
			Dump::ArmWaitFrames, PosDiff, MatDiff,
			bTimeout ? TEXT(" [TIMEOUT]") : TEXT(""),
			*CVarLumenScreenProbeDumpOutputDir);
	}

	// [theozhu++] 通过 WaitForMotion (或未启用等待) 后, 此刻才正式编号.
	// 这样等待相机起动的帧不占用配额, 首帧恒为 frame_000000,
	// 两遍采集无论手速快慢都从同一个"相机刚起步"的时刻开始编号.
	Pending.FrameIndex = Dump::GFrameCounter++;

	// 6 路异步 readback
	FRDGTextureRef Surfaces[Dump::NumSurfaces] =
	{
		CompositedRadiance,
		FilteredRadiance,
		HitDistance,
		SceneDepth,
		WorldNormal,
		WorldPosition
	};
	static const TCHAR* SurfaceNames[Dump::NumSurfaces] =
	{
		TEXT("Theozhu.ProbeDump.InputRadiance"),
		TEXT("Theozhu.ProbeDump.GTRadiance"),
		TEXT("Theozhu.ProbeDump.HitDistance"),
		TEXT("Theozhu.ProbeDump.SceneDepth"),
		TEXT("Theozhu.ProbeDump.WorldNormal"),
		TEXT("Theozhu.ProbeDump.WorldPosition")
	};

	for (uint32 i = 0; i < Dump::NumSurfaces; ++i)
	{
		if (Surfaces[i])
		{
			Pending.Readbacks[i] = MakeUnique<FRHIGPUTextureReadback>(SurfaceNames[i]);
			AddEnqueueCopyPass(GraphBuilder, Pending.Readbacks[i].Get(), Surfaces[i]);
		}
	}

	if (!Dump::bMetaWritten)
	{
		const FString Dir = CVarLumenScreenProbeDumpOutputDir;
		const Dump::FDumpMeta MetaCopy = Pending.Meta;
		AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask, [Dir, MetaCopy]()
		{
			Dump::WriteMetaJson(Dir, MetaCopy);
		});
		Dump::bMetaWritten = true;
	}

	Dump::GPendingDumps.Add(MoveTemp(Pending));
}
// [theozhu--]

// [theozhu++]
void LumenScreenProbeDump::Tick()
{
	namespace Dump = LumenScreenProbeDumpPrivate;

	while (Dump::GPendingDumps.Num() > 0)
	{
		Dump::FPendingDump& Front = Dump::GPendingDumps[0];
		if (!Front.Readbacks[0].IsValid() || !Front.Readbacks[0]->IsReady())
		{
			// 队首未就绪则整体停住, 保持 FIFO 顺序
			break;
		}

		Dump::FPendingDump Pending = MoveTemp(Dump::GPendingDumps[0]);
		Dump::GPendingDumps.RemoveAt(0, 1, EAllowShrinking::No);

		const FString Dir = CVarLumenScreenProbeDumpOutputDir;
		const FString Base = FString::Printf(TEXT("frame_%06d"), Pending.FrameIndex);
		const int32 FrameIdx = Pending.FrameIndex;
		const Dump::FDumpMeta Meta = Pending.Meta;
		const Dump::FCameraInfo Cam = Pending.Camera;

		const int32 FullW = Meta.AtlasSize.X;
		const int32 ValidH = Meta.ValidAtlasSize.Y;
		const int32 ValidW = Meta.ValidAtlasSize.X;
		const int32 ProbeW = Meta.ProbeViewSize.X;
		const int32 ProbeH = Meta.ProbeViewSize.Y;

		if (FullW <= 0 || ValidH <= 0 || ValidW <= 0)
		{
			continue;
		}

		// ---- 两个 radiance: 解码 R11G11B10 -> float32, 裁有效区 ----
		struct FRadianceSurface
		{
			const TCHAR* Tag;
			uint32 Slot;
		};

		const FRadianceSurface RadianceSurfaces[] =
		{
			{ TEXT("in"), (uint32)Dump::EDumpSurface::InputRadiance },
			{ TEXT("gt"), (uint32)Dump::EDumpSurface::GTRadiance },
		};

		for (const FRadianceSurface& S : RadianceSurfaces)
		{
			const TCHAR* Tag = S.Tag;
			const uint32 Slot = S.Slot;

			if (!Pending.Readbacks[Slot].IsValid()) { continue; }

			int32 RowPitchPixels = 0;
			int32 BufferHeight = 0;
			void* Mapped = Pending.Readbacks[Slot]->Lock(RowPitchPixels, &BufferHeight);
			if (!Mapped)
			{
				Pending.Readbacks[Slot]->Unlock();
				continue;
			}

			TArray<uint8> Payload;
			Dump::DecodeRadianceRows(static_cast<const uint8*>(Mapped), RowPitchPixels,
				FullW, ValidH, ValidW, Payload);
			Pending.Readbacks[Slot]->Unlock();

			const FString Path = FPaths::Combine(Dir, FString::Printf(TEXT("%s_%s.npy"), *Base, Tag));
			AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask,
				[Path, Payload = MoveTemp(Payload), ValidH, ValidW]()
			{
				Dump::WriteNPY(Path, Payload, TEXT("<f4"), { ValidH, ValidW, 3 });
			});
		}

		// ---- 其余 4 路: 原始字节, 只按行拷贝 + 裁有效区 ----
		struct FRawSurface
		{
			const TCHAR* Tag;
			uint32 Slot;
			int32 ElemBytes;    // 每像素字节数
			int32 NumChannels;  // 输出 shape 的最后一维
			const TCHAR* Descr;
			bool bProbeLevel;   // true = probe 级尺寸 (probeH, probeW), false = texel 级
		};

		const FRawSurface RawSurfaces[] =
		{
			{ TEXT("hitdist"), (uint32)Dump::EDumpSurface::HitDistance,     1,  1, TEXT("<u1"), false },
			{ TEXT("depth"),   (uint32)Dump::EDumpSurface::SceneDepth,      4,  1, TEXT("<u4"), true  },
			{ TEXT("normal"),  (uint32)Dump::EDumpSurface::WorldNormal,     2,  2, TEXT("<u1"), true  },
			{ TEXT("wpos"),    (uint32)Dump::EDumpSurface::WorldPosition,  16,  4, TEXT("<f4"), true  },
		};

		for (const FRawSurface& S : RawSurfaces)
		{
			if (!Pending.Readbacks[S.Slot].IsValid()) { continue; }

			int32 RowPitchPixels = 0;
			int32 BufferHeight = 0;
			void* Mapped = Pending.Readbacks[S.Slot]->Lock(RowPitchPixels, &BufferHeight);
			if (!Mapped)
			{
				Pending.Readbacks[S.Slot]->Unlock();
				continue;
			}

			const int32 Rows = S.bProbeLevel ? ProbeH : ValidH;
			const int32 Cols = S.bProbeLevel ? ProbeW : ValidW;
			const int32 RowPitchBytes = RowPitchPixels * S.ElemBytes;
			const int32 RowBytes = Cols * S.ElemBytes;

			TArray<uint8> Payload = Dump::CopyRows(static_cast<const uint8*>(Mapped),
				RowPitchBytes, RowBytes, Rows);
			Pending.Readbacks[S.Slot]->Unlock();

			const FString Path = FPaths::Combine(Dir, FString::Printf(TEXT("%s_%s.npy"), *Base, S.Tag));
			const FString Descr = S.Descr;
			TArray<int32> Shape = (S.NumChannels > 1)
				? TArray<int32>{ Rows, Cols, S.NumChannels }
				: TArray<int32>{ Rows, Cols };

			AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask,
				[Path, Payload = MoveTemp(Payload), Descr, Shape = MoveTemp(Shape)]()
			{
				Dump::WriteNPY(Path, Payload, Descr, Shape);
			});
		}

		// ---- 相机 json ----
		AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask,
			[Dir, FrameIdx, Meta, Cam]()
		{
			Dump::WriteFrameJson(Dir, FrameIdx, Meta, Cam);
		});

		if (++Dump::GWrittenCount == 1)
		{
			UE_LOG(LogRenderer, Log,
				TEXT("[theozhu] ScreenProbeDump: first frame written -> %s/%s.npy "
					 "(in/gt %dx%d, probe grid %dx%d)"),
				*Dir, *Base, ValidW, ValidH, ProbeW, ProbeH);
		}
	}
}
// [theozhu--]

// [theozhu++]
void LumenScreenProbeDump::Shutdown()
{
	namespace Dump = LumenScreenProbeDumpPrivate;
	Dump::GPendingDumps.Empty();
	Dump::GFrameCounter = 0;
	Dump::GWrittenCount = 0;
	Dump::bMetaWritten = false;
	Dump::bArmed = false;
	Dump::ArmWaitFrames = 0;
	Dump::GEnqueueCalls = 0;
}
// [theozhu--]

#else // UE_BUILD_SHIPPING —— 空实现, 保证调用点链接通过

bool LumenScreenProbeDump::IsEnabled() { return false; }

void LumenScreenProbeDump::EnqueueDump(
	FRDGBuilder& GraphBuilder,
	const FViewInfo& View,
	const FScreenProbeParameters& ScreenProbeParameters,
	FRDGTextureRef CompositedRadiance,
	FRDGTextureRef FilteredRadiance,
	FRDGTextureRef HitDistance,
	FRDGTextureRef SceneDepth,
	FRDGTextureRef WorldNormal,
	FRDGTextureRef WorldPosition)
{
}

void LumenScreenProbeDump::Tick() {}
void LumenScreenProbeDump::Shutdown() {}

#endif // !UE_BUILD_SHIPPING
