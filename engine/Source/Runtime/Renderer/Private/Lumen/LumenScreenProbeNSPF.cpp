// Copyright Epic Games, Inc. All Rights Reserved.

#include "LumenScreenProbeNSPF.h"

#include "GlobalShader.h"
#include "ShaderCompilerCore.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
// 注意: 没有独立的 ComputeShaderUtils.h —— FComputeShaderUtils 就定义在
// RenderGraphUtils.h 里 (namespace FComputeShaderUtils, 见该文件 341 行).
#include "LumenScreenProbeGather.h"    // FScreenProbeParameters
#include "RendererPrivate.h"           // LogRenderer
#include "ScenePrivate.h"              // FViewInfo

// 注意: 引擎里【不存在】WITH_NNE / WITH_NNE_RUNTIME_HLSL 这类宏 —— 在整个
// Engine/Source 下搜不到任何 #define. 拿它们做条件编译, 条件恒为假,
// 下面的头永远不被包含 -> UE::NNE 命名空间不存在 -> C3083/C2039 一串错.
// Renderer.Build.cs 已加 "NNE" 依赖, 直接包含即可 (NNE 是 Runtime 模块).
#include "NNE.h"
#include "NNEModelData.h"
#include "NNERuntimeRDG.h"
#include "HAL/IConsoleManager.h"   // IConsoleManager / SetOnChangedCallback
#include "RenderingThread.h"       // FlushRenderingCommands (切换后端前同步)

// ------------------------------------------------------------------ 常量 --
// ★ 必须与部署模型的实际输入维度一致!
//   --no-history 训出的模型 in_dim=52 (历史维度被【完全移除】, 不是填 0)
//   带历史训出的模型 in_dim=56.
//   当前部署模型 runs/b_nohist = 52, 见 experiments/02-S4引擎接入/LOG.md.
//   换模型时若维度不同, 这里和 NSPFFeatureBuild.usf 的 stride 必须同步改.
static constexpr uint32 NSPF_IN_DIM     = 52;
static constexpr uint32 NSPF_OUT_DIM    = 3;
static constexpr uint32 NSPF_GROUP_SIZE = 8;

// ------------------------------------------------------------------ CVar --
static int32 CVarLumenScreenProbeNSPFEnable = 0;
static FAutoConsoleVariableRef CVarLumenScreenProbeNSPFEnableRef(
	TEXT("r.Lumen.ScreenProbeNSPF.Enable"),
	CVarLumenScreenProbeNSPFEnable,
	TEXT("[theozhu] NSPF: 用 NNE 替换 Lumen Screen Probe 传统滤波. 0=关(默认) 1=开"),
	ECVF_RenderThreadSafe);

// 推理后端:
//   "NNERuntimeRDGHlsl" —— 通用 HLSL compute shader. 兼容性最好, 但 Gemm 是
//                          通用实现, 无 tensor core / 算子融合, 实测约 22ms.
//   "NNERuntimeORTDml"  —— ONNX Runtime + DirectML, D3D12 原生 GPU 后端.
//                          NNEDenoiser 的 RDG 模式默认就选它
//                          (NNEDenoiserViewExtension.cpp:129); 同为 D3D12,
//                          与渲染共享显存 —— 不需要 CUDA 互操作.
static FString CVarLumenScreenProbeNSPFRuntime = TEXT("NNERuntimeORTDml");
static FAutoConsoleVariableRef CVarLumenScreenProbeNSPFRuntimeRef(
	TEXT("r.Lumen.ScreenProbeNSPF.Runtime"),
	CVarLumenScreenProbeNSPFRuntime,
	TEXT("[theozhu] NSPF: NNE 推理后端 (NNERuntimeRDGHlsl / NNERuntimeORTDml)"),
	ECVF_RenderThreadSafe);

static FString CVarLumenScreenProbeNSPFModel = TEXT("/Game/model_nspf_dyn.model_nspf_dyn");
static FAutoConsoleVariableRef CVarLumenScreenProbeNSPFModelRef(
	TEXT("r.Lumen.ScreenProbeNSPF.Model"),
	CVarLumenScreenProbeNSPFModel,
	TEXT("[theozhu] NSPF: NNE 模型资产路径 (NNEModelData, 由 onnx 导入)"),
	ECVF_RenderThreadSafe);

// ★ 验证开关: 把 NN 输出在写回前整体乘一个系数, 用来【肉眼确认 NN 输出是否
//   真的进入了最终画面】.
//   用法: 设成 3.0 → 若画面 GI 明显变亮, 证明管线通、输出被下游消费;
//         设回 1.0 恢复正常. 默认 1.0 (不做任何缩放).
static float CVarLumenScreenProbeNSPFDebugScale = 1.0f;
static FAutoConsoleVariableRef CVarLumenScreenProbeNSPFDebugScaleRef(
	TEXT("r.Lumen.ScreenProbeNSPF.DebugScale"),
	CVarLumenScreenProbeNSPFDebugScale,
	TEXT("[theozhu] NSPF: 诊断用 —— 缩放 NN 输出以验证其是否生效 (默认 1.0)"),
	ECVF_RenderThreadSafe);

// -------------------------------------------------------------- Shader ---
class FNSPFFeatureBuildCS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FNSPFFeatureBuildCS);
	SHADER_USE_PARAMETER_STRUCT(FNSPFFeatureBuildCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(FIntPoint, AtlasViewSize)
		SHADER_PARAMETER(FIntPoint, ProbeViewSize)
		SHADER_PARAMETER(int32, GatherResolution)
		SHADER_PARAMETER(float, DownsampleFactor)

		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, CompositedRadianceTexture)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, HitDistanceTexture)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, SceneDepthTexture)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, WorldNormalTexture)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, WorldPositionTexture)
		// 没有 sampler: shader 全部用 .Load() 做 texel fetch

		// 必须显式给 Format, 否则 "Format cannot be unknown for typed buffers"
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<float>, OutputFeatureBuffer)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return DoesPlatformSupportLumenGI(Parameters.Platform);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
	}
};
IMPLEMENT_GLOBAL_SHADER(FNSPFFeatureBuildCS, "/Engine/Private/Lumen/NSPFFeatureBuild.usf", "BuildFeaturesCS", SF_Compute);

class FNSPFWriteBackCS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FNSPFWriteBackCS);
	SHADER_USE_PARAMETER_STRUCT(FNSPFWriteBackCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(FIntPoint, AtlasViewSize)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<float>, InputOutputBuffer)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float3>, OutputRadianceTexture)
		SHADER_PARAMETER(float, DebugScale)   // 诊断: 缩放 NN 输出以验证生效
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return DoesPlatformSupportLumenGI(Parameters.Platform);
	}
};
IMPLEMENT_GLOBAL_SHADER(FNSPFWriteBackCS, "/Engine/Private/Lumen/NSPFWriteBack.usf", "WriteBackCS", SF_Compute);

// ------------------------------------------------------- NNE 模型管理 ----
namespace
{
	// ★★ 线程模型 (血泪教训) ★★
	//   LoadObject 会创建 UObject (StaticAllocateObject -> FUObjectArray),
	//   内部断言 IsInGameThread —— 在渲染线程调用直接 Fatal error (实测).
	//   因此:
	//     - 模型加载【只能】在 GameThread 进行:
	//       由 Enable CVar 的 OnChanged 回调触发 (控制台命令在 GameThread 执行)
	//     - 渲染线程只读裸指针, 绝不加载
	//   TSharedPtr 引用计数非线程安全, GameThread 用 owner 持有引用,
	//   渲染线程访问裸指针.
	TSharedPtr<UE::NNE::IModelInstanceRDG> GNSPFModelOwner;   // GameThread 持有
	UE::NNE::IModelInstanceRDG* GNSPFInstance = nullptr;      // 渲染线程只读
	FString GNSPFLoadedPath;
	bool bGNSPFInitFailed  = false;
	bool bGWarnedNotLoaded = false;
	// 输入形状缓存: atlas 尺寸不变时避免每帧重复 SetInputTensorShapes
	uint32 GCachedNumSamples = 0;

	/** 预加载模型. ★ 只能在 GameThread 调用. */
	void LoadModelOnGameThread()
	{
		checkf(IsInGameThread(),
			TEXT("NSPF: 模型加载必须在 GameThread (LoadObject 创建 UObject)"));

		if (GNSPFInstance || bGNSPFInitFailed)
		{
			return;
		}

		const FString Path = CVarLumenScreenProbeNSPFModel;

		UNNEModelData* ModelData = LoadObject<UNNEModelData>(nullptr, *Path);
		if (!ModelData)
		{
			UE_LOG(LogRenderer, Warning,
				TEXT("[theozhu] NSPF: ModelData 未找到 '%s' (先导入 onnx), 沿用传统滤波"), *Path);
			bGNSPFInitFailed = true;
			return;
		}

		// 后端可由 CVar 切换: NNERuntimeRDGHlsl(通用HLSL) / NNERuntimeORTDml(DirectML)
		TWeakInterfacePtr<INNERuntimeRDG> Runtime =
			UE::NNE::GetRuntime<INNERuntimeRDG>(*CVarLumenScreenProbeNSPFRuntime);
		if (!Runtime.IsValid())
		{
			UE_LOG(LogRenderer, Warning,
				TEXT("[theozhu] NSPF: NNE RDG 运行时 '%s' 不可用, 沿用传统滤波"),
				*CVarLumenScreenProbeNSPFRuntime);
			bGNSPFInitFailed = true;
			return;
		}

		TSharedPtr<UE::NNE::IModelRDG> Model = Runtime->CreateModelRDG(ModelData);
		if (!Model.IsValid())
		{
			UE_LOG(LogRenderer, Warning,
				TEXT("[theozhu] NSPF: CreateModelRDG 失败 (算子不支持?), 沿用传统滤波"));
			bGNSPFInitFailed = true;
			return;
		}

		GNSPFModelOwner = Model->CreateModelInstanceRDG();
		GNSPFInstance   = GNSPFModelOwner.Get();
		GNSPFLoadedPath = Path;

		if (GNSPFInstance)
		{
			UE_LOG(LogRenderer, Display,
				TEXT("[theozhu] NSPF: 模型已加载 '%s' (GameThread 预加载)"), *Path);
		}
	}

	/**
	 * 丢弃当前模型并按当前 CVar 重新加载 (切换后端/模型时用).
	 * ★ 只能在 GameThread 调用.
	 */
	void ReloadModelOnGameThread()
	{
		checkf(IsInGameThread(),
			TEXT("NSPF: ReloadModelOnGameThread 必须在 GameThread"));

		// 关键: 必须等渲染线程空闲再销毁旧实例.
		//   否则 Reset owner 会让渲染线程正在使用的裸指针变成悬垂指针 -> 崩溃.
		FlushRenderingCommands();

		GNSPFInstance = nullptr;
		GNSPFModelOwner.Reset();
		GNSPFLoadedPath.Empty();
		bGNSPFInitFailed  = false;
		bGWarnedNotLoaded = false;
		GCachedNumSamples = 0;

		if (CVarLumenScreenProbeNSPFEnable != 0)
		{
			LoadModelOnGameThread();
		}
	}
}

/**
 * Enable CVar 打开时, 在 GameThread 预加载模型.
 * 控制台命令在 GameThread 执行 -> OnChanged 回调也在 GameThread -> 安全.
 */
struct FNSPFEnableHook
{
	FNSPFEnableHook()
	{
		IConsoleVariable* CVar = IConsoleManager::Get().FindConsoleVariable(
			TEXT("r.Lumen.ScreenProbeNSPF.Enable"));
		if (CVar)
		{
			CVar->SetOnChangedCallback(FConsoleVariableDelegate::CreateLambda(
				[](IConsoleVariable*)
				{
					if (CVarLumenScreenProbeNSPFEnable != 0)
					{
						LoadModelOnGameThread();
					}
				}));
		}

		// 切换后端 / 模型路径 -> 热重载 (无需重启编辑器).
		//   LoadModelOnGameThread 有"已加载就跳过"的短路, 所以必须走
		//   ReloadModelOnGameThread 先丢弃旧实例才会真正生效.
		for (const TCHAR* Name : {TEXT("r.Lumen.ScreenProbeNSPF.Runtime"),
		                          TEXT("r.Lumen.ScreenProbeNSPF.Model")})
		{
			if (IConsoleVariable* C = IConsoleManager::Get().FindConsoleVariable(Name))
			{
				C->SetOnChangedCallback(FConsoleVariableDelegate::CreateLambda(
					[](IConsoleVariable*)
					{
						if (IsInGameThread())
						{
							ReloadModelOnGameThread();
						}
					}));
			}
		}
	}
};
static FNSPFEnableHook GNSPFEnableHook;

// ------------------------------------------------------ 主入口 -----------
FRDGTextureRef LumenScreenProbeNSPF::AddNSPFFilterPass(
	FRDGBuilder& GraphBuilder,
	const FViewInfo& View,
	const FScreenProbeParameters& ScreenProbeParameters,
	FRDGTextureRef CompositedScreenProbeRadiance,
	FRDGTextureRef ScreenProbeHitDistance,
	FRDGTextureRef ScreenProbeSceneDepth,
	FRDGTextureRef ScreenProbeWorldNormal,
	FRDGTextureRef ScreenProbeTranslatedWorldPosition)
{
	if (CVarLumenScreenProbeNSPFEnable == 0)
	{
		return nullptr;
	}

	// 渲染线程只读缓存 (模型加载在 GameThread 的 Enable 回调里完成)
	UE::NNE::IModelInstanceRDG* Instance = GNSPFInstance;
	if (!Instance)
	{
		if (!bGWarnedNotLoaded)
		{
			bGWarnedNotLoaded = true;
			UE_LOG(LogRenderer, Warning,
				TEXT("[theozhu] NSPF: 模型未加载, 沿用传统滤波. "
					 "重新执行 r.Lumen.ScreenProbeNSPF.Enable 1 可触发 GameThread 加载"));
		}
		return nullptr;
	}

	// 日志限流: 前 5 帧全打(建链初期), 之后每 300 帧打一次摘要
	static uint32 GNSPFPassCount = 0;
	const bool bVerbose = (GNSPFPassCount < 5) || (GNSPFPassCount % 300 == 0);
	++GNSPFPassCount;
	if (bVerbose)
	{
		UE_LOG(LogRenderer, Display,
			TEXT("[theozhu] NSPF: ---- Pass #%u 开始 (atlas=%dx%d, 后端=%s) ----"),
			GNSPFPassCount,
			CompositedScreenProbeRadiance->Desc.Extent.X,
			CompositedScreenProbeRadiance->Desc.Extent.Y,
			*CVarLumenScreenProbeNSPFRuntime);
	}

	const FIntPoint AtlasSize = FIntPoint(
		CompositedScreenProbeRadiance->Desc.Extent.X,
		CompositedScreenProbeRadiance->Desc.Extent.Y);
	const FIntPoint ProbeViewSize = FIntPoint(
		ScreenProbeParameters.ScreenProbeViewSize.X,
		ScreenProbeParameters.ScreenProbeViewSize.Y);
	const uint32 GatherRes = ScreenProbeParameters.ScreenProbeGatherOctahedronResolution;
	const uint32 NumSamples = (uint32)(AtlasSize.X * AtlasSize.Y);

	if (NumSamples == 0 || GatherRes == 0)
	{
		static uint32 C = 0;
		if (++C <= 3 || C % 300 == 0)
		{
			UE_LOG(LogRenderer, Warning,
				TEXT("[theozhu] NSPF: [失败] NumSamples/GatherRes 为 0 (atlas=%dx%d, gather=%u), 跳过"),
				AtlasSize.X, AtlasSize.Y, GatherRes);
		}
		return nullptr;
	}

	// ---- 输入形状: {NumSamples, NSPF_IN_DIM}, 只在样本数变化时设置 ----
	//   atlas 尺寸固定时每帧相同, 重复设置无意义且有跨线程风险.
	//   只有渲染线程会走到这里 (AddNSPFFilterPass 在渲染线程), 缓存无竞争.
	if (GCachedNumSamples != NumSamples)
	{
		const uint32 InDims[2] = {NumSamples, NSPF_IN_DIM};
		UE::NNE::FTensorShape InShape =
			UE::NNE::FTensorShape::Make(TConstArrayView<uint32>(InDims, 2));
		if (Instance->SetInputTensorShapes(
				TConstArrayView<UE::NNE::FTensorShape>(&InShape, 1))
			!= UE::NNE::IModelInstanceRDG::ESetInputTensorShapesStatus::Ok)
		{
			static uint32 C = 0;
			if (++C <= 3 || C % 300 == 0)
			{
				UE_LOG(LogRenderer, Warning,
					TEXT("[theozhu] NSPF: [失败] SetInputTensorShapes (%ux%u) —— 第 %u 次, 沿用传统滤波"),
					NumSamples, NSPF_IN_DIM, C);
			}
			return nullptr;
		}
		GCachedNumSamples = NumSamples;
		if (bVerbose)
		{
			UE_LOG(LogRenderer, Display,
				TEXT("[theozhu] NSPF:   [step] SetInputTensorShapes OK (%ux%u)"),
				NumSamples, NSPF_IN_DIM);
		}
	}

	// ---- 1) 创建 typed buffer (照搬官方: + EBufferUsageFlags::NNE) ----
	//  NNE Gemm 用 PF_R32_FLOAT 建 SRV, 必须是 typed buffer, 不能是 structured.
	FRDGBufferDesc InDesc =
		FRDGBufferDesc::CreateBufferDesc(sizeof(float), NumSamples * NSPF_IN_DIM);
	InDesc.Usage |= EBufferUsageFlags::NNE;
	FRDGBufferRef InputBuf =
		GraphBuilder.CreateBuffer(InDesc, TEXT("NSPF.Features"));

	FRDGBufferDesc OutDesc =
		FRDGBufferDesc::CreateBufferDesc(sizeof(float), NumSamples * NSPF_OUT_DIM);
	OutDesc.Usage |= EBufferUsageFlags::NNE;
	FRDGBufferRef OutputBuf =
		GraphBuilder.CreateBuffer(OutDesc, TEXT("NSPF.Output"));

	// ---- 2) InputProcess: compute shader 采样纹理 -> 写进 input buffer ----
	{
		FNSPFFeatureBuildCS::FParameters* PassParameters =
			GraphBuilder.AllocParameters<FNSPFFeatureBuildCS::FParameters>();

		PassParameters->AtlasViewSize    = AtlasSize;
		PassParameters->ProbeViewSize    = ProbeViewSize;
		PassParameters->GatherResolution = (int32)GatherRes;
		PassParameters->DownsampleFactor =
			(float)ScreenProbeParameters.ScreenProbeDownsampleFactor;

		PassParameters->CompositedRadianceTexture = CompositedScreenProbeRadiance;
		PassParameters->HitDistanceTexture        = ScreenProbeHitDistance;
		PassParameters->SceneDepthTexture         = ScreenProbeSceneDepth;
		PassParameters->WorldNormalTexture        = ScreenProbeWorldNormal;
		PassParameters->WorldPositionTexture      = ScreenProbeTranslatedWorldPosition;

		// 显式 Format —— 官方 WriteInputBuffer 就这么做
		PassParameters->OutputFeatureBuffer =
			GraphBuilder.CreateUAV(FRDGBufferUAVDesc(InputBuf, PF_R32_FLOAT));

		TShaderMapRef<FNSPFFeatureBuildCS> ComputeShader(View.ShaderMap);

		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("NSPF.BuildFeatures"),
			// NeverCull: 与官方 NNEDenoiser 一致, 防止 pass 被 RDG 剔除
			ERDGPassFlags::Compute | ERDGPassFlags::NeverCull,
			ComputeShader,
			PassParameters,
			FComputeShaderUtils::GetGroupCount(
				FIntVector(AtlasSize.X, AtlasSize.Y, 1), NSPF_GROUP_SIZE));
	}
	if (bVerbose)
	{
		UE_LOG(LogRenderer, Display,
			TEXT("[theozhu] NSPF:   [step] 特征构建 pass 已入队 (in=%u floats, out=%u floats)"),
			NumSamples * NSPF_IN_DIM, NumSamples * NSPF_OUT_DIM);
	}

	// ---- 3) NNE 推理 ----
	{
		TArray<UE::NNE::FTensorBindingRDG> Inputs;
		UE::NNE::FTensorBindingRDG InBinding;
		InBinding.Buffer = InputBuf;
		Inputs.Add(InBinding);

		TArray<UE::NNE::FTensorBindingRDG> Outputs;
		UE::NNE::FTensorBindingRDG OutBinding;
		OutBinding.Buffer = OutputBuf;
		Outputs.Add(OutBinding);

		const auto Status = Instance->EnqueueRDG(GraphBuilder, Inputs, Outputs);
		if (Status != UE::NNE::IModelInstanceRDG::EEnqueueRDGStatus::Ok)
		{
			static uint32 C = 0;
			if (++C <= 3 || C % 300 == 0)
			{
				UE_LOG(LogRenderer, Warning,
					TEXT("[theozhu] NSPF: [失败] EnqueueRDG (%d) —— 第 %u 次, 沿用传统滤波"),
					(int32)Status, C);
			}
			return nullptr;
		}
		if (bVerbose)
		{
			UE_LOG(LogRenderer, Display,
				TEXT("[theozhu] NSPF:   [step] EnqueueRDG OK (NNE 推理已入队)"));
		}
	}

	// ---- 4) OutputProcess: output buffer -> 写回纹理 ----
	FRDGTextureRef NNRadiance = GraphBuilder.CreateTexture(
		CompositedScreenProbeRadiance->Desc, TEXT("NSPF.Radiance"));

	{
		FNSPFWriteBackCS::FParameters* PassParameters =
			GraphBuilder.AllocParameters<FNSPFWriteBackCS::FParameters>();

		PassParameters->AtlasViewSize = AtlasSize;
		PassParameters->InputOutputBuffer =
			GraphBuilder.CreateSRV(FRDGBufferSRVDesc(OutputBuf, PF_R32_FLOAT));
		PassParameters->OutputRadianceTexture =
			GraphBuilder.CreateUAV(NNRadiance);
		PassParameters->DebugScale = CVarLumenScreenProbeNSPFDebugScale;

		TShaderMapRef<FNSPFWriteBackCS> ComputeShader(View.ShaderMap);

		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("NSPF.WriteBack"),
			ERDGPassFlags::Compute | ERDGPassFlags::NeverCull,
			ComputeShader,
			PassParameters,
			FComputeShaderUtils::GetGroupCount(
				FIntVector(AtlasSize.X, AtlasSize.Y, 1), NSPF_GROUP_SIZE));
	}
	if (bVerbose)
	{
		UE_LOG(LogRenderer, Display,
			TEXT("[theozhu] NSPF:   [step] 写回 pass 已入队 (DebugScale=%.2f)"),
			CVarLumenScreenProbeNSPFDebugScale);
	}

	// ★ 诊断日志: 只打一次.
	//   之前只在【失败】时打 warning, 成功时静默 —— 导致无法从日志区分
	//   "NN 没执行(返回nullptr走传统滤波)" 和 "NN 执行了但结果接近".
	//   有了这行, 只要看到它就证明 NN 输出确实被赋给了 GatherParameters.
	static bool bNSPFLoggedOnce = false;
	if (!bNSPFLoggedOnce)
	{
		bNSPFLoggedOnce = true;
		UE_LOG(LogRenderer, Display,
			TEXT("[theozhu] NSPF: ★ NN 滤波已应用 —— atlas=%dx%d, 样本=%u, 后端=%s, DebugScale=%.2f"),
			AtlasSize.X, AtlasSize.Y, NumSamples,
			*CVarLumenScreenProbeNSPFRuntime, CVarLumenScreenProbeNSPFDebugScale);
	}

	return NNRadiance;
}

void LumenScreenProbeNSPF::Shutdown()
{
	// 先清裸指针(渲染线程侧), 再释放 owner(GameThread 侧引用)
	GNSPFInstance = nullptr;
	GNSPFModelOwner.Reset();
	GNSPFLoadedPath.Empty();
	bGNSPFInitFailed  = false;
	bGWarnedNotLoaded = false;
	GCachedNumSamples = 0;
}
