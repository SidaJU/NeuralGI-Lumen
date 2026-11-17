/**
 * NSPFNNEVerify.cpp
 *
 * [theozhu++] NSPF S4.1: 用 UE 原生 NNE(RDG) 对我们的滤波模型做推理验证
 *
 * 目的
 * ----
 * 验证三件事:
 *   1. NNE RDG 运行时能加载我们导入的 ONNX 模型 (Gemm/Relu/Slice/Add)
 *   2. 引擎内推理的结果与 Python(torch FP32) 参考输出一致
 *   3. 推理耗时满足实时预算 (目标 < 1ms 量级)
 *
 * 用法 (编辑器控制台, ~ 键)
 * ------------------------
 *   NSPF.NNEVerify <ModelData资产路径> <输入f32> <期望f32>
 *
 * 验证数据由 tools/verify/gen_verify_data.py 生成 (raw float32):
 *   verify_input.f32     (1024, 56)  真实 56 维特征
 *   verify_expected.f32  (1024, 3)   torch FP32 参考输出
 *
 * 依赖: Build.cs 需加 NNE / NNERuntimeRDG (见 [theozhu++] 标记)
 * [theozhu--]
 */
#include "CoreMinimal.h"
#include "HAL/IConsoleManager.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformProcess.h"
#include "Misc/FileHelper.h"    // FFileHelper::LoadFileToArray
#include "Misc/Paths.h"
#include "Modules/ModuleManager.h"
#include "RenderingThread.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "RenderGraph.h"
#include "RHI.h"
#include "RHIGPUReadback.h"      // FRHIGPUBufferReadback (注意: 文件名不带 Buffer)
#include "NNE.h"
#include "NNEModelData.h"
#include "NNERuntimeRDG.h"
#include "NNERuntimeCPU.h"      // CPU 推理: INNERuntimeCPU / IModelCPU
#include "NNETypes.h"

DEFINE_LOG_CATEGORY_STATIC(LogNSPFNNE, Log, All);

namespace
{
	/** 读取 raw float32 文件 (numpy tofile 的产物). */
	bool LoadRawF32(const FString& Path, TArray<float>& Out)
	{
		TArray<uint8> Bytes;
		if (!FFileHelper::LoadFileToArray(Bytes, *Path))
		{
			UE_LOG(LogNSPFNNE, Error,
				TEXT("[NSPF] 无法读取文件: %s"), *Path);
			return false;
		}
		const int64 NumFloats = Bytes.Num() / sizeof(float);
		Out.SetNumUninitialized(NumFloats);
		FMemory::Memcpy(Out.GetData(), Bytes.GetData(), NumFloats * sizeof(float));
		return true;
	}

	void RunNNEVerify(const FString& AssetPath, const FString& InputFile, const FString& ExpectedFile)
	{
		constexpr uint32 Batch  = 1024;   // 与 gen_verify_data.py 的采样数一致
		constexpr uint32 InDim  = 56;
		constexpr uint32 OutDim = 3;

		// ---------- ① 加载模型数据资产 ----------
		UNNEModelData* ModelData = LoadObject<UNNEModelData>(nullptr, *AssetPath);
		if (!ModelData)
		{
			UE_LOG(LogNSPFNNE, Error,
				TEXT("[NSPF] ModelData 资产未找到: %s\n"
					 "       确认已在编辑器导入 onnx, 且路径为 '包名.资产名' 形式"),
				*AssetPath);
			return;
		}
		UE_LOG(LogNSPFNNE, Display, TEXT("[NSPF] ① 模型资产已加载: %s"), *AssetPath);

		// ---------- ② 获取 RDG 运行时 ----------
		TWeakInterfacePtr<INNERuntimeRDG> RuntimeWeak =
			UE::NNE::GetRuntime<INNERuntimeRDG>(TEXT("NNERuntimeRDGHlsl"));
		if (!RuntimeWeak.IsValid())
		{
			UE_LOG(LogNSPFNNE, Error,
				TEXT("[NSPF] NNE RDG 运行时不可用 ('NNERuntimeRDGHlsl')\n"
					 "       检查: 1) NNERuntimeRDG 插件已启用 2) 项目为 D3D12 + SM6"));
			return;
		}
		INNERuntimeRDG* Runtime = RuntimeWeak.Get();
		UE_LOG(LogNSPFNNE, Display, TEXT("[NSPF] ② RDG 运行时就绪"));

		// ---------- ③ 创建模型与实例 ----------
		if (Runtime->CanCreateModelRDG(ModelData) !=
			INNERuntimeRDG::ECanCreateModelRDGStatus::Ok)
		{
			UE_LOG(LogNSPFNNE, Error,
				TEXT("[NSPF] 该运行时不支持此模型 (算子或形状不被支持)"));
			return;
		}
		TSharedPtr<UE::NNE::IModelRDG> Model = Runtime->CreateModelRDG(ModelData);
		if (!Model.IsValid())
		{
			UE_LOG(LogNSPFNNE, Error, TEXT("[NSPF] CreateModelRDG 失败"));
			return;
		}
		TSharedPtr<UE::NNE::IModelInstanceRDG> Instance = Model->CreateModelInstanceRDG();
		if (!Instance.IsValid())
		{
			UE_LOG(LogNSPFNNE, Error, TEXT("[NSPF] CreateModelInstanceRDG 失败"));
			return;
		}
		UE_LOG(LogNSPFNNE, Display, TEXT("[NSPF] ③ 模型实例已创建"));

		// 注意: Name 是私有成员, 公开接口是 GetName() (返回 const FString&)
		for (const UE::NNE::FTensorDesc& D : Instance->GetInputTensorDescs())
		{
			UE_LOG(LogNSPFNNE, Display, TEXT("[NSPF]    模型输入: %s"), *D.GetName());
		}
		for (const UE::NNE::FTensorDesc& D : Instance->GetOutputTensorDescs())
		{
			UE_LOG(LogNSPFNNE, Display, TEXT("[NSPF]    模型输出: %s"), *D.GetName());
		}

		// ---------- ④ 设置输入形状 ----------
		const uint32 InDims[]  = {Batch, InDim};
		UE::NNE::FTensorShape InShape =
			UE::NNE::FTensorShape::Make(TConstArrayView<uint32>(InDims, 2));
		if (Instance->SetInputTensorShapes(
				TConstArrayView<UE::NNE::FTensorShape>(&InShape, 1)) !=
			UE::NNE::IModelInstanceRDG::ESetInputTensorShapesStatus::Ok)
		{
			UE_LOG(LogNSPFNNE, Error, TEXT("[NSPF] SetInputTensorShapes 失败 (%ux%u)"),
				Batch, InDim);
			return;
		}
		UE_LOG(LogNSPFNNE, Display, TEXT("[NSPF] ④ 输入形状已设置: %ux%u"), Batch, InDim);

		// ---------- ⑤ 读取验证数据 ----------
		TArray<float> Input, Expected;
		if (!LoadRawF32(InputFile, Input) || !LoadRawF32(ExpectedFile, Expected))
		{
			return;
		}
		if (Input.Num() < Batch * InDim || Expected.Num() < Batch * OutDim)
		{
			UE_LOG(LogNSPFNNE, Error,
				TEXT("[NSPF] 验证数据不足: 输入 %d (需 %u), 期望 %d (需 %u)"),
				Input.Num(), Batch * InDim, Expected.Num(), Batch * OutDim);
			return;
		}
		UE_LOG(LogNSPFNNE, Display, TEXT("[NSPF] ⑤ 验证数据就绪, 开始 RDG 推理..."));

		// ---------- ⑥ 渲染线程: 建 RDG → EnqueueRDG → 执行 → 读回 ----------
		FRHIGPUBufferReadback* Readback =
			new FRHIGPUBufferReadback(TEXT("NSPF.NNEVerify.Readback"));
		// 输入端 readback: 用来诊断"输入到底传上 GPU 了没"
		//   若输入 readback 全 0 -> QueueBufferUpload 没生效 (输入源问题)
		//   若输入 readback 与 Input[0..2] 一致 -> 输入 OK, 问题在 NNE 内部
		FRHIGPUBufferReadback* InReadback =
			new FRHIGPUBufferReadback(TEXT("NSPF.NNEVerify.InReadback"));
		const double T0 = FPlatformTime::Seconds();

		ENQUEUE_RENDER_COMMAND(NSPFNNEVerify)(
			// 全部在渲染线程完成: Lock() 走 LockStagingBuffer_RenderThread, 要求渲染线程
			[Instance, Readback, InReadback, Input, Expected, T0](FRHICommandListImmediate& RHICmdList) mutable
		{
			// ==================================================================
			// 完全照搬官方: 上传与使用分离到【两个 Graph】
			//
			// 官方 NNE 内部上传权重就是这么做的 (NNERuntimeRDGModelHlsl.cpp:82-88):
			//   PooledBuffer = RDGBuilder.ConvertToExternalBuffer(TransientRDGBuffer);
			//   RDGBuilder.QueueBufferUpload(TransientRDGBuffer, Data, Size);
			// 然后在【下一个 graph】里:
			//   FRDGBufferRef B = RDGBuilder.RegisterExternalBuffer(PooledBuffer);
			//
			// 我们之前把 create+upload+use 塞进同一个 graph, 实测上传不生效
			// (readback 全 0). 另外官方 NNEDenoiser 里搜 QueueBufferUpload = 0 结果,
			// 它用 InputProcess.AddPasses(shader) 写 buffer —— 说明 CPU 上传本就不是
			// NNE RDG 的主路径. 这里复刻 NNE 自己的上传模式, 是最贴近官方的做法.
			//
			// 两个必备细节 (照抄 NNEDenoiserGenericDenoiser.cpp:42-45):
			//   1. typed buffer —— NNE Gemm 用 PF_R32_FLOAT 建 SRV
			//      (见 NNERuntimeRDGGemm.cpp:138-143), structured buffer 会读到 0
			//   2. Desc.Usage |= EBufferUsageFlags::NNE
			// ==================================================================
			const uint64 InBytes  = (uint64)Batch * InDim  * sizeof(float);
			const uint64 OutBytes = (uint64)Batch * OutDim * sizeof(float);

			TArray<float> Sentinel;
			Sentinel.Init(-1.0f, Batch * OutDim);

			// ---- 阶段 1: 上传并转成 external buffer ----
			TRefCountPtr<FRDGPooledBuffer> PooledInput;
			TRefCountPtr<FRDGPooledBuffer> PooledOutput;
			{
				FRDGBuilder GB(RHICmdList);

				FRDGBufferDesc InDesc =
					FRDGBufferDesc::CreateBufferDesc(sizeof(float), Batch * InDim);
				InDesc.Usage |= EBufferUsageFlags::NNE;
				FRDGBufferRef InB = GB.CreateBuffer(
					InDesc, TEXT("NSPF.NNEVerify.Input"));
				// ★ 顺序照搬官方 (NNERuntimeRDGModelHlsl.cpp:82-85):
				//   先 ConvertToExternalBuffer 把 buffer 标记为"提取到外部",
				//   RDG 才不会把【这个没有任何 pass 引用的 buffer 及其上传】
				//   当作无用资源 culling 掉; 然后再 QueueBufferUpload.
				//   我上一版顺序反了(先上传后转external), 上传疑似被优化掉 -> 全 0.
				PooledInput = GB.ConvertToExternalBuffer(InB);
				GB.QueueBufferUpload(InB, Input.GetData(), InBytes);

				// 输出 buffer 预填哨兵 -1 (跑完仍是 -1 说明 NNE 没写)
				FRDGBufferDesc OutDesc =
					FRDGBufferDesc::CreateBufferDesc(sizeof(float), Batch * OutDim);
				OutDesc.Usage |= EBufferUsageFlags::NNE;
				FRDGBufferRef OutB = GB.CreateBuffer(
					OutDesc, TEXT("NSPF.NNEVerify.Output"));
				PooledOutput = GB.ConvertToExternalBuffer(OutB);
				GB.QueueBufferUpload(OutB, Sentinel.GetData(), OutBytes);

				GB.Execute();
			}
			UE_LOG(LogNSPFNNE, Display,
				TEXT("[NSPF] 阶段1完成: 输入+哨兵已上传为 external buffer"));

			// ---- 阶段 2: 推理 ----
			FRDGBuilder GraphBuilder(RHICmdList);
			FRDGBufferRef InputBuf  = GraphBuilder.RegisterExternalBuffer(PooledInput);
			FRDGBufferRef OutputBuf = GraphBuilder.RegisterExternalBuffer(PooledOutput);

			// [诊断] 输入 readback: 读整体算统计, 不能只读前几个(那里恰好是0)
			AddReadbackBufferPass(GraphBuilder,
				RDG_EVENT_NAME("NSPF.NNEVerify.InReadback"), InputBuf,
				[InReadback, InputBuf](FRDGAsyncTask, FRHICommandList& RHICmdList)
				{
					InReadback->EnqueueCopy(RHICmdList, InputBuf->GetRHI(), 0u);
				});

			// 注: OutputBuf 已在阶段1创建并预填哨兵 -1, 见上方阶段1.
			//   判读: 跑完 ≈-1 -> NNE 没写输出; ≈0 -> 写了0(计算异常); ≈期望 -> 通过

			// NNE 推理入队 (显式构造, 避免聚合初始化的推导差异)
			TArray<UE::NNE::FTensorBindingRDG> Inputs;
			UE::NNE::FTensorBindingRDG InBinding;
			InBinding.Buffer = InputBuf;
			Inputs.Add(InBinding);

			TArray<UE::NNE::FTensorBindingRDG> Outputs;
			UE::NNE::FTensorBindingRDG OutBinding;
			OutBinding.Buffer = OutputBuf;
			Outputs.Add(OutBinding);

			const auto Status = Instance->EnqueueRDG(
				GraphBuilder, Inputs, Outputs);
			if (Status != UE::NNE::IModelInstanceRDG::EEnqueueRDGStatus::Ok)
			{
				UE_LOG(LogNSPFNNE, Error,
					TEXT("[NSPF] EnqueueRDG 失败 (%d)"), (int32)Status);
				delete Readback;
				delete InReadback;
				return;
			}

			// 读回
			AddReadbackBufferPass(GraphBuilder,
				RDG_EVENT_NAME("NSPF.NNEVerify.Readback"), OutputBuf,
				[Readback, OutputBuf](FRDGAsyncTask, FRHICommandList& RHICmdList)
				{
					Readback->EnqueueCopy(RHICmdList, OutputBuf->GetRHI(), 0u);
				});

			GraphBuilder.Execute();
			const double T1 = FPlatformTime::Seconds();

			// [诊断] 输入 readback: 整体统计 + center 位置对比
			const float* InSrc =
				(const float*)InReadback->Lock(Batch * InDim * sizeof(float));
			if (InSrc)
			{
				// 只做统计, 避免逐元素打印刷屏
				double GpuAbsMean = 0.0, GpuMax = 0.0;
				for (uint32 k = 0; k < Batch * InDim; ++k)
				{
					const double A = FMath::Abs((double)InSrc[k]);
					GpuAbsMean += A;
					GpuMax = FMath::Max(GpuMax, A);
				}
				GpuAbsMean /= (Batch * InDim);

				double CpuAbsMean = 0.0, CpuMax = 0.0;
				for (uint32 k = 0; k < Batch * InDim; ++k)
				{
					const double A = FMath::Abs((double)Input[k]);
					CpuAbsMean += A;
					CpuMax = FMath::Max(CpuMax, A);
				}
				CpuAbsMean /= (Batch * InDim);

				UE_LOG(LogNSPFNNE, Display,
					TEXT("[NSPF] 输入 GPU: absmean=%.4f max=%.3f | "
						 "CPU: absmean=%.4f max=%.3f  %s"),
					GpuAbsMean, GpuMax, CpuAbsMean, CpuMax,
					(FMath::Abs(GpuAbsMean - CpuAbsMean) < 1e-3 ?
						TEXT("=> 上传OK") : TEXT("=> ★上传失败★")));

				// center 位置 (feat[12:15]) 必然非 0, 是最可靠的单点对比
				UE_LOG(LogNSPFNNE, Display,
					TEXT("[NSPF] center[12:15] GPU: %+.6f %+.6f %+.6f | "
						 "CPU: %+.6f %+.6f %+.6f"),
					InSrc[12], InSrc[13], InSrc[14],
					Input[12], Input[13], Input[14]);

				InReadback->Unlock();
			}
			delete InReadback;

			// Lock 必须在渲染线程: 它内部走 LockStagingBuffer_RenderThread,
			// 且会自己 ImmediateFlush 等 GPU 完成 (无需我们预先 while 等 IsReady)。
			const float* Src = (const float*)Readback->Lock(Batch * OutDim * sizeof(float));
			if (!Src)
			{
				UE_LOG(LogNSPFNNE, Error, TEXT("[NSPF] readback Lock 失败"));
				delete Readback;
				return;
			}

			// 诊断: 输出也用"整体统计"而不是前 3 个值
			//   (前 3 个可能恰好是 0 区域, 会误判)
			double OutAbsMean = 0.0, OutMax = 0.0, ExpAbsMean = 0.0;
			for (uint32 k = 0; k < Batch * OutDim; ++k)
			{
				const double A = FMath::Abs((double)Src[k]);
				OutAbsMean += A;
				OutMax = FMath::Max(OutMax, A);
				ExpAbsMean += FMath::Abs((double)Expected[k]);
			}
			OutAbsMean /= (Batch * OutDim);
			ExpAbsMean /= (Batch * OutDim);

			UE_LOG(LogNSPFNNE, Display,
				TEXT("[NSPF] 输出: absmean=%.6f max=%.6f | "
					 "期望: absmean=%.6f  %s"),
				OutAbsMean, OutMax, ExpAbsMean,
				(OutAbsMean < 1e-6 ? TEXT("=> ★输出全0★") : TEXT("")));
			UE_LOG(LogNSPFNNE, Display,
				TEXT("[NSPF] 输出样例: %+.6f %+.6f %+.6f"),
				Src[0], Src[1], Src[2]);
			UE_LOG(LogNSPFNNE, Display,
				TEXT("[NSPF] 期望样例: %+.6f %+.6f %+.6f"),
				Expected[0], Expected[1], Expected[2]);

			// 与 torch 参考输出对比
			double MaxErr = 0.0, SumAbs = 0.0, SumRef = 0.0;
			for (uint32 k = 0; k < Batch * OutDim; ++k)
			{
				const double E = FMath::Abs((double)Src[k] - (double)Expected[k]);
				MaxErr = FMath::Max(MaxErr, E);
				SumAbs += E;
				SumRef += FMath::Abs((double)Expected[k]);
			}
			Readback->Unlock();
			delete Readback;

			const double T2 = FPlatformTime::Seconds();
			const double MeanErr = SumAbs / (Batch * OutDim);
			const double RelErr  = SumAbs / (SumRef + 1e-9);
			UE_LOG(LogNSPFNNE, Display,
				TEXT("[NSPF] 推理完成: 建图+执行 %.3f ms | Lock(等GPU) %.3f ms | "
					 "误差: 最大 %.3e, 平均 %.3e, 相对 %.2f%%  (与 torch FP32 参考对比)"),
				(T1 - T0) * 1000.0, (T2 - T1) * 1000.0, MaxErr, MeanErr, RelErr * 100.0);
		});
	}

	/**
	 * CPU 版验证 (NSPF.NNEVerifyCPU)
	 *
	 * 用 NNERuntimeBasicCpu —— 官方为 MLP 优化的 CPU runtime
	 * (插件注释: "performant implementations for simple CPU models such as MLPs").
	 * 它直接吃 CPU 数组(FTensorBindingCPU), 不需要 GPU buffer / RDG / 上传,
	 * 因此完全绕开 GPU 路径的复杂性.
	 *
	 * 价值 —— 把两个问题解耦:
	 *   1. "NNE 是否支持我们的模型(算子) + 数值是否正确" -> CPU 验证即可回答
	 *   2. "GPU(RDG) 路径怎么把数据喂进去"                 -> 单独解决
	 *
	 * 参考: LearningAgents/Private/LearningNeuralNetwork.cpp:344
	 */
	void RunNNEVerifyCPU(const FString& AssetPath, const FString& InputFile,
	                     const FString& ExpectedFile)
	{
		constexpr uint32 Batch  = 1024;
		constexpr uint32 InDim  = 56;
		constexpr uint32 OutDim = 3;

		// 确保 CPU runtime 模块被加载 (动态加载, 不需要 Build.cs 加依赖)
		// 用 NNERuntimeORT(ONNX Runtime): 它支持 onnx.
		// 注意 NNERuntimeBasicCpu 只吃自己的私有格式(需 python 脚本转换),
		// 对 onnx 会报 "does not support Filetype: onnx".
		FModuleManager::Get().LoadModule(TEXT("NNERuntimeORT"));

		UNNEModelData* ModelData = LoadObject<UNNEModelData>(nullptr, *AssetPath);
		if (!ModelData)
		{
			UE_LOG(LogNSPFNNE, Error, TEXT("[NSPF-CPU] ModelData 未找到: %s"), *AssetPath);
			return;
		}
		UE_LOG(LogNSPFNNE, Display, TEXT("[NSPF-CPU] ① 模型资产已加载"));

		TWeakInterfacePtr<INNERuntimeCPU> RuntimeWeak =
			UE::NNE::GetRuntime<INNERuntimeCPU>(TEXT("NNERuntimeORTCpu"));
		if (!RuntimeWeak.IsValid())
		{
			UE_LOG(LogNSPFNNE, Error,
				TEXT("[NSPF-CPU] CPU runtime 不可用 ('NNERuntimeORTCpu')\n"
					 "       检查 uproject 已启用 NNERuntimeORT 插件"));
			return;
		}
		UE_LOG(LogNSPFNNE, Display,
			TEXT("[NSPF-CPU] ② CPU runtime 就绪 (NNERuntimeORTCpu)"));

		TSharedPtr<UE::NNE::IModelCPU> Model = RuntimeWeak->CreateModelCPU(ModelData);
		if (!Model.IsValid())
		{
			UE_LOG(LogNSPFNNE, Error,
				TEXT("[NSPF-CPU] CreateModelCPU 失败 —— 该 runtime 不支持此模型(算子不支持?)"));
			return;
		}
		TSharedPtr<UE::NNE::IModelInstanceCPU> Instance =
			Model->CreateModelInstanceCPU();
		if (!Instance.IsValid())
		{
			UE_LOG(LogNSPFNNE, Error, TEXT("[NSPF-CPU] CreateModelInstanceCPU 失败"));
			return;
		}
		UE_LOG(LogNSPFNNE, Display, TEXT("[NSPF-CPU] ③ 模型实例已创建"));

		const uint32 InDims[] = {Batch, InDim};
		// 注意: 必须先存成局部变量再取地址 —— Make() 返回临时对象(右值),
		// 直接 &Make(...) 会报 C2102 "& 要求左值"
		UE::NNE::FTensorShape InShape =
			UE::NNE::FTensorShape::Make(TConstArrayView<uint32>(InDims, 2));
		Instance->SetInputTensorShapes(
			TConstArrayView<UE::NNE::FTensorShape>(&InShape, 1));
		UE_LOG(LogNSPFNNE, Display,
			TEXT("[NSPF-CPU] ④ 输入形状已设置: %ux%u"), Batch, InDim);

		TArray<float> Input, Expected;
		if (!LoadRawF32(InputFile, Input) || !LoadRawF32(ExpectedFile, Expected))
		{
			return;
		}
		if (Input.Num() < (int32)(Batch * InDim) ||
			Expected.Num() < (int32)(Batch * OutDim))
		{
			UE_LOG(LogNSPFNNE, Error, TEXT("[NSPF-CPU] 验证数据不足"));
			return;
		}
		UE_LOG(LogNSPFNNE, Display,
			TEXT("[NSPF-CPU] ⑤ 验证数据就绪, 开始 CPU 推理..."));

		// 输出先填哨兵 -1: 若 RunSync 失败没写, 我们能看出来
		TArray<float> Output;
		Output.Init(-1.0f, Batch * OutDim);

		const double T0 = FPlatformTime::Seconds();
		// 注意: 用 auto 接返回值, 不引用 ERunSyncStatus 类型名 (避免名字不确定导致编译失败)
		const auto Status = Instance->RunSync(
			{ { (void*)Input.GetData(),  (uint64)Input.Num()  * sizeof(float) } },
			{ { (void*)Output.GetData(), (uint64)Output.Num() * sizeof(float) } });
		const double T1 = FPlatformTime::Seconds();
		UE_LOG(LogNSPFNNE, Display, TEXT("[NSPF-CPU] RunSync status=%d"), (int32)Status);

		double MaxErr = 0.0, SumAbs = 0.0, SumRef = 0.0, OutAbsMean = 0.0;
		for (uint32 k = 0; k < Batch * OutDim; ++k)
		{
			const double E = FMath::Abs((double)Output[k] - (double)Expected[k]);
			MaxErr = FMath::Max(MaxErr, E);
			SumAbs += E;
			SumRef += FMath::Abs((double)Expected[k]);
			OutAbsMean += FMath::Abs((double)Output[k]);
		}
		OutAbsMean /= (Batch * OutDim);

		UE_LOG(LogNSPFNNE, Display,
			TEXT("[NSPF-CPU] 输出样例: %+.6f %+.6f %+.6f"),
			Output[0], Output[1], Output[2]);
		UE_LOG(LogNSPFNNE, Display,
			TEXT("[NSPF-CPU] 期望样例: %+.6f %+.6f %+.6f"),
			Expected[0], Expected[1], Expected[2]);
		UE_LOG(LogNSPFNNE, Display,
			TEXT("[NSPF-CPU] 输出 absmean=%.6f (期望 absmean=%.6f) %s%s"),
			OutAbsMean, SumRef / (Batch * OutDim),
			(OutAbsMean < 1e-6 ? TEXT("★全0★") : TEXT("")),
			(FMath::Abs(OutAbsMean - 1.0) < 1e-4 ? TEXT("★未写入(仍为哨兵-1)★") : TEXT("")));

		const double MeanErr = SumAbs / (Batch * OutDim);
		const double RelErr  = SumAbs / (SumRef + 1e-9);
		UE_LOG(LogNSPFNNE, Display,
			TEXT("[NSPF-CPU] 推理完成: 耗时 %.3f ms | "
				 "误差: 最大 %.3e, 平均 %.3e, 相对 %.2f%%"),
			(T1 - T0) * 1000.0, MaxErr, MeanErr, RelErr * 100.0);
	}
}

static FAutoConsoleCommand GNSPFNNEVerifyCmd(
	TEXT("NSPF.NNEVerify"),
	TEXT("S4.1 NNE RDG 推理验证. 用法: NSPF.NNEVerify <ModelData资产路径> <输入f32> <期望f32>"),
	FConsoleCommandWithArgsDelegate::CreateLambda([](const TArray<FString>& Args)
	{
		const FString Asset = Args.Num() > 0 ? Args[0]
			: TEXT("/Game/model_nne.model_nne");
		const FString InFile = Args.Num() > 1 ? Args[1]
			: TEXT("c:/Users/theozhu/Desktop/拓展/Super Resolution/NeuralGI-Lumen/verify_input.f32");
		const FString ExFile = Args.Num() > 2 ? Args[2]
			: TEXT("c:/Users/theozhu/Desktop/拓展/Super Resolution/NeuralGI-Lumen/verify_expected.f32");
		RunNNEVerify(Asset, InFile, ExFile);
	}));

static FAutoConsoleCommand GNSPFNNEVerifyCPUCmd(
	TEXT("NSPF.NNEVerifyCPU"),
	TEXT("S4.1 NNE CPU 推理验证(用 NNERuntimeBasicCpu, 不需要GPU上传). "
	     "用法: NSPF.NNEVerifyCPU <ModelData资产路径> <输入f32> <期望f32>"),
	FConsoleCommandWithArgsDelegate::CreateLambda([](const TArray<FString>& Args)
	{
		const FString Asset = Args.Num() > 0 ? Args[0]
			: TEXT("/Game/model_nne.model_nne");
		const FString InFile = Args.Num() > 1 ? Args[1]
			: TEXT("c:/Users/theozhu/Desktop/拓展/Super Resolution/NeuralGI-Lumen/verify_input.f32");
		const FString ExFile = Args.Num() > 2 ? Args[2]
			: TEXT("c:/Users/theozhu/Desktop/拓展/Super Resolution/NeuralGI-Lumen/verify_expected.f32");
		RunNNEVerifyCPU(Asset, InFile, ExFile);
	}));
