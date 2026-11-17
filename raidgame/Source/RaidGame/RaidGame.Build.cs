// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;
using System;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;

public class RaidGame : ModuleRules
{
	public RaidGame(ReadOnlyTargetRules Target) : base(Target)
	{
		// 安卓必须优化，否则地址空间超限，无法编出安卓包
        //OptimizeCode =
        //	Target.Platform == UnrealTargetPlatform.Android
        //		? CodeOptimization.Always
        //		: CodeOptimization.InShippingBuildsOnly;
        //OptimizeCode = CodeOptimization.InShippingBuildsOnly;
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PrivatePCHHeaderFile = "RaidGame.h";
		PublicIncludePaths.AddRange(
			new string[] {
				"RaidGame",
				"RaidGame/ESAvatar",
				"RaidGame/ESCamera",
				"RaidGame/ESCharacter",
				"RaidGame/ESDestruction",
				"RaidGame/ESGamePlay",
				"RaidGame/ESInteractions",
				"RaidGame/ESItemSystem",
				"RaidGame/ESLevel",
				"RaidGame/ESRenderExt",
				"RaidGame/ESReplicationGraph",
				"RaidGame/ESSubsystem",
				"RaidGame/ESVehicle",
				"RaidGame/ESWeapon",
				"RaidGame/ESWeaponCore",
			}
			);

		PublicDependencyModuleNames.AddRange(new string[] {
            "Json",
			"JsonUtilities",
            "Core",
            "RaidLogCore",
            "CoreUObject",
            "Engine",
            "Renderer",
            "UMG",
            "Slate",
            "SlateCore",
            "InputCore",
            "EnhancedInput",
            // "SGLua",
            "RUnLua",
            "UnLua",
            "Lua",
            //"SGLuaTest",
            "HTTP",
            "RHI",
            "Navmesh",
            "IrisCore",
            "GameplayTasks",
            "GameplayTags",
			"GameplayAbilities",
            "ESCommon",
			"RenderCore",
            "LevelSequence",
            "MoviePlayer",
            "MovieScene",
            "Foliage",
            "Paper2D",
            "PhysicsCore",
            "Sockets", 
            "Networking",
            "MotionWarping",
            "PhysicsControl",
            "Niagara",
            "Landscape",
            "NetCore",
            "Chaos",
            "FieldSystemEngine",
            "ChaosSolverEngine",
            "LogTrack",
            "AnimGraphRuntime",
            // "WorldPartitionExtend",
            "MovieSceneTracks",
            "ESMeshAdaptationRuntime",
            "ProceduralMeshComponent",
            "CustomPostProcess",
            "AsyncMixin",
            "AnimationCore",
            "AnimGraphRuntime",
            "NiagaraAnimNotifies",
            "GeometryCore",
            "ChaosCore",
            "ChaosVehicles",
            "ChaosVehiclesCore",
            "ControlRig",
            "RigVM",
            "GeometryCollectionEngine",
            "PhysicsCore",
            "Water",
            "MeshDescription",
            "StaticMeshDescription",
            "SkeletalMeshDescription",
            "GameplayCameras",
            "UMG",
            "Eigen",
            "Catch2", 
            "ApplicationCore",
            //@AniperChen -- Add Begin -- 添加OSS模块。
            "OnlineSubsystem",
            "OnlineSubsystemUtils",
            //@AniperChen -- Add End --
            "ReplicationGraph",
            "COSHttpHelper",
            "FieldSystemEngine",
            "ESAnimation",
            "ESGamePredefine",
            "RAbilitySystem",
            "ESAnimationToolkitsRuntime",
            "DeveloperSettings",
            //"Voxel",
            //"VoxelCore",
            //"VoxelGraph",
            "EngineCameras",
			"CustomizableObject",
			"MutableRuntime",
			"Gamelet",
			"NetworkReplayStreaming",
			"HttpNetworkReplayStreaming",
			"AIPD_AIPlugin"
			});

			// Performance authoring is client/editor development tooling; DS/shipping have no pose dependency.
			bool bPerformanceCapture = Target.Platform == UnrealTargetPlatform.Win64 &&
			Target.Type != TargetType.Server && Target.Configuration != UnrealTargetConfiguration.Shipping;
			PrivateDefinitions.Add("WITH_RAID_PERFORMANCE_CAPTURE=" + (bPerformanceCapture ? "1" : "0"));
			if (bPerformanceCapture) PrivateDependencyModuleNames.Add("BloodStainSystem");

			// Shipping excludes DSInsights and its agent runtime dependencies.
		if (Target.Configuration != UnrealTargetConfiguration.Shipping)
		{
			PublicDependencyModuleNames.Add("DSInsights");

			// SeedDS parent must defer DSInsights auto start until a forked child exists.
			if (Target.Type == TargetType.Server && Target.Platform == UnrealTargetPlatform.Linux)
			{
				PrivateDependencyModuleNames.Add("SeedDSAdapter");
			}
		}

		bool bWithRCinePath = Target.Platform == UnrealTargetPlatform.Win64;
		PublicDefinitions.Add($"WITH_RCINEPATH={(bWithRCinePath ? 1 : 0)}");
		if (bWithRCinePath)
		{
			PublicDependencyModuleNames.Add("RCinePath");
		}

		PrivateDependencyModuleNames.AddRange(new string[] {
			"Slate",
			"SlateCore",
			"Navmesh",
			"NavigationSystem",
			"AIModule",
			"ESCommon",
			"SignificanceManager",
			/* disable pxg by kucoolerli
			 "PxgSkillSystem",
			 */
			"ScriptPlugin",
			"Pathfinding",
			"CrashSight",
			"PreLoadScreen",
			"Flow",
			"UltronMLPlugin",
			"Protobuf",
			"LineComponent",
			"SkeletalMerging",
			"GameplayStateTreeModule",
			"StateTreeModule",
			"FlyingNavSystem",
			"NaniteSkinning",
			"MediaAssets",
			//@StewieSong -- Add Begin -- 移动组件并行化
			"StackfulCoro",
			"StackfulCoroAdapter",
			//@StewieSong -- Add End --
			// 音频测试用 Actor ARAudioTestManager 需要在编辑器里用 AssetRegistry 扫路径下的 AkAudioEvent。
			"OFootIK",
			"AssetRegistry",
			// DS Tick Policy: 关闭关卡里 CineCameraActor 在 DS 上的 tick
			"CinematicCamera",
			"StructUtils"
		});
		
		if (Target.bUseGameplayDebugger || Target.bBuildDeveloperTools || (Target.Configuration != UnrealTargetConfiguration.Shipping && Target.Configuration != UnrealTargetConfiguration.Test))
		{
			PrivateDependencyModuleNames.Add("GameplayDebugger");
			PublicDefinitions.Add("WITH_GAMEPLAY_DEBUGGER=1");

		}

		else
		{
			PublicDefinitions.Add("WITH_GAMEPLAY_DEBUGGER=0");
		}

		
        if (Target.Type == TargetRules.TargetType.Editor)
        {
            PublicDependencyModuleNames.AddRange(new string[] {
                "UnrealEd",
                "WorldBrowser",
                "PlacementMode",
                "PropertyEditor",
                "ScriptEditorPlugin",
                "ConsoleVariablesEditor",
                "ConsoleVariablesEditorRuntime",
                "UMGEditor",
                "KismetWidgets",
                "ESAnimationToolkits",
                "LevelEditor",
                "SceneOutliner",
                "InstanceTool", 
                "MovieSceneTools",
                /*"PxgFlowEditor",disable pxg by kucoolerli */
                "WorldPartitionHLODUtilities", 
                "MaterialEditor",
                "RMAFT",
                "Sequencer",
                "UATHelper", 
                //"VoxelBlueprint",
                "Blutility",
            });
            PrivateDependencyModuleNames.Add("GameplayTagsEditor");
        }

		
        // ESGame relative modules
		PublicDependencyModuleNames.AddRange(new string[] {
			"ESCommon",
			"RUI",
            "DSFork",
            "ESGameplayCameras",
            "PhotonDestructible",
        });

		// AkAudio: Required for UHT reflection code generation even on DS
		// Audio resources can exist on DS, playback is simply not executed
		PublicDependencyModuleNames.Add("AkAudio");

		// WwiseSoundEngine: 直接访问 IWwiseSoundEngineAPI（如音频性能监控回调）。
		// 注意：AkAudio 仅 Private 依赖 WwiseSoundEngine，因此使用方必须自行声明依赖。
		PublicDependencyModuleNames.Add("WwiseSoundEngine");

		// HairStrandsCore: client/editor only — DS does not need hair rendering
		if (Target.Type != TargetType.Server)
		{
			PublicDependencyModuleNames.Add("HairStrandsCore");
		}
		PublicDefinitions.Add("CRASHSIGHT_ENABLE=1");

		if (Target.Type != TargetType.Server)
		{
			PublicDependencyModuleNames.Add("GVoiceSDK");
		}
		
		if((Target.Type == TargetType.Editor || Target.Type == TargetType.Game || Target.Type == TargetType.Client) 
		   && (Target.Platform == UnrealTargetPlatform.Win64 ))
		{
			PublicDefinitions.Add("ENABLE_VOICE=1");
			System.Console.WriteLine("ENABLE_VOICE value = 1");

		}else
		{
			PublicDefinitions.Add("ENABLE_VOICE=0");
			System.Console.WriteLine("ENABLE_VOICE value = 0");
		}
		
		// // InGameSDK
		// if ((Target.Type == TargetType.Server || Target.Type == TargetType.Editor) && (Target.Platform == UnrealTargetPlatform.Linux || Target.Platform == UnrealTargetPlatform.Win64))
		// {
		// 	PublicDependencyModuleNames.AddRange(new string[] { "DSSDK" });
		// 	PublicDependencyModuleNames.AddRange(new string[] { "InGameProtocol" });
		// }
		//
		// if (Target.Type == TargetType.Server && Target.Platform == UnrealTargetPlatform.Linux)
		// {
		// 	PublicDependencyModuleNames.AddRange(new string[] { "EngineVirtualization" });
		// }
  //       
		// if(Target.Type == TargetType.Client || Target.Type == TargetType.Game || Target.Type == TargetType.Editor)
		// {
		// 	PublicDependencyModuleNames.AddRange(new string[] { "InGameLogic" });
		// }
		//
		// if (Target.Type == TargetType.Editor)
		// {
		// 	PublicDependencyModuleNames.AddRange(new string[] { "InGameSDK", "InGameSDKLibrary" });
		// }

		// ============================================================
		// ACE Anti-Cheat SDK — Client only, header-only, no lib
		// ------------------------------------------------------------
		// SDK 通过运行时 LoadLibrary("AntiCheatExpert/ACE-Base64.dll") 加载，
		// 因此这里只需要 PublicIncludePaths + WITH_ACE_SDK 编译宏，无 .lib 链接。
		//
		// 启用条件（全部满足）：
		//   1. Win64 平台
		//   2. Target.Name == "RaidGameClient"（主客户端 Target；ClientProfile 明确排除，
		//      因 Profile 包用于 GPU/RHI 分析，ACE 反调试反 attach 会挡工具）
		//   3. Configuration ∈ {Shipping, Test}（Development/Debug 关闭，方便 attach）
		//
		// 头文件位置：Source/RaidGame/ThirdParty/AceSdk/includes/
		// 业务代码必须用 #if WITH_ACE_SDK ... #endif 包夹，且 include 必须在包夹内，
		// 因为 ACE 头文件会拉进 <windows.h> / <strsafe.h>，会污染 UE 编译单元。
		// ============================================================
		bool bAceForClient = Target.Name == "RaidGameClient"
			&& (Target.Configuration == UnrealTargetConfiguration.Shipping
				|| Target.Configuration == UnrealTargetConfiguration.Test);

		bool bWithAceSdk = Target.Platform == UnrealTargetPlatform.Win64
			&& bAceForClient;

		PublicDefinitions.Add("WITH_ACE_SDK=" + (bWithAceSdk ? 1 : 0));
		System.Console.WriteLine("[RaidGame] WITH_ACE_SDK=" + (bWithAceSdk ? 1 : 0)
			+ " (Target=" + Target.Name + ", Config=" + Target.Configuration + ")");

		if (bWithAceSdk)
		{
			PublicIncludePaths.Add(Path.Combine(ModuleDirectory, "ThirdParty", "AceSdk", "includes"));

			// ========================================================================
			// ACE 二进制的 Stage 声明（参照 GCloud 的处理方式）
			// ------------------------------------------------------------------------
			// ACE 的 5 个二进制文件（ACE-BASE.sys / ACE-Base.dat / ACE-Base64.dll /
			// ACE-Service64.exe / ACE-Setup64.exe）作为静态资产 check in 到 P4：
			//     //RaidStreams/Raid/RaidGame/Binaries/Win64/AntiCheatExpert/
			//
			// 这里用【单参数 + StagedFileType】重载，语义是"文件已经在正确位置，
			// 只需登记它需要被 Stage"，不发生任何拷贝，因此不会触发
			//     FilePatternException: 'x.sys' is listed as a source and target file
			// 这与 GCloudCore_Domestic.Build.cs 里
			//     RuntimeDependencies.Add("$(PluginDir)/Config/GCloudInfo", StagedFileType.NonUFS)
			// 是同一种用法。（GCloud 的 .dll 用的是双参数重载，因为它的源在
			// ThirdParty/GCloudLib 下、与目标 $(BinaryOutputDir) 不同，需要真实拷贝。）
			//
			// 关键：本声明位于 if (bWithAceSdk) 内，而 bWithAceSdk 要求
			// Win64 + RaidGameClient + Shipping/Test。因此 LinuxServer(DS)、Android
			// 等非 Win64 Target 编译时这段代码根本不会执行，不会产生任何 Win64
			// RuntimeDependency —— 这正是 GCloud 的 dll 不会进 DS 包的原因：
			// 平台隔离在 Build.cs 里用 Target.Platform 判断完成，而不是靠 ini。
			//
			// 反例（曾经踩过的坑）：写在 DefaultGame.ini 的
			//     +DirectoriesToAlwaysStageAsNonUFS=(Path="../Binaries/Win64/AntiCheatExpert")
			// 对所有平台生效，打 DS 包时 UAT 的 restricted folder 校验会因为路径含
			// "Win64" 抛异常，导致 STAGE 阶段 BUILD FAILED。
			//
			// 用 "..." 递归通配符而非逐个列举：与原 ini 的 AllDirectories 行为一致，
			// 且 ACE 加固版若增减文件无需改这里；通配符匹配 0 个文件也不会让 UBT 报错。
			//
			// 未加固 vs 加固：
			//   - 本地开发 / 联调阶段：P4 里存放的是 ACE SDK 包中的「未加固版本」，
			//     用于业务闭环验证（SDK 接口可调用，但反外挂驱动/服务不激活）。
			//   - 正式发布：加固环节由 ACE 加固工具产出「加固版 exe + 加固版
			//     AntiCheatExpert/ 目录」，同名替换（走同一份 P4 提交流程）。
			//
			// SDK 内部通过 LoadLibrary("AntiCheatExpert\\ACE-Base64.dll") 相对 exe 目录
			// 加载，故子目录层级不可省略；StagedFileType.NonUFS 会保持
			// Binaries/Win64/AntiCheatExpert/ 的相对层级落到包内，与 exe 同级。
			// ========================================================================
			RuntimeDependencies.Add(
				"$(ProjectDir)/Binaries/Win64/AntiCheatExpert/...", StagedFileType.NonUFS);
		}

		// ============================================================
		// AutoTest.ini — 以散文件（NonUFS）形式打进客户端包
		// ------------------------------------------------------------
		// 背景：Config 目录整体由 DefaultGame.ini 的
		//     +DirectoriesToAlwaysStageAsUFS=(Path="Config")
		// 打进 pak，整包用户拿到的包内 AutoTest.ini 恒为 enabled=0 + 全空，
		// 无法在包内修改（pak 只读）。原本用户只能写 bat 用 -autotestini=
		// 指向包外自定义 ini，操作门槛高。
		//
		// 本声明把 AutoTest.ini 额外登记为 NonUFS 散文件，落盘到 exe 旁
		//     <包根>/RaidGame/Config/AutoTest.ini
		// 用户用记事本直接改这个文件（enabled=1 + 填参数）即可触发自动化，
		// 无需 bat、无需命令行参数。
		//
		// 双份并存是安全的：
		//   1. UAT 的 FilesToStage 中 UFS/NonUFS 是两个独立字典，同一输出
		//      路径可同时登记，互不冲突（Version.ini 即 pak + 散文件双份先例）；
		//   2. 读取方全部走 OS 文件 API（Lua io.open、FPaths::FileExists、
		//      FFileHelper::LoadFileToString），读不到 pak 内副本，散文件
		//      即唯一有效副本，pak 内那份永远不生效。
		//
		// 平台隔离（沿用 ACE 的教训，见上方反例注释）：
		//   - Win64 + TargetType.Client：覆盖 RaidGameClient / RaidGameClientProfile，
		//     排除 LinuxServer(DS)、Android、Editor，非 Win64 Target 编译时
		//     本段不产生任何 RuntimeDependency。
		//   - Shipping 剔除自动化功能；仅非 Shipping 客户端保留散文件。
		//
		// 单参数 + StagedFileType 重载：文件已在 $(ProjectDir)/Config/ 下，
		// 只登记 Stage，不发生拷贝。
		// ============================================================
		if (Target.Platform == UnrealTargetPlatform.Win64
			&& Target.Type == TargetType.Client
			&& Target.Configuration != UnrealTargetConfiguration.Shipping)
		{
			RuntimeDependencies.Add(
				"$(ProjectDir)/Config/AutoTest.ini", StagedFileType.NonUFS);
		}

		// [theozhu++] NSPF S4.1: NNE 推理 (见 docs/S4-引擎接入技术方案.md)
		PublicDependencyModuleNames.AddRange(new string[] {
			"NNE",
			"NNERuntimeRDG",
			"RenderCore",
			"RHI",
		});
		// [theozhu--]
    }
}
