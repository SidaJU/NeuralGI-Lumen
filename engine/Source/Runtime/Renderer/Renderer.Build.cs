// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class Renderer : ModuleRules
{
	public Renderer(ReadOnlyTargetRules Target) : base(Target)
	{
		PrivateIncludePaths.AddRange(
			new string[] {
				EngineDirectory + "/Shaders/Private", // For HaltonUtilities.ush
			}
		);

		PrivateIncludePathModuleNames.Add("Shaders");

		PublicDependencyModuleNames.AddRange(
			new string[] {
				"Core",
				"Engine",
			}
		);
		
		// #if LQT_WIND_FIELD
		PublicDependencyModuleNames.Add("DeveloperSettings");
		// #endif

		if (Target.bBuildEditor == true)
        {
			PrivateDependencyModuleNames.AddRange(
				new string[] {
					"TargetPlatform",
					"GeometryCore",
					"NaniteUtilities",
					// #if LQT_LUMEN_BAKE
					"ImageCore",
					"LumenBakeUtilities"
					// #endif
				}
			);
			
			// #if LQT_LUMEN_BAKE
			// EditorReflectionUtilities is dynamically loaded for editor viewport access
			PrivateIncludePathModuleNames.Add("EditorReflectionUtilities");
			DynamicallyLoadedModuleNames.Add("EditorReflectionUtilities");
			// #endif
        }

        // Renderer module builds faster without unity
        // Non-unity also provides faster iteration
		// Not enabled by default as it might harm full rebuild times without XGE
        //bFasterWithoutUnity = true;

        MinFilesUsingPrecompiledHeaderOverride = 4;

		PrivateDependencyModuleNames.AddRange(
			new string[] {
				"CoreUObject",
				"ApplicationCore",
				"RenderCore",
				"ImageWriteQueue",
				// [++Raid] odinlchen VSM cache primitive dump
				"Json",
				// [--Raid]
				"RHI",
				"MaterialShaderQualitySettings",
				// [theozhu++] NSPF S4.2: 引擎内跑 NNE 推理 (替换 Screen Probe 滤波)
				"NNE",
				// [theozhu--]
				"StateStream",
				"TraceLog",
			}
		);

        PrivateIncludePathModuleNames.AddRange(new string[] { "HeadMountedDisplay" });
        DynamicallyLoadedModuleNames.AddRange(new string[] { "HeadMountedDisplay" });
		PrivateIncludePathModuleNames.AddRange(new string[] { "EyeTracker" });
		DynamicallyLoadedModuleNames.AddRange(new string[] { "EyeTracker" });

		RuntimeDependencies.Add("$(EngineDir)/Content/Renderer/TessellationTable.bin");
	}
}
