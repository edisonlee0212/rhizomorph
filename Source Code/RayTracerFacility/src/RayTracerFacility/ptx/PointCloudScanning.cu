#include <RayTracerUtilities.cuh>

namespace RayTracerFacility {
    extern "C" __constant__ PointCloudScanningLaunchParams
            pointCloudScanningLaunchParams;

	struct PointCloudScanningPerRayData {
        bool m_hit;
        Random m_random;
        uint64_t m_handle;
        HitInfo m_hitInfo;
    };

#pragma region Closest hit functions
    extern "C" __global__ void __closesthit__PCS_R() {
        const auto &sbtData = *(const SBT *) optixGetSbtDataPointer();
        const float2 triangleBarycentricsInternal = optixGetTriangleBarycentrics();
        const int primitiveId = optixGetPrimitiveIndex();
        const float3 rayDirectionInternal = optixGetWorldRayDirection();
        glm::vec3 rayDirection = glm::vec3(
                rayDirectionInternal.x, rayDirectionInternal.y, rayDirectionInternal.z);
        auto hitInfo = sbtData.GetHitInfo(rayDirection);

        PointCloudScanningPerRayData &prd = *GetRayDataPointer <PointCloudScanningPerRayData> ();
        prd.m_hit = true;
        prd.m_handle = sbtData.m_handle;
        prd.m_hitInfo = hitInfo;
    }
    extern "C" __global__ void __closesthit__PCS_SS() {}
#pragma endregion
#pragma region Any hit functions
    extern "C" __global__ void __anyhit__PCS_R() {}
    extern "C" __global__ void __anyhit__PCS_SS() {}
#pragma endregion
#pragma region Miss functions
    extern "C" __global__ void __miss__PCS_R() {
        PointCloudScanningPerRayData &prd = *GetRayDataPointer<PointCloudScanningPerRayData> ();
        prd.m_hit = false;
        prd.m_handle = 0;
    }
    extern "C" __global__ void __miss__PCS_SS() {}
#pragma endregion
#pragma region Main ray generation
    extern "C" __global__ void __raygen__PCS() {
        unsigned ix = optixGetLaunchIndex().x;
        auto &samples = pointCloudScanningLaunchParams.m_samples[ix];
        auto start = samples.m_start;
        auto direction = samples.m_direction;
        float3 rayOrigin = make_float3(start.x, start.y, start.z);
        float3 rayDirection = make_float3(direction.x, direction.y, direction.z);

        PointCloudScanningPerRayData perRayData;
        perRayData.m_random.Init(ix, 0);
        perRayData.m_hit = false;
        perRayData.m_hitInfo = HitInfo();
        uint32_t u0, u1;
        PackRayDataPointer(&perRayData, u0, u1);
        optixTrace(
                pointCloudScanningLaunchParams.m_traversable, rayOrigin, rayDirection,
                1e-3f, // tmin
                1e20f, // tmax
                0.0f,  // rayTime
                static_cast<OptixVisibilityMask>(255),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
                static_cast<int>(
                        RayType::Radiance), // SBT offset
                static_cast<int>(
                        RayType::RayTypeCount), // SBT stride
                static_cast<int>(
                        RayType::Radiance), // missSBTIndex
                u0, u1);
        samples.m_handle = perRayData.m_handle;
        samples.m_hit = perRayData.m_hit;
        samples.m_hitInfo = perRayData.m_hitInfo;
    }
#pragma endregion
} // namespace RayTracerFacility
