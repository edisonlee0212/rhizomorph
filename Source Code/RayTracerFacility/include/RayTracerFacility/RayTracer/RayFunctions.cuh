#include "BSDF.cuh"
#include "BSSDF.cuh"
#include "Environment.cuh"
namespace RayTracerFacility {
    static __forceinline__ __device__ void AnyHitFunc() {
        const float3 rayDirectionInternal = optixGetWorldRayDirection();
        glm::vec3 rayDirection = glm::vec3(
                rayDirectionInternal.x, rayDirectionInternal.y, rayDirectionInternal.z);
#pragma region Retrive information
        const auto &sbtData = *(const SBT *) optixGetSbtDataPointer();
        auto hitInfo = sbtData.GetHitInfo(rayDirection);
#pragma endregion
        switch (sbtData.m_materialType) {
            case MaterialType::Default: {
                PerRayData <glm::vec3> &perRayData =
                        *GetRayDataPointer < PerRayData < glm::vec3 >> ();
                glm::vec4 albedoColor =
                        static_cast<SurfaceMaterial *>(sbtData.m_material)->GetAlbedo(hitInfo.m_texCoord);
                if (albedoColor.w <= perRayData.m_random()) optixIgnoreIntersection();
            }
                break;
        }
    }

    static __forceinline__ __device__ void ClosestHitFunc(const RayTracerProperties& rayTracerProperties, OptixTraversableHandle optixTraversableHandle) {
        const float3 rayDirectionInternal = optixGetWorldRayDirection();
        glm::vec3 rayDirection = glm::vec3(
                rayDirectionInternal.x, rayDirectionInternal.y, rayDirectionInternal.z);
#pragma region Retrive information
        const auto &sbtData = *(const SBT *) optixGetSbtDataPointer();
        auto hitInfo = sbtData.GetHitInfo(rayDirection);
#pragma endregion
        PerRayData <glm::vec3> &perRayData =
                *GetRayDataPointer < PerRayData < glm::vec3 >> ();
        unsigned hitCount = perRayData.m_hitCount + 1;

        // start with some ambient term
        auto energy = glm::vec3(0.0f);
        uint32_t u0, u1;
        PackRayDataPointer(&perRayData, u0, u1);
        perRayData.m_hitCount = hitCount;
        perRayData.m_energy = glm::vec3(0.0f);
        auto &environment = rayTracerProperties.m_environment;
        if (sbtData.m_materialType != MaterialType::CompressedBTF) {
            auto *material = static_cast<SurfaceMaterial *>(sbtData.m_material);
            material->ApplyNormalTexture(hitInfo.m_normal, hitInfo.m_texCoord, hitInfo.m_tangent);
            float metallic =
                    material->GetMetallic(hitInfo.m_texCoord);
            float roughness =
                    material->GetRoughness(hitInfo.m_texCoord);
            glm::vec3 albedoColor;
            if (sbtData.m_materialType == MaterialType::Default) {
                albedoColor = material->GetAlbedo(hitInfo.m_texCoord);
            } else {
                albedoColor = hitInfo.m_color;
            }
            energy = glm::vec3(0.0f);
            float f = 1.0f;
            if (metallic >= 0.0f)
                f = (metallic + 2) / (metallic + 1);
            if (environment.m_environmentalLightingType == EnvironmentalLightingType::SingleLightSource) {
                glm::vec3 newRayDirection = RandomSampleHemisphere(perRayData.m_random,
                                                                   environment.m_sunDirection,
                                                                   1.0f -
                                                                   environment.m_lightSize);
                energy += glm::vec3(environment.m_color) * environment.m_ambientLightIntensity * albedoColor;
                const float NdotL = glm::dot(
                        hitInfo.m_normal, newRayDirection);
                if (NdotL > 0.0f) {
                    uint32_t u0, u1;
                    PackRayDataPointer(&perRayData, u0, u1);
                    perRayData.m_energy = glm::vec3(0.0f);
                    optixTrace(
                            optixTraversableHandle,
                            make_float3(hitInfo.m_position.x, hitInfo.m_position.y, hitInfo.m_position.z),
                            make_float3(newRayDirection.x, newRayDirection.y, newRayDirection.z),
                            1e-3f, // tmin
                            1e20f, // tmax
                            0.0f,  // rayTime
                            static_cast<OptixVisibilityMask>(255),
                            OPTIX_RAY_FLAG_DISABLE_ANYHIT
                            | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                            | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                            static_cast<int>(
                                    RayType::Radiance), // SBT offset
                            static_cast<int>(
                                    RayType::RayTypeCount), // SBT stride
                            static_cast<int>(
                                    RayType::Radiance), // missSBTIndex
                            u0, u1);
                    energy += perRayData.m_energy * NdotL * albedoColor;
                }
            } else if (perRayData.m_hitCount <=
                       rayTracerProperties.m_rayProperties
                               .m_bounces) {
                bool needSample = false;
                if (hitCount <= 1 && material->m_materialProperties.m_subsurfaceFactor > 0.0f &&
                    material->m_materialProperties.m_subsurfaceRadius.x > 0.0f) {
                    float3 incidentRayOrigin;
                    float3 newRayDirectionInternal;
                    glm::vec3 outNormal;
                    needSample = BSSRDF(metallic, perRayData.m_random,
                                        material->m_materialProperties.m_subsurfaceRadius.x, sbtData.m_handle,
                                        optixTraversableHandle,
                                        hitInfo.m_position, rayDirection, hitInfo.m_normal,
                                        incidentRayOrigin, newRayDirectionInternal, outNormal);
                    if (needSample) {
                        optixTrace(
                                optixTraversableHandle, incidentRayOrigin,
                                newRayDirectionInternal,
                                1e-3f, // tmin
                                1e20f, // tmax
                                0.0f,  // rayTime
                                static_cast<OptixVisibilityMask>(255), OPTIX_RAY_FLAG_NONE,
                                static_cast<int>(
                                        RayType::Radiance), // SBT offset
                                static_cast<int>(
                                        RayType::RayTypeCount), // SBT stride
                                static_cast<int>(
                                        RayType::Radiance), // missSBTIndex
                                u0, u1);
                        energy += material->m_materialProperties.m_subsurfaceFactor *
                                  material->m_materialProperties.m_subsurfaceColor *
                                  glm::clamp(
                                          glm::abs(glm::dot(outNormal, glm::vec3(newRayDirectionInternal.x,
                                                                                 newRayDirectionInternal.y,
                                                                                 newRayDirectionInternal.z))) *
                                          roughness +
                                          (1.0f - roughness) * f,
                                          0.0f, 1.0f) *
                                  perRayData.m_energy;
                    }
                }
                float3 newRayDirectionInternal;
                BRDF(metallic, perRayData.m_random, rayDirection, hitInfo.m_normal, newRayDirectionInternal);
                optixTrace(
                        optixTraversableHandle,
                        make_float3(hitInfo.m_position.x, hitInfo.m_position.y, hitInfo.m_position.z),
                        newRayDirectionInternal,
                        1e-3f, // tmin
                        1e20f, // tmax
                        0.0f,  // rayTime
                        static_cast<OptixVisibilityMask>(255),
                        OPTIX_RAY_FLAG_NONE,
                        static_cast<int>(
                                RayType::Radiance), // SBT offset
                        static_cast<int>(
                                RayType::RayTypeCount), // SBT stride
                        static_cast<int>(
                                RayType::Radiance), // missSBTIndex
                        u0, u1);
                energy += (1.0f - material->m_materialProperties.m_subsurfaceFactor) *
                          albedoColor *
                          glm::clamp(glm::abs(glm::dot(
                                             hitInfo.m_normal, glm::vec3(newRayDirectionInternal.x,
                                                                         newRayDirectionInternal.y,
                                                                         newRayDirectionInternal.z))) *
                                     roughness +
                                     (1.0f - roughness) * f,
                                     0.0f, 1.0f) *
                          perRayData.m_energy;
            }
            if (hitCount == 1) {
                perRayData.m_normal = hitInfo.m_normal;
                perRayData.m_albedo = albedoColor;
                perRayData.m_position = hitInfo.m_position;
            }
            perRayData.m_energy =
                    energy +
                    static_cast<SurfaceMaterial *>(sbtData.m_material)->m_materialProperties.m_emission *
                    albedoColor;

        } else {
            glm::vec3 btfColor;
            if (perRayData.m_hitCount <=
                rayTracerProperties.m_rayProperties
                        .m_bounces) {
                energy = glm::vec3(0.0f);
                float f = 1.0f;
                glm::vec3 reflected = Reflect(rayDirection, hitInfo.m_normal);
                if (environment.m_environmentalLightingType == EnvironmentalLightingType::SingleLightSource) {
                    glm::vec3 newRayDirection = RandomSampleHemisphere(perRayData.m_random,
                                                                       environment.m_sunDirection,
                                                                       1.0f -
                                                                       environment.m_lightSize);
                    static_cast<SurfaceCompressedBTF *>(sbtData.m_material)
                            ->GetValue(hitInfo.m_texCoord, rayDirection, newRayDirection, hitInfo.m_normal,
                                       hitInfo.m_tangent,
                                       btfColor,
                                       false /*(perRayData.m_printInfo && sampleID == 0)*/);
                    energy += glm::vec3(environment.m_color) * environment.m_ambientLightIntensity * btfColor;
                    const float NdotL = glm::dot(
                            hitInfo.m_normal, newRayDirection);
                    if (NdotL > 0.0f) {

                        auto origin = hitInfo.m_position;
                        origin += hitInfo.m_normal * 1e-3f;
                        float3 incidentRayOrigin = make_float3(origin.x, origin.y, origin.z);
                        float3 newRayDirectionInternal =
                                make_float3(newRayDirection.x, newRayDirection.y, newRayDirection.z);
                        optixTrace(
                                optixTraversableHandle, incidentRayOrigin,
                                newRayDirectionInternal,
                                1e-3f, // tmin
                                1e20f, // tmax
                                0.0f,  // rayTime
                                static_cast<OptixVisibilityMask>(255),
                                OPTIX_RAY_FLAG_DISABLE_ANYHIT
                                | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                                | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                                static_cast<int>(
                                        RayType::Radiance), // SBT offset
                                static_cast<int>(
                                        RayType::RayTypeCount), // SBT stride
                                static_cast<int>(
                                        RayType::Radiance), // missSBTIndex
                                u0, u1);
                        energy += perRayData.m_energy * NdotL * btfColor;
                    }
                } else {
                    glm::vec3 newRayDirection = RandomSampleHemisphere(perRayData.m_random, reflected, 0.0f);
                    static_cast<SurfaceCompressedBTF *>(sbtData.m_material)
                            ->GetValue(hitInfo.m_texCoord, rayDirection, newRayDirection, hitInfo.m_normal,
                                       hitInfo.m_tangent,
                                       btfColor,
                                       false /*(perRayData.m_printInfo && sampleID == 0)*/);
                    auto origin = hitInfo.m_position;
                    origin += hitInfo.m_normal * 1e-3f;
                    float3 incidentRayOrigin = make_float3(origin.x, origin.y, origin.z);
                    float3 newRayDirectionInternal =
                            make_float3(newRayDirection.x, newRayDirection.y, newRayDirection.z);
                    optixTrace(
                            optixTraversableHandle, incidentRayOrigin,
                            newRayDirectionInternal,
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
                    energy += btfColor * perRayData.m_energy;
                }

            }
            if (hitCount == 1) {
                perRayData.m_normal = hitInfo.m_normal;
                perRayData.m_albedo = btfColor;
                perRayData.m_position = hitInfo.m_position;
            }
            perRayData.m_energy = energy;
        }
    }

    static __forceinline__ __device__ void MissFunc(const RayTracerProperties& rayTracerProperties){
        PerRayData <glm::vec3> &perRayData =
                *GetRayDataPointer < PerRayData < glm::vec3 >> ();
        const float3 rayDir = optixGetWorldRayDirection();
        float3 rayOrigin = optixGetWorldRayOrigin();
        glm::vec3 rayOrig = glm::vec3(rayOrigin.x, rayOrigin.y, rayOrigin.z);
        glm::vec3 rayDirection = glm::vec3(rayDir.x, rayDir.y, rayDir.z);
        auto &environment = rayTracerProperties.m_environment;
        glm::vec3 environmentalLightColor = CalculateEnvironmentalLight(
                rayOrig, rayDirection,
                environment);
        perRayData.m_albedo = perRayData.m_energy = environmentalLightColor;
    }
}