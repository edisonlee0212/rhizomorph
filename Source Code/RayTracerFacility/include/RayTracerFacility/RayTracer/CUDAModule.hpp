#pragma once

#include <RayTracer.hpp>
#include <memory>
#include <ray_tracer_facility_export.h>

struct cudaGraphicsResource;
namespace RayTracerFacility {
    class RAY_TRACER_FACILITY_API CudaModule {
#pragma region Class related

        CudaModule() = default;

        CudaModule(CudaModule &&) = default;

        CudaModule(const CudaModule &) = default;

        CudaModule &operator=(CudaModule &&) = default;

        CudaModule &operator=(const CudaModule &) = default;

#pragma endregion
        void *m_optixHandle = nullptr;
        bool m_initialized = false;
        std::unique_ptr<RayTracer> m_rayTracer;

        friend class RayTracerLayer;

    public:
        static std::unique_ptr<RayTracer> &GetRayTracer();

        static CudaModule &GetInstance();

        static void Init();

        static void Terminate();

        static void EstimateIlluminationRayTracing(const EnvironmentProperties &environmentProperties,
                                                   const RayProperties &rayProperties,
                                                   std::vector<IlluminationSampler<glm::vec3>> &lightProbes, unsigned seed,
                                                   float pushNormalDistance);

        static void
        SamplePointCloud(const EnvironmentProperties &environmentProperties,
                         std::vector<PointCloudSample> &samples);
    };
} // namespace RayTracerFacility
