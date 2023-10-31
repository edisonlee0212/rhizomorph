#include <cstdio>
#include <CUDAModule.hpp>
#include <Optix7.hpp>
#include <RayTracer.hpp>
#include <cuda_gl_interop.h>
#include <cuda.h>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <glm/glm.hpp>

using namespace RayTracerFacility;

std::unique_ptr<RayTracer> &CudaModule::GetRayTracer() {
    return GetInstance().m_rayTracer;
}

CudaModule &CudaModule::GetInstance() {
    static CudaModule instance;
    return instance;
}

void CudaModule::Init() {
    // Choose which GPU to run on, change this on a multi-GPU system.
    CUDA_CHECK(SetDevice(0));
    OPTIX_CHECK(optixInitWithHandle(&GetInstance().m_optixHandle));
    GetInstance().m_rayTracer = std::make_unique<RayTracer>();
    GetInstance().m_initialized = true;
}

void CudaModule::Terminate() {
    GetInstance().m_rayTracer.reset();
    OPTIX_CHECK(optixUninitWithHandle(GetInstance().m_optixHandle));
    CUDA_CHECK(DeviceReset());
    GetInstance().m_initialized = false;
}


void CudaModule::EstimateIlluminationRayTracing(const EnvironmentProperties &environmentProperties, const RayProperties& rayProperties,
                                                std::vector<IlluminationSampler<glm::vec3>> &lightProbes, unsigned seed, float pushNormalDistance) {
    auto &cudaModule = GetInstance();
#pragma region Prepare light probes
    size_t size = lightProbes.size();
    CudaBuffer deviceLightProbes;
    deviceLightProbes.Upload(lightProbes);
#pragma endregion
    cudaModule.m_rayTracer->EstimateIllumination(size, environmentProperties, rayProperties, deviceLightProbes, seed, pushNormalDistance);
    deviceLightProbes.Download(lightProbes.data(), size);
    deviceLightProbes.Free();
}

void
CudaModule::SamplePointCloud(const EnvironmentProperties &environmentProperties,
                             std::vector<PointCloudSample> &samples) {
    auto &cudaModule = GetInstance();
#pragma region Prepare light probes
    size_t size = samples.size();
    CudaBuffer deviceSamples;
    deviceSamples.Upload(samples);
#pragma endregion
    cudaModule.m_rayTracer->ScanPointCloud(size, environmentProperties, deviceSamples);
    deviceSamples.Download(samples.data(), size);
    deviceSamples.Free();
}
