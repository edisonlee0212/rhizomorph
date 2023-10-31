#pragma once
#include "LightProbeGroup.hpp"
#include <UniEngine-pch.hpp>
#include <Application.hpp>
#include <ray_tracer_facility_export.h>
#include <CUDAModule.hpp>

using namespace UniEngine;
namespace RayTracerFacility {
    class RAY_TRACER_FACILITY_API TriangleIlluminationEstimator : public IPrivateComponent {
        LightProbeGroup m_lightProbeGroup;
    public:
        void PrepareLightProbeGroup();
        void SampleLightProbeGroup(const RayProperties& rayProperties, int seed, float pushNormalDistance);
        float m_totalArea = 0.0f;
        glm::vec3 m_totalFlux = glm::vec3(0.0f);
        glm::vec3 m_averageFlux = glm::vec3(0.0f);
        void OnInspect() override;

        void Serialize(YAML::Emitter &out) override;
        void Deserialize(const YAML::Node &in) override;
    };


} // namespace SorghumFactory
