//
// Created by lllll on 11/6/2021.
//

#include "LightProbeGroup.hpp"
#include "RayTracerLayer.hpp"
using namespace RayTracerFacility;
void LightProbeGroup::CalculateIllumination(const RayProperties& rayProperties, int seed, float pushNormalDistance) {
    if (m_lightProbes.empty()) return;
    CudaModule::EstimateIlluminationRayTracing(Application::GetLayer<RayTracerLayer>()->m_environmentProperties, rayProperties, m_lightProbes, seed, pushNormalDistance);
}

void LightProbeGroup::OnInspect() {
    ImGui::Text("Light probes size: %llu", m_lightProbes.size());
}
