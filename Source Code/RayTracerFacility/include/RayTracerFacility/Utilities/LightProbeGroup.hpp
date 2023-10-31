#pragma once

#include <UniEngine-pch.hpp>
#include <Application.hpp>
#include <ray_tracer_facility_export.h>
#include <CUDAModule.hpp>

using namespace UniEngine;
namespace RayTracerFacility {
    struct RAY_TRACER_FACILITY_API LightProbeGroup {
        std::vector<IlluminationSampler<glm::vec3>> m_lightProbes;
        void CalculateIllumination(const RayProperties& rayProperties, int seed, float pushNormalDistance);
        void OnInspect();
    };

    template <typename T>
    inline void SaveListAsBinary(const std::string& name, const std::vector<T>& target, YAML::Emitter &out){
        if (!target.empty())
        {
            out << YAML::Key << name << YAML::Value
                << YAML::Binary((const unsigned char *)target.data(), target.size() * sizeof(T));
        }
    }
    template <typename T>
    inline void LoadListFromBinary(const std::string& name, std::vector<T>& target, const YAML::Node &in){
        if (in[name])
        {
            auto binaryList = in[name].as<YAML::Binary>();
            target.resize(binaryList.size() / sizeof(T));
            std::memcpy(target.data(), binaryList.data(), binaryList.size());
        }
    }
}