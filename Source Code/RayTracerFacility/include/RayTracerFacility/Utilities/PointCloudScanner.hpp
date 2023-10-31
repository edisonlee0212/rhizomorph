#pragma once

#include <UniEngine-pch.hpp>
#include <Application.hpp>
#include <ray_tracer_facility_export.h>
#include <CUDAModule.hpp>
#include <PointCloud.hpp>
using namespace UniEngine;
namespace RayTracerFacility {
    class RAY_TRACER_FACILITY_API PointCloudScanner : public IPrivateComponent {
    public:
        float m_rotateAngle = 0.0f;
        glm::vec2 m_size = glm::vec2(8, 4);
        glm::vec2 m_distance = glm::vec2(0.02f, 0.02f);

        std::vector<uint64_t> m_handles;
        std::vector<glm::vec3> m_points;
        std::vector<glm::vec3> m_pointColors;
        void ConstructPointCloud(std::shared_ptr<PointCloud> pointCloud);

        void Scan();

        void OnInspect() override;

        void Serialize(YAML::Emitter &out) override;

        void Deserialize(const YAML::Node &in) override;
    };
}