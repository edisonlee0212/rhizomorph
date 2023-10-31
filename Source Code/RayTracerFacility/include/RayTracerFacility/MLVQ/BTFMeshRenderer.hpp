#pragma once
#include <UniEngine-pch.hpp>
#include <memory>
#include <ray_tracer_facility_export.h>

#include <Entity.hpp>
#include <Mesh.hpp>
#include <Texture2D.hpp>


using namespace UniEngine;
namespace RayTracerFacility {
    class RAY_TRACER_FACILITY_API BTFMeshRenderer : public IPrivateComponent {
    public:
        AssetRef m_mesh;
        AssetRef m_btf;

        void OnInspect() override;
        void Serialize(YAML::Emitter &out) override;
        void Deserialize(const YAML::Node &in) override;

        void CollectAssetRef(std::vector<AssetRef> &list) override;
    };
} // namespace RayTracerFacility
