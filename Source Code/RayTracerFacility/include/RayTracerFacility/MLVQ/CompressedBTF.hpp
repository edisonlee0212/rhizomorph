#pragma once
#include <UniEngine-pch.hpp>
#include <memory>
#include <ray_tracer_facility_export.h>
#include "OpenGLUtils.hpp"
#include "IAsset.hpp"
#include "Scene.hpp"
#include "Transform.hpp"
#include "Vertex.hpp"
#include "filesystem"
#include "BTFBase.cuh"
using namespace UniEngine;
namespace RayTracerFacility {
    class RAY_TRACER_FACILITY_API CompressedBTF : public IAsset {
        std::vector<float> m_sharedCoordinatesBetaAngles;

        std::vector<int> m_pdf6DSlices;
        std::vector<float> m_pdf6DScales;

        std::vector<int> m_pdf4DSlices;
        std::vector<float> m_pdf4DScales;

        std::vector<int> m_pdf3DSlices;
        std::vector<float> m_pdf3DScales;

        std::vector<int> m_indexLuminanceColors;
        std::vector<int> m_pdf2DColors;
        std::vector<float> m_pdf2DScales;
        std::vector<int> m_pdf2DSlices;

        std::vector<int> m_indexAbBasis;

        std::vector<float> m_pdf1DBasis;

        std::vector<float> m_vectorColorBasis;
        void UploadDeviceData();

    public:
        size_t m_version = 0;
        BTFBase m_bTFBase;
        void OnInspect() override;
        bool ImportFromFolder(const std::filesystem::path &path);
        void Serialize(YAML::Emitter &out) override;

        void Deserialize(const YAML::Node &in) override;
    };
}