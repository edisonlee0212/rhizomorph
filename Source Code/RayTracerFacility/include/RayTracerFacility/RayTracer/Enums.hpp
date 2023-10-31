#pragma once
#include <glm/glm.hpp>
namespace RayTracerFacility {
    enum class MaterialType {
        Default,
        VertexColor,
        CompressedBTF
    };

    enum class RendererType {
        Default,
        Instanced,
        Skinned,
        Curve
    };

    enum class GeometryType {
        Custom,
        QuadraticBSpline,
        CubicBSpline,
        Linear,
        CatmullRom,
        Triangle
    };
}