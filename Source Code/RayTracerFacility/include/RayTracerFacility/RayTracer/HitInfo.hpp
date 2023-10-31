#pragma once;
#include <Vertex.hpp>
namespace RayTracerFacility {
    struct HitInfo {
        glm::vec3 m_position = glm::vec3(0.0f);
        glm::vec3 m_normal = glm::vec3(0.0f);
        glm::vec3 m_tangent = glm::vec3(0.0f);
        glm::vec4 m_color = glm::vec4(1.0f);
        glm::vec2 m_texCoord = glm::vec2(0.0f);
        glm::vec4 m_data = glm::vec4(0.0f);
    };
}