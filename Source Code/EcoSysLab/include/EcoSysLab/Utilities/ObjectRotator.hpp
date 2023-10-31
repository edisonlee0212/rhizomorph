#pragma once

#include "ecosyslab_export.h"

using namespace UniEngine;
namespace EcoSysLab {
    class ObjectRotator : public IPrivateComponent {
    public:
        float m_rotateSpeed;
        glm::vec3 m_rotation = glm::vec3(0, 0, 0);

        void OnInspect() override;

        void FixedUpdate() override;

        void Serialize(YAML::Emitter& out) override;

        void Deserialize(const YAML::Node& in) override;
    };
} // namespace PlantFactory
