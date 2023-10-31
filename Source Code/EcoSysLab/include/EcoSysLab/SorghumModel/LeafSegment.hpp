#pragma once

using namespace UniEngine;
namespace EcoSysLab {
class LeafSegment {
public:
  glm::vec3 m_position;
  glm::vec3 m_front;
  glm::vec3 m_up;
  glm::quat m_rotation;
  float m_leafHalfWidth;
  float m_theta;
  float m_stemRadius;
  float m_leftHeightFactor = 1.0f;
  float m_rightHeightFactor = 1.0f;
  bool m_isLeaf;
  LeafSegment(glm::vec3 position, glm::vec3 up, glm::vec3 front, float stemWidth,
              float leafHalfWidth, float theta, bool isLeaf,
              float leftHeightFactor = 1.0f, float rightHeightFactor = 1.0f);

  glm::vec3 GetPoint(float angle);

  glm::vec3 GetNormal(float angle);
};
} // namespace PlantFactory