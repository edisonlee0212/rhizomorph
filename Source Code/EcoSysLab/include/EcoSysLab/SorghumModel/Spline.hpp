#pragma once
#include <Curve.hpp>
#include <LeafSegment.hpp>
#include "ProceduralSorghum.hpp"
using namespace UniEngine;
namespace EcoSysLab {
struct SplineNode {
  glm::vec3 m_position;
  float m_theta;
  float m_stemWidth;
  float m_leafWidth;
  float m_waviness;
  glm::vec3 m_axis;
  bool m_isLeaf;
  float m_range;

  SplineNode(glm::vec3 position, float angle, float stemWidth, float leafWidth, float waviness, glm::vec3 axis,
            bool isLeaf, float range);
  SplineNode();
};
} // namespace EcoSysLab