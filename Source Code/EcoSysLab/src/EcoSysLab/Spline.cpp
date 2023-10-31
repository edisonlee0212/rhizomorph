#include "Spline.hpp"
#include "SorghumLayer.hpp"

using namespace EcoSysLab;

SplineNode::SplineNode() {}
SplineNode::SplineNode(glm::vec3 position, float angle, float stemWidth, float leafWidth,
                       float waviness, glm::vec3 axis, bool isLeaf, float range) {
  m_position = position;
  m_theta = angle;
  m_stemWidth = stemWidth;
  m_leafWidth = leafWidth;
  m_waviness = waviness;
  m_axis = axis;
  m_isLeaf = isLeaf;
  m_range = range;
}
