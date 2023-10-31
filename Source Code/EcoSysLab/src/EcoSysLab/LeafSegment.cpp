#include <LeafSegment.hpp>

using namespace EcoSysLab;

LeafSegment::LeafSegment(glm::vec3 position, glm::vec3 up, glm::vec3 front, float stemWidth,
                         float leafHalfWidth, float theta, bool isLeaf,
                         float leftHeightFactor, float rightHeightFactor) {
  m_isLeaf = isLeaf;
  m_position = position;
  m_up = up;
  m_front = front;
  m_leafHalfWidth = leafHalfWidth;
  m_theta = theta;
  if (isLeaf) {
    m_leftHeightFactor = leftHeightFactor;
    m_rightHeightFactor = rightHeightFactor;
  }
  m_stemRadius = stemWidth;
}

glm::vec3 LeafSegment::GetPoint(float angle) {
  if (glm::abs(m_theta) < 90.0f) {
    const auto radius = m_leafHalfWidth / glm::sin(glm::radians(m_theta));
    float actualHeight = m_stemRadius;
    actualHeight *= angle < 0 ? m_leftHeightFactor : m_rightHeightFactor;
    const auto distanceToCenter = radius - m_stemRadius;
    const auto center =
        m_position + distanceToCenter * m_up;
    const auto direction = glm::rotate(m_up, glm::radians(angle), m_front);
    return center - radius * direction - actualHeight * m_up;
  }
  const auto direction = glm::rotate(m_up, glm::radians(angle), m_front);
  return m_position - m_stemRadius * direction;
}
glm::vec3 LeafSegment::GetNormal(float angle) {
  return glm::normalize(glm::rotate(m_up, glm::radians(angle), m_front));
}
