
#include "IVolume.hpp"

using namespace EcoSysLab;

bool IVolume::InVolume(const GlobalTransform& globalTransform, const glm::vec3& position) { return false; }

bool IVolume::InVolume(const glm::vec3& position) { return false; }

void IVolume::InVolume(const GlobalTransform& globalTransform, const std::vector<glm::vec3>& positions, std::vector<bool>& results) {
    results.resize(positions.size());
    std::vector<std::shared_future<void>> jobs;
    Jobs::ParallelFor(positions.size(), [&](unsigned i) {
        results[i] = InVolume(globalTransform, positions[i]);
        }, jobs);
    for (const auto& i : jobs) {
        i.wait();
    }
}

void IVolume::InVolume(const std::vector<glm::vec3>& positions, std::vector<bool>& results) {
    results.resize(positions.size());
    std::vector<std::shared_future<void>> jobs;
    Jobs::ParallelFor(positions.size(), [&](unsigned i) {
        results[i] = InVolume(positions[i]);
        }, jobs);
    for (const auto& i : jobs) {
        i.wait();
    }
}

glm::vec3 SphericalVolume::GetRandomPoint() {
    return glm::ballRand(1.0f) * m_radius;
}
bool SphericalVolume::InVolume(
    const GlobalTransform& globalTransform, const glm::vec3& position) {
    return false;
}
bool SphericalVolume::InVolume(const glm::vec3& position) {
    auto relativePosition = glm::vec3(position.x / m_radius.x, position.y / m_radius.y, position.z / m_radius.z);
    return glm::length(relativePosition) <= 1.0f;
}
