#include "Curve.hpp"

using namespace EcoSysLab;

void ICurve::GetUniformCurve(size_t pointAmount, std::vector<glm::vec3> &points) const {
    float step = 1.0f / (pointAmount - 1);
    for (size_t i = 0; i <= pointAmount; i++) {
        points.push_back(GetPoint(step * i));
    }
}

BezierCurve::BezierCurve()
{
}


BezierCurve::BezierCurve(glm::vec3 cp0, glm::vec3 cp1, glm::vec3 cp2, glm::vec3 cp3)
        : ICurve(),
          m_p0(cp0),
          m_p1(cp1),
          m_p2(cp2),
          m_p3(cp3) {
}

glm::vec3 BezierCurve::GetPoint(float t) const {
    t = glm::clamp(t, 0.f, 1.f);
    return m_p0 * (1.0f - t) * (1.0f - t) * (1.0f - t)
           + m_p1 * 3.0f * t * (1.0f - t) * (1.0f - t)
           + m_p2 * 3.0f * t * t * (1.0f - t)
           + m_p3 * t * t * t;
}

glm::vec3 BezierCurve::GetAxis(float t) const {
    t = glm::clamp(t, 0.f, 1.f);
    float mt = 1.0f - t;
    return (m_p1 - m_p0) * 3.0f * mt * mt + 6.0f * t * mt * (m_p2 - m_p1) + 3.0f * t * t * (m_p3 - m_p2);
}

glm::vec3 BezierCurve::GetStartAxis() const {
    return glm::normalize(m_p1 - m_p0);
}

glm::vec3 BezierCurve::GetEndAxis() const {
    return glm::normalize(m_p3 - m_p2);
}

float BezierCurve::GetLength() const {
		return glm::distance(m_p0, m_p3);
}

glm::vec3 BezierSpline::EvaluatePointFromCurves(float point) const {
    const float splineU = glm::clamp(point, 0.0f, 1.0f) * float(m_curves.size());

    // Decompose the global u coordinate on the spline
    float integerPart;
    const float fractionalPart = modff(splineU, &integerPart);

    auto curveIndex = int(integerPart);
    auto curveU = fractionalPart;

    // If evaluating the very last point on the spline
    if (curveIndex == m_curves.size() && curveU <= 0.0f) {
        // Flip to the end of the last patch
        curveIndex--;
        curveU = 1.0f;
    }
    return m_curves.at(curveIndex).GetPoint(curveU);
}
void BezierSpline::OnInspect() {
    int size = m_curves.size();
    if (ImGui::DragInt("Size of curves", &size, 0, 10)) {
        size = glm::clamp(size, 0, 10);
        m_curves.resize(size);
    }
    if (ImGui::TreeNode("Curves")) {
        int index = 1;
        for (auto& curve : m_curves) {
            if (ImGui::TreeNode(("Curve " + std::to_string(index)).c_str())) {
                ImGui::DragFloat3("CP0", &curve.m_p0.x, 0.01f);
                ImGui::DragFloat3("CP1", &curve.m_p1.x, 0.01f);
                ImGui::DragFloat3("CP2", &curve.m_p2.x, 0.01f);
                ImGui::DragFloat3("CP3", &curve.m_p3.x, 0.01f);
                ImGui::TreePop();
            }
            index++;
        }
        ImGui::TreePop();
    }
}
void BezierSpline::Serialize(YAML::Emitter& out) {
    if (!m_curves.empty()) {
        out << YAML::Key << "m_curves" << YAML::Value << YAML::BeginSeq;
        for (const auto& i : m_curves) {
            out << YAML::BeginMap;
            out << YAML::Key << "m_p0" << YAML::Value << i.m_p0;
            out << YAML::Key << "m_p1" << YAML::Value << i.m_p1;
            out << YAML::Key << "m_p2" << YAML::Value << i.m_p2;
            out << YAML::Key << "m_p3" << YAML::Value << i.m_p3;
            out << YAML::EndMap;
        }
        out << YAML::EndSeq;
    }
}
void BezierSpline::Deserialize(const YAML::Node& in) {
    if (in["m_curves"]) {
        m_curves.clear();
        for (const auto& iCurve : in["m_curves"]) {
            m_curves.emplace_back(
                iCurve["m_p0"].as<glm::vec3>(), iCurve["m_p1"].as<glm::vec3>(),
                iCurve["m_p2"].as<glm::vec3>(), iCurve["m_p3"].as<glm::vec3>());
        }
    }
}

void BezierSpline::Import(std::ifstream& stream)
{
    int curveAmount;
    stream >> curveAmount;
    m_curves.clear();
    for (int i = 0; i < curveAmount; i++) {
        glm::vec3 cp[4];
        float x, y, z;
        for (auto& j : cp) {
            stream >> x >> z >> y;
            j = glm::vec3(x, y, z);
        }
        m_curves.emplace_back(cp[0], cp[1], cp[2], cp[3]);
    }
}

glm::vec3 BezierSpline::EvaluateAxisFromCurves(float point) const {
    const float splineU = glm::clamp(point, 0.0f, 1.0f) * float(m_curves.size());

    // Decompose the global u coordinate on the spline
    float integerPart;
    const float fractionalPart = modff(splineU, &integerPart);

    auto curveIndex = int(integerPart);
    auto curveU = fractionalPart;

    // If evaluating the very last point on the spline
    if (curveIndex == m_curves.size() && curveU <= 0.0f) {
        // Flip to the end of the last patch
        curveIndex--;
        curveU = 1.0f;
    }
    return m_curves.at(curveIndex).GetAxis(curveU);
}
