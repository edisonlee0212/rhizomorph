//
// Created by lllll on 12/15/2021.
//

#include "PointCloudScanner.hpp"
#include "RayTracerLayer.hpp"
#include "Graphics.hpp"
using namespace RayTracerFacility;

void PointCloudScanner::OnInspect() {
    ImGui::DragFloat("Angle", &m_rotateAngle, 0.1f, -90.0f, 90.0f);
    ImGui::DragFloat2("Size", &m_size.x, 0.1f);
    ImGui::DragFloat2("Distance", &m_distance.x, 0.001f, 1.0f, 0.001f);
    auto scene = GetScene();
    static glm::vec4 color = glm::vec4(0, 1, 0, 0.5);
    ImGui::ColorEdit4("Color", &color.x);
    static bool renderPlane = true;
    ImGui::Checkbox("Render plane", &renderPlane);
    auto gt = scene->GetDataComponent<GlobalTransform>(GetOwner());
    glm::vec3 front = glm::normalize(gt.GetRotation() * glm::vec3(0, 0, -1));
    glm::vec3 up = glm::normalize(gt.GetRotation() * glm::vec3(0, 1, 0));
    glm::vec3 left = glm::normalize(gt.GetRotation() * glm::vec3(1, 0, 0));
    glm::vec3 actualVector = glm::rotate(front, glm::radians(m_rotateAngle), up);
    if (renderPlane) {
        Gizmos::DrawGizmoMesh(DefaultResources::Primitives::Cone, glm::vec4(1, 0, 0, 0.5),
                                     glm::translate(gt.GetPosition() + front * 0.5f) *
                                     glm::mat4_cast(glm::quatLookAt(up, glm::normalize(actualVector))) *
                                     glm::scale(glm::vec3(0.1, 0.5, 0.1f)), 1.0f);
        Gizmos::DrawGizmoMesh(DefaultResources::Primitives::Quad, color,
                                     glm::translate(gt.GetPosition()) * glm::mat4_cast(glm::quatLookAt(up, front)) *
                                     glm::scale(glm::vec3(m_size.x / 2.0f, 1.0, m_size.y / 2.0f)), 1.0f);
    }
    if (ImGui::Button("Scan")) {
        Scan();
    }

    ImGui::Text("Sample amount: %d", m_points.size());
    if(!m_points.empty()){
        if(ImGui::Button("Clear")) {
            m_points.clear();
            m_pointColors.clear();
            m_pointColors.clear();
        }
        static AssetRef pointCloud;
        ImGui::Text("Construct PointCloud");
        ImGui::SameLine();
        if(Editor::DragAndDropButton<PointCloud>(pointCloud, "Here", false)){
            auto ptr = pointCloud.Get<PointCloud>();
            if(ptr){
                ConstructPointCloud(ptr);
            }
            pointCloud.Clear();
        }
    }
}

void PointCloudScanner::Serialize(YAML::Emitter &out) {
    out << YAML::Key << "m_rotateAngle" << YAML::Value << m_rotateAngle;
    out << YAML::Key << "m_size" << YAML::Value << m_size;
    out << YAML::Key << "m_distance" << YAML::Value << m_distance;
}

void PointCloudScanner::Deserialize(const YAML::Node &in) {
    m_rotateAngle = in["m_rotateAngle"].as<float>();
    m_size = in["m_size"].as<glm::vec2>();
    m_distance = in["m_distance"].as<glm::vec2>();

}

void PointCloudScanner::Scan() {
    const auto column = unsigned(m_size.x / m_distance.x);
    const int columnStart = -(int) (column / 2);
    const auto row = unsigned(m_size.y / m_distance.y);
    const int rowStart = -(row / 2);
    const auto size = column * row;
    auto gt = GetScene()->GetDataComponent<GlobalTransform>(GetOwner());
    glm::vec3 center = gt.GetPosition();
    glm::vec3 front = gt.GetRotation() * glm::vec3(0, 0, -1);
    glm::vec3 up = gt.GetRotation() * glm::vec3(0, 1, 0);
    glm::vec3 left = gt.GetRotation() * glm::vec3(1, 0, 0);
    glm::vec3 actualVector = glm::rotate(front, glm::radians(m_rotateAngle), up);
    std::vector<PointCloudSample> pcSamples;
    pcSamples.resize(size);

    std::vector<std::shared_future<void>> results;
    Jobs::ParallelFor(size, [&](unsigned i) {
        const int columnIndex = (int) i / row;
        const int rowIndex = (int) i % row;
        const auto position = center + left * (float) (columnStart + columnIndex) * m_distance.x + up * (float) (rowStart + rowIndex) * m_distance.y;
        pcSamples[i].m_start = position;
        pcSamples[i].m_direction = glm::normalize(actualVector);
    }, results);
    for (const auto &i: results)
        i.wait();

    CudaModule::SamplePointCloud(Application::GetLayer<RayTracerLayer>()->m_environmentProperties, pcSamples);
    for (const auto &sample: pcSamples) {
        if (sample.m_hit) {
            m_points.push_back(sample.m_hitInfo.m_position - gt.GetPosition());
            m_pointColors.push_back(sample.m_hitInfo.m_color);
            m_handles.push_back(sample.m_handle);
        }
    }
}

void PointCloudScanner::ConstructPointCloud(std::shared_ptr<PointCloud> pointCloud) {
    for(int i = 0; i < m_points.size(); i++){
        pointCloud->m_points.push_back(m_points[i]);
    }
}
