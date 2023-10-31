
#include <TriangleIlluminationEstimator.hpp>
#include "RayTracerLayer.hpp"
#include "Graphics.hpp"

using namespace RayTracerFacility;

void ColorDescendentsVertices(const std::shared_ptr<Scene> &scene, const Entity &owner,
															const LightProbeGroup &lightProbeGroup) {
		std::vector<glm::vec4> probeColors;
		for (const auto &probe: lightProbeGroup.m_lightProbes) {
				probeColors.emplace_back(glm::vec4(probe.m_energy, 1.0f));
		}
		auto entities = scene->GetDescendants(owner);
		entities.push_back(owner);
		size_t i = 0;
		for (const auto &entity: entities) {
				if (scene->HasPrivateComponent<MeshRenderer>(entity)) {
						auto meshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(entity).lock();
						auto mesh = meshRenderer->m_mesh.Get<Mesh>();
						auto material = meshRenderer->m_material.Get<Material>();
						if (!mesh || !material) continue;
						std::vector<std::pair<size_t, glm::vec4>> colors;
						colors.resize(mesh->GetVerticesAmount());
						for (auto &color: colors) {
								color.first = 0;
								color.second = glm::vec4(0.0f);
						}
						size_t ti = 0;
						for (const auto &triangle: mesh->UnsafeGetTriangles()) {
								const auto color = probeColors[i];
								colors[triangle.x].first++;
								colors[triangle.y].first++;
								colors[triangle.z].first++;
								colors[triangle.x].second += color;
								colors[triangle.y].second += color;
								colors[triangle.z].second += color;
								ti++;
								i++;
						}
						ti = 0;
						for (auto &vertices: mesh->UnsafeGetVertices()) {
								vertices.m_color = colors[ti].second / static_cast<float>(colors[ti].first);
								ti++;
						}
				}
		}
}

void TriangleIlluminationEstimator::OnInspect() {
		auto scene = GetScene();
		auto owner = GetOwner();
		m_lightProbeGroup.OnInspect();
		static int seed = 0;
		static float pushNormalDistance = 0.001f;
		static RayProperties rayProperties;
		ImGui::DragInt("Seed", &seed);
		ImGui::DragFloat("Normal Distance", &pushNormalDistance, 0.0001f, -1.0f, 1.0f);
		ImGui::DragInt("Samples", &rayProperties.m_samples);
		ImGui::DragInt("Bounces", &rayProperties.m_bounces);
		if (ImGui::Button("Estimate")) {
				PrepareLightProbeGroup();
				SampleLightProbeGroup(rayProperties, seed, pushNormalDistance);
				ColorDescendentsVertices(scene, owner, m_lightProbeGroup);
		}
		if (ImGui::TreeNode("Details")) {
				if (ImGui::Button("Prepare light probe group")) {
						PrepareLightProbeGroup();
				}
				if (ImGui::Button("Sample light probe group")) {
						SampleLightProbeGroup(rayProperties, seed, pushNormalDistance);
				}
				if (ImGui::Button("Color vertices")) {
						ColorDescendentsVertices(scene, owner, m_lightProbeGroup);
				}
				ImGui::TreePop();
		}

		ImGui::Text("%s", ("Surface area: " + std::to_string(m_totalArea)).c_str());
		ImGui::Text("%s", ("Total energy: " + std::to_string(glm::length(m_totalFlux))).c_str());
		ImGui::Text("%s", ("Radiant flux: " + std::to_string(glm::length(m_averageFlux))).c_str());
}

void TriangleIlluminationEstimator::SampleLightProbeGroup(const RayProperties &rayProperties, int seed,
																													float pushNormalDistance) {
		m_lightProbeGroup.CalculateIllumination(rayProperties, seed, pushNormalDistance);
		m_totalFlux = glm::vec3(0.0f);
		for (const auto &probe: m_lightProbeGroup.m_lightProbes) {
				m_totalFlux += probe.m_energy * probe.GetArea();
		}
		m_averageFlux = m_totalFlux / m_totalArea;
}

void TriangleIlluminationEstimator::PrepareLightProbeGroup() {
		m_totalArea = 0.0f;
		m_lightProbeGroup.m_lightProbes.clear();
		auto scene = GetScene();
		auto entities = scene->GetDescendants(GetOwner());
		entities.push_back(GetOwner());
		for (const auto &entity: entities) {
				if (scene->HasPrivateComponent<MeshRenderer>(entity)) {
						auto globalTransform = scene->GetDataComponent<GlobalTransform>(entity);
						auto meshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(entity).lock();
						auto mesh = meshRenderer->m_mesh.Get<Mesh>();
						auto material = meshRenderer->m_material.Get<Material>();
						if (!mesh || !material) continue;
						for (const auto &triangle: mesh->UnsafeGetTriangles()) {
								auto &vertices = mesh->UnsafeGetVertices();
								IlluminationSampler<glm::vec3> lightProbe;
								lightProbe.m_a = vertices[triangle.x];
								lightProbe.m_b = vertices[triangle.y];
								lightProbe.m_c = vertices[triangle.z];
								lightProbe.m_a.m_position = globalTransform.m_value * glm::vec4(lightProbe.m_a.m_position, 1.0f);
								lightProbe.m_b.m_position = globalTransform.m_value * glm::vec4(lightProbe.m_b.m_position, 1.0f);
								lightProbe.m_c.m_position = globalTransform.m_value * glm::vec4(lightProbe.m_c.m_position, 1.0f);
								lightProbe.m_a.m_normal = globalTransform.m_value * glm::vec4(lightProbe.m_a.m_normal, 0.0f);
								lightProbe.m_b.m_normal = globalTransform.m_value * glm::vec4(lightProbe.m_b.m_normal, 0.0f);
								lightProbe.m_c.m_normal = globalTransform.m_value * glm::vec4(lightProbe.m_c.m_normal, 0.0f);
								auto area = lightProbe.GetArea();
								lightProbe.m_direction = glm::vec3(0.0f);
								lightProbe.m_energy = glm::vec3(0.0f);
								if (!material->m_drawSettings.m_cullFace) {
										lightProbe.m_frontFace = lightProbe.m_backFace = true;
										m_totalArea += 2.0f * area;
								} else if (material->m_drawSettings.m_cullFaceMode == OpenGLCullFace::Front) {
										lightProbe.m_backFace = true;
										lightProbe.m_frontFace = false;
										m_totalArea += area;
								} else if (material->m_drawSettings.m_cullFaceMode == OpenGLCullFace::Back) {
										lightProbe.m_frontFace = true;
										lightProbe.m_backFace = false;
										m_totalArea += area;
								} else if (material->m_drawSettings.m_cullFaceMode == OpenGLCullFace::FrontAndBack) {
										lightProbe.m_frontFace = lightProbe.m_backFace = false;
								}
								m_lightProbeGroup.m_lightProbes.push_back(lightProbe);
						}
				}
		}
}

void TriangleIlluminationEstimator::Serialize(YAML::Emitter &out) {
		out << YAML::Key << "m_totalArea" << YAML::Value << m_totalArea;
		out << YAML::Key << "m_totalFlux" << YAML::Value << m_totalFlux;
		out << YAML::Key << "m_averageFlux" << YAML::Value << m_averageFlux;
}

void TriangleIlluminationEstimator::Deserialize(const YAML::Node &in) {
		m_totalArea = in["m_totalArea"].as<float>();
		m_totalFlux = in["m_totalFlux"].as<glm::vec3>();
		m_averageFlux = in["m_averageFlux"].as<glm::vec3>();
}
