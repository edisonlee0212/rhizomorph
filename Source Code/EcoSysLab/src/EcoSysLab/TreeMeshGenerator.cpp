//
// Created by lllll on 10/27/2022.
//

#include "TreeMeshGenerator.hpp"


using namespace EcoSysLab;

RingSegment::RingSegment(glm::vec3 startPosition, glm::vec3 endPosition, glm::vec3 startAxis,
	glm::vec3 endAxis, float startRadius, float endRadius)
	: m_startPosition(startPosition),
	m_endPosition(endPosition),
	m_startAxis(startAxis),
	m_endAxis(endAxis),
	m_startRadius(startRadius),
	m_endRadius(endRadius) {
}

void RingSegment::AppendPoints(std::vector<Vertex>& vertices, glm::vec3& normalDir, int step) {
	std::vector<Vertex> startRing;
	std::vector<Vertex> endRing;

	float angleStep = 360.0f / (float)(step);
	Vertex archetype;
	for (int i = 0; i < step; i++) {
		archetype.m_position = GetPoint(normalDir, angleStep * i, true);
		startRing.push_back(archetype);
	}
	for (int i = 0; i < step; i++) {
		archetype.m_position = GetPoint(normalDir, angleStep * i, false);
		endRing.push_back(archetype);
	}
	float textureXstep = 1.0f / step * 4;
	for (int i = 0; i < step - 1; i++) {
		float x = (i % step) * textureXstep;
		startRing[i].m_texCoord = glm::vec2(x, 0.0f);
		startRing[i + 1].m_texCoord = glm::vec2(x + textureXstep, 0.0f);
		endRing[i].m_texCoord = glm::vec2(x, 1.0f);
		endRing[i + 1].m_texCoord = glm::vec2(x + textureXstep, 1.0f);
		vertices.push_back(startRing[i]);
		vertices.push_back(startRing[i + 1]);
		vertices.push_back(endRing[i]);
		vertices.push_back(endRing[i + 1]);
		vertices.push_back(endRing[i]);
		vertices.push_back(startRing[i + 1]);
	}
	startRing[step - 1].m_texCoord = glm::vec2(1.0f - textureXstep, 0.0f);
	startRing[0].m_texCoord = glm::vec2(1.0f, 0.0f);
	endRing[step - 1].m_texCoord = glm::vec2(1.0f - textureXstep, 1.0f);
	endRing[0].m_texCoord = glm::vec2(1.0f, 1.0f);
	vertices.push_back(startRing[step - 1]);
	vertices.push_back(startRing[0]);
	vertices.push_back(endRing[step - 1]);
	vertices.push_back(endRing[0]);
	vertices.push_back(endRing[step - 1]);
	vertices.push_back(startRing[0]);
}

glm::vec3 RingSegment::GetPoint(const glm::vec3& normalDir, const float angle, const bool isStart) const
{
	glm::vec3 direction = glm::cross(normalDir, isStart ? this->m_startAxis : this->m_endAxis);
	direction = glm::rotate(direction, glm::radians(angle), isStart ? this->m_startAxis : this->m_endAxis);
	direction = glm::normalize(direction);
	const glm::vec3 position = (isStart ? m_startPosition : m_endPosition) + direction * (
		isStart ? m_startRadius : m_endRadius);
	return position;
}

void TreeMeshGeneratorSettings::Save(const std::string& name, YAML::Emitter& out) {
	out << YAML::Key << name << YAML::Value << YAML::BeginMap;
	out << YAML::Key << "m_resolution" << YAML::Value << m_resolution;
	out << YAML::Key << "m_subdivision" << YAML::Value << m_subdivision;
	out << YAML::Key << "m_vertexColorOnly" << YAML::Value << m_vertexColorOnly;
	out << YAML::Key << "m_enableFoliage" << YAML::Value << m_enableFoliage;
	out << YAML::Key << "m_enableBranch" << YAML::Value << m_enableBranch;
	out << YAML::Key << "m_smoothness" << YAML::Value << m_smoothness;
	out << YAML::Key << "m_overrideRadius" << YAML::Value << m_overrideRadius;
	out << YAML::Key << "m_boundaryRadius" << YAML::Value << m_radius;
	out << YAML::Key << "m_baseControlPointRatio" << YAML::Value << m_baseControlPointRatio;
	out << YAML::Key << "m_branchControlPointRatio" << YAML::Value << m_branchControlPointRatio;
	out << YAML::Key << "m_lineLengthFactor" << YAML::Value << m_lineLengthFactor;
	out << YAML::Key << "m_overrideVertexColor" << YAML::Value << m_overrideVertexColor;
	out << YAML::Key << "m_markJunctions" << YAML::Value << m_markJunctions;
	out << YAML::Key << "m_junctionUpperRatio" << YAML::Value << m_junctionUpperRatio;
	out << YAML::Key << "m_junctionLowerRatio" << YAML::Value << m_junctionLowerRatio;
	out << YAML::Key << "m_branchVertexColor" << YAML::Value << m_branchVertexColor;
	out << YAML::Key << "m_foliageVertexColor" << YAML::Value << m_foliageVertexColor;

	out << YAML::Key << "m_autoLevel" << YAML::Value << m_autoLevel;
	out << YAML::Key << "m_voxelSubdivisionLevel" << YAML::Value << m_voxelSubdivisionLevel;
	out << YAML::Key << "m_voxelSmoothIteration" << YAML::Value << m_voxelSmoothIteration;
	out << YAML::Key << "m_removeDuplicate" << YAML::Value << m_removeDuplicate;
	out << YAML::Key << "m_rootVertexColor" << YAML::Value << m_rootVertexColor;

	out << YAML::Key << "m_branchMeshType" << YAML::Value << m_branchMeshType;
	out << YAML::Key << "m_rootMeshType" << YAML::Value << m_rootMeshType;

	out << YAML::Key << "m_detailedFoliage" << YAML::Value << m_detailedFoliage;
	out << YAML::EndMap;
}

void TreeMeshGeneratorSettings::Load(const std::string& name, const YAML::Node& in) {
	if (in[name]) {
		const auto& ms = in[name];
		if (ms["m_resolution"]) m_resolution = ms["m_resolution"].as<float>();
		if (ms["m_subdivision"]) m_subdivision = ms["m_subdivision"].as<float>();
		if (ms["m_vertexColorOnly"]) m_vertexColorOnly = ms["m_vertexColorOnly"].as<bool>();
		if (ms["m_enableFoliage"]) m_enableFoliage = ms["m_enableFoliage"].as<bool>();
		if (ms["m_enableBranch"]) m_enableBranch = ms["m_enableBranch"].as<bool>();
		if (ms["m_smoothness"]) m_smoothness = ms["m_smoothness"].as<bool>();
		if (ms["m_overrideRadius"]) m_overrideRadius = ms["m_overrideRadius"].as<bool>();
		if (ms["m_boundaryRadius"]) m_radius = ms["m_boundaryRadius"].as<float>();
		if (ms["m_baseControlPointRatio"]) m_baseControlPointRatio = ms["m_baseControlPointRatio"].as<float>();
		if (ms["m_branchControlPointRatio"]) m_branchControlPointRatio = ms["m_branchControlPointRatio"].as<float>();
		if (ms["m_lineLengthFactor"]) m_lineLengthFactor = ms["m_lineLengthFactor"].as<float>();
		if (ms["m_overrideVertexColor"]) m_overrideVertexColor = ms["m_overrideVertexColor"].as<bool>();
		if (ms["m_markJunctions"]) m_markJunctions = ms["m_markJunctions"].as<bool>();
		if (ms["m_junctionUpperRatio"]) m_junctionUpperRatio = ms["m_junctionUpperRatio"].as<float>();
		if (ms["m_junctionLowerRatio"]) m_junctionLowerRatio = ms["m_junctionLowerRatio"].as<float>();
		if (ms["m_branchVertexColor"]) m_branchVertexColor = ms["m_branchVertexColor"].as<glm::vec3>();
		if (ms["m_foliageVertexColor"]) m_foliageVertexColor = ms["m_foliageVertexColor"].as<glm::vec3>();


		if (ms["m_autoLevel"]) m_autoLevel = ms["m_autoLevel"].as<bool>();
		if (ms["m_voxelSubdivisionLevel"]) m_voxelSubdivisionLevel = ms["m_voxelSubdivisionLevel"].as<int>();
		if (ms["m_voxelSmoothIteration"]) m_voxelSmoothIteration = ms["m_voxelSmoothIteration"].as<int>();
		if (ms["m_removeDuplicate"]) m_removeDuplicate = ms["m_removeDuplicate"].as<bool>();
		if (ms["m_rootVertexColor"]) m_rootVertexColor = ms["m_rootVertexColor"].as<glm::vec3>();

		if (ms["m_branchMeshType"]) m_branchMeshType = ms["m_branchMeshType"].as<unsigned>();
		if (ms["m_rootMeshType"]) m_rootMeshType = ms["m_rootMeshType"].as<unsigned>();

		if (ms["m_detailedFoliage"]) m_detailedFoliage = ms["m_detailedFoliage"].as<bool>();
	}
}

void TreeMeshGeneratorSettings::OnInspect() {
	if (ImGui::TreeNodeEx("Mesh Generator settings")) {
		ImGui::Checkbox("Vertex color only", &m_vertexColorOnly);
		ImGui::Checkbox("Branch", &m_enableBranch);
		ImGui::Checkbox("Fruit", &m_enableFruit);
		ImGui::Checkbox("Foliage", &m_enableFoliage);
		ImGui::Checkbox("Root", &m_enableRoot);
		ImGui::Checkbox("Fine Root", &m_enableFineRoot);
		ImGui::Combo("Branch mesh mode", { "Cylindrical", "Marching cubes" }, m_branchMeshType);
		ImGui::Combo("Root mesh mode", { "Cylindrical", "Marching cubes" }, m_rootMeshType);
		if(ImGui::TreeNode("Cylindrical mesh settings"))
		{
			ImGui::DragFloat("Resolution", &m_resolution, 0.00001f, 0.00001f, 1.0f, "%.3f");
			ImGui::DragFloat("Subdivision", &m_subdivision, 1.0f, 1.0f, 16.0f);
			ImGui::Checkbox("Smoothness", &m_smoothness);
			if (m_smoothness) {
				ImGui::DragFloat("Base control point ratio", &m_baseControlPointRatio, 0.001f, 0.0f, 1.0f);
				ImGui::DragFloat("Branch control point ratio", &m_branchControlPointRatio, 0.001f, 0.0f, 1.0f);
			}
			else {
				ImGui::DragFloat("Line length factor", &m_lineLengthFactor, 0.001f, 0.0f, 1.0f);
			}
			ImGui::Checkbox("Override radius", &m_overrideRadius);
			if (m_overrideRadius) ImGui::DragFloat("Radius", &m_radius);
			ImGui::Checkbox("Override vertex color", &m_overrideVertexColor);
			ImGui::Checkbox("Mark Junctions", &m_markJunctions);
			if (m_markJunctions) {
				ImGui::DragFloat("Junction Lower Ratio", &m_junctionLowerRatio, 0.01f, 0.0f, 0.5f);
				ImGui::DragFloat("Junction Upper Ratio", &m_junctionUpperRatio, 0.01f, 0.0f, 0.5f);
			}
			ImGui::TreePop();
		}
		if(ImGui::TreeNode("Marching cubes settings"))
		{
			ImGui::Checkbox("Auto set level", &m_autoLevel);
			if (!m_autoLevel) ImGui::DragInt("Voxel subdivision level", &m_voxelSubdivisionLevel, 1, 5, 16);
			ImGui::DragInt("Smooth iteration", &m_voxelSmoothIteration, 0, 0, 10);
			if (m_voxelSmoothIteration == 0) ImGui::Checkbox("Remove duplicate", &m_removeDuplicate);
			ImGui::TreePop();
		}
		if (m_enableBranch && ImGui::TreeNode("Branch settings")) {
			
			if (m_overrideVertexColor) {
				ImGui::ColorEdit3("Branch vertex color", &m_branchVertexColor.x);
			}
			
			ImGui::TreePop();
		}
		if (m_enableFoliage && ImGui::TreeNode("Foliage settings"))
		{
			if (m_overrideVertexColor) {
				ImGui::ColorEdit3("Foliage vertex color", &m_foliageVertexColor.x);
			}
			ImGui::Checkbox("Detailed foliage", &m_detailedFoliage);
			ImGui::TreePop();
		}
		if (m_enableRoot && ImGui::TreeNode("Root settings"))
		{
			if (m_overrideVertexColor) {
				ImGui::ColorEdit3("Root vertex color", &m_rootVertexColor.x);
			}
			ImGui::TreePop();
		}
		ImGui::Checkbox("Override presentation", &m_overridePresentation);
		if (m_overridePresentation && ImGui::TreeNodeEx("Override settings"))
		{
			ImGui::DragFloat3("Leaf size", &m_presentationOverrideSettings.m_leafSize.x, 0.01f);
			ImGui::DragInt("Leaf per node", &m_presentationOverrideSettings.m_leafCountPerInternode);
			ImGui::DragFloat("Distance to end limit", &m_presentationOverrideSettings.m_distanceToEndLimit, 0.01f);
			ImGui::DragFloat("Position variance", &m_presentationOverrideSettings.m_positionVariance, 0.01f);
			ImGui::DragFloat("Phototropism", &m_presentationOverrideSettings.m_phototropism, 0.01f);
			ImGui::Checkbox("Limit max thickness", &m_presentationOverrideSettings.m_limitMaxThickness);
			ImGui::ColorEdit3("Root color", &m_presentationOverrideSettings.m_rootOverrideColor.x);
			ImGui::ColorEdit3("Branch color", &m_presentationOverrideSettings.m_branchOverrideColor.x);
			ImGui::ColorEdit3("Foliage color", &m_presentationOverrideSettings.m_foliageOverrideColor.x);
			Editor::DragAndDropButton<Texture2D>(m_foliageTexture, "Foliage tex");


			ImGui::TreePop();
		}
		ImGui::TreePop();
	}
}


