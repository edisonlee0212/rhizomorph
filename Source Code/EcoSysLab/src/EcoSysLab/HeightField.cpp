#include "HeightField.hpp"

#include "glm/gtc/noise.hpp"

using namespace EcoSysLab;

float HeightField::GetValue(const glm::vec2& position)
{
	float retVal = 0.0f;
	if(position.x < 0)
	{
		//retVal += glm::max(position.x, -5.0f);
	}
	retVal += m_noises2D.GetValue(position);
	return retVal;
}

void HeightField::OnInspect()
{
	bool changed = false;
	changed = ImGui::DragInt("Precision level", &m_precisionLevel) || changed;
	changed = m_noises2D.OnInspect() | changed;
	if (changed) m_saved = false;
}

void HeightField::Serialize(YAML::Emitter& out)
{
	out << YAML::Key << "m_precisionLevel" << YAML::Value << m_precisionLevel;
	m_noises2D.Save("m_noises2D", out);
}

void HeightField::Deserialize(const YAML::Node& in)
{
	if (in["m_precisionLevel"])
		m_precisionLevel = in["m_precisionLevel"].as<int>();
	m_noises2D.Load("m_noises2D", in);
}

void HeightField::GenerateMesh(const glm::vec2& start, const glm::uvec2& resolution, float unitSize, std::vector<Vertex>& vertices, std::vector<glm::uvec3>& triangles, float xDepth, float zDepth)
{
	for (unsigned i = 0; i < resolution.x * m_precisionLevel; i++) {
		for (unsigned j = 0; j < resolution.y * m_precisionLevel; j++) {
			Vertex archetype;
			archetype.m_position.x = start.x + unitSize * i / m_precisionLevel;
			archetype.m_position.z = start.y + unitSize * j / m_precisionLevel;
			archetype.m_position.y = GetValue({ archetype.m_position.x , archetype.m_position.z });
			archetype.m_texCoord = glm::vec2(static_cast<float>(i) / (resolution.x * m_precisionLevel),
				static_cast<float>(j) / (resolution.y * m_precisionLevel));
			vertices.push_back(archetype);
		}
	}

	for (int i = 0; i < resolution.x * m_precisionLevel - 1; i++) {
		for (int j = 0; j < resolution.y * m_precisionLevel - 1; j++) {
			if (static_cast<float>(i) / (resolution.x * m_precisionLevel - 2) > (1.0 - zDepth) && static_cast<float>(j) / (resolution.y * m_precisionLevel - 2) < xDepth) continue;
			const int n = resolution.x * m_precisionLevel;
			triangles.emplace_back(i + j * n, i + 1 + j * n, i + (j + 1) * n);
			triangles.emplace_back(i + 1 + (j + 1) * n, i + (j + 1) * n,
				i + 1 + j * n);
		}
	}
}

