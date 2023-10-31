#include "Noises.hpp"
#include "glm/gtc/noise.hpp"
using namespace EcoSysLab;


Noises2D::Noises2D() {
	m_minMax = glm::vec2(-1000, 1000);
	m_noiseDescriptors.clear();
	m_noiseDescriptors.emplace_back();
}
Noises3D::Noises3D()
{
	m_minMax = glm::vec2(-1000, 1000);
	m_noiseDescriptors.clear();
	m_noiseDescriptors.emplace_back();
}

bool Noises2D::OnInspect() {
	bool changed = false;
	if (ImGui::DragFloat2("Global Min/max", &m_minMax.x, 0, -1000, 1000)) { changed = true; }
	if (ImGui::Button("New start descriptor")) {
		changed = true;
		m_noiseDescriptors.emplace_back();
	}
	for (int i = 0; i < m_noiseDescriptors.size(); i++)
	{
		if (ImGui::TreeNodeEx(("No." + std::to_string(i)).c_str(), ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (ImGui::Button("Remove"))
			{
				m_noiseDescriptors.erase(m_noiseDescriptors.begin() + i);
				changed = true;
				ImGui::TreePop();
				continue;
			}
			changed = ImGui::Combo("Type", { "Constant", "Linear", "Simplex", "Perlin" }, m_noiseDescriptors[i].m_type) || changed;
			switch (static_cast<NoiseType>(m_noiseDescriptors[i].m_type))
			{
			case NoiseType::Perlin:
				changed = ImGui::DragFloat("Frequency", &m_noiseDescriptors[i].m_frequency, 0.00001f, 0, 0, "%.5f") || changed;
				changed = ImGui::DragFloat("Intensity", &m_noiseDescriptors[i].m_intensity, 0.00001f, 0, 0, "%.5f") || changed;
				changed = ImGui::DragFloat("Multiplier", &m_noiseDescriptors[i].m_multiplier, 0.00001f, 0, 0, "%.5f") || changed;
				if (ImGui::DragFloat("Min", &m_noiseDescriptors[i].m_min, 0.01f, -99999, m_noiseDescriptors[i].m_max))
				{
					changed = true;
					m_noiseDescriptors[i].m_min = glm::min(m_noiseDescriptors[i].m_min, m_noiseDescriptors[i].m_max);
				}
				if (ImGui::DragFloat("Max", &m_noiseDescriptors[i].m_max, 0.01f, m_noiseDescriptors[i].m_min, 99999))
				{
					changed = true;
					m_noiseDescriptors[i].m_max = glm::max(m_noiseDescriptors[i].m_min, m_noiseDescriptors[i].m_max);
				}
				changed = ImGui::DragFloat("Offset", &m_noiseDescriptors[i].m_offset, 0.01f) || changed;
				break;
			case NoiseType::Simplex:
				changed = ImGui::DragFloat("Frequency", &m_noiseDescriptors[i].m_frequency, 0.00001f, 0, 0, "%.5f") || changed;
				changed = ImGui::DragFloat("Intensity", &m_noiseDescriptors[i].m_intensity, 0.00001f, 0, 0, "%.5f") || changed;
				changed = ImGui::DragFloat("Multiplier", &m_noiseDescriptors[i].m_multiplier, 0.00001f, 0, 0, "%.5f") || changed;
				if (ImGui::DragFloat("Min", &m_noiseDescriptors[i].m_min, 0.01f, -99999, m_noiseDescriptors[i].m_max))
				{
					changed = true;
					m_noiseDescriptors[i].m_min = glm::min(m_noiseDescriptors[i].m_min, m_noiseDescriptors[i].m_max);
				}
				if (ImGui::DragFloat("Max", &m_noiseDescriptors[i].m_max, 0.01f, m_noiseDescriptors[i].m_min, 99999))
				{
					changed = true;
					m_noiseDescriptors[i].m_max = glm::max(m_noiseDescriptors[i].m_min, m_noiseDescriptors[i].m_max);
				}
				changed = ImGui::DragFloat("Offset", &m_noiseDescriptors[i].m_offset, 0.01f) || changed;
				break;
			case NoiseType::Constant:
				changed = ImGui::DragFloat("Value", &m_noiseDescriptors[i].m_offset, 0.00001f, 0, 0, "%.5f") || changed;
				break;
			case NoiseType::Linear:
				changed = ImGui::DragFloat("X multiplier", &m_noiseDescriptors[i].m_frequency, 0.00001f, 0, 0, "%.5f") || changed;
				changed = ImGui::DragFloat("Y multiplier", &m_noiseDescriptors[i].m_intensity, 0.00001f, 0, 0, "%.5f") || changed;
				changed = ImGui::DragFloat("Base", &m_noiseDescriptors[i].m_offset, 0.00001f, 0, 0, "%.5f") || changed;
				if (ImGui::DragFloat("Min", &m_noiseDescriptors[i].m_min, 0.01f, -99999, m_noiseDescriptors[i].m_max))
				{
					changed = true;
					m_noiseDescriptors[i].m_min = glm::min(m_noiseDescriptors[i].m_min, m_noiseDescriptors[i].m_max);
				}
				if (ImGui::DragFloat("Max", &m_noiseDescriptors[i].m_max, 0.01f, m_noiseDescriptors[i].m_min, 99999))
				{
					changed = true;
					m_noiseDescriptors[i].m_max = glm::max(m_noiseDescriptors[i].m_min, m_noiseDescriptors[i].m_max);
				}
				break;
			}
			ImGui::TreePop();
		}
	}
	return changed;
}

void Noises2D::Save(const std::string& name, YAML::Emitter& out) const
{
	out << YAML::Key << name << YAML::Value << YAML::BeginMap;
	out << YAML::Key << "m_minMax" << YAML::Value << m_minMax;
	if (!m_noiseDescriptors.empty())
	{
		out << YAML::Key << "m_noiseDescriptors" << YAML::Value
			<< YAML::Binary((const unsigned char*)m_noiseDescriptors.data(), m_noiseDescriptors.size() * sizeof(NoiseDescriptor));
	}
	out << YAML::EndMap;
}

void Noises2D::Load(const std::string& name, const YAML::Node& in)
{
	if (in[name])
	{
		const auto& n = in[name];
		if (n["m_minMax"])
			m_minMax = n["m_minMax"].as<glm::vec2>();


		if (n["m_noiseDescriptors"])
		{
			const auto& ds = n["m_noiseDescriptors"].as<YAML::Binary>();
			m_noiseDescriptors.resize(ds.size() / sizeof(NoiseDescriptor));
			std::memcpy(m_noiseDescriptors.data(), ds.data(), ds.size());
		}
	}
}
void Noises3D::Save(const std::string& name, YAML::Emitter& out) const
{
	out << YAML::Key << name << YAML::Value << YAML::BeginMap;
	out << YAML::Key << "m_minMax" << YAML::Value << m_minMax;
	if (!m_noiseDescriptors.empty())
	{
		out << YAML::Key << "m_noiseDescriptors" << YAML::Value
			<< YAML::Binary((const unsigned char*)m_noiseDescriptors.data(), m_noiseDescriptors.size() * sizeof(NoiseDescriptor));
	}
	out << YAML::EndMap;
}

void Noises3D::Load(const std::string& name, const YAML::Node& in)
{
	if (in[name])
	{
		const auto& n = in[name];
		if (n["m_minMax"])
			m_minMax = n["m_minMax"].as<glm::vec2>();
		if (n["m_noiseDescriptors"])
		{
			const auto& ds = n["m_noiseDescriptors"].as<YAML::Binary>();
			m_noiseDescriptors.resize(ds.size() / sizeof(NoiseDescriptor));
			std::memcpy(m_noiseDescriptors.data(), ds.data(), ds.size());
		}
	}
}

float Noises2D::GetValue(const glm::vec2& position) const
{
	float retVal = 0;
	for (const auto& noiseDescriptor : m_noiseDescriptors)
	{
		float noise = 0;
		switch (static_cast<NoiseType>(noiseDescriptor.m_type))
		{
		case NoiseType::Perlin:
			noise = glm::pow(glm::perlin(noiseDescriptor.m_frequency * position +
				glm::vec2(noiseDescriptor.m_offset)), noiseDescriptor.m_intensity) * noiseDescriptor.m_multiplier;
			noise = glm::clamp(noise, noiseDescriptor.m_min, noiseDescriptor.m_max);
			break;
		case NoiseType::Simplex:
			noise = glm::pow(glm::simplex(noiseDescriptor.m_frequency * position +
				glm::vec2(noiseDescriptor.m_offset)), noiseDescriptor.m_intensity) * noiseDescriptor.m_multiplier;
			noise = glm::clamp(noise, noiseDescriptor.m_min, noiseDescriptor.m_max);
			break;
		case NoiseType::Constant:
			noise = noiseDescriptor.m_offset;
			break;
		case NoiseType::Linear:
			noise = noiseDescriptor.m_offset + noiseDescriptor.m_frequency * position.x + noiseDescriptor.m_intensity * position.y;
			noise = glm::clamp(noise, noiseDescriptor.m_min, noiseDescriptor.m_max);
			break;
		}
		retVal += noise;
	}
	return glm::clamp(retVal, m_minMax.x, m_minMax.y);
}


bool Noises3D::OnInspect() {
	bool changed = false;
	if (ImGui::DragFloat2("Global Min/max", &m_minMax.x, 0, -1000, 1000)) { changed = true; }
	if (ImGui::Button("New start descriptor")) {
		changed = true;
		m_noiseDescriptors.emplace_back();
	}
	for (int i = 0; i < m_noiseDescriptors.size(); i++)
	{
		if (ImGui::TreeNodeEx(("No." + std::to_string(i)).c_str(), ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (ImGui::Button("Remove"))
			{
				m_noiseDescriptors.erase(m_noiseDescriptors.begin() + i);
				changed = true;
				ImGui::TreePop();
				continue;
			}
			changed = ImGui::Combo("Type", { "Constant", "Linear", "Simplex", "Perlin" }, m_noiseDescriptors[i].m_type) || changed;
			switch (static_cast<NoiseType>(m_noiseDescriptors[i].m_type))
			{
			case NoiseType::Perlin:
				changed = ImGui::DragFloat("Frequency", &m_noiseDescriptors[i].m_frequency, 0.00001f, 0, 0, "%.5f") || changed;
				changed = ImGui::DragFloat("Intensity", &m_noiseDescriptors[i].m_intensity, 0.00001f, 0, 0, "%.5f") || changed;
				changed = ImGui::DragFloat("Multiplier", &m_noiseDescriptors[i].m_multiplier, 0.00001f, 0, 0, "%.5f") || changed;
				if (ImGui::DragFloat("Min", &m_noiseDescriptors[i].m_min, 0.01f, -99999, m_noiseDescriptors[i].m_max))
				{
					changed = true;
					m_noiseDescriptors[i].m_min = glm::min(m_noiseDescriptors[i].m_min, m_noiseDescriptors[i].m_max);
				}
				if (ImGui::DragFloat("Max", &m_noiseDescriptors[i].m_max, 0.01f, m_noiseDescriptors[i].m_min, 99999))
				{
					changed = true;
					m_noiseDescriptors[i].m_max = glm::max(m_noiseDescriptors[i].m_min, m_noiseDescriptors[i].m_max);
				}
				changed = ImGui::DragFloat("Offset", &m_noiseDescriptors[i].m_offset, 0.00001f, 0, 0, "%.5f") || changed;
				break;
			case NoiseType::Simplex:
				changed = ImGui::DragFloat("Frequency", &m_noiseDescriptors[i].m_frequency, 0.00001f, 0, 0, "%.5f") || changed;
				changed = ImGui::DragFloat("Intensity", &m_noiseDescriptors[i].m_intensity, 0.00001f, 0, 0, "%.5f") || changed;
				changed = ImGui::DragFloat("Multiplier", &m_noiseDescriptors[i].m_multiplier, 0.00001f, 0, 0, "%.5f") || changed;
				if (ImGui::DragFloat("Min", &m_noiseDescriptors[i].m_min, 0.01f, -99999, m_noiseDescriptors[i].m_max))
				{
					changed = true;
					m_noiseDescriptors[i].m_min = glm::min(m_noiseDescriptors[i].m_min, m_noiseDescriptors[i].m_max);
				}
				if (ImGui::DragFloat("Max", &m_noiseDescriptors[i].m_max, 0.01f, m_noiseDescriptors[i].m_min, 99999))
				{
					changed = true;
					m_noiseDescriptors[i].m_max = glm::max(m_noiseDescriptors[i].m_min, m_noiseDescriptors[i].m_max);
				}
				changed = ImGui::DragFloat("Offset", &m_noiseDescriptors[i].m_offset, 0.00001f, 0, 0, "%.5f") || changed;
				break;
			case NoiseType::Constant:
				changed = ImGui::DragFloat("Value", &m_noiseDescriptors[i].m_offset, 0.00001f, 0, 0, "%.5f") || changed;
				break;
			case NoiseType::Linear:
				changed = ImGui::DragFloat("X multiplier", &m_noiseDescriptors[i].m_frequency, 0.00001f, 0, 0, "%.5f") || changed;
				changed = ImGui::DragFloat("Y multiplier", &m_noiseDescriptors[i].m_intensity, 0.00001f, 0, 0, "%.5f") || changed;
				changed = ImGui::DragFloat("Z multiplier", &m_noiseDescriptors[i].m_multiplier, 0.00001f, 0, 0, "%.5f") || changed;
				changed = ImGui::DragFloat("Base", &m_noiseDescriptors[i].m_offset, 0.00001f, 0, 0, "%.5f") || changed;
				if (ImGui::DragFloat("Min", &m_noiseDescriptors[i].m_min, 0.01f, -99999, m_noiseDescriptors[i].m_max))
				{
					changed = true;
					m_noiseDescriptors[i].m_min = glm::min(m_noiseDescriptors[i].m_min, m_noiseDescriptors[i].m_max);
				}
				if (ImGui::DragFloat("Max", &m_noiseDescriptors[i].m_max, 0.01f, m_noiseDescriptors[i].m_min, 99999))
				{
					changed = true;
					m_noiseDescriptors[i].m_max = glm::max(m_noiseDescriptors[i].m_min, m_noiseDescriptors[i].m_max);
				}
				break;
			}
			ImGui::TreePop();
		}
	}
	return changed;
}


float Noises3D::GetValue(const glm::vec3& position) const
{
	float retVal = 0;
	for (const auto& noiseDescriptor : m_noiseDescriptors)
	{
		float noise = 0;
		switch (static_cast<NoiseType>(noiseDescriptor.m_type))
		{
		case NoiseType::Perlin:
			noise = glm::pow(glm::perlin(noiseDescriptor.m_frequency * position +
				glm::vec3(noiseDescriptor.m_offset)), noiseDescriptor.m_intensity) * noiseDescriptor.m_multiplier;
			noise = glm::clamp(noise, noiseDescriptor.m_min, noiseDescriptor.m_max);
			break;
		case NoiseType::Simplex:
			noise = glm::pow(glm::simplex(noiseDescriptor.m_frequency * position +
				glm::vec3(noiseDescriptor.m_offset)), noiseDescriptor.m_intensity) * noiseDescriptor.m_multiplier;
			noise = glm::clamp(noise, noiseDescriptor.m_min, noiseDescriptor.m_max);
			break;
		case NoiseType::Constant:
			noise = noiseDescriptor.m_offset;
			break;
		case NoiseType::Linear:
			noise = noiseDescriptor.m_offset + noiseDescriptor.m_frequency * position.x + noiseDescriptor.m_intensity * position.y + noiseDescriptor.m_multiplier * position.z;
			noise = glm::clamp(noise, noiseDescriptor.m_min, noiseDescriptor.m_max);
			break;
		}
		retVal += noise;
	}
	return glm::clamp(retVal, m_minMax.x, m_minMax.y);
}