#pragma once

#include <glm/glm.hpp>
#include <vector>
#include "ecosyslab_export.h"
#include "Noises.hpp"
using namespace UniEngine;

namespace EcoSysLab
{
	class HeightField : public IAsset
	{
	public:
		Noises2D m_noises2D;
		int m_precisionLevel = 2;
		[[nodiscard]] float GetValue(const glm::vec2& position);

		void OnInspect() override;
		void Serialize(YAML::Emitter& out) override;
		void Deserialize(const YAML::Node& in) override;
		void GenerateMesh(const glm::vec2& start, const glm::uvec2& resolution, float unitSize, std::vector<Vertex>& vertices, std::vector<glm::uvec3>& triangles, float xDepth = 1.0f, float zDepth = 1.0f);
	};
}