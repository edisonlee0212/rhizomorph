#pragma once

#include "TreeModel.hpp"
#include "Curve.hpp"
#include "Octree.hpp"
using namespace UniEngine;
namespace EcoSysLab {
	struct RingSegment {
		glm::vec3 m_startPosition, m_endPosition;
		glm::vec3 m_startAxis, m_endAxis;
		float m_startRadius, m_endRadius;

		RingSegment() {}

		RingSegment(glm::vec3 startPosition, glm::vec3 endPosition,
			glm::vec3 startAxis, glm::vec3 endAxis,
			float startRadius, float endRadius);

		void AppendPoints(std::vector<Vertex>& vertices, glm::vec3& normalDir,
			int step);

		[[nodiscard]] glm::vec3 GetPoint(const glm::vec3& normalDir, float angle, bool isStart) const;
	};

	struct PresentationOverrideSettings
	{
		glm::vec3 m_leafSize = glm::vec3(0.03f, 1.0f, 0.03f);
		int m_leafCountPerInternode = 8;
		float m_distanceToEndLimit = 2.f;
		float m_positionVariance = 0.2f;
		float m_phototropism = 0.9f;
		glm::vec3 m_rootOverrideColor = glm::vec3(80, 60, 50) / 255.0f;
		glm::vec3 m_branchOverrideColor = glm::vec3(109, 79, 75) / 255.0f;
		glm::vec3 m_foliageOverrideColor = glm::vec3(152 / 255.0f, 203 / 255.0f, 0 / 255.0f);

		bool m_limitMaxThickness = true;
	};

	struct TreeMeshGeneratorSettings {
		bool m_vertexColorOnly = false;
		bool m_enableFoliage = true;
		bool m_enableFruit = true;
		bool m_enableBranch = true;
		bool m_enableRoot = true;
		bool m_enableFineRoot = true;
		bool m_overridePresentation = false;
		PresentationOverrideSettings m_presentationOverrideSettings;
		AssetRef m_foliageTexture;

		float m_resolution = 0.0002f;
		float m_subdivision = 3.0f;
		bool m_overrideRadius = false;
		float m_radius = 0.01f;
		bool m_overrideVertexColor = false;
		bool m_markJunctions = true;
		float m_junctionLowerRatio = 0.4f;
		float m_junctionUpperRatio = 0.0f;

		float m_baseControlPointRatio = 0.5f;
		float m_branchControlPointRatio = 0.5f;
		float m_lineLengthFactor = 1.0f;
		bool m_smoothness = true;

		bool m_autoLevel = true;
		int m_voxelSubdivisionLevel = 10;
		int m_voxelSmoothIteration = 5;
		bool m_removeDuplicate = true;

		glm::vec3 m_branchVertexColor = glm::vec3(1.0f);
		glm::vec3 m_foliageVertexColor = glm::vec3(1.0f);
		glm::vec3 m_rootVertexColor = glm::vec3(1.0f);

		unsigned m_branchMeshType = 0;
		unsigned m_rootMeshType = 0;

		bool m_detailedFoliage = false;

		void OnInspect();

		void Save(const std::string& name, YAML::Emitter& out);

		void Load(const std::string& name, const YAML::Node& in);
	};


	template<typename SkeletonData, typename FlowData, typename NodeData>
	class CylindricalMeshGenerator {
	public:
		void Generate(const Skeleton<SkeletonData, FlowData, NodeData>& treeSkeleton, std::vector<Vertex>& vertices,
			std::vector<unsigned int>& indices, const TreeMeshGeneratorSettings& settings, float maxThickness) const;
	};
	template<typename SkeletonData, typename FlowData, typename NodeData>
	class VoxelMeshGenerator {
	public:
		void Generate(const Skeleton<SkeletonData, FlowData, NodeData>& treeSkeleton, std::vector<Vertex>& vertices,
			std::vector<unsigned int>& indices, const TreeMeshGeneratorSettings& settings, float minRadius) const;
	};
	template<typename SkeletonData, typename FlowData, typename NodeData>
	void CylindricalMeshGenerator<SkeletonData, FlowData, NodeData>::Generate(const 
		Skeleton<SkeletonData, FlowData, NodeData>& treeSkeleton, std::vector<Vertex>& vertices,
		std::vector<unsigned int>& indices, const TreeMeshGeneratorSettings& settings, float maxThickness) const {
		int parentStep = -1;
		const auto& sortedInternodeList = treeSkeleton.RefSortedNodeList();
		std::vector<std::vector<RingSegment>> ringsList;
		std::vector<int> steps;
		ringsList.resize(sortedInternodeList.size());
		steps.resize(sortedInternodeList.size());

		std::vector<std::shared_future<void>> results;
		Jobs::ParallelFor(sortedInternodeList.size(), [&](unsigned internodeIndex) {
			auto internodeHandle = sortedInternodeList[internodeIndex];
		const auto& internode = treeSkeleton.PeekNode(internodeHandle);
		const auto& internodeInfo = internode.m_info;
		auto& rings = ringsList[internodeIndex];
		rings.clear();

		glm::vec3 directionStart =
			internodeInfo.m_globalRotation * glm::vec3(0, 0, -1);
		glm::vec3 directionEnd = directionStart;
		glm::vec3 positionStart = internodeInfo.m_globalPosition;
		glm::vec3 positionEnd =
			positionStart + internodeInfo.m_length * settings.m_lineLengthFactor * directionStart;
		float thicknessStart = internodeInfo.m_thickness;
		float thicknessEnd = internodeInfo.m_thickness;
		
		if (internode.GetParentHandle() != -1) {
			const auto& parentInternode = treeSkeleton.PeekNode(internode.GetParentHandle());
			thicknessStart = parentInternode.m_info.m_thickness;
			directionStart =
				parentInternode.m_info.m_globalRotation *
				glm::vec3(0, 0, -1);
		}

		if (settings.m_overrideRadius) {
			thicknessStart = settings.m_radius;
			thicknessEnd = settings.m_radius;
		}

		if (settings.m_overridePresentation && settings.m_presentationOverrideSettings.m_limitMaxThickness)
		{
			thicknessStart = glm::min(thicknessStart, maxThickness);
			thicknessEnd = glm::min(thicknessEnd, maxThickness);
		}

#pragma region Subdivision internode here.
		int step = thicknessStart / settings.m_resolution;
		if (step < 4)
			step = 4;
		if (step % 2 != 0)
			step++;
		steps[internodeIndex] = step;
		int amount = static_cast<int>(0.5f +
			internodeInfo.m_length * settings.m_subdivision);
		if (amount % 2 != 0)
			amount++;
		BezierCurve curve = BezierCurve(
			positionStart,
			positionStart +
			(settings.m_smoothness ? internodeInfo.m_length * settings.m_baseControlPointRatio : 0.0f) * directionStart,
			positionEnd -
			(settings.m_smoothness ? internodeInfo.m_length * settings.m_branchControlPointRatio : 0.0f) * directionEnd,
			positionEnd);
		float posStep = 1.0f / static_cast<float>(amount);
		glm::vec3 dirStep = (directionEnd - directionStart) / static_cast<float>(amount);
		float radiusStep = (thicknessEnd - thicknessStart) /
			static_cast<float>(amount);

		for (int ringIndex = 1; ringIndex < amount; ringIndex++) {
			float startThickness = static_cast<float>(ringIndex - 1) * radiusStep;
			float endThickness = static_cast<float>(ringIndex) * radiusStep;
			if (settings.m_smoothness) {
				rings.emplace_back(
					curve.GetPoint(posStep * (ringIndex - 1)), curve.GetPoint(posStep * ringIndex),
					directionStart + static_cast<float>(ringIndex - 1) * dirStep,
					directionStart + static_cast<float>(ringIndex) * dirStep,
					thicknessStart + startThickness, thicknessStart + endThickness);
			}
			else {
				rings.emplace_back(
					curve.GetPoint(posStep * (ringIndex - 1)), curve.GetPoint(posStep * ringIndex),
					directionEnd,
					directionEnd,
					thicknessStart + startThickness, thicknessStart + endThickness);
			}
		}
		if (amount > 1)
			rings.emplace_back(
				curve.GetPoint(1.0f - posStep), positionEnd, directionEnd - dirStep,
				directionEnd,
				thicknessEnd - radiusStep,
				thicknessEnd);
		else
			rings.emplace_back(positionStart, positionEnd,
				directionStart, directionEnd, thicknessStart,
				thicknessEnd);
#pragma endregion
			}, results);
		for (auto& i : results) i.wait();

		for (int internodeIndex = 0; internodeIndex < sortedInternodeList.size(); internodeIndex++) {
			auto internodeHandle = sortedInternodeList[internodeIndex];
			const auto& internode = treeSkeleton.PeekNode(internodeHandle);
			const auto& internodeInfo = internode.m_info;
			auto parentInternodeHandle = internode.GetParentHandle();
			const glm::vec3 up = internodeInfo.m_regulatedGlobalRotation * glm::vec3(0, 1, 0);
			auto& rings = ringsList[internodeIndex];
			if (rings.empty()) {
				continue;
			}
			auto step = steps[internodeIndex];
			// For stitching
			const int pStep = parentStep > 0 ? parentStep : step;
			parentStep = step;

			float angleStep = 360.0f / static_cast<float>(pStep);
			int vertexIndex = vertices.size();
			Vertex archetype;
			if (settings.m_overrideVertexColor) archetype.m_color = glm::vec4(settings.m_branchVertexColor, 1.0f);
			//else archetype.m_color = branchColors.at(internodeHandle);

			float textureXStep = 1.0f / pStep * 4.0f;

			const auto startPosition = rings.at(0).m_startPosition;
			const auto endPosition = rings.back().m_endPosition;
			for (int p = 0; p < pStep; p++) {
				archetype.m_position =
					rings.at(0).GetPoint(up, angleStep * p, true);
				float distanceToStart = 0;
				float distanceToEnd = 1;
				const float x =
					p < pStep / 2 ? p * textureXStep : (pStep - p) * textureXStep;
				archetype.m_texCoord = glm::vec2(x, 0.0f);
				vertices.push_back(archetype);
			}
			std::vector<float> angles;
			angles.resize(step);
			std::vector<float> pAngles;
			pAngles.resize(pStep);

			for (auto p = 0; p < pStep; p++) {
				pAngles[p] = angleStep * p;
			}
			angleStep = 360.0f / static_cast<float>(step);
			for (auto s = 0; s < step; s++) {
				angles[s] = angleStep * s;
			}

			std::vector<unsigned> pTarget;
			std::vector<unsigned> target;
			pTarget.resize(pStep);
			target.resize(step);
			for (int p = 0; p < pStep; p++) {
				// First we allocate nearest vertices for parent.
				auto minAngleDiff = 360.0f;
				for (auto j = 0; j < step; j++) {
					const float diff = glm::abs(pAngles[p] - angles[j]);
					if (diff < minAngleDiff) {
						minAngleDiff = diff;
						pTarget[p] = j;
					}
				}
			}
			for (int s = 0; s < step; s++) {
				// Second we allocate nearest vertices for child
				float minAngleDiff = 360.0f;
				for (int j = 0; j < pStep; j++) {
					const float diff = glm::abs(angles[s] - pAngles[j]);
					if (diff < minAngleDiff) {
						minAngleDiff = diff;
						target[s] = j;
					}
				}
			}
			for (int p = 0; p < pStep; p++) {
				if (pTarget[p] == pTarget[p == pStep - 1 ? 0 : p + 1]) {
					indices.push_back(vertexIndex + p);
					indices.push_back(vertexIndex + (p == pStep - 1 ? 0 : p + 1));
					indices.push_back(vertexIndex + pStep + pTarget[p]);
				}
				else {
					indices.push_back(vertexIndex + p);
					indices.push_back(vertexIndex + (p == pStep - 1 ? 0 : p + 1));
					indices.push_back(vertexIndex + pStep + pTarget[p]);

					indices.push_back(vertexIndex + pStep +
						pTarget[p == pStep - 1 ? 0 : p + 1]);
					indices.push_back(vertexIndex + pStep + pTarget[p]);
					indices.push_back(vertexIndex + (p == pStep - 1 ? 0 : p + 1));
				}
			}

			vertexIndex += pStep;
			textureXStep = 1.0f / step * 4.0f;
			int ringSize = rings.size();
			for (auto ringIndex = 0; ringIndex < ringSize; ringIndex++) {
				for (auto s = 0; s < step; s++) {
					archetype.m_position = rings.at(ringIndex).GetPoint(
						up, angleStep * s, false);
					float distanceToStart = glm::distance(
						rings.at(ringIndex).m_endPosition, startPosition);
					float distanceToEnd = glm::distance(
						rings.at(ringIndex).m_endPosition, endPosition);
					const auto x =
						s < (step / 2) ? s * textureXStep : (step - s) * textureXStep;
					const auto y = ringIndex % 2 == 0 ? 1.0f : 0.0f;
					archetype.m_texCoord = glm::vec2(x, y);
					vertices.push_back(archetype);
				}
				if (ringIndex != 0) {
					for (int s = 0; s < step - 1; s++) {
						// Down triangle
						indices.push_back(vertexIndex + (ringIndex - 1) * step + s);
						indices.push_back(vertexIndex + (ringIndex - 1) * step + s + 1);
						indices.push_back(vertexIndex + (ringIndex)*step + s);
						// Up triangle
						indices.push_back(vertexIndex + (ringIndex)*step + s + 1);
						indices.push_back(vertexIndex + (ringIndex)*step + s);
						indices.push_back(vertexIndex + (ringIndex - 1) * step + s + 1);
					}
					// Down triangle
					indices.push_back(vertexIndex + (ringIndex - 1) * step + step - 1);
					indices.push_back(vertexIndex + (ringIndex - 1) * step);
					indices.push_back(vertexIndex + (ringIndex)*step + step - 1);
					// Up triangle
					indices.push_back(vertexIndex + (ringIndex)*step);
					indices.push_back(vertexIndex + (ringIndex)*step + step - 1);
					indices.push_back(vertexIndex + (ringIndex - 1) * step);
				}
			}
		}
	}

	template <typename SkeletonData, typename FlowData, typename NodeData>
	void VoxelMeshGenerator<SkeletonData, FlowData, NodeData>::Generate(const
		Skeleton<SkeletonData, FlowData, NodeData>& treeSkeleton, std::vector<Vertex>& vertices,
		std::vector<unsigned>& indices, const TreeMeshGeneratorSettings& settings, float minRadius) const
	{
		const auto boxSize = treeSkeleton.m_max - treeSkeleton.m_min;
		Octree<bool> octree;
		if (settings.m_autoLevel)
		{
			const float maxRadius = glm::max(glm::max(boxSize.x, boxSize.y), boxSize.z) * 0.5f + 2.0f * minRadius;
			int subdivisionLevel = -1;
			float testRadius = minRadius;
			while (testRadius <= maxRadius)
			{
				subdivisionLevel++;
				testRadius *= 2.f;
			}
			UNIENGINE_LOG("Root mesh formation: Auto set level to " + std::to_string(subdivisionLevel))

				octree.Reset(maxRadius, subdivisionLevel, (treeSkeleton.m_min + treeSkeleton.m_max) * 0.5f);
		}
		else {
			octree.Reset(glm::max((boxSize.x, boxSize.y), glm::max(boxSize.y, boxSize.z)) * 0.5f,
				glm::clamp(settings.m_voxelSubdivisionLevel, 4, 16), (treeSkeleton.m_min + treeSkeleton.m_max) / 2.0f);
		}
		auto& nodeList = treeSkeleton.RefSortedNodeList();
		for (const auto& nodeIndex : nodeList)
		{
			const auto& node = treeSkeleton.PeekNode(nodeIndex);
			const auto& info = node.m_info;
			auto thickness = info.m_thickness;
			if (node.GetParentHandle() > 0)
			{
				thickness = (thickness + treeSkeleton.PeekNode(node.GetParentHandle()).m_info.m_thickness) / 2.0f;
			}
			octree.Occupy(info.m_globalPosition, info.m_globalRotation, info.m_length, thickness, [](OctreeNode<bool>&) {});
		}
		octree.TriangulateField(vertices, indices, settings.m_removeDuplicate, settings.m_voxelSmoothIteration);
	}
}
