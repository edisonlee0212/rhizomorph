#pragma once

#include "ecosyslab_export.h"
#include "glm/gtx/quaternion.hpp"
#include "MarchingCubes.hpp"
using namespace UniEngine;
namespace EcoSysLab
{
	template <typename NodeData>
	class OctreeNode
	{
		float m_radius = 0.0f;
		unsigned m_level = 0;
		glm::vec3 m_center = glm::vec3(0.0f);
		int m_children[8] = { -1 , -1, -1, -1, -1, -1, -1, -1 };

		template<typename ND>
		friend
			class Octree;
	public:
		NodeData m_data = {};
		[[nodiscard]] float GetRadius() const { return m_radius; }
		[[nodiscard]] unsigned GetLevel() const { return m_level; }
		[[nodiscard]] glm::vec3 GetCenter() const { return m_center; }
		/*
		int m_leftUpBack = -1;
		int m_leftUpFront = -1;
		int m_leftDownBack = -1;
		int m_leftDownFront = -1;
		int m_rightUpBack = -1;
		int m_rightUpFront = -1;
		int m_rightDownBack = -1;
		int m_rightDownFront = -1;
		*/
	};

	template <typename NodeData>
	class Octree
	{
		std::vector<OctreeNode<NodeData>> m_octreeNodes;
		int NewNode(float radius, unsigned level, const glm::vec3 &center);
		float m_chunkRadius = 16;
		unsigned m_maxSubdivisionLevel = 10;
		float m_minimumNodeRadius = 0.015625f;
		glm::vec3 m_center;
	public:
		Octree();
		[[nodiscard]] float GetMinRadius() const;
		Octree(float radius, unsigned maxSubdivisionLevel, const glm::vec3& center);
		void IterateLeaves(const std::function<void(const OctreeNode<NodeData>& octreeNode)>& func) const;
		[[nodiscard]] bool Occupied(const glm::vec3& position) const;
		void Reset(float radius, unsigned maxSubdivisionLevel, const glm::vec3& center);
		[[nodiscard]] int GetIndex(const glm::vec3& position) const;
		[[nodiscard]] const OctreeNode<NodeData>& RefNode(int index) const;
		void Occupy(const glm::vec3& position, const std::function<void(OctreeNode<NodeData>&)>& occupiedNodes);
		void Occupy(const glm::vec3& position, const glm::quat& rotation, float length, float radius, const std::function<void(OctreeNode<NodeData>&)>& occupiedNodes);
		void Occupy(const glm::vec3& min, const glm::vec3 &max, const std::function<bool(const glm::vec3& boxCenter)>& collisionHandle, const std::function<void(OctreeNode<NodeData>&)>& occupiedNodes);

		void GetVoxels(std::vector<glm::mat4>& voxels) const;

		void TriangulateField(std::vector<Vertex>& vertices, std::vector<unsigned>& indices, bool removeDuplicate, int smoothMeshIteration) const;
	};

	template <typename NodeData>
	int Octree<NodeData>::NewNode(float radius, unsigned level, const glm::vec3& center)
	{
		m_octreeNodes.emplace_back();
		m_octreeNodes.back().m_radius = radius;
		m_octreeNodes.back().m_level = level;
		m_octreeNodes.back().m_center = center;
		return m_octreeNodes.size() - 1;
	}

	template <typename NodeData>
	Octree<NodeData>::Octree()
	{
		Reset(16, 10, glm::vec3(0.0f));
	}
	template <typename NodeData>
	float Octree<NodeData>::GetMinRadius() const
	{
		return m_minimumNodeRadius;
	}
	template <typename NodeData>
	Octree<NodeData>::Octree(float radius, unsigned maxSubdivisionLevel, const glm::vec3& center)
	{
		Reset(radius, maxSubdivisionLevel, center);
	}


	template <typename NodeData>
	bool Octree<NodeData>::Occupied(const glm::vec3& position) const
	{
		float currentRadius = m_chunkRadius;
		glm::vec3 center = m_center;
		int octreeNodeIndex = 0;
		for (int subdivision = 0; subdivision < m_maxSubdivisionLevel; subdivision++)
		{
			currentRadius /= 2.f;
			const auto& octreeNode = m_octreeNodes[octreeNodeIndex];
			const int index = 4 * (position.x > center.x ? 0 : 1) + 2 * (position.y > center.y ? 0 : 1) + (position.z > center.z ? 0 : 1);
			if (octreeNode.m_children[index] == -1)
			{
				return false;
			}
			octreeNodeIndex = octreeNode.m_children[index];
			center.x += position.x > center.x ? currentRadius : -currentRadius;
			center.y += position.y > center.y ? currentRadius : -currentRadius;
			center.z += position.z > center.z ? currentRadius : -currentRadius;
		}
		return true;
	}
	template <typename NodeData>
	void Octree<NodeData>::Reset(float radius, unsigned maxSubdivisionLevel, const glm::vec3& center)
	{
		m_chunkRadius = m_minimumNodeRadius = radius;
		m_maxSubdivisionLevel = maxSubdivisionLevel;
		m_center = center;
		m_octreeNodes.clear();
		for (int subdivision = 0; subdivision < m_maxSubdivisionLevel; subdivision++)
		{
			m_minimumNodeRadius /= 2.f;
		}
		NewNode(m_chunkRadius, -1, center);
	}
	template <typename NodeData>
	int Octree<NodeData>::GetIndex(const glm::vec3& position) const
	{
		float currentRadius = m_chunkRadius;
		glm::vec3 center = m_center;
		int octreeNodeIndex = 0;
		for (int subdivision = 0; subdivision < m_maxSubdivisionLevel; subdivision++)
		{
			currentRadius /= 2.f;
			const auto& octreeNode = m_octreeNodes[octreeNodeIndex];
			const int index = 4 * (position.x > center.x ? 0 : 1) + 2 * (position.y > center.y ? 0 : 1) + (position.z > center.z ? 0 : 1);
			if (octreeNode.m_children[index] == -1)
			{
				return -1;
			}
			octreeNodeIndex = octreeNode.m_children[index];
			center.x += position.x > center.x ? currentRadius : -currentRadius;
			center.y += position.y > center.y ? currentRadius : -currentRadius;
			center.z += position.z > center.z ? currentRadius : -currentRadius;
		}
		return octreeNodeIndex;
	}
	template <typename NodeData>
	const OctreeNode<NodeData>& Octree<NodeData>::RefNode(const int index) const
	{
		return m_octreeNodes[index];
	}

	template <typename NodeData>
	void Octree<NodeData>::Occupy(const glm::vec3& position, const std::function<void(OctreeNode<NodeData>&)>& occupiedNodes)
	{
		float currentRadius = m_chunkRadius;
		glm::vec3 center = m_center;
		int octreeNodeIndex = 0;
		for (int subdivision = 0; subdivision < m_maxSubdivisionLevel; subdivision++)
		{
			currentRadius /= 2.f;
			const auto& octreeNode = m_octreeNodes[octreeNodeIndex];
			const int index = 4 * (position.x > center.x ? 0 : 1) + 2 * (position.y > center.y ? 0 : 1) + (position.z > center.z ? 0 : 1);
			center.x += position.x > center.x ? currentRadius : -currentRadius;
			center.y += position.y > center.y ? currentRadius : -currentRadius;
			center.z += position.z > center.z ? currentRadius : -currentRadius;
			if (octreeNode.m_children[index] == -1)
			{
				const auto newIndex = NewNode(currentRadius, subdivision, center);
				m_octreeNodes[octreeNodeIndex].m_children[index] = newIndex;
				octreeNodeIndex = newIndex;
			}
			else octreeNodeIndex = octreeNode.m_children[index];
		}
		occupiedNodes(m_octreeNodes[octreeNodeIndex]);
	}

	template <typename NodeData>
	void Octree<NodeData>::Occupy(const glm::vec3& position, const glm::quat& rotation, float length, float radius, const std::function<void(OctreeNode<NodeData>&)>& occupiedNodes)
	{
		const float maxRadius = glm::max(length, radius);
		Occupy(glm::vec3(position - glm::vec3(maxRadius)), glm::vec3(position + glm::vec3(maxRadius)), [&](const glm::vec3& boxCenter)
			{
				const auto relativePos = glm::rotate(glm::inverse(rotation), boxCenter - position);
				return glm::abs(relativePos.z) <= length && glm::length(glm::vec2(relativePos.x, relativePos.y)) <= radius;
			}, occupiedNodes);
	}

	template <typename NodeData>
	void Octree<NodeData>::Occupy(const glm::vec3& min, const glm::vec3& max,
		const std::function<bool(const glm::vec3& boxCenter)>& collisionHandle,
		const std::function<void(OctreeNode<NodeData>&)>& occupiedNodes)
	{
		for (float x = min.x - m_minimumNodeRadius; x < max.x + m_minimumNodeRadius; x += m_minimumNodeRadius)
		{
			for (float y = min.y - m_minimumNodeRadius; y < max.y + m_minimumNodeRadius; y += m_minimumNodeRadius)
			{
				for (float z = min.z - m_minimumNodeRadius; z < max.z + m_minimumNodeRadius; z += m_minimumNodeRadius)
				{
					if (collisionHandle(glm::vec3(x, y, z)))
					{
						Occupy(glm::vec3(x, y, z), occupiedNodes);
					}
				}
			}
		}
	}

	template <typename NodeData>
	void Octree<NodeData>::IterateLeaves(const std::function<void(const OctreeNode<NodeData>& octreeNode)>& func) const
	{
		for (const auto& node : m_octreeNodes)
		{
			if (node.m_level == m_maxSubdivisionLevel - 1)
			{
				func(node);
			}
		}
	}
	template <typename NodeData>
	void Octree<NodeData>::GetVoxels(std::vector<glm::mat4>& voxels) const
	{
		voxels.clear();
		IterateLeaves([&](const OctreeNode<NodeData>& octreeNode)
			{
				voxels.push_back(glm::translate(octreeNode.m_center) * glm::scale(glm::vec3(m_minimumNodeRadius)));
			});
	}
	template <typename NodeData>
	void Octree<NodeData>::TriangulateField(std::vector<Vertex>& vertices, std::vector<unsigned>& indices, const bool removeDuplicate, const int smoothMeshIteration) const
	{
		std::vector<glm::vec3> testingCells;
		std::vector<glm::vec3> validateCells;
		IterateLeaves([&](const OctreeNode<NodeData>& octreeNode)
			{
				testingCells.push_back(octreeNode.m_center);
			});

		MarchingCubes::TriangulateField(m_center, [&](const glm::vec3& samplePoint)
			{
				return Occupied(samplePoint) ? 1.0f : 0.0f;
			}, 0.5f, m_minimumNodeRadius, testingCells, vertices, indices, removeDuplicate, smoothMeshIteration);
	}


}