#pragma once
namespace EcoSysLab {
	/* Coordinate system

	The voxel position is its center.
	Each voxel is dx wide.

					<-dx ->
					-------------------------
					|     |     |     |     |
					|  x  |  x  |  x  |  x  |
					|     |     |     |     |
					-------------------------
					   |     |     |     |
					   |     |     |     |
	X-Coordinate:   -- 0 --- 1 --- 2 --- 3 -----

	The "m_minBound" stores the lower left corner of the lower left voxel.
	I.e. for m_volumePositionMin=(0, 0) and m_resolution=(2, 2), and m_dx=1,
	the voxel centers are at 0.5 and 1.5.

	*/
	template <typename VoxelData>
	class VoxelGrid
	{
		std::vector<VoxelData> m_data;
		glm::vec3 m_minBound = glm::vec3(0.0f);
		float m_voxelSize = 1.0f;
		glm::ivec3 m_resolution = { 0, 0, 0 };
	public:
		void Initialize(float voxelSize, const glm::ivec3& resolution, const glm::vec3& minBound, const VoxelData& defaultData = {});
		void Initialize(float voxelSize, const glm::vec3& minBound, const glm::vec3& maxBound, const VoxelData& defaultData = {});
		[[nodiscard]] size_t GetVoxelSize() const;
		[[nodiscard]] glm::ivec3 GetResolution() const;
		[[nodiscard]] glm::vec3 GetMinBound() const;
		[[nodiscard]] float GetVoxelDiameter() const;

		[[nodiscard]] VoxelData& Ref(int index);
		[[nodiscard]] const VoxelData& Peek(int index) const;
		[[nodiscard]] VoxelData& Ref(const glm::ivec3& coordinate);
		[[nodiscard]] const VoxelData& Peek(const glm::ivec3& coordinate) const;
		[[nodiscard]] VoxelData& Ref(const glm::vec3& position);
		[[nodiscard]] const VoxelData& Peek(const glm::vec3& position) const;

		[[nodiscard]] int GetIndex(const glm::ivec3& coordinate) const;
		[[nodiscard]] int GetIndex(const glm::vec3& position) const;
		[[nodiscard]] glm::ivec3 GetCoordinate(int index) const;
		[[nodiscard]] glm::ivec3 GetCoordinate(const glm::vec3& position) const;
		[[nodiscard]] glm::vec3	GetPosition(int index) const;
		[[nodiscard]] glm::vec3	GetPosition(const glm::ivec3& coordinate) const;

		[[nodiscard]] void ForEach(const glm::vec3& minBound, const glm::vec3& maxBound, const std::function<void(VoxelData& data)>& func);
		[[nodiscard]] void ForEach(const glm::vec3& center, float radius, const std::function<void(VoxelData& data)>& func);
		[[nodiscard]] bool IsValid(const glm::vec3& position) const;
	};

	template <typename VoxelData>
	void VoxelGrid<VoxelData>::Initialize(const float voxelSize, const glm::ivec3& resolution, const glm::vec3& minBound, const VoxelData& defaultData)
	{
		m_resolution = resolution;
		m_voxelSize = voxelSize;
		m_minBound = minBound;
		auto numVoxels = m_resolution.x * m_resolution.y * m_resolution.z;
		m_data.resize(numVoxels);
		std::fill(m_data.begin(), m_data.end(), defaultData);
	}

	template <typename VoxelData>
	void VoxelGrid<VoxelData>::Initialize(const float voxelSize, const glm::vec3& minBound, const glm::vec3& maxBound, const VoxelData& defaultData)
	{
		Initialize(voxelSize,
			glm::ivec3(
				glm::ceil((maxBound.x - minBound.x) / voxelSize),
				glm::ceil((maxBound.y - minBound.y) / voxelSize),
				glm::ceil((maxBound.z - minBound.z) / voxelSize)), minBound, defaultData);
	}

	template <typename VoxelData>
	size_t VoxelGrid<VoxelData>::GetVoxelSize() const
	{
		return m_data.size();
	}

	template <typename VoxelData>
	glm::ivec3 VoxelGrid<VoxelData>::GetResolution() const
	{
		return m_resolution;
	}

	template <typename VoxelData>
	glm::vec3 VoxelGrid<VoxelData>::GetMinBound() const
	{
		return m_minBound;
	}

	template <typename VoxelData>
	float VoxelGrid<VoxelData>::GetVoxelDiameter() const
	{
		return m_voxelSize;
	}

	template <typename VoxelData>
	VoxelData& VoxelGrid<VoxelData>::Ref(const int index)
	{
		return m_data[index];
	}

	template <typename VoxelData>
	const VoxelData& VoxelGrid<VoxelData>::Peek(const int index) const
	{
		return m_data[index];
	}

	template <typename VoxelData>
	VoxelData& VoxelGrid<VoxelData>::Ref(const glm::ivec3& coordinate)
	{
		return Ref(GetIndex(coordinate));
	}

	template <typename VoxelData>
	const VoxelData& VoxelGrid<VoxelData>::Peek(const glm::ivec3& coordinate) const
	{
		return Peek(GetIndex(coordinate));
	}

	template <typename VoxelData>
	VoxelData& VoxelGrid<VoxelData>::Ref(const glm::vec3& position)
	{
		return Ref(GetIndex(position));
	}

	template <typename VoxelData>
	const VoxelData& VoxelGrid<VoxelData>::Peek(const glm::vec3& position) const
	{
		return Peek(GetIndex(position));
	}

	template <typename VoxelData>
	int VoxelGrid<VoxelData>::GetIndex(const glm::ivec3& coordinate) const
	{
		return coordinate.x + coordinate.y * m_resolution.x + coordinate.z * m_resolution.x * m_resolution.y;
	}

	template <typename VoxelData>
	int VoxelGrid<VoxelData>::GetIndex(const glm::vec3& position) const
	{
		return GetIndex(GetCoordinate(position));
	}

	template <typename VoxelData>
	glm::ivec3 VoxelGrid<VoxelData>::GetCoordinate(int index) const
	{
		return {
			index % m_resolution.x,
			index % (m_resolution.x * m_resolution.y) / m_resolution.x,
			index / (m_resolution.x * m_resolution.y) };
	}

	template <typename VoxelData>
	glm::ivec3 VoxelGrid<VoxelData>::GetCoordinate(const glm::vec3& position) const
	{
		return {
			floor((position.x - m_minBound.x) / m_voxelSize),
			floor((position.y - m_minBound.y) / m_voxelSize),
			floor((position.z - m_minBound.z) / m_voxelSize)
		};
	}

	template <typename VoxelData>
	glm::vec3 VoxelGrid<VoxelData>::GetPosition(const glm::ivec3& coordinate) const
	{
		return {
			m_minBound.x + m_voxelSize / 2.0 + coordinate.x * m_voxelSize,
			m_minBound.y + m_voxelSize / 2.0 + coordinate.y * m_voxelSize,
			m_minBound.z + m_voxelSize / 2.0 + coordinate.z * m_voxelSize
		};
	}

	template <typename VoxelData>
	void VoxelGrid<VoxelData>::ForEach(const glm::vec3& minBound, const glm::vec3& maxBound,
		const std::function<void(VoxelData& data)>& func)
	{
		const glm::vec3 boundSize = maxBound - minBound;
		const glm::ivec3 start = glm::max(glm::ivec3(glm::floor(minBound / glm::vec3(m_voxelSize))), glm::ivec3(0));
		const glm::ivec3 span = glm::ceil(boundSize / glm::vec3(m_voxelSize));
		for (int i = start.x; i <= start.x + span.x; i++) {
			for (int j = start.y; j <= start.y + span.y; j++) {
				for (int k = start.z; k <= start.z + span.z; k++) {
					auto index = GetIndex(glm::ivec3(i, j, k));
					if (index >= m_data.size()) continue;
					func(Ref(index));
				}
			}
		}
	}

	template <typename VoxelData>
	void VoxelGrid<VoxelData>::ForEach(const glm::vec3& center, float radius,
		const std::function<void(VoxelData& data)>& func)
	{
		const glm::vec3 minBound = center - glm::vec3(radius);
		const glm::vec3 maxBound = center + glm::vec3(radius);
		const glm::vec3 boundSize = maxBound - minBound;
		const glm::ivec3 start = glm::max(glm::ivec3(glm::floor(minBound / glm::vec3(m_voxelSize))), glm::ivec3(0));
		const glm::ivec3 span = glm::ceil(boundSize / glm::vec3(m_voxelSize));
		for (int i = start.x; i <= start.x + span.x; i++) {
			for (int j = start.y; j <= start.y + span.y; j++) {
				for (int k = start.z; k <= start.z + span.z; k++) {
					auto index = GetIndex(glm::ivec3(i, j, k));
					if (index >= m_data.size()) continue;
					if (glm::distance(GetPosition(index), center) > radius + m_voxelSize / 2.0f) continue;
					func(Ref(index));
				}
			}
		}
	}

	template <typename VoxelData>
	bool VoxelGrid<VoxelData>::IsValid(const glm::vec3& position) const
	{
		const auto maxBound = m_minBound + m_voxelSize * glm::vec3(m_resolution);
		if (position.x < m_minBound.x || position.y < m_minBound.y || position.z < m_minBound.z
			|| position.x >= maxBound.x || position.y >= maxBound.y || position.z >= maxBound.z) return false;
		return true;
	}

	template <typename VoxelData>
	glm::vec3 VoxelGrid<VoxelData>::GetPosition(const int index) const
	{
		return GetPosition(GetCoordinate(index));
	}
}
