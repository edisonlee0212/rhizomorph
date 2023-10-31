#pragma once

#include "ecosyslab_export.h"
#include "VoxelGrid.hpp"
#include "PlantStructure.hpp"
#include "TreeMeshGenerator.hpp"
#include "Curve.hpp"

using namespace UniEngine;
#define TREEPOINTCLOUD_CLEAN
namespace EcoSysLab {
	typedef int PointHandle;
	typedef int BranchHandle;
	typedef int TreePartHandle;
	struct ScatteredPoint {
		PointHandle m_handle = -1;
		std::vector<PointHandle> m_neighbors;
#ifndef TREEPOINTCLOUD_CLEAN
		std::vector<BranchHandle> m_neighborBranchStarts;
#endif
		std::vector<BranchHandle> m_neighborBranchEnds;
		glm::vec3 m_position = glm::vec3(0.0f);
	};
	struct AllocatedPoint {
		glm::vec3 m_position;
		PointHandle m_handle = -1;
		TreePartHandle m_treePartHandle = -1;
		BranchHandle m_branchHandle = -1;
		NodeHandle m_nodeHandle = -1;
		int m_skeletonIndex = -1;
	};
	struct ScannedBranch {
		TreePartHandle m_treePartHandle = -1;
		BranchHandle m_handle = -1;
		BezierCurve m_bezierCurve;
		float m_startThickness = 0.0f;
		float m_endThickness = 0.0f;
		std::vector<PointHandle> m_startNeighbors;
#ifndef TREEPOINTCLOUD_CLEAN
		std::vector<PointHandle> m_endNeighbors;
		std::vector<BranchHandle> m_neighborBranchStarts;
#endif
		std::vector<BranchHandle> m_neighborBranchEnds;
		BranchHandle m_parentHandle = -1;
		std::vector<BranchHandle> m_childHandles;

	};

	struct OperatingBranch{
		TreePartHandle m_treePartHandle = -1;
		BranchHandle m_handle = -1;
		BezierCurve m_bezierCurve;
		float m_startThickness = 0.0f;
		float m_endThickness = 0.0f;

		BranchHandle m_parentHandle = -1;
		std::vector<BranchHandle> m_childHandles;

		int m_skeletonIndex = -1;
		std::vector<NodeHandle> m_chainNodeHandles;
		void Apply(const ScannedBranch& target);
	};

	struct TreePart {
		TreePartHandle m_handle = -1;
		std::vector<PointHandle> m_allocatedPoints;
		std::vector<BranchHandle> m_branchHandles;
	};

	struct ConnectivityGraphSettings {
		float m_scatterPointConnectionMaxLength = 0.05f;

		float m_edgeExtendStep = 0.05f;
		float m_edgeLength = 0.1f;
		int m_maxTimeout = 60;
		float m_forceConnectionAngleLimit = 135.0f;
		float m_forceConnectionRatio = 0.0f;
		float m_absoluteAngleLimit = 60.0f;
		float m_branchShortening = 0.15f;

		void OnInspect();
	};

	enum class PointCloudVoxelType {
		ScatteredPoint,
		BranchStart,
		BranchEnd
	};

	struct PointCloudVoxel {
		PointCloudVoxelType m_type;
		glm::vec3 m_position = glm::vec3(0.0f);
		int m_handle = -1;
	};

	struct ReconstructionSettings {
		float m_internodeLength = 0.03f;
		float m_minHeight = 0.15f;
		float m_maxTreeDistance = 0.05f;
		float m_branchShortening = 0.3f;

		void OnInspect();
	};

	struct ReconstructionSkeletonData {

	};
	struct ReconstructionFlowData {

	};
	struct ReconstructionNodeData {
		std::vector<PointHandle> m_allocatedPoints;
		std::vector<PointHandle> m_filteredPoints;
		BranchHandle m_branchHandle;
	};
	typedef Skeleton<ReconstructionSkeletonData, ReconstructionFlowData, ReconstructionNodeData> ReconstructionSkeleton;

	class TreePointCloud : public IPrivateComponent {
		void FindPoints(const glm::vec3 &position, VoxelGrid<std::vector<PointCloudVoxel>> &pointVoxelGrid, float radius,
										const std::function<void(const PointCloudVoxel &voxel)> &func) const;

		public:
		void ImportGraph(const std::filesystem::path &path, float scaleFactor = 0.1f);

		glm::vec3 m_min;
		glm::vec3 m_max;
		std::vector<ScatteredPoint> m_scatteredPoints;
		std::vector<AllocatedPoint> m_allocatedPoints;
		std::vector<ScannedBranch> m_scannedBranches;

		std::vector<OperatingBranch> m_operatingBranches;
		std::vector<TreePart> m_treeParts;

		void OnInspect() override;

		std::vector<ReconstructionSkeleton> m_skeletons;

		std::vector<std::pair<glm::vec3, glm::vec3>> m_scatterPointToBranchEndConnections;
		std::vector<std::pair<glm::vec3, glm::vec3>> m_scatterPointToBranchStartConnections;
		std::vector<std::pair<glm::vec3, glm::vec3>> m_scatterPointsConnections;
		std::vector<std::pair<glm::vec3, glm::vec3>> m_branchConnections;
		std::vector<std::pair<glm::vec3, glm::vec3>> m_filteredBranchConnections;

		void EstablishConnectivityGraph(const ConnectivityGraphSettings &otherPointHandle);

		void BuildTreeStructure(const ReconstructionSettings &reconstructionSettings);

		void ClearMeshes() const;

		void GenerateMeshes(const TreeMeshGeneratorSettings &meshGeneratorSettings);
	};
}