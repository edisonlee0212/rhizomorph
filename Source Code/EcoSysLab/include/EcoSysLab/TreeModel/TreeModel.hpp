#pragma once
#include "SoilModel.hpp"
#include "ClimateModel.hpp"
#include "Octree.hpp"
#include "TreeGrowthController.hpp"
using namespace UniEngine;
namespace EcoSysLab {
	struct TreeGrowthSettings
	{
		int m_flowNodeLimit = 10;

		bool m_enableRoot = true;
		bool m_enableShoot = true;

		bool m_autoBalance = true;
		bool m_collectLight = true;
		bool m_collectWater = true;
		bool m_collectNitrite = true;

		float m_leafMaintenanceVigorFillingRate = 1.0f;
		float m_leafDevelopmentalVigorFillingRate = 1.0f;
		float m_fruitMaintenanceVigorFillingRate = 1.0f;
		float m_fruitDevelopmentalVigorFillingRate = 1.0f;
		float m_nodeDevelopmentalVigorFillingRate = 1.0f;


		bool m_enableRootCollisionDetection = false;
		bool m_enableBranchCollisionDetection = false;
	};

	

	class TreeModel {
#pragma region Root Growth

		bool ElongateRoot(SoilModel& soilModel, float extendLength, NodeHandle rootNodeHandle,
			const RootGrowthController& rootGrowthParameters, float& collectedAuxin);

		inline bool GrowRootNode(SoilModel& soilModel, NodeHandle rootNodeHandle, const RootGrowthController& rootGrowthParameters);

		inline void CalculateThickness(NodeHandle rootNodeHandle,
			const RootGrowthController& rootGrowthParameters);

		

		inline void AggregateRootVigorRequirement(const RootGrowthController& rootGrowthParameters);

		inline void AllocateRootVigor(const RootGrowthController& rootGrowthParameters);

		inline void CalculateVigorRequirement(const RootGrowthController& rootGrowthParameters, PlantGrowthRequirement& newRootGrowthNutrientsRequirement);
		inline void SampleNitrite(const glm::mat4& globalTransform, SoilModel& soilModel);
#pragma endregion
#pragma region Tree Growth
		inline void AggregateInternodeVigorRequirement(const ShootGrowthController& shootGrowthParameters);

		inline void CalculateVigorRequirement(const ShootGrowthController& shootGrowthParameters, PlantGrowthRequirement& newTreeGrowthNutrientsRequirement);

		inline void AllocateShootVigor(const ShootGrowthController& shootGrowthParameters);

		inline bool PruneInternodes(float maxDistance, NodeHandle internodeHandle,
			const ShootGrowthController& shootGrowthParameters);

		inline void CalculateThicknessAndSagging(NodeHandle internodeHandle,
			const ShootGrowthController& shootGrowthParameters);

		inline bool GrowInternode(ClimateModel& climateModel, NodeHandle internodeHandle, const ShootGrowthController& shootGrowthParameters);

		bool ElongateInternode(float extendLength, NodeHandle internodeHandle,
			const ShootGrowthController& shootGrowthParameters, float& collectedInhibitor);

		friend class Tree;

		
#pragma endregion

		void Initialize(const ShootGrowthController& shootGrowthParameters, const RootGrowthController& rootGrowthParameters);

		bool m_initialized = false;

		ShootSkeleton m_shootSkeleton;
		RootSkeleton m_rootSkeleton;

		
		std::deque<std::pair<ShootSkeleton, RootSkeleton>> m_history;

		/**
		 * Grow one iteration of the branches, given the climate model and the procedural parameters.
		 * @param globalTransform The plant's world transform.
		 * @param climateModel The climate model.
		 * @param shootGrowthParameters The procedural parameters that guides the growth.
		 * @param newShootGrowthRequirement Growth requirements from shoots.
		 * @return Whether the growth caused a structural change during the growth.
		 */
		bool GrowShoots(const glm::mat4& globalTransform, ClimateModel& climateModel,
			const ShootGrowthController& shootGrowthParameters, PlantGrowthRequirement& newShootGrowthRequirement);

		/**
		 * Grow one iteration of the roots, given the soil model and the procedural parameters.
		 * @param globalTransform The plant's world transform.
		 * @param soilModel The soil model
		 * @param rootGrowthParameters The procedural parameters that guides the growth.
		 * @param newRootGrowthRequirement Growth requirements from roots.
		 * @return Whether the growth caused a structural change during the growth.
		 */
		bool GrowRoots(const glm::mat4& globalTransform, SoilModel& soilModel,
			const RootGrowthController& rootGrowthParameters, PlantGrowthRequirement& newRootGrowthRequirement);

		inline void PlantVigorAllocation();

		int m_leafCount = 0;
		int m_fruitCount = 0;
		int m_fineRootCount = 0;

		float m_age = 0;
		int m_ageInYear = 0;
		float m_internodeDevelopmentRate = 1.0f;
		float m_rootNodeDevelopmentRate = 1.0f;
		float m_currentDeltaTime = 1.0f;

		bool m_enableRoot = true;
		bool m_enableShoot = true;

		void ResetReproductiveModule();

		
	public:

		void PruneInternode(NodeHandle internodeHandle);
		void PruneRootNode(NodeHandle rootNodeHandle);

		inline void CollectRootFlux(const glm::mat4& globalTransform, SoilModel& soilModel,
			const RootGrowthController& rootGrowthParameters);
		inline void CollectShootFlux(const glm::mat4& globalTransform, ClimateModel& climateModel,
			const ShootGrowthController& shootGrowthParameters);
		void HarvestFruits(const std::function<bool(const ReproductiveModule& fruit)>& harvestFunction);

		int m_iteration = 0;


		static void ApplyTropism(const glm::vec3& targetDir, float tropism, glm::vec3& front, glm::vec3& up);

		static void ApplyTropism(const glm::vec3& targetDir, float tropism, glm::quat& rotation);

		std::vector<int> m_internodeOrderCounts;
		std::vector<int> m_rootNodeOrderCounts;

		template <typename SkeletonData, typename FlowData, typename NodeData>
		void CollisionDetection(float minRadius, Octree<TreeVoxelData>& octree, Skeleton<SkeletonData, FlowData, NodeData>& skeleton);

		TreeGrowthSettings m_treeGrowthSettings;

		//TreeSphericalVolume m_shootVolume;
		//IlluminationEstimationSettings m_illuminationEstimationSettings;


		ShootRootVigorRatio m_vigorRatio;
		glm::vec3 m_currentGravityDirection = glm::vec3(0, -1, 0);

		/**
		 * Erase the entire tree.
		 */
		void Clear();

		[[nodiscard]] int GetLeafCount() const;
		[[nodiscard]] int GetFruitCount() const;
		[[nodiscard]] int GetFineRootCount() const;
		/**
		 * Grow one iteration of the tree, given the nutrients and the procedural parameters.
		 * @param deltaTime The real world time for this iteration
		 * @param globalTransform The global transform of tree in world space.
		 * @param soilModel The soil model
		 * @param climateModel The climate model
		 * @param rootGrowthParameters The procedural parameters that guides the growth of the roots.
		 * @param shootGrowthParameters The procedural parameters that guides the growth of the branches.
		 * @return Whether the growth caused a structural change during the growth.
		 */
		bool Grow(float deltaTime, const glm::mat4& globalTransform, SoilModel& soilModel, ClimateModel& climateModel,
			const RootGrowthController& rootGrowthParameters, const ShootGrowthController& shootGrowthParameters);

		int m_historyLimit = -1;

		void SampleTemperature(const glm::mat4& globalTransform, ClimateModel& climateModel);
		void SampleSoilDensity(const glm::mat4& globalTransform, SoilModel& soilModel);
		[[nodiscard]] ShootSkeleton& RefShootSkeleton();

		[[nodiscard]] const ShootSkeleton& PeekShootSkeleton(int iteration) const;

		[[nodiscard]] RootSkeleton&
			RefRootSkeleton();


		[[nodiscard]] const RootSkeleton& PeekRootSkeleton(int iteration) const;

		void ClearHistory();

		void Step();

		void Pop();

		[[nodiscard]] int CurrentIteration() const;

		void Reverse(int iteration);
	};

	template <typename SkeletonData, typename FlowData, typename NodeData>
	void TreeModel::CollisionDetection(float minRadius, Octree<TreeVoxelData>& octree,
		Skeleton<SkeletonData, FlowData, NodeData>& skeleton)
	{
		const auto boxSize = skeleton.m_max - skeleton.m_min;
		const float maxRadius = glm::max(glm::max(boxSize.x, boxSize.y), boxSize.z) + 2.0f * minRadius;
		int subdivisionLevel = 0;
		float testRadius = minRadius;
		while (testRadius <= maxRadius)
		{
			subdivisionLevel++;
			testRadius *= 2.f;
		}
		octree.Reset(maxRadius, subdivisionLevel, (skeleton.m_min + skeleton.m_max) * 0.5f);
		const auto& sortedRootNodeList = skeleton.RefSortedNodeList();
		const auto& sortedRootFlowList = skeleton.RefSortedFlowList();
		int collisionCount = 0;
		int flowCollisionCount = 0;
		std::unordered_map<int, int> nodeCollisionCollection;
		std::unordered_map<int, int> flowCollisionCollection;
		for (const auto& nodeHandle : sortedRootNodeList)
		{
			const auto& node = skeleton.RefNode(nodeHandle);
			const auto& info = node.m_info;
			octree.Occupy(info.m_globalPosition, info.m_globalRotation, info.m_length * 0.9f, info.m_thickness, [&](OctreeNode<TreeVoxelData>& octreeNode)
				{
					if (octreeNode.m_data.m_nodeHandle == nodeHandle) return;
			if (octreeNode.m_data.m_nodeHandle == node.GetParentHandle()) return;
			for (const auto& i : node.RefChildHandles()) if (octreeNode.m_data.m_nodeHandle == i) return;
			auto flowHandle = node.GetFlowHandle();
			if (octreeNode.m_data.m_referenceCount != 0)
			{
				if (octreeNode.m_data.m_nodeHandle > nodeHandle)
				{
					nodeCollisionCollection[octreeNode.m_data.m_nodeHandle] = nodeHandle;
				}
				else
				{
					nodeCollisionCollection[nodeHandle] = octreeNode.m_data.m_nodeHandle;
				}
				if (octreeNode.m_data.m_flowHandle != flowHandle) {
					if (octreeNode.m_data.m_flowHandle > flowHandle)
					{
						flowCollisionCollection[octreeNode.m_data.m_flowHandle] = flowHandle;
					}
					else
					{
						flowCollisionCollection[flowHandle] = octreeNode.m_data.m_flowHandle;
					}
				}
			}
			else {
				octreeNode.m_data.m_flowHandle = flowHandle;
				octreeNode.m_data.m_nodeHandle = nodeHandle;
			}
			octreeNode.m_data.m_referenceCount++;
				});

		}
		collisionCount = nodeCollisionCollection.size();
		flowCollisionCount = flowCollisionCollection.size();
		std::vector<int> collisionStat;
		collisionStat.resize(200);
		for (auto& i : collisionStat) i = 0;
		int totalVoxel = 0;
		octree.IterateLeaves([&](const OctreeNode<TreeVoxelData>& leaf)
			{
				collisionStat[leaf.m_data.m_referenceCount]++;
		totalVoxel++;
			});

		std::string report = "Collision: [" + std::to_string(collisionCount) + "/" + std::to_string(sortedRootNodeList.size()) + "], [" + std::to_string(flowCollisionCount) + "/" + std::to_string(sortedRootFlowList.size()) + "], ";
		report += "total occupied: " + std::to_string(totalVoxel) + ", collision stat: ";

		std::string appendStat;
		for (int i = 199; i > 0; i--)
		{
			if (collisionStat[i] != 0)
			{
				appendStat += "[" + std::to_string(i) + "]->" + std::to_string(collisionStat[i]) + "; ";
			}
		}
		if (appendStat.empty()) appendStat = "No collision";

		UNIENGINE_LOG(report + appendStat);
	}
}
