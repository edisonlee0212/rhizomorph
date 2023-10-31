#pragma once
#include "PlantStructure.hpp"
#include "VigorSink.hpp"
#include "TreeIlluminationEstimator.hpp"
using namespace UniEngine;
namespace EcoSysLab
{
#pragma region Utilities
	enum class BudType {
		Apical,
		Lateral,
		Leaf,
		Fruit
	};

	enum class BudStatus {
		Dormant,
		Flushed,
		Died,
		Removed
	};

	struct ReproductiveModule
	{
		float m_maturity = 0.0f;
		float m_health = 1.0f;
		glm::mat4 m_transform = glm::mat4(0.0f);
		void Reset();
	};

	class Bud {
	public:
		BudType m_type = BudType::Apical;
		BudStatus m_status = BudStatus::Dormant;

		VigorSink m_vigorSink;

		glm::quat m_localRotation = glm::vec3(0.0f);

		//-1.0 means the no fruit.
		ReproductiveModule m_reproductiveModule;

		float m_shootFlux = 0.0f;
	};

	struct RootFlux {
		float m_nitrite = 0.0f;
		float m_water = 0.0f;
	};

	struct ShootFlux {
		float m_lightEnergy = 0.0f;
	};

	struct PlantGrowthRequirement
	{
		float m_leafMaintenanceVigor = 0.0f;
		float m_leafDevelopmentalVigor = 0.0f;
		float m_fruitMaintenanceVigor = 0.0f;
		float m_fruitDevelopmentalVigor = 0.0f;
		float m_nodeDevelopmentalVigor = 0.0f;
	};

	struct TreeVoxelData
	{
		NodeHandle m_nodeHandle = -1;
		NodeHandle m_flowHandle = -1;
		unsigned m_referenceCount = 0;
	};

	struct ShootRootVigorRatio
	{
		float m_rootVigorWeight = 1.0f;
		float m_shootVigorWeight = 1.0f;
	};

	

	
#pragma endregion

	struct InternodeGrowthData {
		bool m_isMaxChild = false;
		bool m_lateral = false;
		float m_startAge = 0;
		float m_inhibitor = 0;
		glm::quat m_desiredLocalRotation = glm::vec3(0.0f);
		float m_sagging = 0;

		float m_maxDistanceToAnyBranchEnd = 0;
		int m_order = 0;
		float m_descendentTotalBiomass = 0;
		float m_biomass = 0;
		float m_extraMass = 0.0f;
		float m_rootDistance = 0;

		float m_temperature = 0.0f;

		glm::vec3 m_lightDirection = glm::vec3(0, 1, 0);
		float m_lightIntensity = 1.0f;

		float m_lightEnergy = 0.0f;
		/**
		 * List of buds, first one will always be the apical bud which points forward.
		 */
		std::vector<Bud> m_buds;
		VigorFlow m_vigorFlow;
		std::vector<glm::mat4> m_leaves;
		std::vector<glm::mat4> m_fruits;		
	};

	struct RootNodeGrowthData {
		bool m_isMaxChild = false;
		bool m_lateral = false;
		float m_soilDensity = 0.0f;
		float m_startAge = 0;
		float m_maxDistanceToAnyBranchEnd = 0;
		float m_descendentTotalBiomass = 0;
		float m_biomass = 0;
		float m_extraMass = 0.0f;

		float m_rootDistance = 0;
		int m_order = 0;

		float m_nitrite = 1.0f;
		float m_water = 1.0f;

		float m_inhibitor = 0;

		float m_horizontalTropism = 0.0f;
		float m_verticalTropism = 0.0f;
		VigorFlow m_vigorFlow;
		/*
		 * The allocated total resource for maintenance and development of this module.
		 */
		VigorSink m_vigorSink;

		std::vector<glm::vec4> m_fineRootAnchors;
	};

	struct ShootStemGrowthData {
		int m_order = 0;
	};

	struct RootStemGrowthData {
		int m_order = 0;
	};

	

	struct ShootGrowthData {
		Octree<TreeVoxelData> m_octree = {};

		TreeIlluminationEstimator m_treeIlluminationEstimator;
		PlantGrowthRequirement m_vigorRequirement = {};
		ShootFlux m_shootFlux = {};

		std::vector<ReproductiveModule> m_droppedLeaves;
		std::vector<ReproductiveModule> m_droppedFruits;

		float m_vigor = 0;
	};

	struct RootGrowthData {
		Octree<TreeVoxelData> m_octree = {};
		PlantGrowthRequirement m_vigorRequirement = {};
		RootFlux m_rootFlux = {};

		float m_vigor = 0;
	};

	typedef Skeleton<ShootGrowthData, ShootStemGrowthData, InternodeGrowthData> ShootSkeleton;
	typedef Skeleton<RootGrowthData, RootStemGrowthData, RootNodeGrowthData> RootSkeleton;
}