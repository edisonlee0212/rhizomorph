#pragma once

#include "ecosyslab_export.h"
#include "Tree.hpp"
#include "Soil.hpp"
#include "Climate.hpp"

using namespace UniEngine;
namespace EcoSysLab {
	struct Fruit {
		GlobalTransform m_globalTransform;
		float m_maturity = 0.0f;
		float m_health = 1.0f;
	};

	struct Leaf {
		GlobalTransform m_globalTransform;
		float m_maturity = 0.0f;
		float m_health = 1.0f;
	};

	class EcoSysLabLayer : public ILayer {
		friend class TreeVisualizer;
		bool m_displayShootStem = true;
		bool m_displayFoliage = true;
		bool m_displayFruit = true;
		bool m_displayFineRoot = false;
		bool m_displayRootStem = true;
		bool m_displayBoundingBox = false;
		bool m_displaySoil = false;
		bool m_displayGroundFruit = true;
		bool m_displayGroundLeaves = true;

		bool m_debugVisualization = true;
		std::vector<int> m_shootVersions;
		std::vector<int> m_rootVersions;
		std::vector<glm::vec3> m_randomColors;

		std::vector<glm::uint> m_shootStemSegments;
		std::vector<StrandPoint> m_shootStemPoints;
		std::vector<glm::uint> m_rootStemSegments;
		std::vector<StrandPoint> m_rootStemPoints;

		std::vector<glm::uint> m_fineRootSegments;
		std::vector<StrandPoint> m_fineRootPoints;

		AssetRef m_shootStemStrands;
		AssetRef m_rootStemStrands;
		AssetRef m_fineRootStrands;

		std::vector<glm::mat4> m_boundingBoxMatrices;
		std::vector<glm::vec4> m_boundingBoxColors;

		std::shared_ptr<ParticleMatrices> m_foliageMatrices;
		std::shared_ptr<ParticleMatrices> m_fruitMatrices;

		std::shared_ptr<ParticleMatrices> m_groundFruitMatrices;
		std::shared_ptr<ParticleMatrices> m_groundLeafMatrices;

		float m_lastUsedTime = 0.0f;
		float m_totalTime = 0.0f;
		int m_internodeSize = 0;
		int m_leafSize = 0;
		int m_fruitSize = 0;
		int m_shootStemSize = 0;
		int m_rootNodeSize = 0;
		int m_rootStemSize = 0;
		bool m_needFullFlowUpdate = false;
		bool m_needFlowUpdateForSelection = false;
		int m_lastSelectedTreeIndex = -1;
		bool m_lockTreeSelection = false;
		bool m_autoGrow = false;
		bool m_autoGrowWithSoilStep = false;
		bool m_autoClearFruitAndLeaves = true;
		int m_soilVersion = -1;
		bool m_vectorEnable = false;
		bool m_scalarEnable = true;
		bool m_updateVectorMatrices = false;
		bool m_updateVectorColors = false;
		bool m_updateScalarMatrices = false;
		bool m_updateScalarColors = false;
		float m_vectorMultiplier = 50.0f;
		glm::vec4 m_vectorBaseColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.8f);
		unsigned m_vectorSoilProperty = 4;
		float m_vectorLineWidthFactor = 0.1f;
		float m_vectorLineMaxWidth = 0.1f;
		std::vector<glm::mat4> m_vectorMatrices;
		std::vector<glm::vec4> m_vectorColors;

		float m_scalarMultiplier = 1.0f;
		float m_scalarBoxSize = 1.0f;
		float m_scalarMinAlpha = 0.00f;

		std::vector<glm::vec4> m_soilLayerColors;

		friend class Soil;

		float m_soilCutoutXDepth = 0.0f;
		float m_soilCutoutZDepth = 0.0f;

		glm::vec3 m_scalarBaseColor = glm::vec3(0.0f, 0.0f, 1.0f);
		unsigned m_scalarSoilProperty = 1;
		std::vector<glm::mat4> m_scalarMatrices;
		std::vector<glm::vec4> m_scalarColors;

		void UpdateVisualizationCamera();

		void Update() override;

		void OnCreate() override;

		void OnDestroy() override;

		void LateUpdate() override;

		void OnInspect() override;

		void OnSoilVisualizationMenu();


		void UpdateFlows(const std::vector<Entity> *treeEntities, const std::shared_ptr<Strands> &branchStrands,
										 const std::shared_ptr<Strands> &rootStrands, const std::shared_ptr<Strands> &fineRootStrands,
										 int targetTreeIndex = -1);

		void ClearGroundFruitAndLeaf();

		void UpdateGroundFruitAndLeaves() const;

		// helper functions to structure code a bit
		void SoilVisualization();

		void SoilVisualizationScalar(SoilModel &soilModel); // called during LateUpdate()
		void SoilVisualizationVector(SoilModel &soilModel); // called during LateUpdate()

		float m_time;
		float m_deltaTime = 0.01918f;

		std::vector<Fruit> m_fruits;
		std::vector<Leaf> m_leaves;


		std::shared_ptr<Camera> m_visualizationCamera;

		glm::vec2 m_visualizationCameraMousePosition;
		bool m_visualizationCameraWindowFocused = false;
		public:
		int m_visualizationCameraResolutionX = 1;
		int m_visualizationCameraResolutionY = 1;
		float m_lastX = 0;
		float m_lastY = 0;
		float m_lastScrollY = 0;
		bool m_startMouse = false;
		bool m_startScroll = false;
		bool m_rightMouseButtonHold = false;


		IlluminationEstimationSettings m_shadowEstimationSettings;


		TreeMeshGeneratorSettings m_meshGeneratorSettings;
		Entity m_selectedTree = {};

		EntityRef m_shootStemStrandsHolder;
		EntityRef m_rootStemStrandsHolder;
		EntityRef m_fineRootStrandsHolder;
		EntityRef m_foliageHolder;
		EntityRef m_fruitHolder;
		EntityRef m_groundLeavesHolder;
		EntityRef m_groundFruitsHolder;
		TreeVisualizer m_treeVisualizer;

		PrivateComponentRef m_soilHolder;
		PrivateComponentRef m_climateHolder;

		void Simulate(float deltaTime);

		void GenerateMeshes(const TreeMeshGeneratorSettings &meshGeneratorSettings) const;

		void ClearGeometries() const;

		void ResetAllTrees(const std::vector<Entity> *treeEntities);

		const std::vector<glm::vec3> &RandomColors();
	};


}
