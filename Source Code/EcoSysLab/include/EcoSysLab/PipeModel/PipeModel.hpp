#pragma once
#include "PipeModelParameters.hpp"
using namespace UniEngine;
namespace EcoSysLab
{
	struct CellSortSettings
	{
		bool m_flatCut = false;
	};
	struct CellSqueezeSettings
	{
		
	};
	class PipeModel
	{
		void CalculatePipeLocalPositions(const PipeModelSkeleton& targetSkeleton, const PipeModelParameters& pipeModelParameters);
		void CalculatePipeTransforms(const PipeModelSkeleton& targetSkeleton, const PipeModelParameters& pipeModelParameters);
		void DistributePipes(PipeModelBaseHexagonGrid baseGrid, PipeModelSkeleton& targetSkeleton, const PipeModelParameters& pipeModelParameters);
		static void SqueezeCells(const CellSqueezeSettings& cellSqueezeSettings, const PipeModelHexagonGrid& prevGrid, const PipeModelHexagonGrid& newGrid);
		static void SortCells(const CellSortSettings& cellSortSettings, std::multimap<float, CellHandle>& sortedCellHandles, const std::set<CellHandle>& availableCellHandles, const PipeModelHexagonGrid& prevGrid, const glm::vec2& direction);
		static void ExtractCells(int cellCount, const std::multimap<float, CellHandle>& sortedCellHandles,
			std::set<CellHandle>& availableCellHandles,
		                         const PipeModelHexagonGrid& prevGrid,
		                         PipeModelHexagonGrid& childNewGrid);
		void SplitPipes(std::unordered_map<NodeHandle, HexagonGridHandle>& gridHandleMap, PipeModelHexagonGridGroup& gridGroup, 
			PipeModelSkeleton& targetSkeleton, NodeHandle nodeHandle, HexagonGridHandle newGridHandle, const PipeModelParameters& pipeModelParameters);
	public:
		PipeModelPipeGroup m_pipeGroup;
		template <typename SkeletonData, typename FlowData, typename NodeData>
		void InitializePipes(const Skeleton<SkeletonData, FlowData, NodeData> &targetSkeleton, PipeModelBaseHexagonGrid baseGrid, const PipeModelParameters& pipeModelParameters);
	};

	template <typename SkeletonData, typename FlowData, typename NodeData>
	void PipeModel::InitializePipes(const Skeleton<SkeletonData, FlowData, NodeData>& targetSkeleton, PipeModelBaseHexagonGrid baseGrid,
		const PipeModelParameters& pipeModelParameters)
	{
		if (baseGrid.GetCellCount() == 0) return;
		PipeModelSkeleton clonedSkeleton;
		clonedSkeleton.Clone<SkeletonData, FlowData, NodeData>(targetSkeleton, [&](NodeHandle srcNodeHandle, NodeHandle dstNodeHandle){});
		const auto& flowList = clonedSkeleton.RefSortedFlowList();
		if (!flowList.empty())
		{
			DistributePipes(baseGrid, clonedSkeleton, pipeModelParameters);
			CalculatePipeLocalPositions(clonedSkeleton, pipeModelParameters);
			clonedSkeleton.CalculateTransforms();
			CalculatePipeTransforms(clonedSkeleton, pipeModelParameters);
		}
	}
}
