#pragma once
#include "PlantStructure.hpp"
#include "PipeStructure.hpp"
#include "HexagonGrid.hpp"
using namespace UniEngine;
namespace EcoSysLab
{
	struct BaseHexagonGridCellData
	{
		PipeHandle m_pipeHandle = -1;
	};

	struct BaseHexagonGridData
	{
	};

	struct HexagonGridCellData
	{
		PipeHandle m_pipeHandle = -1;
	};

	struct HexagonGridData
	{
		NodeHandle m_nodeHandle = -1;
	};

	typedef HexagonGrid<BaseHexagonGridData, BaseHexagonGridCellData> PipeModelBaseHexagonGrid;
	typedef HexagonGrid<HexagonGridData, HexagonGridCellData> PipeModelHexagonGrid;
	typedef HexagonGridGroup<HexagonGridData, HexagonGridCellData> PipeModelHexagonGridGroup;

	struct PipeModelPipeGroupData
	{
	};

	struct PipeModelPipeData
	{
		PipeNodeInfo m_baseInfo;
	};

	struct PipeModelPipeNodeData
	{
		NodeHandle m_nodeHandle = -1;
	};

	typedef PipeGroup<PipeModelPipeGroupData, PipeModelPipeData, PipeModelPipeNodeData> PipeModelPipeGroup;

	struct PipeModelNodeData
	{
		int m_endNodeCount = 0;
	};

	struct PipeModelFlowData
	{
		
	};

	struct PipeModelSkeletonData
	{
	};
	typedef Skeleton<PipeModelSkeletonData, PipeModelFlowData, PipeModelNodeData> PipeModelSkeleton;
}