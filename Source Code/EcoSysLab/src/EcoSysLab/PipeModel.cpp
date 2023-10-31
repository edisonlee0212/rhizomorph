#include "PipeModel.hpp"

using namespace EcoSysLab;

void PipeModel::CalculatePipeLocalPositions(const PipeModelSkeleton& targetSkeleton,
                                            const PipeModelParameters& pipeModelParameters)
{
	auto& pipeGroup = m_pipeGroup;
	for (auto& pipeNode : pipeGroup.RefPipeNodes())
	{
		if (pipeNode.IsRecycled()) continue;
		const auto& node = targetSkeleton.PeekNode(pipeNode.m_data.m_nodeHandle);
		auto& pipeInfo = pipeNode.m_info;
		pipeInfo.m_thickness = pipeModelParameters.m_pipeRadius;
	}

	for (auto& pipe : pipeGroup.RefPipes())
	{
		if (pipe.IsRecycled()) continue;
		pipe.m_data.m_baseInfo.m_thickness = pipeModelParameters.m_pipeRadius;
		pipe.m_info.m_color = glm::vec4(1.0f);
	}
}

void PipeModel::CalculatePipeTransforms(const PipeModelSkeleton& targetSkeleton, const PipeModelParameters& pipeModelParameters)
{
	auto& pipeGroup = m_pipeGroup;
	for (auto& pipeNode : pipeGroup.RefPipeNodes())
	{
		if (pipeNode.IsRecycled()) continue;
		const auto& node = targetSkeleton.PeekNode(pipeNode.m_data.m_nodeHandle);
		const auto& nodeInfo = node.m_info;
		const glm::vec3 left = nodeInfo.m_regulatedGlobalRotation * glm::vec3(1, 0, 0);
		const glm::vec3 up = nodeInfo.m_regulatedGlobalRotation * glm::vec3(0, 1, 0);
		const glm::vec3 front = nodeInfo.m_regulatedGlobalRotation * glm::vec3(0, 0, -1);
		auto& pipeNodeInfo = pipeNode.m_info;
		pipeNodeInfo.m_globalPosition = nodeInfo.m_globalPosition + front * nodeInfo.m_length + left * pipeNodeInfo.m_localPosition.x + up * pipeNodeInfo.m_localPosition.y;
		pipeNodeInfo.m_globalRotation = nodeInfo.m_regulatedGlobalRotation;
	}
	for(auto& pipe : pipeGroup.RefPipes())
	{
		if (pipe.IsRecycled()) continue;
		auto& pipeInfo = pipe.m_data.m_baseInfo;
		const glm::vec3 left = pipeInfo.m_globalRotation * glm::vec3(1, 0, 0);
		const glm::vec3 up = pipeInfo.m_globalRotation * glm::vec3(0, 1, 0);
		pipeInfo.m_globalPosition = pipeInfo.m_globalPosition + left * pipeInfo.m_localPosition.x + up * pipeInfo.m_localPosition.y;
	}
}

void PipeModel::DistributePipes(PipeModelBaseHexagonGrid baseGrid, PipeModelSkeleton& targetSkeleton, const PipeModelParameters& pipeModelParameters)
{
	//1. Reverse calculate number of distribution ratio.
	const auto nodeList = targetSkeleton.RefSortedNodeList();
	for (auto it = nodeList.rbegin(); it != nodeList.rend(); ++it)
	{
		auto& node = targetSkeleton.RefNode(*it);
		auto& nodeData = node.m_data;
		if (node.IsEndNode()) nodeData.m_endNodeCount = 1;
		else
		{
			nodeData.m_endNodeCount = 0;
			for (const auto& childNodeHandle : node.RefChildHandles())
			{
				const auto& childNode = targetSkeleton.RefNode(childNodeHandle);
				nodeData.m_endNodeCount += childNode.m_data.m_endNodeCount;
			}
		}
	}
	m_pipeGroup = {};
	auto& pipeGroup = m_pipeGroup;
	PipeModelHexagonGridGroup gridGroup;
	std::unordered_map<NodeHandle, HexagonGridHandle> gridHandleMap;
	//2. Allocate pipe for target skeleton. Also create new grid for first node and copy cells from base grid.
	const auto firstGridHandle = gridGroup.Allocate();
	auto& firstGrid = gridGroup.RefGrid(firstGridHandle);
	for (const auto& readOnlyCell : baseGrid.PeekCells())
	{
		if (readOnlyCell.IsRecycled()) continue;
		auto& cell = baseGrid.RefCell(readOnlyCell.GetHandle());
		const auto newPipeHandle = pipeGroup.AllocatePipe();
		cell.m_data.m_pipeHandle = newPipeHandle;

		auto& firstNode = targetSkeleton.RefNode(0);
		const auto firstGridCellHandle = firstGrid.Allocate(cell.GetCoordinate());
		auto& firstGridCell = firstGrid.RefCell(firstGridCellHandle);
		firstGridCell.m_data.m_pipeHandle = newPipeHandle;
		gridHandleMap[0] = firstGridHandle;
		const auto firstPipeNodeHandle = pipeGroup.Extend(newPipeHandle);
		auto& firstPipeNode = pipeGroup.RefPipeNode(firstPipeNodeHandle);
		firstPipeNode.m_info.m_localPosition = pipeModelParameters.m_pipeRadius * 2.0f * firstGrid.GetPosition(cell.GetCoordinate());
		firstPipeNode.m_data.m_nodeHandle = 0;

		auto& firstPipe = pipeGroup.RefPipe(newPipeHandle);
		firstPipe.m_data.m_baseInfo.m_localPosition = pipeModelParameters.m_pipeRadius * 2.0f * firstGrid.GetPosition(cell.GetCoordinate());
		firstPipe.m_data.m_baseInfo.m_globalRotation = firstNode.m_info.m_regulatedGlobalRotation;
	}
	//3. Create traverse graph and setup pipes.
	for (const auto& nodeHandle : nodeList)
	{
		auto& node = targetSkeleton.RefNode(nodeHandle);
		auto& nodeData = node.m_data;
		if(gridHandleMap.find(nodeHandle) == gridHandleMap.end()) continue;
		//Create a hexagon grid for every node that has multiple child, and a hexagon grid for each child.
		const auto currentGridHandle = gridHandleMap.at(nodeHandle);
		//No pipe left for this node.
		if(currentGridHandle < 0) continue;
		if (node.RefChildHandles().size() > 1) {
			const auto newGridHandle = gridGroup.Allocate();
			gridHandleMap[nodeHandle] = newGridHandle;
			auto& newGrid = gridGroup.RefGrid(newGridHandle);
			//Copy all cells from parent grid.
			const auto& previousGrid = gridGroup.PeekGrid(currentGridHandle);
			for(const auto& parentCell : previousGrid.PeekCells())
			{
				if(parentCell.IsRecycled()) continue;
				const auto newCellHandle = newGrid.Allocate(parentCell.GetCoordinate());
				auto& newCell = newGrid.RefCell(newCellHandle);
				newCell.m_data.m_pipeHandle = parentCell.m_data.m_pipeHandle;

				const auto newPipeNodeHandle = pipeGroup.Extend(newCell.m_data.m_pipeHandle);
				auto& newPipeNode = pipeGroup.RefPipeNode(newPipeNodeHandle);

				newPipeNode.m_info.m_localPosition = pipeModelParameters.m_pipeRadius * 2.0f * firstGrid.GetPosition(newCell.GetCoordinate());
				newPipeNode.m_data.m_nodeHandle = nodeHandle;
			}
			SplitPipes(gridHandleMap, gridGroup, targetSkeleton, nodeHandle, newGridHandle, pipeModelParameters);
		}
		else if(node.RefChildHandles().size() == 1)
		{
			const auto childNodeHandle = node.RefChildHandles()[0];
			auto& childNode = targetSkeleton.RefNode(childNodeHandle);
			auto& childNodeData = childNode.m_data;
			//Extend all pipe nodes from parent.
			gridHandleMap[childNodeHandle] = currentGridHandle;
			const auto& previousGrid = gridGroup.PeekGrid(currentGridHandle);
			for (const auto& parentCell : previousGrid.PeekCells())
			{
				if (parentCell.IsRecycled()) continue;

				const auto newPipeNodeHandle = pipeGroup.Extend(parentCell.m_data.m_pipeHandle);
				auto& newPipeNode = pipeGroup.RefPipeNode(newPipeNodeHandle);
				newPipeNode.m_info.m_localPosition = pipeModelParameters.m_pipeRadius * 2.0f * firstGrid.GetPosition(parentCell.GetCoordinate());
				newPipeNode.m_data.m_nodeHandle = childNodeHandle;
			}
		}
	}
}

void PipeModel::SqueezeCells(const CellSqueezeSettings& cellSqueezeSettings, const PipeModelHexagonGrid& prevGrid,
	const PipeModelHexagonGrid& newGrid)
{

}


void PipeModel::SplitPipes(std::unordered_map<NodeHandle, HexagonGridHandle>& gridHandleMap, 
                           PipeModelHexagonGridGroup& gridGroup, PipeModelSkeleton& targetSkeleton, 
                           NodeHandle nodeHandle, HexagonGridHandle newGridHandle, const PipeModelParameters& pipeModelParameters)
{
	auto& pipeGroup = m_pipeGroup;
	auto& node = targetSkeleton.RefNode(nodeHandle);
	auto& newGrid = gridGroup.RefGrid(newGridHandle);
	const auto nodeUp = node.m_info.m_regulatedGlobalRotation * glm::vec3(0, 1, 0);
	const auto nodeLeft = node.m_info.m_regulatedGlobalRotation * glm::vec3(1, 0, 0);
	//Sort child by their end node sizes.
	int totalAllocatedCellCount = 0;
	std::multimap<int, std::pair<NodeHandle, int>> childNodeHandles;
	for (int i = 1; i < node.RefChildHandles().size(); i++)
	{
		auto childNodeHandle = node.RefChildHandles()[i];
		const auto& childNode = targetSkeleton.RefNode(childNodeHandle);
		int cellCount = static_cast<float>(newGrid.GetCellCount()) * childNode.m_data.m_endNodeCount / node.m_data.m_endNodeCount;
		childNodeHandles.insert({ childNode.m_data.m_endNodeCount, { childNodeHandle , cellCount } });
		totalAllocatedCellCount += cellCount;
	}
	int mainChildCellCount = newGrid.GetCellCount() - totalAllocatedCellCount;
	const auto& newGridCellMap = newGrid.PeekCellMap();
	std::set<CellHandle> availableCellHandles;
	for(const auto& i : newGridCellMap)
	{
		availableCellHandles.insert(i.second);
	}
	for (auto it = childNodeHandles.rbegin(); it != childNodeHandles.rend(); ++it)
	{
		const auto cellCount = it->second.second;
		auto& childNode = targetSkeleton.RefNode(it->second.first);
		if (cellCount == 0) continue;
		const auto childNewGridHandle = gridGroup.Allocate();
		gridHandleMap[it->second.first] = childNewGridHandle;
		auto& childNewGrid = gridGroup.RefGrid(childNewGridHandle);
		const auto& prevGrid = gridGroup.RefGrid(newGridHandle);
		//1. Sort cells.
		std::multimap<float, CellHandle> sortedCellHandles;
		auto childNodeFront = glm::inverse(node.m_info.m_regulatedGlobalRotation) * childNode.m_info.m_regulatedGlobalRotation * glm::vec3(0, 0, -1);
		glm::vec2 direction = glm::normalize(glm::vec2(childNodeFront.x, childNodeFront.y));
		CellSortSettings cellSortSettings;
		SortCells(cellSortSettings, sortedCellHandles, availableCellHandles, prevGrid, direction);
		//2. Extract cells based on distance.
		ExtractCells(cellCount, sortedCellHandles, availableCellHandles, prevGrid, childNewGrid);
		glm::ivec2 extractedCellSumCoordinate = glm::ivec2(0, 0);
		for (const auto& cell : childNewGrid.PeekCells())
		{
			if (cell.IsRecycled()) continue;
			auto& childNewCell = childNewGrid.RefCell(cell.GetHandle());
			extractedCellSumCoordinate += childNewCell.GetCoordinate();
		}
		const auto shiftPosition = childNewGrid.GetPosition(extractedCellSumCoordinate / cellCount);
		childNewGrid.ShiftCoordinate(-extractedCellSumCoordinate / cellCount);

		childNode.m_info.m_localPosition = pipeModelParameters.m_pipeRadius * 2.0f * (nodeLeft * shiftPosition.x + nodeUp * shiftPosition.y);
		for (const auto& cell : childNewGrid.PeekCells())
		{
			if (cell.IsRecycled()) continue;
			auto& childNewCell = childNewGrid.RefCell(cell.GetHandle());
			const auto childNewPipeNodeHandle = pipeGroup.Extend(childNewCell.m_data.m_pipeHandle);
			auto& childNewPipeNode = pipeGroup.RefPipeNode(childNewPipeNodeHandle);
			childNewPipeNode.m_info.m_localPosition = pipeModelParameters.m_pipeRadius * 2.0f * childNewGrid.GetPosition(cell.GetHandle());
			childNewPipeNode.m_data.m_nodeHandle = it->second.first;
		}
	}
	//Put the rest cell to main branches.
	if(!node.RefChildHandles().empty() && mainChildCellCount != 0)
	{
		auto childNodeHandle = node.RefChildHandles()[0];
		auto& childNode = targetSkeleton.RefNode(childNodeHandle);
		const auto childNewGridHandle = gridGroup.Allocate();
		gridHandleMap[childNodeHandle] = childNewGridHandle;
		auto& childNewGrid = gridGroup.RefGrid(childNewGridHandle);
		const auto& prevGrid = gridGroup.RefGrid(newGridHandle);
		for (const auto& cellHandle : availableCellHandles)
		{
			const auto& allocatedCell = prevGrid.PeekCell(cellHandle);
			const auto childNewCellHandle = childNewGrid.Allocate(prevGrid.GetPosition(cellHandle));
			auto& childNewCell = childNewGrid.RefCell(childNewCellHandle);
			childNewCell.m_data.m_pipeHandle = allocatedCell.m_data.m_pipeHandle;
		}
		glm::ivec2 extractedCellSumCoordinate = glm::ivec2(0, 0);
		for (const auto& cell : childNewGrid.PeekCells())
		{
			if (cell.IsRecycled()) continue;
			auto& childNewCell = childNewGrid.RefCell(cell.GetHandle());
			extractedCellSumCoordinate += childNewCell.GetCoordinate();
		}
		const auto shiftPosition = childNewGrid.GetPosition(extractedCellSumCoordinate / mainChildCellCount);
		childNewGrid.ShiftCoordinate(-extractedCellSumCoordinate / mainChildCellCount);

		childNode.m_info.m_localPosition = pipeModelParameters.m_pipeRadius * 2.0f * (nodeLeft * shiftPosition.x + nodeUp * shiftPosition.y);
		for (const auto& cell : childNewGrid.PeekCells())
		{
			if (cell.IsRecycled()) continue;
			auto& childNewCell = childNewGrid.RefCell(cell.GetHandle());
			const auto childNewPipeNodeHandle = pipeGroup.Extend(childNewCell.m_data.m_pipeHandle);
			auto& childNewPipeNode = pipeGroup.RefPipeNode(childNewPipeNodeHandle);
			childNewPipeNode.m_info.m_localPosition = pipeModelParameters.m_pipeRadius * 2.0f * childNewGrid.GetPosition(cell.GetHandle());
			childNewPipeNode.m_data.m_nodeHandle = childNodeHandle;
		}
	}
}

void PipeModel::ExtractCells(const int cellCount, const std::multimap<float, CellHandle>& sortedCellHandles,
	std::set<CellHandle>& availableCellHandles,
                             const PipeModelHexagonGrid& prevGrid,
                             PipeModelHexagonGrid& childNewGrid)
{
	auto it = sortedCellHandles.begin();
	for (int i = 0; i < cellCount; i++)
	{
		const auto allocatedCellHandle = it->second;
		++it;
		const auto& allocatedCell = prevGrid.PeekCell(allocatedCellHandle);
		availableCellHandles.erase(allocatedCellHandle);
		const auto childNewCellHandle = childNewGrid.Allocate(prevGrid.GetPosition(allocatedCellHandle));
		auto& childNewCell = childNewGrid.RefCell(childNewCellHandle);
		childNewCell.m_data.m_pipeHandle = allocatedCell.m_data.m_pipeHandle;
	}
}

void PipeModel::SortCells(const CellSortSettings& cellSortSettings, std::multimap<float, CellHandle>& sortedCellHandles, const std::set<CellHandle>& availableCellHandles, const PipeModelHexagonGrid& prevGrid, const glm::vec2& direction)
{
	//1. Find start cell.
	std::multimap<float, CellHandle> distanceSortedCellHandles;
	auto avgPosition = glm::vec2(0.0f);
	for (const auto& i : availableCellHandles) {
		avgPosition += prevGrid.GetPosition(i);
	}
	avgPosition /= availableCellHandles.size();
	for (const auto& i : availableCellHandles) {
		auto position = prevGrid.GetPosition(i) - avgPosition;
		const float distance = glm::dot(position, direction) / glm::length(direction);
		distanceSortedCellHandles.insert({ -distance, i });
	}
	if(cellSortSettings.m_flatCut)
	{
		sortedCellHandles = distanceSortedCellHandles;
		return;
	}
	//2. Sort cell based on distance to the start cell
	const auto baseCellHandle = distanceSortedCellHandles.begin()->second;
	const auto& baseCell = prevGrid.PeekCell(baseCellHandle);
	const auto basePosition = prevGrid.GetPosition(baseCell.GetCoordinate());
	for (const auto& i : availableCellHandles) {
		auto position = prevGrid.GetPosition(i);
		float distance = glm::distance(position, basePosition);
		sortedCellHandles.insert({ distance, i });
	}
}


