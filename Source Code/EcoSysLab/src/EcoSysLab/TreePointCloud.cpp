#include <unordered_set>
#include "TreePointCloud.hpp"
#include "Graphics.hpp"
#include "EcoSysLabLayer.hpp"
#include "rapidcsv.h"

using namespace EcoSysLab;

void TreePointCloud::FindPoints(const glm::vec3 &position, VoxelGrid<std::vector<PointCloudVoxel>> &pointVoxelGrid,
																float radius,
																const std::function<void(const PointCloudVoxel &voxel)> &func) const {
		pointVoxelGrid.ForEach(position, radius * 2.0f, [&](const std::vector<PointCloudVoxel> &voxels) {
			for (const auto &voxel: voxels) {
					if (glm::distance(position, voxel.m_position) > radius) continue;
					func(voxel);
			}
		});
}

void TreePointCloud::ImportGraph(const std::filesystem::path &path, float scaleFactor) {
		if (!std::filesystem::exists(path)) {
				UNIENGINE_ERROR("Not exist!");
				return;
		}
		try {
				std::ifstream stream(path.string());
				std::stringstream stringStream;
				stringStream << stream.rdbuf();
				YAML::Node in = YAML::Load(stringStream.str());

				const auto &tree = in["Tree"];
				const auto &scatterPoints = tree["Scatter Points"];
				const auto &treeParts = tree["Tree Parts"];

				m_scatteredPoints.resize(scatterPoints.size());

				for (int i = 0; i < scatterPoints.size(); i++) {
						auto &point = m_scatteredPoints[i];
						point.m_position = scatterPoints[i].as<glm::vec3>() * scaleFactor;

						point.m_handle = i;
						point.m_neighbors.clear();
				}

				m_scannedBranches.clear();
				m_operatingBranches.clear();
				m_treeParts.clear();
				m_allocatedPoints.clear();
				m_skeletons.clear();
				m_scatterPointToBranchEndConnections.clear();
				m_scatterPointToBranchStartConnections.clear();
				m_scatterPointsConnections.clear();
				m_branchConnections.clear();
				m_filteredBranchConnections.clear();
				m_min = glm::vec3(FLT_MAX);
				m_max = glm::vec3(FLT_MIN);
				float minHeight = 999.0f;
				for (int i = 0; i < treeParts.size(); i++) {
						const auto &inTreeParts = treeParts[i];
						auto &treePart = m_treeParts.emplace_back();
						treePart.m_handle = m_treeParts.size() - 1;
						for (const auto &inBranch: inTreeParts["Branches"]) {
								auto &branch = m_scannedBranches.emplace_back();
								branch.m_bezierCurve.m_p0 = inBranch["Start Pos"].as<glm::vec3>() * scaleFactor;
								branch.m_bezierCurve.m_p3 = inBranch["End Pos"].as<glm::vec3>() * scaleFactor;

								auto cPLength = glm::distance(branch.m_bezierCurve.m_p0, branch.m_bezierCurve.m_p3) * 0.3f;
								branch.m_bezierCurve.m_p1 =
												glm::normalize(inBranch["Start Dir"].as<glm::vec3>()) * cPLength + branch.m_bezierCurve.m_p0;
								branch.m_bezierCurve.m_p2 =
												branch.m_bezierCurve.m_p3 - glm::normalize(inBranch["End Dir"].as<glm::vec3>()) * cPLength;
								/*
								branch.m_bezierCurve.m_p1 = glm::mix(branch.m_bezierCurve.m_p0, branch.m_bezierCurve.m_p3, 0.3f);
								branch.m_bezierCurve.m_p2 = glm::mix(branch.m_bezierCurve.m_p0, branch.m_bezierCurve.m_p3, 0.7f);
								*/
								branch.m_startThickness = inBranch["Start Radius"].as<float>() * scaleFactor;
								branch.m_endThickness = inBranch["End Radius"].as<float>() * scaleFactor;
								branch.m_handle = m_scannedBranches.size() - 1;
								treePart.m_branchHandles.emplace_back(branch.m_handle);
								branch.m_treePartHandle = treePart.m_handle;
								if (branch.m_bezierCurve.m_p0.y >= branch.m_bezierCurve.m_p3.y) {
										auto p0 = branch.m_bezierCurve.m_p3;
										branch.m_bezierCurve.m_p3 = branch.m_bezierCurve.m_p0;
										branch.m_bezierCurve.m_p0 = p0;
										auto p1 = branch.m_bezierCurve.m_p2;
										branch.m_bezierCurve.m_p2 = branch.m_bezierCurve.m_p1;
										branch.m_bezierCurve.m_p1 = p1;
										auto startT = branch.m_startThickness;
										branch.m_startThickness = branch.m_endThickness;
										branch.m_endThickness = startT;
								}
								minHeight = glm::min(minHeight, branch.m_bezierCurve.m_p0.y);
								minHeight = glm::min(minHeight, branch.m_bezierCurve.m_p3.y);
						}
						for (const auto &inAllocatedPoint: inTreeParts["Allocated Points"]) {
								auto &allocatedPoint = m_allocatedPoints.emplace_back();
								allocatedPoint.m_position = inAllocatedPoint.as<glm::vec3>() * scaleFactor;
								allocatedPoint.m_handle = m_allocatedPoints.size() - 1;
								allocatedPoint.m_treePartHandle = treePart.m_handle;
								allocatedPoint.m_branchHandle = -1;
								treePart.m_allocatedPoints.emplace_back(allocatedPoint.m_handle);
						}
				}
				for (auto &scatterPoint: m_scatteredPoints) {
						scatterPoint.m_position.y -= minHeight;

						m_min = glm::min(m_min, scatterPoint.m_position);
						m_max = glm::max(m_max, scatterPoint.m_position);
				}
				for (auto &allocatedPoint: m_allocatedPoints) {
						allocatedPoint.m_position.y -= minHeight;

						m_min = glm::min(m_min, allocatedPoint.m_position);
						m_max = glm::max(m_max, allocatedPoint.m_position);
				}
				for (auto &scannedBranch: m_scannedBranches) {
						scannedBranch.m_bezierCurve.m_p0.y -= minHeight;
						scannedBranch.m_bezierCurve.m_p1.y -= minHeight;
						scannedBranch.m_bezierCurve.m_p2.y -= minHeight;
						scannedBranch.m_bezierCurve.m_p3.y -= minHeight;

						m_min = glm::min(m_min, scannedBranch.m_bezierCurve.m_p0);
						m_max = glm::max(m_max, scannedBranch.m_bezierCurve.m_p0);
						m_min = glm::min(m_min, scannedBranch.m_bezierCurve.m_p3);
						m_max = glm::max(m_max, scannedBranch.m_bezierCurve.m_p3);
				}
		}
		catch (std::exception e) {
				UNIENGINE_ERROR("Failed to load!");
				return;
		}
}

void TreePointCloud::OnInspect() {
		static Handle previousHandle = 0;
		static std::vector<glm::mat4> allocatedPointMatrices;
		static std::vector<glm::vec4> allocatedPointColors;

		static std::vector<glm::mat4> scatterPointMatrices;
		static std::vector<glm::mat4> nodeMatrices;
		static std::vector<glm::vec4> nodeColors;
		static std::vector<glm::vec3> scatteredPointConnectionsStarts;
		static std::vector<glm::vec3> scatterConnectionEnds;
		static std::vector<glm::vec3> branchConnectionStarts;
		static std::vector<glm::vec3> branchConnectionEnds;
		static std::vector<glm::vec3> filteredBranchConnectionStarts;
		static std::vector<glm::vec3> filteredBranchConnectionEnds;
		static std::vector<glm::vec3> scatterPointToBranchConnectionStarts;
		static std::vector<glm::vec3> scatterPointToBranchConnectionEnds;


		static std::vector<glm::vec3> scannedBranchStarts;
		static std::vector<glm::vec3> scannedBranchEnds;
		static std::vector<glm::vec4> scannedBranchColors;
		static bool enableDebugRendering = true;

		static bool drawAllocatedPoints = true;
		static bool drawScannedBranches = true;
		static bool drawScatteredPoints = true;
		static bool drawScatteredPointConnections = false;
		static bool drawBranchConnections = false;
		static bool drawFilteredConnections = true;
		static bool drawScatterPointToBranchConnections = false;
		static bool drawNode = true;
		static float scannedBranchWidth = 0.005f;
		static float connectionWidth = 0.001f;
		static float pointSize = 0.002f;

		static float nodeSize = 1.0f;
		bool refreshData = false;

		static int colorMode = 0;


		static glm::vec4 scatterPointToBranchConnectionColor = glm::vec4(1, 0, 1, 1);
		static glm::vec4 scatterPointColor = glm::vec4(0, 1, 0, 1);
		static glm::vec4 scatteredPointConnectionColor = glm::vec4(1, 1, 1, 1);
		static glm::vec4 branchConnectionColor = glm::vec4(1, 0, 0, 1);
		static glm::vec4 filteredBranchConnectionColor = glm::vec4(1, 1, 0, 1);

		static float importScale = 0.1f;
		ImGui::DragFloat("Import scale", &importScale, 0.01f, 0.01f, 10.0f);
		FileUtils::OpenFile("Load YAML", "YAML", {".yml"}, [&](const std::filesystem::path &path) {
			ImportGraph(path, importScale);
			refreshData = true;
		}, false);

		if (!m_scatteredPoints.empty()) {
				static ConnectivityGraphSettings connectivityGraphSettings;
				if (ImGui::TreeNodeEx("Graph Settings")) {
						connectivityGraphSettings.OnInspect();
						ImGui::TreePop();
				}
				static ReconstructionSettings reconstructionSettings;
				if (ImGui::TreeNodeEx("Reconstruction Settings")) {
						reconstructionSettings.OnInspect();
						ImGui::TreePop();
				}
				if (ImGui::Button("Build Skeleton")) {
						EstablishConnectivityGraph(connectivityGraphSettings);
						BuildTreeStructure(reconstructionSettings);
						refreshData = true;
				}
				static TreeMeshGeneratorSettings meshGeneratorSettings;
				meshGeneratorSettings.OnInspect();
				if (ImGui::Button("Form tree mesh")) {
						if (m_filteredBranchConnections.empty()) {
								m_skeletons.clear();
								EstablishConnectivityGraph(connectivityGraphSettings);
						}
						if (m_skeletons.empty()) BuildTreeStructure(reconstructionSettings);
						GenerateMeshes(meshGeneratorSettings);
				}
				if (ImGui::Button("Refresh Data")) {
						refreshData = true;
				}
		}

		ImGui::Checkbox("Debug Rendering", &enableDebugRendering);
		if (enableDebugRendering) {
				GizmoSettings gizmoSettings;
				if (ImGui::Combo("Color mode", {"TreePart", "Branch", "Node"}, colorMode)) refreshData = true;
				if (ImGui::TreeNode("Render settings")) {
						ImGui::DragFloat("Branch width", &scannedBranchWidth, 0.0001f, 0.0001f, 1.0f, "%.4f");
						ImGui::DragFloat("Connection width", &connectionWidth, 0.0001f, 0.0001f, 1.0f, "%.4f");
						ImGui::DragFloat("Point size", &pointSize, 0.0001f, 0.0001f, 1.0f, "%.4f");
						ImGui::DragFloat("Node size", &nodeSize, 0.01f, 0.1f, 10.0f);

						ImGui::Checkbox("Render allocated point", &drawAllocatedPoints);
						ImGui::Checkbox("Render scattered point", &drawScatteredPoints);
						ImGui::Checkbox("Render scatter point connections", &drawScatteredPointConnections);
						ImGui::Checkbox("Render branch", &drawScannedBranches);
						ImGui::Checkbox("Render all junction connections", &drawBranchConnections);
						ImGui::Checkbox("Render filtered junction connections", &drawFilteredConnections);
						ImGui::Checkbox("Render scatter point to branch connections", &drawScatterPointToBranchConnections);
						ImGui::Checkbox("Render nodes", &drawNode);
						ImGui::ColorEdit4("Scatter Point Color", &scatterPointColor.x);
						if (drawScatteredPoints) ImGui::ColorEdit4("Scatter Connection Color", &scatteredPointConnectionColor.x);
						if (drawBranchConnections) ImGui::ColorEdit4("Junction Connection Color", &branchConnectionColor.x);
						if (drawFilteredConnections)
								ImGui::ColorEdit4("Filtered Scatter Connection Color", &filteredBranchConnectionColor.x);
						if (drawScatterPointToBranchConnections)
								ImGui::ColorEdit4("Scattered point to branch connection color", &scatterPointToBranchConnectionColor.x);

						gizmoSettings.m_drawSettings.OnInspect();

						ImGui::TreePop();
				}
				if (GetHandle() != previousHandle) refreshData = true;

				if (refreshData) {
						previousHandle = GetHandle();
						const auto ecoSysLabLayer = Application::GetLayer<EcoSysLabLayer>();

						allocatedPointColors.resize(m_allocatedPoints.size());
						allocatedPointMatrices.resize(m_allocatedPoints.size());

						scannedBranchStarts.resize(m_scannedBranches.size());
						scannedBranchEnds.resize(m_scannedBranches.size());
						scannedBranchColors.resize(m_scannedBranches.size());

						nodeMatrices.clear();
						nodeColors.clear();
						switch (colorMode) {
								case 0: {
										//TreePart
										for (int i = 0; i < m_allocatedPoints.size(); i++) {
												allocatedPointMatrices[i] =
																glm::translate(m_allocatedPoints[i].m_position) * glm::scale(glm::vec3(1.0f));
												allocatedPointColors[i] = glm::vec4(
																ecoSysLabLayer->RandomColors()[m_allocatedPoints[i].m_treePartHandle], 1.0f);
										}
										for (int i = 0; i < m_scannedBranches.size(); i++) {
												scannedBranchStarts[i] = m_scannedBranches[i].m_bezierCurve.m_p0;
												scannedBranchEnds[i] = m_scannedBranches[i].m_bezierCurve.m_p3;
												scannedBranchColors[i] = glm::vec4(
																ecoSysLabLayer->RandomColors()[m_scannedBranches[i].m_treePartHandle], 1.0f);
										}
										for (const auto &skeleton: m_skeletons) {
												const auto &nodeList = skeleton.RefSortedNodeList();
												auto startIndex = nodeMatrices.size();
												nodeMatrices.resize(startIndex + nodeList.size());
												nodeColors.resize(startIndex + nodeList.size());
												for (int i = 0; i < nodeList.size(); i++) {
														const auto &node = skeleton.PeekNode(nodeList[i]);
														nodeMatrices[startIndex + i] = glm::translate(node.m_info.m_globalPosition) *
																													 glm::scale(glm::vec3(node.m_info.m_thickness));
														nodeColors[startIndex + i] = glm::vec4(1.0f);
												}
										}
								}
										break;
								case 1: {
										//Branch
										for (int i = 0; i < m_allocatedPoints.size(); i++) {
												allocatedPointMatrices[i] =
																glm::translate(m_allocatedPoints[i].m_position) * glm::scale(glm::vec3(1.0f));
												if (m_allocatedPoints[i].m_branchHandle >= 0) {
														allocatedPointColors[i] = glm::vec4(
																		ecoSysLabLayer->RandomColors()[m_allocatedPoints[i].m_branchHandle], 1.0f);
												} else {
														allocatedPointColors[i] = glm::vec4(
																		ecoSysLabLayer->RandomColors()[m_allocatedPoints[i].m_treePartHandle], 1.0f);
												}
										}

										for (int i = 0; i < m_scannedBranches.size(); i++) {
												scannedBranchStarts[i] = m_scannedBranches[i].m_bezierCurve.m_p0;
												scannedBranchEnds[i] = m_scannedBranches[i].m_bezierCurve.m_p3;
												scannedBranchColors[i] = glm::vec4(
																ecoSysLabLayer->RandomColors()[m_scannedBranches[i].m_handle], 1.0f);
										}
										for (const auto &skeleton: m_skeletons) {
												const auto &nodeList = skeleton.RefSortedNodeList();
												auto startIndex = nodeMatrices.size();
												nodeMatrices.resize(startIndex + nodeList.size());
												nodeColors.resize(startIndex + nodeList.size());
												for (int i = 0; i < nodeList.size(); i++) {
														const auto &node = skeleton.PeekNode(nodeList[i]);
														nodeMatrices[startIndex + i] = glm::translate(node.m_info.m_globalPosition) *
																													 glm::scale(glm::vec3(node.m_info.m_thickness));
														nodeColors[startIndex + i] = glm::vec4(
																		ecoSysLabLayer->RandomColors()[node.m_data.m_branchHandle],
																		1.0f);
												}
										}
								}
										break;
								case 2: {
										//Node
										for (int i = 0; i < m_allocatedPoints.size(); i++) {
												allocatedPointMatrices[i] =
																glm::translate(m_allocatedPoints[i].m_position) * glm::scale(glm::vec3(1.0f));
												if (m_allocatedPoints[i].m_nodeHandle >= 0) {
														allocatedPointColors[i] = glm::vec4(
																		ecoSysLabLayer->RandomColors()[m_allocatedPoints[i].m_nodeHandle], 1.0f);
												} else {
														allocatedPointColors[i] = glm::vec4(
																		ecoSysLabLayer->RandomColors()[m_allocatedPoints[i].m_treePartHandle], 1.0f);
												}
										}

										for (int i = 0; i < m_scannedBranches.size(); i++) {
												scannedBranchStarts[i] = m_scannedBranches[i].m_bezierCurve.m_p0;
												scannedBranchEnds[i] = m_scannedBranches[i].m_bezierCurve.m_p3;
												scannedBranchColors[i] = glm::vec4(1.0f);
										}
										for (const auto &skeleton: m_skeletons) {
												const auto &nodeList = skeleton.RefSortedNodeList();
												auto startIndex = nodeMatrices.size();
												nodeMatrices.resize(startIndex + nodeList.size());
												nodeColors.resize(startIndex + nodeList.size());
												for (int i = 0; i < nodeList.size(); i++) {
														const auto &node = skeleton.PeekNode(nodeList[i]);
														nodeMatrices[startIndex + i] = glm::translate(node.m_info.m_globalPosition) *
																													 glm::scale(glm::vec3(node.m_info.m_thickness));
														nodeColors[startIndex + i] = glm::vec4(ecoSysLabLayer->RandomColors()[node.GetHandle()],
																																	 1.0f);
												}
										}
								}
										break;
						}

						scatterPointMatrices.resize(m_scatteredPoints.size());
						for (int i = 0; i < m_scatteredPoints.size(); i++) {
								scatterPointMatrices[i] = glm::translate(m_scatteredPoints[i].m_position) * glm::scale(glm::vec3(1.0f));
						}
						scatteredPointConnectionsStarts.resize(m_scatterPointsConnections.size());
						scatterConnectionEnds.resize(m_scatterPointsConnections.size());
						for (int i = 0; i < m_scatterPointsConnections.size(); i++) {
								scatteredPointConnectionsStarts[i] = m_scatterPointsConnections[i].first;
								scatterConnectionEnds[i] = m_scatterPointsConnections[i].second;
						}

						branchConnectionStarts.resize(m_branchConnections.size());
						branchConnectionEnds.resize(m_branchConnections.size());
						for (int i = 0; i < m_branchConnections.size(); i++) {
								branchConnectionStarts[i] = m_branchConnections[i].first;
								branchConnectionEnds[i] = m_branchConnections[i].second;
						}

						filteredBranchConnectionStarts.resize(m_filteredBranchConnections.size());
						filteredBranchConnectionEnds.resize(m_filteredBranchConnections.size());
						for (int i = 0; i < m_filteredBranchConnections.size(); i++) {
								filteredBranchConnectionStarts[i] = m_filteredBranchConnections[i].first;
								filteredBranchConnectionEnds[i] = m_filteredBranchConnections[i].second;
						}

						scatterPointToBranchConnectionStarts.resize(
										m_scatterPointToBranchStartConnections.size() + m_scatterPointToBranchEndConnections.size());
						scatterPointToBranchConnectionEnds.resize(
										m_scatterPointToBranchStartConnections.size() + m_scatterPointToBranchEndConnections.size());
						for (int i = 0; i < m_scatterPointToBranchStartConnections.size(); i++) {
								scatterPointToBranchConnectionStarts[i] = m_scatterPointToBranchStartConnections[i].first;
								scatterPointToBranchConnectionEnds[i] = m_scatterPointToBranchStartConnections[i].second;
						}
						for (int i = m_scatterPointToBranchStartConnections.size();
								 i < m_scatterPointToBranchStartConnections.size() + m_scatterPointToBranchEndConnections.size(); i++) {
								scatterPointToBranchConnectionStarts[i] = m_scatterPointToBranchEndConnections[i -
																																															 m_scatterPointToBranchStartConnections.size()].first;
								scatterPointToBranchConnectionEnds[i] = m_scatterPointToBranchEndConnections[i -
																																														 m_scatterPointToBranchStartConnections.size()].second;
						}
				}
				if (!scatterPointMatrices.empty()) {
						if (drawScatteredPoints) {
								Gizmos::DrawGizmoMeshInstanced(DefaultResources::Primitives::Sphere, scatterPointColor,
																							 scatterPointMatrices,
																							 glm::mat4(1.0f),
																							 pointSize, gizmoSettings);
						}
						if (drawAllocatedPoints) {
								Gizmos::DrawGizmoMeshInstancedColored(DefaultResources::Primitives::Sphere, allocatedPointColors,
																											allocatedPointMatrices,
																											glm::mat4(1.0f),
																											pointSize, gizmoSettings);
						}
						if (drawScannedBranches)
								Gizmos::DrawGizmoRays(scannedBranchColors, scannedBranchStarts, scannedBranchEnds,
																			scannedBranchWidth, gizmoSettings);
						if (drawScatteredPointConnections)
								Gizmos::DrawGizmoRays(scatteredPointConnectionColor, scatteredPointConnectionsStarts,
																			scatterConnectionEnds,
																			connectionWidth, gizmoSettings);
						if (drawBranchConnections)
								Gizmos::DrawGizmoRays(branchConnectionColor, branchConnectionStarts, branchConnectionEnds,
																			connectionWidth * 1.5f, gizmoSettings);
						if (drawFilteredConnections)
								Gizmos::DrawGizmoRays(filteredBranchConnectionColor, filteredBranchConnectionStarts,
																			filteredBranchConnectionEnds,
																			connectionWidth * 2.0f, gizmoSettings);
						if (drawScatterPointToBranchConnections)
								Gizmos::DrawGizmoRays(scatterPointToBranchConnectionColor, scatterPointToBranchConnectionStarts,
																			scatterPointToBranchConnectionEnds,
																			connectionWidth, gizmoSettings);
						if (drawNode) {
								Gizmos::DrawGizmoMeshInstancedColored(DefaultResources::Primitives::Sphere, nodeColors, nodeMatrices,
																											glm::mat4(1.0f), nodeSize, gizmoSettings);
						}
				}
		}


}

void TreePointCloud::EstablishConnectivityGraph(const ConnectivityGraphSettings &settings) {
		VoxelGrid<std::vector<PointCloudVoxel>> pointVoxelGrid;
		pointVoxelGrid.Initialize(3.0f * settings.m_edgeLength, m_min, m_max);
		for (auto &point: m_allocatedPoints) {
				point.m_branchHandle = point.m_nodeHandle = point.m_skeletonIndex = -1;
		}
		for (auto &point: m_scatteredPoints) {
				point.m_neighbors.clear();
				point.m_neighborBranchEnds.clear();
#ifndef TREEPOINTCLOUD_CLEAN
				point.m_neighborBranchStarts.clear();
#endif

				PointCloudVoxel voxel;
				voxel.m_handle = point.m_handle;
				voxel.m_position = point.m_position;
				voxel.m_type = PointCloudVoxelType::ScatteredPoint;
				pointVoxelGrid.Ref(point.m_position).emplace_back(voxel);
		}

		for (auto &scannedBranch: m_scannedBranches) {
				scannedBranch.m_startNeighbors.clear();
				scannedBranch.m_neighborBranchEnds.clear();
				scannedBranch.m_parentHandle = -1;
				scannedBranch.m_childHandles.clear();

#ifndef TREEPOINTCLOUD_CLEAN
				scannedBranch.m_endNeighbors.clear();
				scannedBranch.m_neighborBranchStarts.clear();
#endif

				PointCloudVoxel voxel;
				voxel.m_handle = scannedBranch.m_handle;
				auto shortenedP0 = scannedBranch.m_bezierCurve.GetPoint(settings.m_branchShortening);
				auto shortenedP3 = scannedBranch.m_bezierCurve.GetPoint(1.0f - settings.m_branchShortening);
				voxel.m_position = shortenedP0;
				voxel.m_type = PointCloudVoxelType::BranchStart;
				pointVoxelGrid.Ref(shortenedP0).emplace_back(voxel);
				voxel.m_position = shortenedP3;
				voxel.m_type = PointCloudVoxelType::BranchEnd;
				pointVoxelGrid.Ref(shortenedP3).emplace_back(voxel);
		}
		m_scatterPointsConnections.clear();
		m_filteredBranchConnections.clear();
		m_branchConnections.clear();
		m_scatterPointToBranchStartConnections.clear();
		m_scatterPointToBranchEndConnections.clear();

		for (auto &point: m_scatteredPoints) {
				if (m_scatterPointsConnections.size() > 1000000) {
						UNIENGINE_ERROR("Too much connections!");
						return;
				}
				FindPoints(point.m_position, pointVoxelGrid, settings.m_scatterPointConnectionMaxLength,
									 [&](const PointCloudVoxel &voxel) {
										 if (voxel.m_type != PointCloudVoxelType::ScatteredPoint) return;
										 if (voxel.m_handle == point.m_handle) return;
										 for (const auto &neighbor: point.m_neighbors) {
												 if (voxel.m_handle == neighbor) return;
										 }
										 auto &otherPoint = m_scatteredPoints[voxel.m_handle];
										 point.m_neighbors.emplace_back(voxel.m_handle);
										 otherPoint.m_neighbors.emplace_back(point.m_handle);
										 m_scatterPointsConnections.emplace_back(point.m_position, otherPoint.m_position);
									 });
		}

		for (auto &branch: m_scannedBranches) {
				float currentEdgeLength = settings.m_edgeLength;
				auto shortenedP0 = branch.m_bezierCurve.GetPoint(settings.m_branchShortening);
				auto shortenedP3 = branch.m_bezierCurve.GetPoint(1.0f - settings.m_branchShortening);

				float branchLength = glm::distance(shortenedP0, shortenedP3);
				int timeout = 0;
				bool findScatterPoint = false;
				while (!findScatterPoint && timeout < settings.m_maxTimeout) {
						FindPoints(shortenedP0, pointVoxelGrid, currentEdgeLength,
											 [&](const PointCloudVoxel &voxel) {
												 if (voxel.m_type == PointCloudVoxelType::BranchStart) return;
												 if (voxel.m_type == PointCloudVoxelType::ScatteredPoint) {
														 findScatterPoint = true;
														 for (const auto &i: branch.m_startNeighbors) {
																 if (i == voxel.m_handle) return;
														 }
														 branch.m_startNeighbors.emplace_back(voxel.m_handle);
#ifndef TREEPOINTCLOUD_CLEAN
														 auto &otherPoint = m_scatteredPoints[voxel.m_handle];
														 otherPoint.m_neighborBranchStarts.emplace_back(branch.m_handle);
#endif
														 m_scatterPointToBranchStartConnections.emplace_back(branch.m_bezierCurve.m_p0,
																																								 voxel.m_position);
												 } else {
														 auto dotP = glm::dot(glm::normalize(voxel.m_position - shortenedP0),
																									glm::normalize(shortenedP0 - shortenedP3));
														 if (dotP < glm::cos(glm::radians(settings.m_absoluteAngleLimit))) return;
														 if (glm::distance(voxel.m_position, shortenedP0) >
																 settings.m_forceConnectionRatio * branchLength &&
																 dotP < glm::cos(glm::radians(settings.m_forceConnectionAngleLimit)))
																 return;
														 for (const auto &i: branch.m_neighborBranchEnds) {
																 if (i == voxel.m_handle) return;
														 }
														 auto &otherBranch = m_scannedBranches[voxel.m_handle];
														 branch.m_neighborBranchEnds.emplace_back(voxel.m_handle);
#ifndef TREEPOINTCLOUD_CLEAN
														 otherBranch.m_neighborBranchStarts.emplace_back(branch.m_handle);
#endif
														 m_branchConnections.emplace_back(branch.m_bezierCurve.m_p0,
																															otherBranch.m_bezierCurve.m_p3);
												 }
											 });
						currentEdgeLength += settings.m_edgeExtendStep;
						timeout++;
				}

				currentEdgeLength = settings.m_edgeLength;
				timeout = 0;
				findScatterPoint = false;
				while (!findScatterPoint && timeout < settings.m_maxTimeout) {
						FindPoints(shortenedP3, pointVoxelGrid, currentEdgeLength,
											 [&](const PointCloudVoxel &voxel) {
												 if (voxel.m_type == PointCloudVoxelType::BranchEnd) return;
												 if (voxel.m_type == PointCloudVoxelType::ScatteredPoint) {
														 findScatterPoint = true;
														 auto &otherPoint = m_scatteredPoints[voxel.m_handle];
														 for (const auto &i: otherPoint.m_neighborBranchEnds) {
																 if (i == branch.m_handle) return;
														 }
#ifndef TREEPOINTCLOUD_CLEAN
														 branch.m_endNeighbors.emplace_back(voxel.m_handle);
#endif
														 otherPoint.m_neighborBranchEnds.emplace_back(branch.m_handle);
														 m_scatterPointToBranchEndConnections.emplace_back(branch.m_bezierCurve.m_p3,
																																							 voxel.m_position);
												 } else {
														 auto dotP = glm::dot(glm::normalize(voxel.m_position - shortenedP3),
																									glm::normalize(shortenedP3 - shortenedP0));
														 if (dotP < glm::cos(glm::radians(settings.m_absoluteAngleLimit)))
																 return;
														 if (glm::distance(voxel.m_position, shortenedP3) >
																 settings.m_forceConnectionRatio * branchLength &&
																 dotP < glm::cos(glm::radians(settings.m_forceConnectionAngleLimit)))
																 return;
														 auto &otherBranch = m_scannedBranches[voxel.m_handle];
														 for (const auto &i: otherBranch.m_neighborBranchEnds) {
																 if (i == branch.m_handle) return;
														 }
#ifndef TREEPOINTCLOUD_CLEAN
														 branch.m_neighborBranchStarts.emplace_back(voxel.m_handle);
#endif
														 otherBranch.m_neighborBranchEnds.emplace_back(branch.m_handle);
														 m_branchConnections.emplace_back(branch.m_bezierCurve.m_p3,
																															otherBranch.m_bezierCurve.m_p0);
												 }
											 });
						currentEdgeLength += settings.m_edgeExtendStep;
						timeout++;
				}
		}
		for (auto &branch: m_scannedBranches) {
				std::unordered_set<PointHandle> visitedPoints;
				std::vector<PointHandle> processingPoints = branch.m_startNeighbors;
				for (const auto &i: processingPoints) {
						visitedPoints.emplace(i);
				}
				while (!processingPoints.empty()) {
						auto currentPointHandle = processingPoints.back();
						visitedPoints.emplace(currentPointHandle);
						processingPoints.pop_back();
						auto &currentPoint = m_scatteredPoints[currentPointHandle];
						for (const auto &neighborHandle: currentPoint.m_neighbors) {
								if (visitedPoints.find(neighborHandle) != visitedPoints.end()) continue;
								auto &neighbor = m_scatteredPoints[neighborHandle];
								//We stop search if the point is junction point.
								for (const auto &branchEndHandle: neighbor.m_neighborBranchEnds) {
										bool skip = false;
										for (const auto &i: branch.m_neighborBranchEnds) if (branchEndHandle == i) skip = true;
										auto &otherBranch = m_scannedBranches[branchEndHandle];
										auto shortenedP0 = branch.m_bezierCurve.GetPoint(settings.m_branchShortening);
										auto shortenedP3 = branch.m_bezierCurve.GetPoint(1.0f - settings.m_branchShortening);
										float branchLength = glm::distance(shortenedP0, shortenedP3);
										auto otherBranchShortenedP3 = otherBranch.m_bezierCurve.GetPoint(
														1.0f - settings.m_branchShortening);
										auto dotP = glm::dot(glm::normalize(otherBranchShortenedP3 - shortenedP0),
																				 glm::normalize(shortenedP0 - shortenedP3));
										if (dotP < glm::cos(glm::radians(settings.m_absoluteAngleLimit))) skip = true;
										if (!skip) {
												branch.m_neighborBranchEnds.emplace_back(branchEndHandle);
#ifndef TREEPOINTCLOUD_CLEAN
												otherBranch.m_neighborBranchStarts.emplace_back(branch.m_handle);
#endif
												m_branchConnections.emplace_back(branch.m_bezierCurve.m_p0, otherBranch.m_bezierCurve.m_p3);
										}
								}
								processingPoints.emplace_back(neighborHandle);
						}
				}
		}
		for (auto &branch: m_scannedBranches) {
				//Detect connection to branch start.
				if (branch.m_neighborBranchEnds.empty()) {
						continue;
				}
				std::unordered_set<BranchHandle> availableCandidates;
				for (const auto &branchEndHandle: branch.m_neighborBranchEnds) {
						availableCandidates.emplace(branchEndHandle);
				}
				float minDistance = 999.0f;
				BranchHandle bestCandidate;
				for (const auto &candidateHandle: availableCandidates) {
						auto &candidate = m_scannedBranches[candidateHandle];
						auto shortenedP0 = branch.m_bezierCurve.GetPoint(settings.m_branchShortening);
						auto shortenedP3 = candidate.m_bezierCurve.GetPoint(1.0f - settings.m_branchShortening);
						auto distance = glm::distance(branch.m_bezierCurve.m_p0, candidate.m_bezierCurve.m_p3);
						if (distance < minDistance) {
								minDistance = distance;
								bestCandidate = candidateHandle;
						}
				}
				m_scannedBranches[bestCandidate].m_childHandles.emplace_back(branch.m_handle);
				branch.m_parentHandle = bestCandidate;
				m_filteredBranchConnections.emplace_back(branch.m_bezierCurve.m_p0,
																								 m_scannedBranches[bestCandidate].m_bezierCurve.m_p3);
		}

}

void ApplyCurve(ReconstructionSkeleton &skeleton, OperatingBranch &branch) {
		const auto chainAmount = branch.m_chainNodeHandles.size();
		for (int i = 0; i < chainAmount; i++) {
				auto &node = skeleton.RefNode(branch.m_chainNodeHandles[i]);
				node.m_info.m_globalPosition = branch.m_bezierCurve.GetPoint(
								static_cast<float>(i) / chainAmount);
				node.m_info.m_length = glm::distance(
								branch.m_bezierCurve.GetPoint(static_cast<float>(i) / chainAmount),
								branch.m_bezierCurve.GetPoint(static_cast<float>(i + 1) / chainAmount));
				node.m_info.m_localPosition = glm::normalize(
								branch.m_bezierCurve.GetPoint(static_cast<float>(i + 1) / chainAmount) -
								branch.m_bezierCurve.GetPoint(static_cast<float>(i) / chainAmount));
				node.m_info.m_thickness = glm::mix(branch.m_startThickness, branch.m_endThickness,
																					 static_cast<float>(i) / chainAmount);
				node.m_data.m_branchHandle = branch.m_handle;
		}
}

void TreePointCloud::BuildTreeStructure(const ReconstructionSettings &reconstructionSettings) {
		m_skeletons.clear();
		std::unordered_set<BranchHandle> allocatedBranchHandles;

		m_operatingBranches.resize(m_scannedBranches.size());
		for (int i = 0; i < m_scannedBranches.size(); i++) {
				m_operatingBranches[i].Apply(m_scannedBranches[i]);
		}

		auto &operatingBranches = m_operatingBranches;
		std::vector<BranchHandle> rootBranchHandles;
		for (auto &branch: operatingBranches) {
				branch.m_chainNodeHandles.clear();
				const auto branchStart = branch.m_bezierCurve.m_p0;
				if (branchStart.y < reconstructionSettings.m_minHeight) {
						bool replaced = false;
						for (int i = 0; i < rootBranchHandles.size(); i++) {
								const auto &rootBranchStart = operatingBranches[rootBranchHandles[i]].m_bezierCurve.m_p0;
								if (glm::distance(glm::vec2(rootBranchStart.x, rootBranchStart.z),
																	glm::vec2(branchStart.x, branchStart.z)) < reconstructionSettings.m_maxTreeDistance
										&& rootBranchStart.y > branchStart.y) {
										replaced = true;
										rootBranchHandles[i] = branch.m_handle;
								}
						}
						if (!replaced) rootBranchHandles.emplace_back(branch.m_handle);
				} else {
						auto shortenedP0 = branch.m_bezierCurve.GetPoint(reconstructionSettings.m_branchShortening);
						auto shortenedP3 = branch.m_bezierCurve.GetPoint(1.0f - reconstructionSettings.m_branchShortening);
						auto shortenedLength = glm::distance(shortenedP0, shortenedP3);
						branch.m_bezierCurve.m_p1 = shortenedP0 +
																				branch.m_bezierCurve.GetAxis(reconstructionSettings.m_branchShortening) *
																				shortenedLength * 0.25f;
						branch.m_bezierCurve.m_p2 = shortenedP3 -
																				branch.m_bezierCurve.GetAxis(1.0f - reconstructionSettings.m_branchShortening) *
																				shortenedLength * 0.25f;
						branch.m_bezierCurve.m_p0 = shortenedP0;
						branch.m_bezierCurve.m_p3 = shortenedP3;
				}
		}
		for (const auto &rootBranchHandle: rootBranchHandles) {
				allocatedBranchHandles.emplace(rootBranchHandle);
		}
		for (const auto &rootBranchHandle: rootBranchHandles) {
				auto &skeleton = m_skeletons.emplace_back();
				std::queue<BranchHandle> processingBranchHandles;
				processingBranchHandles.emplace(rootBranchHandle);
				while (!processingBranchHandles.empty()) {
						auto processingBranchHandle = processingBranchHandles.front();
						processingBranchHandles.pop();
						for (const auto &childBranchHandle: operatingBranches[processingBranchHandle].m_childHandles) {
								if (allocatedBranchHandles.find(childBranchHandle) != allocatedBranchHandles.end()) continue;
								processingBranchHandles.emplace(childBranchHandle);
						}
						bool onlyChild = true;
						NodeHandle prevNodeHandle = -1;
						if (operatingBranches[processingBranchHandle].m_handle == rootBranchHandle) {
								prevNodeHandle = 0;
								operatingBranches[processingBranchHandle].m_chainNodeHandles.emplace_back(0);
						} else {
								operatingBranches.emplace_back();
								auto &processingBranch = operatingBranches[processingBranchHandle];
								auto &connectionBranch = operatingBranches.back();
								auto &parentBranch = operatingBranches[processingBranch.m_parentHandle];
								connectionBranch.m_handle = operatingBranches.size() - 1;
								connectionBranch.m_bezierCurve.m_p0 = parentBranch.m_bezierCurve.m_p3;
								connectionBranch.m_bezierCurve.m_p3 = processingBranch.m_bezierCurve.m_p0;
								auto connectionLength = glm::distance(connectionBranch.m_bezierCurve.m_p0,
																											connectionBranch.m_bezierCurve.m_p3);
								/*
								connectionBranch.m_bezierCurve.m_p1 = parentBranch.m_bezierCurve.m_p3 +
																											(parentBranch.m_bezierCurve.m_p3 -
																											 parentBranch.m_bezierCurve.m_p2) * 0.1f * connectionLength;
								connectionBranch.m_bezierCurve.m_p2 = processingBranch.m_bezierCurve.m_p0 +
																											(processingBranch.m_bezierCurve.m_p0 -
																											 processingBranch.m_bezierCurve.m_p1) * 0.1f * connectionLength;
								*/
								connectionBranch.m_bezierCurve.m_p1 = glm::mix(connectionBranch.m_bezierCurve.m_p0,
																															 connectionBranch.m_bezierCurve.m_p3, 0.25f);
								connectionBranch.m_bezierCurve.m_p2 = glm::mix(connectionBranch.m_bezierCurve.m_p0,
																															 connectionBranch.m_bezierCurve.m_p3, 0.75f);
								connectionBranch.m_startThickness = parentBranch.m_endThickness;
								connectionBranch.m_endThickness = processingBranch.m_startThickness;
								connectionBranch.m_childHandles.emplace_back(processingBranch.m_handle);
								connectionBranch.m_parentHandle = processingBranch.m_parentHandle;
								processingBranch.m_parentHandle = connectionBranch.m_handle;
								for (int i = 0; i < parentBranch.m_childHandles.size(); i++) {
										if (parentBranch.m_childHandles[i] == processingBranch.m_handle) {
												parentBranch.m_childHandles[i] = connectionBranch.m_handle;
												break;
										}
								}
								if (parentBranch.m_childHandles.size() > 1) onlyChild = false;
								prevNodeHandle = parentBranch.m_chainNodeHandles.back();
								auto connectionFirstNodeHandle = skeleton.Extend(prevNodeHandle, !onlyChild);
								connectionBranch.m_chainNodeHandles.emplace_back(connectionFirstNodeHandle);
								prevNodeHandle = connectionFirstNodeHandle;

								float connectionChainLength = connectionBranch.m_bezierCurve.GetLength();
								int connectionChainAmount = glm::max(2, (int) (connectionChainLength /
																															 reconstructionSettings.m_internodeLength));
								for (int i = 1; i < connectionChainAmount; i++) {
										auto newNodeHandle = skeleton.Extend(prevNodeHandle, false);
										connectionBranch.m_chainNodeHandles.emplace_back(newNodeHandle);
										prevNodeHandle = newNodeHandle;
								}
								ApplyCurve(skeleton, connectionBranch);
								auto chainFirstNodeHandle = skeleton.Extend(prevNodeHandle, false);
								processingBranch.m_chainNodeHandles.emplace_back(chainFirstNodeHandle);
								prevNodeHandle = chainFirstNodeHandle;
						}
						auto &processingBranch = operatingBranches[processingBranchHandle];
						processingBranch.m_skeletonIndex = m_skeletons.size() - 1;
						float chainLength = processingBranch.m_bezierCurve.GetLength();
						int chainAmount = glm::max(2, (int) (chainLength /
																								 reconstructionSettings.m_internodeLength));
						for (int i = 1; i < chainAmount; i++) {
								auto newNodeHandle = skeleton.Extend(prevNodeHandle, false);
								processingBranch.m_chainNodeHandles.emplace_back(newNodeHandle);
								prevNodeHandle = newNodeHandle;
						}
						ApplyCurve(skeleton, processingBranch);

				}
				skeleton.SortLists();
				auto &sortedNodeList = skeleton.RefSortedNodeList();
				auto &rootNode = skeleton.RefNode(0);
				rootNode.m_info.m_localRotation = glm::vec3(0.0f);
				rootNode.m_info.m_globalRotation = rootNode.m_info.m_regulatedGlobalRotation = glm::quatLookAt(
								rootNode.m_info.m_localPosition, glm::vec3(0, 0, -1));
				rootNode.m_info.m_localPosition = glm::vec3(0.0f);

				for (const auto &nodeHandle: sortedNodeList) {
						if (nodeHandle == 0) continue;
						auto &node = skeleton.RefNode(nodeHandle);
						auto &nodeInfo = node.m_info;
						auto &parentNode = skeleton.RefNode(node.GetParentHandle());
						auto front = glm::normalize(nodeInfo.m_localPosition);
						nodeInfo.m_localPosition = glm::vec3(0.0f);
						auto parentUp = parentNode.m_info.m_globalRotation * glm::vec3(0, 1, 0);
						auto regulatedUp = glm::normalize(glm::cross(glm::cross(front, parentUp), front));
						nodeInfo.m_globalRotation = glm::quatLookAt(front, regulatedUp);
						nodeInfo.m_localRotation = glm::inverse(parentNode.m_info.m_globalRotation) * nodeInfo.m_globalRotation;
				}

				for (auto i = sortedNodeList.rbegin(); i != sortedNodeList.rend(); i++) {
						auto &node = skeleton.RefNode(*i);
						auto &nodeInfo = node.m_info;
						for (const auto &childHandle: node.RefChildHandles()) {
								const auto &childNode = skeleton.RefNode(childHandle);
								nodeInfo.m_thickness = glm::max(childNode.m_info.m_thickness, nodeInfo.m_thickness);
						}
				}
				skeleton.CalculateTransforms();
				skeleton.CalculateFlows();
		}


		for (auto &allocatedPoint: m_allocatedPoints) {
				const auto &treePart = m_treeParts[allocatedPoint.m_treePartHandle];
				float minDistance = 999.f;
				NodeHandle closestNodeHandle = -1;
				BranchHandle cloeseBranchHandle = -1;
				int closestSkeletonIndex = -1;
				for (const auto &branchHandle: treePart.m_branchHandles) {
						auto &branch = operatingBranches[branchHandle];
						for (const auto &nodeHandle: branch.m_chainNodeHandles) {
								auto &node = m_skeletons[branch.m_skeletonIndex].RefNode(nodeHandle);
								auto distance = glm::distance(node.m_info.m_globalPosition, allocatedPoint.m_position);
								if (distance < minDistance) {
										minDistance = distance;
										closestNodeHandle = nodeHandle;
										closestSkeletonIndex = branch.m_skeletonIndex;
										cloeseBranchHandle = branchHandle;
								}
						}
				}
				allocatedPoint.m_nodeHandle = closestNodeHandle;
				allocatedPoint.m_branchHandle = cloeseBranchHandle;
				allocatedPoint.m_skeletonIndex = closestSkeletonIndex;
				if (allocatedPoint.m_skeletonIndex != -1)
						m_skeletons[allocatedPoint.m_skeletonIndex].RefNode(
										closestNodeHandle).m_data.m_allocatedPoints.emplace_back(allocatedPoint.m_handle);
		}
}

void TreePointCloud::ClearMeshes() const {
		const auto scene = GetScene();
		const auto self = GetOwner();
		const auto children = scene->GetChildren(self);
		for (const auto &child: children) {
				auto name = scene->GetEntityName(child);
				if (name.substr(0, 4) == "Tree") {
						scene->DeleteEntity(child);
				}
		}
}

void TreePointCloud::GenerateMeshes(const TreeMeshGeneratorSettings &meshGeneratorSettings) {
		const auto scene = GetScene();
		const auto self = GetOwner();
		const auto children = scene->GetChildren(self);

		ClearMeshes();
		for (int i = 0; i < m_skeletons.size(); i++) {
				Entity treeEntity;
				treeEntity = scene->CreateEntity("Tree " + std::to_string(i));
				scene->SetParent(treeEntity, self);
				if (meshGeneratorSettings.m_enableBranch) {
						Entity branchEntity;
						branchEntity = scene->CreateEntity("Branch Mesh");
						scene->SetParent(branchEntity, treeEntity);

						std::vector<Vertex> vertices;
						std::vector<unsigned int> indices;
						CylindricalMeshGenerator<ReconstructionSkeletonData, ReconstructionFlowData, ReconstructionNodeData> meshGenerator;
						meshGenerator.Generate(m_skeletons[i], vertices, indices, meshGeneratorSettings, 999.0f);

						auto mesh = ProjectManager::CreateTemporaryAsset<Mesh>();
						auto material = ProjectManager::CreateTemporaryAsset<Material>();
						material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
						mesh->SetVertices(17, vertices, indices);
						auto meshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(branchEntity).lock();
						if (meshGeneratorSettings.m_overridePresentation) {
								material->m_materialProperties.m_albedoColor = meshGeneratorSettings.m_presentationOverrideSettings.m_branchOverrideColor;
						} else {
								material->m_materialProperties.m_albedoColor = glm::vec3(109, 79, 75) / 255.0f;
						}
						material->m_materialProperties.m_roughness = 1.0f;
						material->m_materialProperties.m_metallic = 0.0f;
						meshRenderer->m_mesh = mesh;
						meshRenderer->m_material = material;
				}
		}
}

void ConnectivityGraphSettings::OnInspect() {
	ImGui::DragFloat("Scatter point connection length", &m_scatterPointConnectionMaxLength, 0.01f, 0.01f, 1.0f);
		ImGui::DragInt("Branch finder timeout", &m_maxTimeout, 1, 1, 30);
		ImGui::DragFloat("Branch finder length", &m_edgeLength, 0.01f, 0.01f, 1.0f);
		ImGui::DragFloat("Branch finder step", &m_edgeExtendStep, 0.01f, 0.01f, 1.0f);
		ImGui::DragFloat("Absolute angle limit", &m_absoluteAngleLimit, 0.01f, 0.0f, 180.0f);
		ImGui::DragFloat("Branch shortening", &m_branchShortening, 0.01f, 0.00f, 0.5f);
		ImGui::DragFloat("Force connection ratio", &m_forceConnectionRatio, 0.01f, 0.01f, 1.0f);
		if (m_forceConnectionRatio > 0.0f) {
				ImGui::DragFloat("Force connection angle limit", &m_forceConnectionAngleLimit, 0.01f, 0.01f, 90.0f);
		}
}

void OperatingBranch::Apply(const ScannedBranch &target) {
		m_treePartHandle = target.m_treePartHandle;
		m_handle = target.m_handle;
		m_bezierCurve = target.m_bezierCurve;
		m_startThickness = target.m_startThickness;
		m_endThickness = target.m_endThickness;
		m_parentHandle = target.m_parentHandle;
		m_childHandles = target.m_childHandles;
		m_skeletonIndex = -1;
		m_chainNodeHandles.clear();
}

void ReconstructionSettings::OnInspect() {
		ImGui::DragFloat("Internode length", &m_internodeLength, 0.01f, 0.01f, 1.0f);
		ImGui::DragFloat("Root node max height", &m_minHeight, 0.01f, 0.01f, 1.0f);
		ImGui::DragFloat("Tree distance limit", &m_maxTreeDistance, 0.01f, 0.01f, 1.0f);
		ImGui::DragFloat("Branch shortening", &m_branchShortening, 0.01f, 0.01f, 0.5f);
}
