//
// Created by lllll on 11/20/2022.
//

#include "TreeVisualizer.hpp"
#include "Utilities.hpp"
#include "Application.hpp"
#include "EcoSysLabLayer.hpp"
using namespace EcoSysLab;

bool TreeVisualizer::DrawInternodeInspectionGui(
				TreeModel &treeModel,
				NodeHandle internodeHandle,
				bool &deleted,
				const unsigned &hierarchyLevel) {
		auto &treeSkeleton = treeModel.RefShootSkeleton();
		const int index = m_selectedInternodeHierarchyList.size() - hierarchyLevel - 1;
		if (!m_selectedInternodeHierarchyList.empty() && index >= 0 &&
				index < m_selectedInternodeHierarchyList.size() &&
				m_selectedInternodeHierarchyList[index] == internodeHandle) {
				ImGui::SetNextItemOpen(true);
		}
		const bool opened = ImGui::TreeNodeEx(("Handle: " + std::to_string(internodeHandle)).c_str(),
																					ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_OpenOnArrow |
																					ImGuiTreeNodeFlags_NoAutoOpenOnLog |
																					(m_selectedInternodeHandle == internodeHandle ? ImGuiTreeNodeFlags_Framed
																																												: ImGuiTreeNodeFlags_FramePadding));
		if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
				SetSelectedNode(treeSkeleton, internodeHandle, m_selectedInternodeHandle, m_selectedInternodeHierarchyList);
		}

		if (ImGui::BeginPopupContextItem(std::to_string(internodeHandle).c_str())) {
				ImGui::Text(("Handle: " + std::to_string(internodeHandle)).c_str());
				if (ImGui::Button("Delete")) {
						deleted = true;
				}
				ImGui::EndPopup();
		}
		bool modified = deleted;
		if (opened && !deleted) {
				ImGui::TreePush();
				const auto &internodeChildren = treeSkeleton.RefNode(internodeHandle).RefChildHandles();
				for (const auto &child: internodeChildren) {
						bool childDeleted = false;
						DrawInternodeInspectionGui(treeModel, child, childDeleted, hierarchyLevel + 1);
						if (childDeleted) {
								treeModel.Step();
								treeModel.PruneInternode(child);

								treeSkeleton.SortLists();
								m_iteration = treeModel.CurrentIteration();
								modified = true;
								break;
						}
				}
				ImGui::TreePop();
		}
		return modified;
}

bool
TreeVisualizer::OnInspect(
				TreeModel &treeModel,
				PipeModel &pipeModel,
				const GlobalTransform &globalTransform) {
		bool updated = false;
		if (ImGui::Begin("Tree Inspector")) {
				if (ImGui::Combo("Shoot Color mode",
												 {"Default", "LightIntensity", "LightDirection", "IsMaxChild", "AllocatedVigor"},
												 m_settings.m_shootVisualizationMode)) {
						m_needShootColorUpdate = true;
				}
				if (ImGui::Combo("Root Color mode", {"Default", "AllocatedVigor"}, m_settings.m_rootVisualizationMode)) {
						m_needRootColorUpdate = true;
				}
				if (ImGui::TreeNode("Shoot Color settings")) {
						switch (static_cast<ShootVisualizerMode>(m_settings.m_shootVisualizationMode)) {
								case ShootVisualizerMode::LightIntensity:
										ImGui::DragFloat("Light intensity multiplier", &m_settings.m_shootColorMultiplier, 0.001f);
										m_needShootColorUpdate = true;
										break;
								case ShootVisualizerMode::AllocatedVigor:
										ImGui::DragFloat("Vigor multiplier", &m_settings.m_shootColorMultiplier, 0.001f);
										m_needShootColorUpdate = true;
										break;
								default:
										break;
						}
						ImGui::TreePop();
				}
				if (ImGui::TreeNode("Root Color settings")) {
						switch (static_cast<RootVisualizerMode>(m_settings.m_rootVisualizationMode)) {
								case RootVisualizerMode::AllocatedVigor:
										ImGui::DragFloat("Vigor multiplier", &m_settings.m_rootColorMultiplier, 0.001f);
										m_needRootColorUpdate = true;
										break;
								default:
										break;
						}
						ImGui::TreePop();
				}
				if (treeModel.CurrentIteration() > 0) {
						if (ImGui::TreeNodeEx("History", ImGuiTreeNodeFlags_DefaultOpen)) {
								ImGui::DragInt("History Limit", &treeModel.m_historyLimit, 1, -1, 1024);
								if (ImGui::SliderInt("Iteration", &m_iteration, 0, treeModel.CurrentIteration())) {
										m_iteration = glm::clamp(m_iteration, 0, treeModel.CurrentIteration());
										m_selectedInternodeHandle = -1;
										m_selectedInternodeHierarchyList.clear();
										m_selectedRootNodeHandle = -1;
										m_selectedRootNodeHierarchyList.clear();
										m_needUpdate = true;
								}
								if (m_iteration != treeModel.CurrentIteration() && ImGui::Button("Reverse")) {
										treeModel.Reverse(m_iteration);
										m_needUpdate = true;
								}
								if (ImGui::Button("Clear history")) {
										m_iteration = 0;
										treeModel.ClearHistory();
								}
								ImGui::TreePop();
						}
				}

				if (ImGui::TreeNodeEx("Settings")) {
						ImGui::Checkbox("Visualization", &m_visualization);
						ImGui::Checkbox("Hexagon grid", &m_hexagonGridGui);
						ImGui::Checkbox("Tree Hierarchy", &m_treeHierarchyGui);
						ImGui::Checkbox("Root Hierarchy", &m_rootHierarchyGui);
						ImGui::TreePop();
				}
				if (m_selectedInternodeHandle >= 0) {
						if (m_iteration == treeModel.CurrentIteration()) {
								InspectInternode(treeModel.RefShootSkeleton(), m_selectedInternodeHandle);
						} else {
								PeekInternode(treeModel.PeekShootSkeleton(m_iteration), m_selectedInternodeHandle);
						}
				}
				if (m_selectedRootNodeHandle >= 0) {
						if (m_iteration == treeModel.CurrentIteration()) {
								InspectRootNode(treeModel.RefRootSkeleton(), m_selectedRootNodeHandle);
						} else {
								PeekRootNode(treeModel.PeekRootSkeleton(m_iteration), m_selectedRootNodeHandle);
						}
				}


				if (m_visualization) {
						const auto &treeSkeleton = treeModel.PeekShootSkeleton(m_iteration);
						const auto &rootSkeleton = treeModel.PeekRootSkeleton(m_iteration);
						const auto editorLayer = Application::GetLayer<EditorLayer>();
						const auto &sortedBranchList = treeSkeleton.RefSortedFlowList();
						const auto &sortedInternodeList = treeSkeleton.RefSortedNodeList();
						ImGui::Text("Internode count: %d", sortedInternodeList.size());
						ImGui::Text("Shoot stem count: %d", sortedBranchList.size());

						const auto &sortedRootFlowList = rootSkeleton.RefSortedFlowList();
						const auto &sortedRootNodeList = rootSkeleton.RefSortedNodeList();
						ImGui::Text("Root node count: %d", sortedRootNodeList.size());
						ImGui::Text("Root stem count: %d", sortedRootFlowList.size());

						static bool enableStroke = false;
						if (ImGui::Checkbox("Enable stroke", &enableStroke)) {
								if (enableStroke) {
										m_mode = PruningMode::Stroke;
								} else {
										m_mode = PruningMode::None;
								}
								m_storedMousePositions.clear();
						}
				}

				if (m_treeHierarchyGui) {
						if (ImGui::TreeNodeEx("Tree Hierarchy")) {
								bool deleted = false;
								auto tempSelection = m_selectedInternodeHandle;
								if (m_iteration == treeModel.CurrentIteration()) {
										if (DrawInternodeInspectionGui(treeModel, 0, deleted, 0)) {
												m_needUpdate = true;
												updated = true;
										}
								} else
										PeekNodeInspectionGui(treeModel.PeekShootSkeleton(m_iteration), 0, m_selectedInternodeHandle,
																					m_selectedInternodeHierarchyList, 0);
								m_selectedInternodeHierarchyList.clear();
								if (tempSelection != m_selectedInternodeHandle) {
										m_selectedRootNodeHandle = -1;
										m_selectedRootNodeHierarchyList.clear();
								}
								ImGui::TreePop();
						}

				}
				if (m_rootHierarchyGui) {
						if (ImGui::TreeNodeEx("Root Hierarchy")) {
								bool deleted = false;
								const auto tempSelection = m_selectedRootNodeHandle;
								if (m_iteration == treeModel.CurrentIteration()) {
										if (DrawRootNodeInspectionGui(treeModel, 0, deleted, 0)) {
												m_needUpdate = true;
												updated = true;
										}
								} else
										PeekNodeInspectionGui(treeModel.PeekRootSkeleton(m_iteration), 0, m_selectedRootNodeHandle,
																					m_selectedRootNodeHierarchyList, 0);
								m_selectedRootNodeHierarchyList.clear();
								if (tempSelection != m_selectedRootNodeHandle) {
										m_selectedInternodeHandle = -1;
										m_selectedInternodeHierarchyList.clear();
								}
								ImGui::TreePop();
						}

				}
		}
		ImGui::End();
		if (m_hexagonGridGui) {
				if (ImGui::Begin("Hexagon Grid")) {
						/*
						if (m_selectedInternodeHandle >= 0 && pipeModel.m_shootSkeleton.RefSortedNodeList().size() > m_selectedInternodeHandle)
						{
							auto& shootSkeleton = pipeModel.m_shootSkeleton;
							//auto& grid = shootSkeleton.m_data.m_hexagonGridGroup.RefGrid(shootSkeleton.RefNode(pipeModel.m_shootSkeletonLinks[m_selectedInternodeHandle]).m_data.m_gridHandle);
							//VisualizeGrid(shootSkeleton, grid);
						}else if (m_selectedRootNodeHandle >= 0 && pipeModel.m_rootSkeleton.RefSortedNodeList().size() > m_selectedRootNodeHandle)
						{
							auto& rootSkeleton = pipeModel.m_rootSkeleton;
							//auto& grid = rootSkeleton.m_data.m_hexagonGridGroup.RefGrid(rootSkeleton.RefNode(pipeModel.m_rootSkeletonLinks[m_selectedRootNodeHandle]).m_data.m_gridHandle);
							//VisualizeGrid(rootSkeleton, grid);
						}
						else
						{
							ImGui::Text("No node selected or pipe model is invalid!");
						}
						*/
				}
				ImGui::End();
		}
		return updated;
}

bool TreeVisualizer::Visualize(TreeModel &treeModel,
															 const GlobalTransform &globalTransform) {
		bool updated = false;
		auto ecoSysLabLayer = Application::GetLayer<EcoSysLabLayer>();
		const auto &treeSkeleton = treeModel.PeekShootSkeleton(m_iteration);
		const auto &rootSkeleton = treeModel.PeekRootSkeleton(m_iteration);
		if (m_visualization) {
				const auto editorLayer = Application::GetLayer<EditorLayer>();
				if(ecoSysLabLayer->m_visualizationCameraWindowFocused) {
						switch (m_mode) {
								case PruningMode::None: {
										if (Inputs::GetMouseInternal(GLFW_MOUSE_BUTTON_LEFT, Windows::GetWindow())) {
												if (RayCastSelection(ecoSysLabLayer->m_visualizationCamera, ecoSysLabLayer->m_visualizationCameraMousePosition, treeSkeleton, globalTransform, m_selectedInternodeHandle,
																						 m_selectedInternodeHierarchyList, m_selectedInternodeLengthFactor)) {
														m_needUpdate = true;
														updated = true;
														m_selectedRootNodeHandle = -1;
														m_selectedRootNodeHierarchyList.clear();
												} else if (RayCastSelection(ecoSysLabLayer->m_visualizationCamera, ecoSysLabLayer->m_visualizationCameraMousePosition, rootSkeleton, globalTransform, m_selectedRootNodeHandle,
																										m_selectedRootNodeHierarchyList, m_selectedRootNodeLengthFactor)) {
														m_needUpdate = true;
														updated = true;
														m_selectedInternodeHandle = -1;
														m_selectedInternodeHierarchyList.clear();
												}
										}
										if (m_iteration == treeModel.CurrentIteration() &&
												Inputs::GetKeyInternal(GLFW_KEY_DELETE,
																							 Windows::GetWindow())) {
												if (m_selectedInternodeHandle > 0) {
														treeModel.Step();
														auto &skeleton = treeModel.RefShootSkeleton();
														auto &pruningInternode = skeleton.RefNode(m_selectedInternodeHandle);
														auto childHandles = pruningInternode.RefChildHandles();
														for (const auto &childHandle: childHandles) {
																treeModel.PruneInternode(childHandle);
														}
														pruningInternode.m_info.m_length *= m_selectedInternodeLengthFactor;
														m_selectedInternodeLengthFactor = 1.0f;
														for (auto &bud: pruningInternode.m_data.m_buds) {
																bud.m_status = BudStatus::Died;
														}
														skeleton.SortLists();
														m_iteration = treeModel.CurrentIteration();
														m_needUpdate = true;
														updated = true;
												}
												if (m_selectedRootNodeHandle > 0) {
														treeModel.Step();
														auto &skeleton = treeModel.RefRootSkeleton();
														auto &pruningRootNode = skeleton.RefNode(m_selectedRootNodeHandle);
														auto childHandles = pruningRootNode.RefChildHandles();
														for (const auto &childHandle: childHandles) {
																treeModel.PruneRootNode(childHandle);
														}
														pruningRootNode.m_info.m_length *= m_selectedRootNodeLengthFactor;
														m_selectedRootNodeLengthFactor = 1.0f;
														skeleton.SortLists();
														m_iteration = treeModel.CurrentIteration();
														m_needUpdate = true;
														updated = true;
												}
										}
								}
										break;
								case PruningMode::Stroke: {
										if (m_iteration == treeModel.CurrentIteration()) {
												if (Inputs::GetMouseInternal(GLFW_MOUSE_BUTTON_LEFT, Windows::GetWindow())) {
														glm::vec2 mousePosition = editorLayer->GetMouseScreenPosition();
														const float halfX = editorLayer->m_sceneCamera->GetResolution().x / 2.0f;
														const float halfY = editorLayer->m_sceneCamera->GetResolution().y / 2.0f;
														mousePosition = {-1.0f * (mousePosition.x - halfX) / halfX,
																						 -1.0f * (mousePosition.y - halfY) / halfY};
														if (mousePosition.x > -1.0f && mousePosition.x < 1.0f && mousePosition.y > -1.0f &&
																mousePosition.y < 1.0f &&
																(m_storedMousePositions.empty() || mousePosition != m_storedMousePositions.back())) {
																m_storedMousePositions.emplace_back(mousePosition);
														}
												} else {
														//Once released, check if empty.
														if (!m_storedMousePositions.empty()) {
																treeModel.Step();
																auto &skeleton = treeModel.RefShootSkeleton();
																bool changed = ScreenCurvePruning(
																				[&](NodeHandle nodeHandle) { treeModel.PruneInternode(nodeHandle); }, skeleton,
																				globalTransform, m_selectedInternodeHandle, m_selectedInternodeHierarchyList);
																if (changed) {
																		skeleton.SortLists();
																		m_iteration = treeModel.CurrentIteration();
																		m_needUpdate = true;
																		updated = true;
																} else {
																		treeModel.Pop();
																}
																m_storedMousePositions.clear();
														}
												}
										}
								}
										break;
						}
				}

				if (m_needUpdate) {
						SyncMatrices(treeSkeleton, m_internodeMatrices, m_internodeColors, m_selectedInternodeHandle,
												 m_selectedInternodeLengthFactor);
						SyncMatrices(rootSkeleton, m_rootNodeMatrices, m_rootNodeColors, m_selectedRootNodeHandle,
												 m_selectedRootNodeLengthFactor);
						SyncColors(treeSkeleton, m_selectedInternodeHandle);
						SyncColors(rootSkeleton, m_selectedRootNodeHandle);
						m_needUpdate = false;
				} else {
						if (m_needShootColorUpdate) {
								SyncColors(treeSkeleton, m_selectedInternodeHandle);
								m_needShootColorUpdate = false;
						}
						if (m_needRootColorUpdate) {
								SyncColors(rootSkeleton, m_selectedRootNodeHandle);
								m_needRootColorUpdate = false;
						}
				}
				if (!m_internodeMatrices.empty()) {
						GizmoSettings gizmoSettings;
						gizmoSettings.m_drawSettings.m_blending = true;
						if (m_selectedInternodeHandle == -1) {
								m_internodeMatrices[0] = glm::translate(glm::vec3(1.0f)) * glm::scale(glm::vec3(0.0f));
								m_internodeColors[0] = glm::vec4(0.0f);
						}
						Gizmos::DrawGizmoMeshInstancedColored(
										DefaultResources::Primitives::Cylinder, ecoSysLabLayer->m_visualizationCamera,
										editorLayer->m_sceneCameraPosition,
										editorLayer->m_sceneCameraRotation,
										m_internodeColors,
										m_internodeMatrices,
										globalTransform.m_value, 1.0f, gizmoSettings);

				}
				if (!m_rootNodeMatrices.empty()) {
						GizmoSettings gizmoSettings;
						gizmoSettings.m_drawSettings.m_blending = true;
						if (m_selectedRootNodeHandle == -1) {
								m_rootNodeMatrices[0] = glm::translate(glm::vec3(1.0f)) * glm::scale(glm::vec3(0.0f));
								m_rootNodeColors[0] = glm::vec4(0.0f);
						}
						Gizmos::DrawGizmoMeshInstancedColored(
										DefaultResources::Primitives::Cylinder, ecoSysLabLayer->m_visualizationCamera,
										editorLayer->m_sceneCameraPosition,
										editorLayer->m_sceneCameraRotation,
										m_rootNodeColors,
										m_rootNodeMatrices,
										globalTransform.m_value, 1.0f, gizmoSettings);
				}
		}
		return updated;
}

bool
TreeVisualizer::InspectInternode(
				ShootSkeleton &shootSkeleton,
				NodeHandle internodeHandle) {
		bool changed = false;

		const auto &internode = shootSkeleton.RefNode(internodeHandle);
		if (ImGui::TreeNode("Internode info")) {
				ImGui::Checkbox("Is max child", (bool *) &internode.m_data.m_isMaxChild);
				ImGui::Text("Thickness: %.3f", internode.m_info.m_thickness);
				ImGui::Text("Length: %.3f", internode.m_info.m_length);
				ImGui::InputFloat3("Position", (float *) &internode.m_info.m_globalPosition.x, "%.3f",
													 ImGuiInputTextFlags_ReadOnly);
				auto globalRotationAngle = glm::eulerAngles(internode.m_info.m_globalRotation);
				ImGui::InputFloat3("Global rotation", (float *) &globalRotationAngle.x, "%.3f",
													 ImGuiInputTextFlags_ReadOnly);
				auto localRotationAngle = glm::eulerAngles(internode.m_info.m_localRotation);
				ImGui::InputFloat3("Local rotation", (float *) &localRotationAngle.x, "%.3f",
													 ImGuiInputTextFlags_ReadOnly);
				auto &internodeData = internode.m_data;
				ImGui::InputFloat("Start Age", (float *) &internodeData.m_startAge, 1, 100, "%.3f",
													ImGuiInputTextFlags_ReadOnly);
				ImGui::InputFloat("Distance to end", (float *) &internodeData.m_maxDistanceToAnyBranchEnd, 1, 100,
													"%.3f",
													ImGuiInputTextFlags_ReadOnly);
				ImGui::InputFloat("Descendent biomass", (float *) &internodeData.m_descendentTotalBiomass, 1, 100, "%.3f",
													ImGuiInputTextFlags_ReadOnly);
				ImGui::InputFloat("Biomass", (float *) &internodeData.m_biomass, 1, 100, "%.3f",
													ImGuiInputTextFlags_ReadOnly);

				ImGui::InputFloat("Root distance", (float *) &internodeData.m_rootDistance, 1, 100, "%.3f",
													ImGuiInputTextFlags_ReadOnly);
				ImGui::InputFloat3("Light dir", (float *) &internodeData.m_lightDirection.x, "%.3f",
													 ImGuiInputTextFlags_ReadOnly);
				ImGui::InputFloat("Light intensity", (float *) &internodeData.m_lightIntensity, 1, 100, "%.3f",
													ImGuiInputTextFlags_ReadOnly);

				if (ImGui::DragFloat("Sagging", (float *) &internodeData.m_sagging)) {
						changed = true;
				}
				if (ImGui::DragFloat("Extra mass", (float *) &internodeData.m_extraMass)) {
						changed = true;
				}


				if (ImGui::TreeNodeEx("Buds", ImGuiTreeNodeFlags_DefaultOpen)) {
						int index = 1;
						for (auto &bud: internodeData.m_buds) {
								if (ImGui::TreeNode(("Bud " + std::to_string(index)).c_str())) {
										switch (bud.m_type) {
												case BudType::Apical:
														ImGui::Text("Apical");
														break;
												case BudType::Lateral:
														ImGui::Text("Lateral");
														break;
												case BudType::Leaf:
														ImGui::Text("Leaf");
														break;
												case BudType::Fruit:
														ImGui::Text("Fruit");
														break;
										}
										switch (bud.m_status) {
												case BudStatus::Dormant:
														ImGui::Text("Dormant");
														break;
												case BudStatus::Flushed:
														ImGui::Text("Flushed");
														break;
												case BudStatus::Died:
														ImGui::Text("Died");
														break;
										}

										auto budRotationAngle = glm::eulerAngles(bud.m_localRotation);
										ImGui::InputFloat3("Rotation", &budRotationAngle.x, "%.3f", ImGuiInputTextFlags_ReadOnly);
										/*
										ImGui::InputFloat("Base resource requirement", (float *) &bud.m_maintenanceVigorRequirementWeight, 1, 100,
															"%.3f", ImGuiInputTextFlags_ReadOnly);
															*/
										ImGui::TreePop();
								}
								index++;
						}
						ImGui::TreePop();
				}
				ImGui::TreePop();
		}
		if (ImGui::TreeNodeEx("Flow info", ImGuiTreeNodeFlags_DefaultOpen)) {
				const auto &flow = shootSkeleton.PeekFlow(internode.GetFlowHandle());
				ImGui::Text("Child flow size: %d", flow.RefChildHandles().size());
				ImGui::Text("Internode size: %d", flow.RefNodeHandles().size());
				if (ImGui::TreeNode("Internodes")) {
						int i = 0;
						for (const auto &chainedInternodeHandle: flow.RefNodeHandles()) {
								ImGui::Text("No.%d: Handle: %d", i, chainedInternodeHandle);
								i++;
						}
						ImGui::TreePop();
				}
				ImGui::TreePop();
		}
		return changed;
}

bool TreeVisualizer::DrawRootNodeInspectionGui(TreeModel &treeModel, NodeHandle rootNodeHandle, bool &deleted,
																							 const unsigned &hierarchyLevel) {
		auto &rootSkeleton = treeModel.RefRootSkeleton();
		const int index = m_selectedRootNodeHierarchyList.size() - hierarchyLevel - 1;
		if (!m_selectedRootNodeHierarchyList.empty() && index >= 0 &&
				index < m_selectedRootNodeHierarchyList.size() &&
				m_selectedRootNodeHierarchyList[index] == rootNodeHandle) {
				ImGui::SetNextItemOpen(true);
		}
		const bool opened = ImGui::TreeNodeEx(("Handle: " + std::to_string(rootNodeHandle)).c_str(),
																					ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_OpenOnArrow |
																					ImGuiTreeNodeFlags_NoAutoOpenOnLog |
																					(m_selectedRootNodeHandle == rootNodeHandle ? ImGuiTreeNodeFlags_Framed
																																											: ImGuiTreeNodeFlags_FramePadding));
		if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
				SetSelectedNode(rootSkeleton, rootNodeHandle, m_selectedRootNodeHandle, m_selectedRootNodeHierarchyList);
		}

		if (ImGui::BeginPopupContextItem(std::to_string(rootNodeHandle).c_str())) {
				ImGui::Text(("Handle: " + std::to_string(rootNodeHandle)).c_str());
				if (ImGui::Button("Delete")) {
						deleted = true;
				}
				ImGui::EndPopup();
		}
		bool modified = deleted;
		if (opened && !deleted) {
				ImGui::TreePush();
				const auto &rootNodeChildren = rootSkeleton.RefNode(rootNodeHandle).RefChildHandles();
				for (const auto &child: rootNodeChildren) {
						bool childDeleted = false;
						DrawRootNodeInspectionGui(treeModel, child, childDeleted, hierarchyLevel + 1);
						if (childDeleted) {
								treeModel.Step();
								treeModel.PruneRootNode(child);
								rootSkeleton.SortLists();
								m_iteration = treeModel.CurrentIteration();
								modified = true;
								break;
						}
				}
				ImGui::TreePop();
		}
		return modified;
}

void
TreeVisualizer::PeekInternode(const ShootSkeleton &shootSkeleton, NodeHandle internodeHandle) const {
		const auto &internode = shootSkeleton.PeekNode(internodeHandle);
		if (ImGui::TreeNode("Internode info")) {
				ImGui::Checkbox("Is max child", (bool *) &internode.m_data.m_isMaxChild);
				ImGui::Text("Thickness: %.3f", internode.m_info.m_thickness);
				ImGui::Text("Length: %.3f", internode.m_info.m_length);
				ImGui::InputFloat3("Position", (float *) &internode.m_info.m_globalPosition.x, "%.3f",
													 ImGuiInputTextFlags_ReadOnly);
				auto globalRotationAngle = glm::eulerAngles(internode.m_info.m_globalRotation);
				ImGui::InputFloat3("Global rotation", (float *) &globalRotationAngle.x, "%.3f",
													 ImGuiInputTextFlags_ReadOnly);
				auto localRotationAngle = glm::eulerAngles(internode.m_info.m_localRotation);
				ImGui::InputFloat3("Local rotation", (float *) &localRotationAngle.x, "%.3f",
													 ImGuiInputTextFlags_ReadOnly);
				auto &internodeData = internode.m_data;
				ImGui::InputInt("Start Age", (int *) &internodeData.m_startAge, 1, 100, ImGuiInputTextFlags_ReadOnly);
				ImGui::InputFloat("Sagging", (float *) &internodeData.m_sagging, 1, 100, "%.3f",
													ImGuiInputTextFlags_ReadOnly);
				ImGui::InputFloat("Distance to end", (float *) &internodeData.m_maxDistanceToAnyBranchEnd, 1, 100,
													"%.3f",
													ImGuiInputTextFlags_ReadOnly);
				ImGui::InputFloat("Descendent biomass", (float *) &internodeData.m_descendentTotalBiomass, 1, 100, "%.3f",
													ImGuiInputTextFlags_ReadOnly);
				ImGui::InputFloat("Biomass", (float *) &internodeData.m_biomass, 1, 100, "%.3f",
													ImGuiInputTextFlags_ReadOnly);
				ImGui::InputFloat("Root distance", (float *) &internodeData.m_rootDistance, 1, 100, "%.3f",
													ImGuiInputTextFlags_ReadOnly);
				ImGui::InputFloat3("Light dir", (float *) &internodeData.m_lightDirection.x, "%.3f",
													 ImGuiInputTextFlags_ReadOnly);
				ImGui::InputFloat("Light intensity", (float *) &internodeData.m_lightIntensity, 1, 100, "%.3f",
													ImGuiInputTextFlags_ReadOnly);

				if (ImGui::TreeNodeEx("Buds")) {
						int index = 1;
						for (auto &bud: internodeData.m_buds) {
								if (ImGui::TreeNode(("Bud " + std::to_string(index)).c_str())) {
										switch (bud.m_type) {
												case BudType::Apical:
														ImGui::Text("Apical");
														break;
												case BudType::Lateral:
														ImGui::Text("Lateral");
														break;
												case BudType::Leaf:
														ImGui::Text("Leaf");
														break;
												case BudType::Fruit:
														ImGui::Text("Fruit");
														break;
										}
										switch (bud.m_status) {
												case BudStatus::Dormant:
														ImGui::Text("Dormant");
														break;
												case BudStatus::Flushed:
														ImGui::Text("Flushed");
														break;
												case BudStatus::Died:
														ImGui::Text("Died");
														break;
										}

										auto budRotationAngle = glm::eulerAngles(bud.m_localRotation);
										ImGui::InputFloat3("Rotation", &budRotationAngle.x, "%.3f", ImGuiInputTextFlags_ReadOnly);
										/*
										ImGui::InputFloat("Base resource requirement", (float *) &bud.m_maintenanceVigorRequirementWeight, 1, 100,
															"%.3f", ImGuiInputTextFlags_ReadOnly);
															*/
										ImGui::TreePop();
								}
								index++;
						}
						ImGui::TreePop();
				}
				ImGui::TreePop();
		}
		if (ImGui::TreeNodeEx("Stem info", ImGuiTreeNodeFlags_DefaultOpen)) {
				const auto &flow = shootSkeleton.PeekFlow(internode.GetFlowHandle());
				ImGui::Text("Child stem size: %d", flow.RefChildHandles().size());
				ImGui::Text("Internode size: %d", flow.RefNodeHandles().size());
				if (ImGui::TreeNode("Internodes")) {
						int i = 0;
						for (const auto &chainedInternodeHandle: flow.RefNodeHandles()) {
								ImGui::Text("No.%d: Handle: %d", i, chainedInternodeHandle);
								i++;
						}
						ImGui::TreePop();
				}
				ImGui::TreePop();
		}
}

void TreeVisualizer::Reset(
				TreeModel &treeModel) {
		m_selectedInternodeHandle = -1;
		m_selectedInternodeHierarchyList.clear();
		m_selectedRootNodeHandle = -1;
		m_selectedRootNodeHierarchyList.clear();
		m_iteration = treeModel.CurrentIteration();
		m_internodeMatrices.clear();
		m_rootNodeMatrices.clear();
		m_needUpdate = true;
}

void TreeVisualizer::Clear() {
		m_selectedInternodeHandle = -1;
		m_selectedInternodeHierarchyList.clear();
		m_selectedRootNodeHandle = -1;
		m_selectedRootNodeHierarchyList.clear();
		m_iteration = 0;
		m_internodeMatrices.clear();
		m_rootNodeMatrices.clear();
}


void TreeVisualizer::PeekRootNode(
				const RootSkeleton &rootSkeleton,
				NodeHandle rootNodeHandle) const {

		const auto &rootNode = rootSkeleton.PeekNode(rootNodeHandle);
		if (ImGui::TreeNode("Root node info")) {
				ImGui::Checkbox("Is max child", (bool *) &rootNode.m_data.m_isMaxChild);
				ImGui::Text("Thickness: %.3f", rootNode.m_info.m_thickness);
				ImGui::Text("Length: %.3f", rootNode.m_info.m_length);
				ImGui::InputFloat3("Position", (float *) &rootNode.m_info.m_globalPosition.x, "%.3f",
													 ImGuiInputTextFlags_ReadOnly);
				auto globalRotationAngle = glm::eulerAngles(rootNode.m_info.m_globalRotation);
				ImGui::InputFloat3("Global rotation", (float *) &globalRotationAngle.x, "%.3f",
													 ImGuiInputTextFlags_ReadOnly);
				auto localRotationAngle = glm::eulerAngles(rootNode.m_info.m_localRotation);
				ImGui::InputFloat3("Local rotation", (float *) &localRotationAngle.x, "%.3f",
													 ImGuiInputTextFlags_ReadOnly);
				auto &rootNodeData = rootNode.m_data;
				ImGui::InputFloat("Nitrite", (float *) &rootNodeData.m_nitrite, 1, 100, "%.3f",
													ImGuiInputTextFlags_ReadOnly);
				ImGui::InputFloat("Soil Density", (float *) &rootNodeData.m_soilDensity, 1, 100, "%.3f",
													ImGuiInputTextFlags_ReadOnly);

				ImGui::InputFloat("Root flux", (float *) &rootNodeData.m_water, 1, 100, "%.3f",
													ImGuiInputTextFlags_ReadOnly);
				ImGui::InputFloat("Auxin", (float *) &rootNodeData.m_inhibitor, 1, 100, "%.3f",
													ImGuiInputTextFlags_ReadOnly);

				ImGui::InputFloat("Horizontal tropism", (float *) &rootNodeData.m_horizontalTropism, 1, 100,
													"%.3f",
													ImGuiInputTextFlags_ReadOnly);

				ImGui::InputFloat("Vertical tropism", (float *) &rootNodeData.m_verticalTropism, 1, 100,
													"%.3f",
													ImGuiInputTextFlags_ReadOnly);
				ImGui::TreePop();
		}
		if (ImGui::TreeNodeEx("Stem info", ImGuiTreeNodeFlags_DefaultOpen)) {
				const auto &flow = rootSkeleton.PeekFlow(rootNode.GetFlowHandle());
				ImGui::Text("Child stem size: %d", flow.RefChildHandles().size());
				ImGui::Text("Root node size: %d", flow.RefNodeHandles().size());
				if (ImGui::TreeNode("Root nodes")) {
						int i = 0;
						for (const auto &chainedInternodeHandle: flow.RefNodeHandles()) {
								ImGui::Text("No.%d: Handle: %d", i, chainedInternodeHandle);
								i++;
						}
						ImGui::TreePop();
				}
				ImGui::TreePop();
		}
}

bool TreeVisualizer::InspectRootNode(
				RootSkeleton &rootSkeleton,
				NodeHandle rootNodeHandle) {
		bool changed = false;

		const auto &rootNode = rootSkeleton.RefNode(rootNodeHandle);
		if (ImGui::TreeNode("Root node info")) {
				ImGui::Checkbox("Is max child", (bool *) &rootNode.m_data.m_isMaxChild);
				ImGui::Text("Thickness: %.3f", rootNode.m_info.m_thickness);
				ImGui::Text("Length: %.3f", rootNode.m_info.m_length);
				ImGui::InputFloat3("Position", (float *) &rootNode.m_info.m_globalPosition.x, "%.3f",
													 ImGuiInputTextFlags_ReadOnly);
				auto globalRotationAngle = glm::eulerAngles(rootNode.m_info.m_globalRotation);
				ImGui::InputFloat3("Global rotation", (float *) &globalRotationAngle.x, "%.3f",
													 ImGuiInputTextFlags_ReadOnly);
				auto localRotationAngle = glm::eulerAngles(rootNode.m_info.m_localRotation);
				ImGui::InputFloat3("Local rotation", (float *) &localRotationAngle.x, "%.3f",
													 ImGuiInputTextFlags_ReadOnly);
				auto &rootNodeData = rootNode.m_data;

				ImGui::InputFloat("Root distance", (float *) &rootNodeData.m_rootDistance, 1, 100, "%.3f",
													ImGuiInputTextFlags_ReadOnly);
				ImGui::InputFloat("Soil density", (float *) &rootNodeData.m_soilDensity, 1, 100, "%.3f",
													ImGuiInputTextFlags_ReadOnly);


				ImGui::InputFloat("Nitrite", (float *) &rootNodeData.m_nitrite, 1, 100, "%.3f",
													ImGuiInputTextFlags_ReadOnly);
				ImGui::InputFloat("Root flux", (float *) &rootNodeData.m_water, 1, 100, "%.3f",
													ImGuiInputTextFlags_ReadOnly);

				if (ImGui::DragFloat("Inhibitor", (float *) &rootNodeData.m_inhibitor)) {
						changed = true;
				}
				ImGui::InputFloat("Horizontal tropism", (float *) &rootNodeData.m_horizontalTropism, 1, 100,
													"%.3f",
													ImGuiInputTextFlags_ReadOnly);

				ImGui::InputFloat("Vertical tropism", (float *) &rootNodeData.m_verticalTropism, 1, 100,
													"%.3f",
													ImGuiInputTextFlags_ReadOnly);
				ImGui::TreePop();
		}
		if (ImGui::TreeNodeEx("Flow info", ImGuiTreeNodeFlags_DefaultOpen)) {
				const auto &flow = rootSkeleton.PeekFlow(rootNode.GetFlowHandle());
				ImGui::Text("Child flow size: %d", flow.RefChildHandles().size());
				ImGui::Text("Root node size: %d", flow.RefNodeHandles().size());
				if (ImGui::TreeNode("Root nodes")) {
						int i = 0;
						for (const auto &chainedInternodeHandle: flow.RefNodeHandles()) {
								ImGui::Text("No.%d: Handle: %d", i, chainedInternodeHandle);
								i++;
						}
						ImGui::TreePop();
				}
				ImGui::TreePop();
		}
		return changed;
}

void TreeVisualizer::SyncColors(const ShootSkeleton &shootSkeleton, NodeHandle &selectedNodeHandle) {
		if (m_randomColors.empty()) {
				for (int i = 0; i < 1000; i++) {
						m_randomColors.emplace_back(glm::ballRand(1.0f), 1.0f);
				}
		}

		const auto &sortedNodeList = shootSkeleton.RefSortedNodeList();
		m_internodeColors.resize(sortedNodeList.size() + 1);
		std::vector<std::shared_future<void>> results;
		Jobs::ParallelFor(sortedNodeList.size(), [&](unsigned i) {
			const auto nodeHandle = sortedNodeList[i];
			const auto &node = shootSkeleton.PeekNode(nodeHandle);
			if (nodeHandle == selectedNodeHandle) {
					m_internodeColors[i + 1] = glm::vec4(1, 0, 0, 1);
			} else {
					switch (static_cast<ShootVisualizerMode>(m_settings.m_shootVisualizationMode)) {
							case ShootVisualizerMode::LightIntensity:
									m_internodeColors[i + 1] = glm::vec4(
													glm::clamp(node.m_data.m_lightIntensity * m_settings.m_shootColorMultiplier, 0.0f, 1.f));
									break;
							case ShootVisualizerMode::LightDirection:
									m_internodeColors[i + 1] = glm::vec4(glm::vec3(glm::clamp(node.m_data.m_lightDirection, 0.0f, 1.f)),
																											 1.0f);
									break;
							case ShootVisualizerMode::IsMaxChild:
									m_internodeColors[i + 1] = glm::vec4(glm::vec3(node.m_data.m_isMaxChild ? 1.0f : 0.0f), 1.0f);
									break;
							case ShootVisualizerMode::AllocatedVigor:
									m_internodeColors[i + 1] = glm::vec4(glm::clamp(
													glm::vec3(node.m_data.m_vigorFlow.m_allocatedVigor * m_settings.m_shootColorMultiplier), 0.0f,
													1.f), 1.0f);
									break;
							default:
									m_internodeColors[i + 1] = m_randomColors[node.m_data.m_order];
									break;
					}
					m_internodeColors[i + 1].a = 1.0f;
					if (selectedNodeHandle != -1) m_internodeColors[i + 1].a = 0.3f;
			}
		}, results);
		for (auto &i: results) i.wait();
}

void TreeVisualizer::SyncColors(const RootSkeleton &rootSkeleton, const NodeHandle &selectedNodeHandle) {
		if (m_randomColors.empty()) {
				for (int i = 0; i < 1000; i++) {
						m_randomColors.emplace_back(glm::ballRand(1.0f), 1.0f);
				}
		}

		const auto &sortedNodeList = rootSkeleton.RefSortedNodeList();
		m_rootNodeColors.resize(sortedNodeList.size() + 1);
		std::vector<std::shared_future<void>> results;
		Jobs::ParallelFor(sortedNodeList.size(), [&](unsigned i) {
			const auto nodeHandle = sortedNodeList[i];
			const auto &node = rootSkeleton.PeekNode(nodeHandle);
			if (nodeHandle == selectedNodeHandle) {
					m_rootNodeColors[i + 1] = glm::vec4(1, 0, 0, 1);
			} else {
					switch (static_cast<RootVisualizerMode>(m_settings.m_rootVisualizationMode)) {
							case RootVisualizerMode::AllocatedVigor:
									m_internodeColors[i + 1] = glm::vec4(
													glm::clamp(glm::vec3(node.m_data.m_vigorSink.GetVigor() * m_settings.m_rootColorMultiplier),
																		 0.0f, 1.f), 1.0f);
									break;
							default:
									m_rootNodeColors[i + 1] = m_randomColors[node.m_data.m_order];
									break;
					}
					m_rootNodeColors[i + 1].a = 1.0f;
					if (selectedNodeHandle != -1) m_rootNodeColors[i + 1].a = 0.3f;
			}
		}, results);
		for (auto &i: results) i.wait();
}