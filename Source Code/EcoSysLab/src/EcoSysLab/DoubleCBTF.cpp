//
// Created by lllll on 11/16/2022.
//

#include "DoubleCBTF.hpp"
#ifdef RAYTRACERFACILITY
#include "CompressedBTF.hpp"
using namespace RayTracerFacility;
#endif
using namespace EcoSysLab;
void DoubleCBTF::OnInspect() {
  Editor::DragAndDropButton<CompressedBTF>(m_top, "Top");
  Editor::DragAndDropButton<CompressedBTF>(m_bottom, "Bottom");
}
void DoubleCBTF::CollectAssetRef(std::vector<AssetRef> &list) {
  list.push_back(m_top);
  list.push_back(m_bottom);
}
void DoubleCBTF::Serialize(YAML::Emitter &out) {
  m_top.Save("m_top", out);
  m_bottom.Save("m_bottom", out);
}
void DoubleCBTF::Deserialize(const YAML::Node &in) {
  m_top.Load("m_top", in);
  m_bottom.Load("m_bottom", in);
}
