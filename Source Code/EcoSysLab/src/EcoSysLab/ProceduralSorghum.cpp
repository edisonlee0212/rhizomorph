//
// Created by lllll on 1/8/2022.
//
#include "ProceduralSorghum.hpp"
#include "SorghumData.hpp"
#include "SorghumLayer.hpp"
#include "SorghumStateGenerator.hpp"
#include "Utilities.hpp"
#include "rapidcsv.h"
#include <utility>
using namespace EcoSysLab;
static const char *StateModes[]{"Default", "Cubic-Bezier"};

SorghumStatePair ProceduralSorghum::Get(float time) const {
  SorghumStatePair retVal;
  retVal.m_mode = m_mode;
  if (m_sorghumStates.empty())
    return retVal;
  auto actualTime = glm::clamp(time, 0.0f, 99999.0f);
  float previousTime = m_sorghumStates.begin()->first;
  SorghumState previousState = m_sorghumStates.begin()->second;
  if (actualTime < previousTime) {
    // Get from zero state to first state.
    retVal.m_left = SorghumState();
    retVal.m_left.m_leaves.clear();
    retVal.m_left.m_stem.m_length = 0;
    retVal.m_right = m_sorghumStates.begin()->second;
    retVal.m_a = actualTime / previousTime;
    return retVal;
  }
  for (auto it = (++m_sorghumStates.begin()); it != m_sorghumStates.end();
       it++) {
    if (it->first > actualTime) {
      retVal.m_left = previousState;
      retVal.m_right = it->second;
      retVal.m_a = (actualTime - previousTime) / (it->first - previousTime);
      return retVal;
    }
    previousTime = it->first;
    previousState = it->second;
  }

  retVal.m_left = (--m_sorghumStates.end())->second;
  retVal.m_right = (--m_sorghumStates.end())->second;
  retVal.m_a = 1.0f;
  return retVal;
}

bool ProceduralPanicleState::OnInspect() {
  bool changed = false;
  if (ImGui::DragFloat("Panicle width", &m_panicleSize.x, 0.001f)) {
    changed = true;
    m_panicleSize.z = m_panicleSize.x;
  }
  if (ImGui::DragFloat("Panicle height", &m_panicleSize.y, 0.001f))
    changed = true;
  if (ImGui::DragInt("Num of seeds", &m_seedAmount, 1.0f))
    changed = true;
  if (ImGui::DragFloat("Seed radius", &m_seedRadius, 0.0001f))
    changed = true;
  if (changed)
    m_saved = false;
  return changed;
}
void ProceduralPanicleState::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_panicleSize" << YAML::Value << m_panicleSize;
  out << YAML::Key << "m_seedAmount" << YAML::Value << m_seedAmount;
  out << YAML::Key << "m_seedRadius" << YAML::Value << m_seedRadius;
  m_saved = true;
}
void ProceduralPanicleState::Deserialize(const YAML::Node &in) {
  if (in["m_panicleSize"])
    m_panicleSize = in["m_panicleSize"].as<glm::vec3>();
  if (in["m_seedAmount"])
    m_seedAmount = in["m_seedAmount"].as<int>();
  if (in["m_seedRadius"])
    m_seedRadius = in["m_seedRadius"].as<float>();
  m_saved = true;
}

ProceduralPanicleState::ProceduralPanicleState() {
  m_panicleSize = glm::vec3(0, 0, 0);
  m_seedAmount = 0;
  m_seedRadius = 0.002f;
  m_saved = false;
}
void ProceduralStemState::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_direction" << YAML::Value << m_direction;
  m_widthAlongStem.Serialize("m_widthAlongStem", out);
  out << YAML::Key << "m_length" << YAML::Value << m_length;
  out << YAML::Key << "m_spline" << YAML::Value << YAML::BeginMap;
  m_spline.Serialize(out);
  out << YAML::EndMap;

  m_saved = true;
}
void ProceduralStemState::Deserialize(const YAML::Node &in) {
  if (in["m_spline"]) {
    m_spline.Deserialize(in["m_spline"]);
  }

  if (in["m_direction"])
    m_direction = in["m_direction"].as<glm::vec3>();
  if (in["m_length"])
    m_length = in["m_length"].as<float>();
  m_widthAlongStem.Deserialize("m_widthAlongStem", in);

  m_saved = true;
}
bool ProceduralStemState::OnInspect(int mode) {
  bool changed = false;
  switch ((StateMode)mode) {
  case StateMode::Default:
    // ImGui::DragFloat3("Direction", &m_direction.x, 0.01f);
    if (ImGui::DragFloat("Length", &m_length, 0.01f))
      changed = true;
    break;
  case StateMode::CubicBezier:
    if (ImGui::TreeNode("Spline")) {
      m_spline.OnInspect();
      ImGui::TreePop();
    }
    break;
  }
  if (m_widthAlongStem.OnInspect("Width along stem"))
    changed = true;

  if (changed)
    m_saved = false;
  return changed;
}
bool ProceduralLeafState::OnInspect(int mode) {
  bool changed = false;
  if (ImGui::Checkbox("Dead", &m_dead)) {
    changed = true;
    if (!m_dead && m_length == 0.0f)
      m_length = 0.35f;
  }
  if (!m_dead) {
    if (ImGui::InputFloat("Starting point", &m_startingPoint)) {
      m_startingPoint = glm::clamp(m_startingPoint, 0.0f, 1.0f);
      changed = true;
    }
    switch ((StateMode)mode) {
    case StateMode::Default:
      if (ImGui::TreeNodeEx("Geometric", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::DragFloat("Length", &m_length, 0.01f, 0.0f, 999.0f))
          changed = true;
        if (ImGui::TreeNodeEx("Angles", ImGuiTreeNodeFlags_DefaultOpen)) {
          if (ImGui::DragFloat("Roll angle", &m_rollAngle, 1.0f, -999.0f,
                               999.0f))
            changed = true;
          if (ImGui::InputFloat("Branching angle", &m_branchingAngle)) {
            m_branchingAngle = glm::clamp(m_branchingAngle, 0.0f, 180.0f);
            changed = true;
          }
          ImGui::TreePop();
        }
        ImGui::TreePop();
      }
      break;
    case StateMode::CubicBezier:
      if (ImGui::TreeNodeEx("Geometric", ImGuiTreeNodeFlags_DefaultOpen)) {
        m_spline.OnInspect();
        ImGui::TreePop();
      }
      break;
    }

    if (ImGui::TreeNodeEx("Others")) {
      if (m_widthAlongLeaf.OnInspect("Width"))
        changed = true;
      if (m_curlingAlongLeaf.OnInspect("Rolling"))
        changed = true;

      static CurveDescriptorSettings leafBending = {
          1.0f, false, true,
          "The bending of the leaf, controls how leaves bend because of "
          "gravity. Positive value results in leaf bending towards the "
          "ground, negative value results in leaf bend towards the sky"};

      if (m_bendingAlongLeaf.OnInspect("Bending along leaf", leafBending)) {
        changed = true;
        m_bendingAlongLeaf.m_curve.UnsafeGetValues()[1].y = 0.5f;
      }
      if (m_wavinessAlongLeaf.OnInspect("Waviness along leaf"))
        changed = true;

      if (ImGui::DragFloat2("Waviness frequency", &m_wavinessFrequency.x, 0.01f,
                            0.0f, 999.0f))
        changed = true;
      if (ImGui::DragFloat2("Waviness start period", &m_wavinessPeriodStart.x,
                            0.01f, 0.0f, 999.0f))
        changed = true;
      ImGui::TreePop();
    }
  }
  if (changed)
    m_saved = false;
  return changed;
}
void ProceduralLeafState::Serialize(YAML::Emitter &out) {

  out << YAML::Key << "m_dead" << YAML::Value << m_dead;
  out << YAML::Key << "m_index" << YAML::Value << m_index;
  if (!m_dead) {
    out << YAML::Key << "m_spline" << YAML::Value << YAML::BeginMap;
    m_spline.Serialize(out);
    out << YAML::EndMap;

    out << YAML::Key << "m_startingPoint" << YAML::Value << m_startingPoint;
    out << YAML::Key << "m_length" << YAML::Value << m_length;
    m_curlingAlongLeaf.Serialize("m_curlingAlongLeaf", out);
    m_widthAlongLeaf.Serialize("m_widthAlongLeaf", out);
    out << YAML::Key << "m_rollAngle" << YAML::Value << m_rollAngle;
    out << YAML::Key << "m_branchingAngle" << YAML::Value << m_branchingAngle;
    m_bendingAlongLeaf.Serialize("m_bendingAlongLeaf", out);
    m_wavinessAlongLeaf.Serialize("m_wavinessAlongLeaf", out);
    out << YAML::Key << "m_wavinessFrequency" << YAML::Value
        << m_wavinessFrequency;
    out << YAML::Key << "m_wavinessPeriodStart" << YAML::Value
        << m_wavinessPeriodStart;
  }

  m_saved = true;
}
void ProceduralLeafState::Deserialize(const YAML::Node &in) {
  if (in["m_index"])
    m_index = in["m_index"].as<int>();
  if (in["m_dead"])
    m_dead = in["m_dead"].as<bool>();
  if (!m_dead) {
    if (in["m_spline"]) {
      m_spline.Deserialize(in["m_spline"]);
    }

    if (in["m_startingPoint"])
      m_startingPoint = in["m_startingPoint"].as<float>();
    if (in["m_length"])
      m_length = in["m_length"].as<float>();
    if (in["m_rollAngle"])
      m_rollAngle = in["m_rollAngle"].as<float>();
    if (in["m_branchingAngle"])
      m_branchingAngle = in["m_branchingAngle"].as<float>();
    if (in["m_wavinessFrequency"])
      m_wavinessFrequency = in["m_wavinessFrequency"].as<glm::vec2>();
    if (in["m_wavinessPeriodStart"])
      m_wavinessPeriodStart = in["m_wavinessPeriodStart"].as<glm::vec2>();

    m_curlingAlongLeaf.Deserialize("m_curlingAlongLeaf", in);
    m_bendingAlongLeaf.Deserialize("m_bendingAlongLeaf", in);
    m_widthAlongLeaf.Deserialize("m_widthAlongLeaf", in);
    m_wavinessAlongLeaf.Deserialize("m_wavinessAlongLeaf", in);
  }

  m_saved = true;
}
ProceduralStemState::ProceduralStemState() {
  m_length = 0.35f;
  m_widthAlongStem = {0.0f, 0.015f, {0.6f, 0.4f, {0, 0}, {1, 1}}};

  m_saved = false;
}

ProceduralLeafState::ProceduralLeafState() {
  m_dead = false;
  m_wavinessAlongLeaf = {0.0f, 5.0f, {0.0f, 0.5f, {0, 0}, {1, 1}}};
  m_wavinessFrequency = {30.0f, 30.0f};
  m_wavinessPeriodStart = {0.0f, 0.0f};
  m_widthAlongLeaf = {0.0f, 0.02f, {0.5f, 0.1f, {0, 0}, {1, 1}}};
  auto &pairs = m_widthAlongLeaf.m_curve.UnsafeGetValues();
  pairs.clear();
  pairs.emplace_back(-0.1, 0.0f);
  pairs.emplace_back(0, 0.5);
  pairs.emplace_back(0.11196319, 0.111996889);

  pairs.emplace_back(-0.0687116608, 0);
  pairs.emplace_back(0.268404901, 0.92331290);
  pairs.emplace_back(0.100000001, 0.0f);

  pairs.emplace_back(-0.100000001, 0);
  pairs.emplace_back(0.519368708, 1);
  pairs.emplace_back(0.100000001, 0);

  pairs.emplace_back(-0.100000001, 0.0f);
  pairs.emplace_back(1, 0.1);
  pairs.emplace_back(0.1, 0.0f);

  m_bendingAlongLeaf = {-180.0f, 180.0f, {0.5f, 0.5, {0, 0}, {1, 1}}};
  m_curlingAlongLeaf = {0.0f, 90.0f, {0.3f, 0.3f, {0, 0}, {1, 1}}};
  m_length = 0.35f;
  m_branchingAngle = 30.0f;

  m_saved = false;
}
void ProceduralLeafState::CopyShape(const ProceduralLeafState &another) {
  m_spline = another.m_spline;
  m_widthAlongLeaf.m_curve = another.m_widthAlongLeaf.m_curve;
  m_curlingAlongLeaf = another.m_curlingAlongLeaf;
  m_bendingAlongLeaf = another.m_bendingAlongLeaf;
  m_wavinessAlongLeaf = another.m_wavinessAlongLeaf;
  m_wavinessPeriodStart = another.m_wavinessPeriodStart;
  m_wavinessFrequency = another.m_wavinessFrequency;

  m_saved = false;
}

bool SorghumState::OnInspect(int mode) {
  bool changed = false;
  if (ImGui::TreeNodeEx((std::string("Stem")).c_str())) {
    if (m_stem.OnInspect(mode))
      changed = true;
    ImGui::TreePop();
  }

  if (ImGui::TreeNodeEx("Leaves")) {
    int leafSize = m_leaves.size();
    if (ImGui::InputInt("Number of leaves", &leafSize)) {
      changed = true;
      leafSize = glm::clamp(leafSize, 0, 999);
      auto previousSize = m_leaves.size();
      m_leaves.resize(leafSize);
      for (int i = 0; i < leafSize; i++) {
        if (i >= previousSize) {
          if (i - 1 >= 0) {
            m_leaves[i] = m_leaves[i - 1];
            m_leaves[i].m_rollAngle =
                glm::mod(m_leaves[i - 1].m_rollAngle + 180.0f, 360.0f);
            m_leaves[i].m_startingPoint =
                m_leaves[i - 1].m_startingPoint + 0.1f;
          } else {
            m_leaves[i] = ProceduralLeafState();
            m_leaves[i].m_rollAngle = 0;
            m_leaves[i].m_startingPoint = 0.1f;
          }
        }
        m_leaves[i].m_index = i;
      }
    }
    for (auto &leaf : m_leaves) {
      if (ImGui::TreeNode(
              ("Leaf No." + std::to_string(leaf.m_index + 1) +
               (leaf.m_length == 0.0f || leaf.m_dead ? " (Dead)" : ""))
                  .c_str())) {
        if (leaf.OnInspect(mode))
          changed = true;
        ImGui::TreePop();
      }
    }
    ImGui::TreePop();
  }

  if (ImGui::TreeNodeEx((std::string("Panicle")).c_str())) {
    if (m_panicle.OnInspect())
      changed = true;
    ImGui::TreePop();
  }
  if (mode == (int)StateMode::CubicBezier) {
    FileUtils::OpenFile(
        "Import...", "TXT", {".txt"},
        [&](const std::filesystem::path &path) {
          std::ifstream file(path, std::fstream::in);
          if (!file.is_open()) {
            UNIENGINE_LOG("Failed to open file!");
            return;
          }
          changed = true;
          // Number of leaves in the file
          int leafCount;
          file >> leafCount;
          m_stem = ProceduralStemState();
          m_stem.m_spline.Import(file);
          /*
          // Recenter plant:
          glm::vec3 posSum = m_stem.m_spline.m_curves.front().m_p0;
          for (auto &curve : m_stem.m_spline.m_curves) {
            curve.m_p0 -= posSum;
            curve.m_p1 -= posSum;
            curve.m_p2 -= posSum;
            curve.m_p3 -= posSum;
          }
          */
          m_leaves.resize(leafCount);
          for (int i = 0; i < leafCount; i++) {
            float startingPoint;
            file >> startingPoint;
            m_leaves[i] = ProceduralLeafState();
            m_leaves[i].m_startingPoint = startingPoint;
            m_leaves[i].m_spline.Import(file);
            m_leaves[i].m_spline.m_curves[0].m_p0 =
                m_stem.m_spline.EvaluatePointFromCurves(startingPoint);
          }

          for (int i = 0; i < leafCount; i++) {
            m_leaves[i].m_index = i;
          }
        },
        false);
  }
  if (changed)
    m_saved = false;
  return changed;
}

void SorghumState::Serialize(YAML::Emitter &out) {

  out << YAML::Key << "m_version" << YAML::Value << m_version;
  out << YAML::Key << "m_name" << YAML::Value << m_name;
  out << YAML::Key << "m_panicle" << YAML::Value << YAML::BeginMap;
  m_panicle.Serialize(out);
  out << YAML::EndMap;
  out << YAML::Key << "m_stem" << YAML::Value << YAML::BeginMap;
  m_stem.Serialize(out);
  out << YAML::EndMap;

  if (!m_leaves.empty()) {
    out << YAML::Key << "m_leaves" << YAML::Value << YAML::BeginSeq;
    for (auto &i : m_leaves) {
      out << YAML::BeginMap;
      i.Serialize(out);
      out << YAML::EndMap;
    }
    out << YAML::EndSeq;
  }

  m_saved = true;
}
void SorghumState::Deserialize(const YAML::Node &in) {
  if (in["m_version"])
    m_version = in["m_version"].as<unsigned>();
  if (in["m_name"])
    m_name = in["m_name"].as<std::string>();
  if (in["m_panicle"])
    m_panicle.Deserialize(in["m_panicle"]);

  if (in["m_stem"])
    m_stem.Deserialize(in["m_stem"]);

  if (in["m_leaves"]) {
    for (const auto &i : in["m_leaves"]) {
      ProceduralLeafState leaf;
      leaf.Deserialize(i);
      m_leaves.push_back(leaf);
    }
  }
  m_saved = true;
}
SorghumState::SorghumState() {
  m_saved = false;
  m_name = "Unnamed";
}

void ProceduralSorghum::OnInspect() {
  if (ImGui::Button("Instantiate")) {
    auto sorghum = Application::GetLayer<SorghumLayer>()->CreateSorghum(
        std::dynamic_pointer_cast<ProceduralSorghum>(m_self.lock()));
    Application::GetActiveScene()->SetEntityName(
        sorghum, m_self.lock()->GetAssetRecord().lock()->GetAssetFileName());
  }
  static bool autoSave = false;
  ImGui::Checkbox("Auto save", &autoSave);
  if (!autoSave) {
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));
    ImGui::Text("[Auto save disabled!]");
    ImGui::PopStyleColor();
  } else {
    static double lastAutoSaveTime = 0;
    static float autoSaveInterval = 10;
    if (ImGui::TreeNodeEx("Auto save settings")) {
      if (ImGui::DragFloat("Time interval", &autoSaveInterval, 1.0f, 2.0f,
                           300.0f)) {
        autoSaveInterval = glm::clamp(autoSaveInterval, 5.0f, 300.0f);
      }
      ImGui::TreePop();
    }
    if (lastAutoSaveTime == 0) {
      lastAutoSaveTime = Application::Time().CurrentTime();
    } else if (lastAutoSaveTime + autoSaveInterval <
               Application::Time().CurrentTime()) {
      lastAutoSaveTime = Application::Time().CurrentTime();
      if (!m_saved) {
        Save();
        UNIENGINE_LOG(GetTypeName() + " autosaved!");
      }
    }
  }
  if (!m_saved) {
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));
    ImGui::Text("[Changed unsaved!]");
    ImGui::PopStyleColor();
  }
  bool changed = false;
  FileUtils::OpenFile(
      "Import CSV", "CSV", {".csv", ".CSV"},
      [&](const std::filesystem::path &path) { changed = ImportCSV(path); },
      false);

  if (ImGui::Combo("Mode", &m_mode, StateModes, IM_ARRAYSIZE(StateModes))) {
    changed = false;
  }
  if (ImGui::TreeNodeEx("States", ImGuiTreeNodeFlags_DefaultOpen)) {
    float startTime =
        m_sorghumStates.empty() ? 1.0f : m_sorghumStates.begin()->first;
    if (startTime >= 0.01f) {
      if (ImGui::Button("New start state")) {
        changed = true;
        if (m_sorghumStates.empty()) {
          Add(0.0f, SorghumState());
        } else {
          Add(0.0f, m_sorghumStates.begin()->second);
        }
      }
    }

    float previousTime = 0.0f;
    int stateIndex = 1;
    for (auto it = m_sorghumStates.begin(); it != m_sorghumStates.end(); ++it) {
      if (ImGui::TreeNodeEx(
              ("State " + std::to_string(stateIndex) + ": " + it->second.m_name)
                  .c_str())) {
        const std::string tag = "##SorghumState" + std::to_string(stateIndex);
        if (ImGui::BeginPopupContextItem(tag.c_str())) {
          if (ImGui::BeginMenu(("Rename" + tag).c_str())) {
            static char newName[256];
            ImGui::InputText(("New name" + tag).c_str(), newName, 256);
            if (ImGui::Button(("Confirm" + tag).c_str())) {
              it->second.m_name = newName;
              memset(newName, 0, 256);
            }
            ImGui::EndMenu();
          }
          ImGui::EndPopup();
        }
        if (stateIndex != 1) {
          if (ImGui::Button("Copy prev leaves shape")) {
            for (int i = 0; i < (it - 1)->second.m_leaves.size() &&
                            i < it->second.m_leaves.size();
                 i++) {
              it->second.m_leaves[i].CopyShape((it - 1)->second.m_leaves[i]);
              it->second.m_saved = false;
              changed = true;
            }
          }
          ImGui::SameLine();
          if (ImGui::Button("Duplicate prev")) {
            it->second = (it - 1)->second;
            it->second.m_saved = false;
            for (auto &leafState : it->second.m_leaves)
              leafState.m_saved = false;
            it->second.m_panicle.m_saved = false;
            it->second.m_stem.m_saved = false;
            changed = true;
          }
        }
        if (it != (--m_sorghumStates.end())) {
          auto tit = it;
          ++tit;
          float nextTime = tit->first - 0.01f;
          float currentTime = it->first;
          if (ImGui::InputFloat("Time", &currentTime)) {
            it->first = glm::clamp(currentTime, previousTime, nextTime);
            changed = true;
          }

        } else {
          float currentTime = it->first;
          if (ImGui::InputFloat("Time", &currentTime)) {
            it->first = glm::clamp(currentTime, previousTime, 99999.0f);
            changed = true;
          }
        }

        if (it->second.OnInspect(m_mode)) {
          changed = true;
        }

        ImGui::TreePop();
      }
      previousTime = it->first + 0.01f;
      stateIndex++;
    }

    if (!m_sorghumStates.empty()) {
      if (ImGui::Button("New end state")) {
        changed = true;
        float endTime = (--m_sorghumStates.end())->first;
        Add(endTime + 0.01f, (--m_sorghumStates.end())->second);
      }
      ImGui::SameLine();
      if (ImGui::Button("Remove end state")) {
        changed = true;
        m_sorghumStates.erase(--m_sorghumStates.end());
      }
    }
    ImGui::TreePop();
  }

  if (ImGui::TreeNode("Import state...")) {
    static int seed = 0;
    ImGui::DragInt("Using seed", &seed);
    static AssetRef descriptor;
    Editor::DragAndDropButton<SorghumStateGenerator>(
        descriptor, "Drag SPD here to add end state");
    auto temp = descriptor.Get<SorghumStateGenerator>();
    if (temp) {
      float endTime =
          m_sorghumStates.empty() ? -0.01f : (--m_sorghumStates.end())->first;
      Add(endTime + 0.01f, temp->Generate(seed));
      descriptor.Clear();
      changed = true;
    }
    ImGui::TreePop();
  }

  if (changed) {
    m_saved = false;
    m_version++;
  }
}

void ProceduralSorghum::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_mode" << YAML::Value << m_mode;
  out << YAML::Key << "m_version" << YAML::Value << m_version;
  out << YAML::Key << "m_sorghumStates" << YAML::Value << YAML::BeginSeq;
  for (auto &state : m_sorghumStates) {
    out << YAML::BeginMap;
    out << YAML::Key << "Time" << YAML::Value << state.first;
    state.second.Serialize(out);
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;
}

void ProceduralSorghum::Deserialize(const YAML::Node &in) {
  if (in["m_mode"])
    m_mode = in["m_mode"].as<int>();
  if (in["m_version"])
    m_version = in["m_version"].as<unsigned>();
  if (in["m_sorghumStates"]) {
    m_sorghumStates.clear();
    for (const auto &inState : in["m_sorghumStates"]) {
      SorghumState state;
      state.Deserialize(inState);
      m_sorghumStates.emplace_back(inState["Time"].as<float>(), state);
    }
  }
}

void ProceduralSorghum::Add(float time, const SorghumState &state) {
  for (auto it = m_sorghumStates.begin(); it != m_sorghumStates.end(); ++it) {
    if (it->first == time) {
      it->second = state;
      return;
    }
    if (it->first > time) {
      m_sorghumStates.insert(it, {time, state});
      return;
    }
  }
  m_sorghumStates.emplace_back(time, state);
  m_sorghumStates.back().second.m_name = "Unnamed";
}

void ProceduralSorghum::ResetTime(float previousTime, float newTime) {
  for (auto &i : m_sorghumStates) {
    if (i.first == previousTime) {
      i.first = newTime;
      return;
    }
  }
  UNIENGINE_ERROR("Failed: State at previous time not exists!");
}
void ProceduralSorghum::Remove(float time) {
  for (auto it = m_sorghumStates.begin(); it != m_sorghumStates.end(); ++it) {
    if (it->first == time) {
      m_sorghumStates.erase(it);
      return;
    }
  }
}
float ProceduralSorghum::GetCurrentStartTime() const {
  if (m_sorghumStates.empty()) {
    return 0.0f;
  }
  return m_sorghumStates.begin()->first;
}
float ProceduralSorghum::GetCurrentEndTime() const {
  if (m_sorghumStates.empty()) {
    return 0.0f;
  }
  return (--m_sorghumStates.end())->first;
}

unsigned ProceduralSorghum::GetVersion() const { return m_version; }

bool ProceduralSorghum::ImportCSV(const std::filesystem::path &filePath) {
  try {
    rapidcsv::Document doc(filePath.string());
    std::vector<std::string> timePoints =
        doc.GetColumn<std::string>("Time Point");
    std::vector<float> stemHeights = doc.GetColumn<float>("Stem Height");
    std::vector<float> stemWidth = doc.GetColumn<float>("Stem Width");
    std::vector<float> leafIndex = doc.GetColumn<float>("Leaf Number");
    std::vector<float> leafLength = doc.GetColumn<float>("Leaf Length");
    std::vector<float> leafWidth = doc.GetColumn<float>("Leaf Width");
    std::vector<float> leafHeight = doc.GetColumn<float>("Leaf Height");
    std::vector<float> startingPoint = doc.GetColumn<float>("Start Point");
    std::vector<float> branchingAngle = doc.GetColumn<float>("Branching Angle");
    std::vector<float> panicleLength = doc.GetColumn<float>("Panicle Height");
    std::vector<float> panicleWidth = doc.GetColumn<float>("Panicle Width");

    m_sorghumStates.clear();

    std::map<std::string, std::pair<int, int>> columnIndices;
    int currentIndex = 0;
    for (int row = 0; row < timePoints.size(); row++) {
      auto &timePoint = timePoints[row];
      if (columnIndices.find(timePoint) == columnIndices.end()) {
        columnIndices[timePoint].first = currentIndex;
        currentIndex++;
      }
      if (columnIndices[timePoint].second < leafIndex[row])
        columnIndices[timePoint].second = leafIndex[row];
    }

    m_sorghumStates.resize(currentIndex);
    for (int row = 0; row < timePoints.size(); row++) {
      int stateIndex = columnIndices.at(timePoints[row]).first;
      auto &statePair = m_sorghumStates[stateIndex];
      auto &state = statePair.second;
      if (state.m_leaves.empty()) {
        statePair.first = stateIndex;
        state.m_name = timePoints[row];
        state.m_leaves.resize(columnIndices.at(timePoints[row]).second);
        for (auto &leaf : state.m_leaves)
          leaf.m_dead = true;
        state.m_stem.m_length = stemHeights[row] / 100.0f;
        state.m_stem.m_widthAlongStem.m_minValue = 0.0f;
        state.m_stem.m_widthAlongStem.m_maxValue = stemWidth[row] * 2.0f;
        state.m_panicle.m_panicleSize.x = state.m_panicle.m_panicleSize.z = panicleWidth[row] / 100.0f;
        state.m_panicle.m_panicleSize.y = panicleLength[row] / 100.0f;
        state.m_panicle.m_seedAmount = state.m_panicle.m_panicleSize.x * state.m_panicle.m_panicleSize.y * state.m_panicle.m_panicleSize.z / 0.001f;
      }
      auto &leaf = state.m_leaves[leafIndex[row] - 1];
      leaf.m_index = leafIndex[row] - 1;
      leaf.m_length = leafLength[row] / 100.0f;
      if (leaf.m_length == 0)
        leaf.m_dead = true;
      else {
        leaf.m_dead = false;
        leaf.m_rollAngle = leaf.m_index % 2 * 180.0f;
        leaf.m_widthAlongLeaf.m_maxValue = leafWidth[row] / 100.0f;
        leaf.m_startingPoint = leafHeight[row] / stemHeights[row];
        leaf.m_branchingAngle = branchingAngle[row];
      }
    }

    for (auto &sorghumState : m_sorghumStates) {
      sorghumState.second.m_saved = false;
      int leafIndex = 0;
      for (auto &leafState : sorghumState.second.m_leaves) {
        leafState.m_saved = false;
        leafState.m_index = leafIndex;
        leafIndex++;
      }
      sorghumState.second.m_stem.m_saved = false;
      sorghumState.second.m_panicle.m_saved = false;
    }
    m_saved = false;
  } catch (std::exception e) {
    return false;
  }
  return true;
}

glm::vec3 ProceduralStemState::GetPoint(float point) const {
  return m_direction * point * m_length;
}

int SorghumStatePair::GetLeafSize() const {
  if (m_left.m_leaves.size() <= m_right.m_leaves.size()) {
    return m_left.m_leaves.size() +
           glm::ceil((m_right.m_leaves.size() - m_left.m_leaves.size()) * m_a);
  } else
    return m_left.m_leaves.size();
}
float SorghumStatePair::GetStemLength() const {
  float leftLength, rightLength;
  switch ((StateMode)m_mode) {
  case StateMode::Default:
    leftLength = m_left.m_stem.m_length;
    rightLength = m_right.m_stem.m_length;
    break;
  case StateMode::CubicBezier:
    if (!m_left.m_stem.m_spline.m_curves.empty()) {
      leftLength = glm::distance(m_left.m_stem.m_spline.m_curves.front().m_p0,
                                 m_left.m_stem.m_spline.m_curves.back().m_p3);
    } else {
      leftLength = 0.0f;
    }
    if (!m_right.m_stem.m_spline.m_curves.empty()) {
      rightLength = glm::distance(m_right.m_stem.m_spline.m_curves.front().m_p0,
                                  m_right.m_stem.m_spline.m_curves.back().m_p3);
    } else {
      rightLength = 0.0f;
    }
    break;
  }
  return glm::mix(leftLength, rightLength, m_a);
}
glm::vec3 SorghumStatePair::GetStemDirection() const {
  glm::vec3 leftDir, rightDir;
  switch ((StateMode)m_mode) {
  case StateMode::Default:
    leftDir = glm::normalize(m_left.m_stem.m_direction);
    rightDir = glm::normalize(m_right.m_stem.m_direction);
    break;
  case StateMode::CubicBezier:
    if (!m_left.m_stem.m_spline.m_curves.empty()) {
      leftDir = glm::vec3(0.0f, 1.0f, 0.0f);
    } else {
      leftDir = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    if (!m_right.m_stem.m_spline.m_curves.empty()) {
      rightDir = glm::vec3(0.0f, 1.0f, 0.0f);
    } else {
      rightDir = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    break;
  }

  return glm::normalize(glm::mix(leftDir, rightDir, m_a));
}
glm::vec3 SorghumStatePair::GetStemPoint(float point) const {
  glm::vec3 leftPoint, rightPoint;
  switch ((StateMode)m_mode) {
  case StateMode::Default:
    leftPoint = glm::normalize(m_left.m_stem.m_direction) * point * m_left.m_stem.m_length;
    rightPoint = glm::normalize(m_right.m_stem.m_direction) * point * m_right.m_stem.m_length;
    break;
  case StateMode::CubicBezier:
    if (!m_left.m_stem.m_spline.m_curves.empty()) {
      leftPoint = m_left.m_stem.m_spline.EvaluatePointFromCurves(point);
    } else {
      leftPoint = glm::vec3(0.0f, 0.0f, 0.0f);
    }
    if (!m_right.m_stem.m_spline.m_curves.empty()) {
      rightPoint = m_right.m_stem.m_spline.EvaluatePointFromCurves(point);
    } else {
      rightPoint = glm::vec3(0.0f, 0.0f, 0.0f);
    }
    break;
  }

  return glm::mix(leftPoint, rightPoint, m_a);
}
