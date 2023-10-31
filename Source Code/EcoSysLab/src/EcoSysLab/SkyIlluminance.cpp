//
// Created by lllll on 2/23/2022.
//
#include "rapidcsv.h"
#include "SkyIlluminance.hpp"
#ifdef RAYTRACERFACILITY
#include "RayTracerLayer.hpp"
#endif

#ifdef RAYTRACERFACILITY
using namespace RayTracerFacility;
#endif
using namespace EcoSysLab;SkyIlluminanceSnapshot
SkyIlluminanceSnapshotLerp(const SkyIlluminanceSnapshot &l,
                           const SkyIlluminanceSnapshot &r, float a) {
  if (a < 0.0f)
    return l;
  if (a > 1.0f)
    return r;
  SkyIlluminanceSnapshot snapshot;
  snapshot.m_ghi = l.m_ghi * a + r.m_ghi * (1.0f - a);
  snapshot.m_azimuth = l.m_azimuth * a + r.m_azimuth * (1.0f - a);
  snapshot.m_zenith = l.m_zenith * a + r.m_zenith * (1.0f - a);
  return snapshot;
}

SkyIlluminanceSnapshot SkyIlluminance::Get(float time) {
  if (m_snapshots.empty()) {
    return {};
  }
  if (time <= m_snapshots.begin()->first)
    return m_snapshots.begin()->second;
  SkyIlluminanceSnapshot lastShot;
  lastShot = m_snapshots.begin()->second;
  float lastTime = m_snapshots.begin()->first;
  for (const auto &pair : m_snapshots) {
    if (time < pair.first) {
      if (pair.first - lastTime == 0)
        return lastShot;
      return SkyIlluminanceSnapshotLerp(
          lastShot, pair.second, (time - lastTime) / (pair.first - lastTime));
    }
    lastShot = pair.second;
    lastTime = pair.first;
  }
  return std::prev(m_snapshots.end())->second;
}
void SkyIlluminance::ImportCSV(const std::filesystem::path& path) {
  rapidcsv::Document doc(path.string());
  std::vector<float> timeSeries = doc.GetColumn<float>("Time");
  std::vector<float> ghiSeries = doc.GetColumn<float>("SunLightDensity");
  std::vector<float> azimuthSeries = doc.GetColumn<float>("Azimuth");
  std::vector<float> zenithSeries = doc.GetColumn<float>("Zenith");
  assert(timeSeries.size() == ghiSeries.size() && azimuthSeries.size() == zenithSeries.size() && timeSeries.size() == azimuthSeries.size());
  m_snapshots.clear();
  m_maxTime = 0;
  m_minTime = 999999;
  for(int i = 0; i < timeSeries.size(); i++) {
    SkyIlluminanceSnapshot snapshot;
    snapshot.m_ghi = ghiSeries[i];
    snapshot.m_azimuth = azimuthSeries[i];
    snapshot.m_zenith = zenithSeries[i];
    auto time = timeSeries[i];
    m_snapshots[time] = snapshot;
    if (m_maxTime < time) {
      m_maxTime = time;
    }
    if (m_minTime > time) {
      m_minTime = time;
    }
  }

}
void SkyIlluminance::OnInspect() {
  FileUtils::OpenFile("Import CSV", "CSV", {".csv"}, [&](const std::filesystem::path &path){
    ImportCSV(path);
  }, false);
  static float time;
  static SkyIlluminanceSnapshot snapshot;
  static bool autoApply = false;
  ImGui::Checkbox("Auto Apply", &autoApply);
  if(ImGui::SliderFloat("Time", &time, m_minTime, m_maxTime)){
    snapshot = Get(time);
#ifdef RAYTRACERFACILITY
    if(autoApply){
      auto& envProp = Application::GetLayer<RayTracerLayer>()->m_environmentProperties;
      envProp.m_sunDirection = snapshot.GetSunDirection();
      envProp.m_skylightIntensity = snapshot.GetSunIntensity();
    }
#endif
  }
  ImGui::Text("Ghi: %.3f", snapshot.m_ghi);
  ImGui::Text("Azimuth: %.3f", snapshot.m_azimuth);
  ImGui::Text("Zenith: %.3f", snapshot.m_zenith);

}
void SkyIlluminance::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_minTime" << YAML::Value << m_minTime;
  out << YAML::Key << "m_maxTime" << YAML::Value << m_maxTime;
  if(!m_snapshots.empty()) {
    out << YAML::Key << "m_snapshots" << YAML::Value << YAML::BeginSeq;
    for (const auto &pair : m_snapshots) {
      out << YAML::BeginMap;
      out << YAML::Key << "time" << YAML::Value << pair.first;
      out << YAML::Key << "m_ghi" << YAML::Value << pair.second.m_ghi;
      out << YAML::Key << "m_azimuth" << YAML::Value << pair.second.m_azimuth;
      out << YAML::Key << "m_zenith" << YAML::Value << pair.second.m_zenith;
      out << YAML::EndMap;
    }
    out << YAML::EndSeq;
  }
}
void SkyIlluminance::Deserialize(const YAML::Node &in) {
  if(in["m_minTime"]) m_minTime = in["m_minTime"].as<float>();
  if(in["m_maxTime"]) m_maxTime = in["m_maxTime"].as<float>();
  if(in["m_snapshots"]) {
    m_snapshots.clear();
    for(const auto& data : in["m_snapshots"]){
      SkyIlluminanceSnapshot snapshot;
      snapshot.m_ghi = data["m_ghi"].as<float>();
      snapshot.m_azimuth = data["m_azimuth"].as<float>();
      snapshot.m_zenith = data["m_zenith"].as<float>();
      m_snapshots[data["time"].as<float>()] = snapshot;
    }
  }
}

glm::vec3 SkyIlluminanceSnapshot::GetSunDirection() {
  return glm::quat(glm::radians(glm::vec3(90.0f - m_zenith, m_azimuth, 0))) * glm::vec3(0, 0, -1);
}
float SkyIlluminanceSnapshot::GetSunIntensity() { return m_ghi; }
