#include "Climate.hpp"

using namespace EcoSysLab;

void ClimateDescriptor::OnInspect()
{
	if (ImGui::Button("Instantiate")) {
		auto scene = Application::GetActiveScene();
		auto climateEntity = scene->CreateEntity(GetTitle());
		auto climate = scene->GetOrSetPrivateComponent<Climate>(climateEntity).lock();
		climate->m_climateDescriptor = ProjectManager::GetAsset(GetHandle());
	}
}

void ClimateDescriptor::Serialize(YAML::Emitter& out)
{
	
}

void ClimateDescriptor::Deserialize(const YAML::Node& in)
{
	
}

void Climate::OnInspect()
{
	if(Editor::DragAndDropButton<ClimateDescriptor>(m_climateDescriptor, "ClimateDescriptor", true))
	{
		InitializeClimateModel();
	}

	if (m_climateDescriptor.Get<ClimateDescriptor>())
	{

	}
}

void Climate::Serialize(YAML::Emitter& out)
{
	m_climateDescriptor.Save("m_climateDescriptor", out);
}

void Climate::CollectAssetRef(std::vector<AssetRef>& list)
{
	list.push_back(m_climateDescriptor);
}

void Climate::InitializeClimateModel()
{
	auto climateDescriptor = m_climateDescriptor.Get<ClimateDescriptor>();
	if (climateDescriptor)
	{
		auto params = climateDescriptor->m_climateParameters;
		m_climateModel.Initialize(params);
	}
}

void Climate::Deserialize(const YAML::Node& in)
{
	m_climateDescriptor.Load("m_climateDescriptor", in);
}
