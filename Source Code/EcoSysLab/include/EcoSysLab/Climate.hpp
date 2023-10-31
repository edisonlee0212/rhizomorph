#pragma once

#include "ecosyslab_export.h"
#include "ClimateModel.hpp"
using namespace UniEngine;
namespace EcoSysLab
{
	class ClimateDescriptor : public IAsset {
	public:
		ClimateParameters m_climateParameters;

		void OnInspect() override;

		void Serialize(YAML::Emitter& out) override;

		void Deserialize(const YAML::Node& in) override;
	};
	class Climate : public IPrivateComponent {

	public:
		ClimateModel m_climateModel;
		AssetRef m_climateDescriptor;

		/**ImGui menu goes to here.Also you can take care you visualization with Gizmos here.
		 * Note that the visualization will only be activated while you are inspecting the soil private component in the entity inspector.
		 */
		void OnInspect() override;
		void Serialize(YAML::Emitter& out) override;

		void Deserialize(const YAML::Node& in) override;

		void CollectAssetRef(std::vector<AssetRef>& list) override;

		void InitializeClimateModel();
	};
}
