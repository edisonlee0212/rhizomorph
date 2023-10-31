#pragma once

#include "ecosyslab_export.h"

using namespace UniEngine;
namespace EcoSysLab
{
	class VigorSink
	{
		/*
		 * The desired resource needed for maintaining current plant structure.
		 * Depending on the size of fruit/leaf.
		 */
		float m_desiredMaintenanceVigorRequirement = 0.0f;
		/*
		 * The desired resource needed for reproduction (forming shoot/leaf/fruit) of this bud.
		 * Depending on the size of fruit/leaf.
		 */
		float m_desiredDevelopmentalVigorRequirement = 0.0f;
		float m_vigor = 0.0f;
	public:
		void SetDesiredMaintenanceVigorRequirement(float value);
		void SetDesiredDevelopmentalVigorRequirement(float value);
		[[nodiscard]] float GetDesiredMaintenanceVigorRequirement() const;
		[[nodiscard]] float GetDesiredDevelopmentalVigorRequirement() const;
		[[nodiscard]] float GetMaintenanceVigorRequirement() const;
		[[nodiscard]] float GetMaxVigorRequirement() const;

		void AddVigor(float value);
		[[nodiscard]] float GetVigor() const;
		[[nodiscard]] float SubtractAllDevelopmentalVigor();
		[[nodiscard]] float SubtractAllVigor();
		[[nodiscard]] float SubtractDevelopmentalVigor(float maxValue);
		[[nodiscard]] float SubtractVigor(float maxValue);
		[[nodiscard]] float GetAvailableDevelopmentalVigor() const;
		[[nodiscard]] float GetAvailableMaintenanceVigor() const;
		void EmptyVigor();
	};

	struct VigorFlow
	{
		float m_vigorRequirementWeight = 0.0f;

		float m_subtreeVigorRequirementWeight = 0.0f;

		float m_allocatedVigor = 0.0f;
		/*
		 * The allocated total resource for maintenance and development of all descendents.
		 */
		float m_subTreeAllocatedVigor = 0.0f;
	};
}