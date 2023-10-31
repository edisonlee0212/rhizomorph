#include "VigorSink.hpp"
using namespace EcoSysLab;

void VigorSink::SetDesiredMaintenanceVigorRequirement(const float value)
{
	m_desiredMaintenanceVigorRequirement = value;
}

void VigorSink::SetDesiredDevelopmentalVigorRequirement(const float value)
{
	m_desiredDevelopmentalVigorRequirement = value;
}

float VigorSink::GetDesiredMaintenanceVigorRequirement() const
{
	return m_desiredMaintenanceVigorRequirement;
}

float VigorSink::GetDesiredDevelopmentalVigorRequirement() const
{
	return m_desiredDevelopmentalVigorRequirement;
}

float VigorSink::GetMaintenanceVigorRequirement() const
{
	return glm::max(0.0f, m_desiredMaintenanceVigorRequirement - m_vigor);
}

float VigorSink::GetMaxVigorRequirement() const
{
	return glm::max(0.0f, m_desiredDevelopmentalVigorRequirement + m_desiredMaintenanceVigorRequirement - m_vigor);
}

void VigorSink::AddVigor(const float value)
{
	m_vigor += value;
}

float VigorSink::GetVigor() const
{
	return m_vigor;
}

float VigorSink::SubtractAllDevelopmentalVigor()
{
	const auto retVal = glm::max(0.0f, m_vigor - m_desiredMaintenanceVigorRequirement);
	m_vigor -= retVal;
	return retVal;
}

float VigorSink::SubtractAllVigor()
{
	const auto retVal = m_vigor;
	m_vigor = 0.0f;
	return retVal;
}

float VigorSink::SubtractDevelopmentalVigor(const float maxValue)
{
	const auto retVal = glm::max(maxValue, glm::max(0.0f, m_vigor - m_desiredMaintenanceVigorRequirement));
	m_vigor -= retVal;
	return retVal;
}

float VigorSink::SubtractVigor(const float maxValue)
{
	const auto retVal = glm::min(maxValue, m_vigor);
	m_vigor -= retVal;
	return retVal;
}

float VigorSink::GetAvailableDevelopmentalVigor() const
{
	return glm::max(0.0f, m_vigor - m_desiredMaintenanceVigorRequirement);
}

float VigorSink::GetAvailableMaintenanceVigor() const
{
	return glm::clamp(m_vigor, 0.0f, m_desiredMaintenanceVigorRequirement);
}

void VigorSink::EmptyVigor()
{
	m_vigor = 0;
}
