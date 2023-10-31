#include "ClimateModel.hpp"

using namespace EcoSysLab;

float ClimateModel::GetTemperature(const glm::vec3& position) const
{
	int month = static_cast<int>(m_time * 365) / 30 % 12;
	int days = static_cast<int>(m_time * 365) % 30;
	
	int startIndex = month - 1;
	int endIndex = month + 1;
	if (startIndex < 0) startIndex += 12;
	if (endIndex > 11) endIndex -= 12;
	float startTemp = m_monthAvgTemp[startIndex];
	float avgTemp = m_monthAvgTemp[month];
	float endTemp = m_monthAvgTemp[endIndex];
	if(days < 15)
	{
		return glm::mix(startTemp, avgTemp, days / 15.0f);
	}
	if(days > 15)
	{
		return glm::mix(avgTemp, endTemp, (days - 15) / 15.0f);
	}
	return avgTemp;
}

float ClimateModel::GetSolarIntensity(const glm::vec3& position) const 
{
	return 1.0f;
}

void ClimateModel::Initialize(const ClimateParameters& climateParameters)
{
	m_time = 0;
}
