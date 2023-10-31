// PlantFactory.cpp : This file contains the 'main' function. Program execution
// begins and ends there.
//
#include <Application.hpp>
#ifdef RAYTRACERFACILITY
#include <RayTracerLayer.hpp>
#include "BTFMeshRenderer.hpp"
#include "TriangleIlluminationEstimator.hpp"
#endif


#ifdef RAYTRACERFACILITY
using namespace RayTracerFacility;
#endif

int main() {
    const bool enableRayTracing = true;
    ApplicationConfigs applicationConfigs;
    Application::Create(applicationConfigs);
#ifdef RAYTRACERFACILITY
    Application::PushLayer<RayTracerLayer>();
#endif
#pragma region Engine Loop
    Application::Start();
#pragma endregion
    Application::End();
}
