cmake_minimum_required(VERSION 3.17)
project(
        EcoSysLab
        VERSION 0.1
        DESCRIPTION "An interactive application developed for my research purposes based on UniEngine"
)

option(BUILD_RAY_TRACER_FACILITY "Build Ray Tracer Facility" ON)
set(BUILD_RAY_TRACER_FACILITY ON)
include(GenerateExportHeader)

# Set a default build type if none was specified
set(default_build_type "Release")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release")
endif ()

set(CMAKE_CXX_STANDARD 17)
if (WIN32)
    # Compiler settings for Windows platform
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
elseif (UNIX AND NOT APPLE)
    if (${CMAKE_BUILD_TYPE} STREQUAL Release)
        set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    elseif (${CMAKE_BUILD_TYPE} STREQUAL Debug)
        set(CMAKE_CXX_FLAGS_DEBUG "-g")
    endif ()
elseif (APPLE)
endif ()

# ------------------------------------------------------------------
# 3rd Party libraries
# ------------------------------------------------------------------
set(UNIENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../UniEngine)
if(BUILD_RAY_TRACER_FACILITY) 
    set(RAYTRACERFACILITY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../RayTracerFacility)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT TARGET uniengine)
    add_subdirectory(${UNIENGINE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/UniEngine)
endif ()
if(BUILD_RAY_TRACER_FACILITY) 
    if (NOT TARGET RayTracerFacility)
        add_subdirectory(${RAYTRACERFACILITY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/RayTracerFacility)
    endif ()
endif()

include_directories(${UNIENGINE_INCLUDES})
include_directories(${CMAKE_CURRENT_BINARY_DIR})


# ------------------------------------------------------------------
# EcoSysLab
# ------------------------------------------------------------------
file(GLOB ECOSYSLAB_SOURCES_LOCAL "src/EcoSysLab/*.cpp")
add_library(EcoSysLab
        SHARED
        ${ECOSYSLAB_SOURCES_LOCAL}
        )
set(ECOSYSLAB_PCH_LOCAL
        ${UNIENGINE_PCH}
        ${RAY_TRACER_FACILITY_PCH}
        ${CMAKE_CURRENT_SOURCE_DIR}/include/EcoSysLab/EcoSysLab-pch.hpp
        )
generate_export_header(EcoSysLab
        BASE_NAME ECOSYSLAB
        EXPORT_MACRO_NAME ECOSYSLAB_API
        )

target_precompile_headers(EcoSysLab
        PRIVATE
        ${ECOSYSLAB_PCH_LOCAL}
        )
if (BUILD_RAY_TRACER_FACILITY)
    target_compile_definitions(EcoSysLab
            PRIVATE
            GLAD_GLAPI_EXPORT
            NOMINMAX
            RAYTRACERFACILITY
            )
    set(ECOSYSLAB_INCLUDES_LOCAL
            ${RAY_TRACER_FACILITY_INCLUDES}
            ${CMAKE_CURRENT_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/include/EcoSysLab
            ${CMAKE_CURRENT_SOURCE_DIR}/include/EcoSysLab/SorghumModel
            ${CMAKE_CURRENT_SOURCE_DIR}/include/EcoSysLab/ClimateModel
            ${CMAKE_CURRENT_SOURCE_DIR}/include/EcoSysLab/SoilModel
            ${CMAKE_CURRENT_SOURCE_DIR}/include/EcoSysLab/TreeModel
            ${CMAKE_CURRENT_SOURCE_DIR}/include/EcoSysLab/PipeModel
            ${CMAKE_CURRENT_SOURCE_DIR}/include/EcoSysLab/Utilities
            ${CMAKE_CURRENT_SOURCE_DIR}/include/EcoSysLab/Structures
            ${CMAKE_CURRENT_SOURCE_DIR}/include/EcoSysLab/Reconstruction
            )
    target_link_libraries(EcoSysLab
            RayTracerFacility
            )
else ()
    target_compile_definitions(EcoSysLab
            PRIVATE
            GLAD_GLAPI_EXPORT
            NOMINMAX
            )
    set(ECOSYSLAB_INCLUDES_LOCAL
            ${UNIENGINE_INCLUDES}
            ${CMAKE_CURRENT_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/include/EcoSysLab
            ${CMAKE_CURRENT_SOURCE_DIR}/include/EcoSysLab/SorghumModel
            ${CMAKE_CURRENT_SOURCE_DIR}/include/EcoSysLab/ClimateModel
            ${CMAKE_CURRENT_SOURCE_DIR}/include/EcoSysLab/SoilModel
            ${CMAKE_CURRENT_SOURCE_DIR}/include/EcoSysLab/PipeModel
            ${CMAKE_CURRENT_SOURCE_DIR}/include/EcoSysLab/TreeModel
            ${CMAKE_CURRENT_SOURCE_DIR}/include/EcoSysLab/Utilities
            ${CMAKE_CURRENT_SOURCE_DIR}/include/EcoSysLab/Structures
            ${CMAKE_CURRENT_SOURCE_DIR}/include/EcoSysLab/Reconstruction
            )
    target_link_libraries(EcoSysLab
            uniengine
            )
endif ()
target_include_directories(EcoSysLab
        PUBLIC
        ${ECOSYSLAB_INCLUDES_LOCAL}
        )
# ------------------------------------------------------------------
# Example program
# ------------------------------------------------------------------
add_executable(EcoSysLabApp
        "src/app/Application.cpp")
target_precompile_headers(EcoSysLabApp
        PRIVATE
        ${ECOSYSLAB_PCH_LOCAL}
        )
if (BUILD_RAY_TRACER_FACILITY)
    target_link_libraries(EcoSysLabApp
            RayTracerFacility
            EcoSysLab
            )
    target_compile_definitions(EcoSysLabApp
            PRIVATE
            RAYTRACERFACILITY
            NOMINMAX
            )
else ()
    target_link_libraries(EcoSysLabApp
            EcoSysLab
            )
    target_compile_definitions(EcoSysLabApp
            PRIVATE
            NOMINMAX
            )
endif ()

# ------------------------------------------------------------------
# Copy Internal resources
# ------------------------------------------------------------------
add_custom_command(TARGET EcoSysLab POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Internals
        ${CMAKE_BINARY_DIR})

file(COPY src/app/imgui.ini DESTINATION ${CMAKE_CURRENT_BINARY_DIR})