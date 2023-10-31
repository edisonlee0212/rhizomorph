cmake_minimum_required(VERSION 3.17)
project(
        RayTracerFacility
        VERSION 0.1
        DESCRIPTION "An implementation of MLVQ library in CUDA with OptiX ray tracer."
)
option(BUILD_RAY_TRACER_FACILITY "Build Ray Tracer Facility" ON)
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

# ------------------------------------------------------------------
# 3rd Party libraries
# ------------------------------------------------------------------
if (BUILD_RAY_TRACER_FACILITY)
    if (NOT WIN32)
        set(OptiX_INSTALL_DIR "~/NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64")
    endif ()
endif ()
# We need a CMAKE_DIR with some code to find external dependencies
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(UNIENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../UniEngine)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (NOT TARGET uniengine)
    add_subdirectory(${UNIENGINE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/UniEngine)
endif ()
include_directories(${UNIENGINE_INCLUDES})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

if (BUILD_RAY_TRACER_FACILITY)
    include(${CMAKE_MODULE_PATH}/configure_build_type.cmake)
    include(${CMAKE_MODULE_PATH}/configure_optix.cmake)
    mark_as_advanced(CUDA_SDK_ROOT_DIR)
    include_directories(${OptiX_INCLUDE})
endif ()

# ------------------------------------------------------------------
# RayTracer
# ------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
if (BUILD_RAY_TRACER_FACILITY)
    enable_language(CUDA)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/RayTracerFacility/MLVQ)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/RayTracerFacility/RayTracer)

    file(GLOB RAY_TRACER_FACILITY_SOURCES_LOCAL
            "src/RayTracerFacility/*.cpp"
            "src/RayTracerFacility/MLVQ/*.cpp"
            )
    file(GLOB RAY_TRACER_FACILITY_CUDA_SOURCES_LOCAL
            "src/RayTracerFacility/*.cu"
            )

    cuda_compile_and_embed(CAMERA_RENDERING_PTX ${CMAKE_CURRENT_SOURCE_DIR}/src/RayTracerFacility/ptx/CameraRendering.cu)
    cuda_compile_and_embed(ILLUMINATION_ESTIMATION_PTX ${CMAKE_CURRENT_SOURCE_DIR}/src/RayTracerFacility/ptx/IlluminationEstimation.cu)
    cuda_compile_and_embed(POINT_CLOUD_SCANNING_PTX ${CMAKE_CURRENT_SOURCE_DIR}/src/RayTracerFacility/ptx/PointCloudScanning.cu)

    add_library(RayTracerFacility
            SHARED
            #Default
            ${CAMERA_RENDERING_PTX}
            ${ILLUMINATION_ESTIMATION_PTX}
            ${POINT_CLOUD_SCANNING_PTX}

            #C++
            ${RAY_TRACER_FACILITY_SOURCES_LOCAL}
            ${RAY_TRACER_FACILITY_CUDA_SOURCES_LOCAL}
            ${CMAKE_MODULE_PATH}/configure_optix.cmake
            ${CMAKE_MODULE_PATH}/FindOptiX.cmake
            )

    set_property(TARGET RayTracerFacility PROPERTY CUDA_ARCHITECTURES 61-real 61-virtual)

    generate_export_header(RayTracerFacility
            BASE_NAME RAY_TRACER_FACILITY
            EXPORT_MACRO_NAME RAY_TRACER_FACILITY_API
            )

    if (NOT WIN32)
        message(STATUS CUDALIB=${CUDA_TOOLKIT_ROOT_DIR})
        set(CUDA_CUDA_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)
    endif ()
    message(STATUS OptiX_INCLUDE:${OptiX_INCLUDE})
    message(STATUS CUDA_LIBRARIES:${CUDA_LIBRARIES})
    message(STATUS CUDA_CUDA_LIBRARY:${CUDA_CUDA_LIBRARY})
    # Linker settings for all platforms
    target_link_libraries(RayTracerFacility
            uniengine
            ${CUDA_LIBRARIES}
            ${CUDA_CUDA_LIBRARY}
            )
    set(RAY_TRACER_FACILITY_INCLUDES_LOCAL
            ${OptiX_INCLUDE}
            ${CUDA_TOOLKIT_ROOT_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/include/RayTracerFacility
            ${CMAKE_CURRENT_SOURCE_DIR}/include/RayTracerFacility/MLVQ
            ${CMAKE_CURRENT_SOURCE_DIR}/include/RayTracerFacility/RayTracer
            ${CMAKE_CURRENT_SOURCE_DIR}/include/RayTracerFacility/Utilities
            ${UNIENGINE_INCLUDES}
            ${CMAKE_CURRENT_BINARY_DIR}
            )

    target_include_directories(RayTracerFacility
            PRIVATE
            ${RAY_TRACER_FACILITY_INCLUDES_LOCAL}
            )
    if (WIN32)
        target_compile_definitions(RayTracerFacility
                PRIVATE
                RAY_TRACER_FACILITY_EXPORTS
                )
        set(CUDA_PROPAGATE_HOST_FLAGS ON)
    endif ()

endif ()
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
# Example program
# ------------------------------------------------------------------
if (${CMAKE_BINARY_DIR} STREQUAL ${PROJECT_BINARY_DIR})
    add_executable(Application
            "src/app/Application.cpp"
            src/RayTracerFacility/PointCloudScanner.cpp include/RayTracerFacility/Utilities/PointCloudScanner.hpp)
    target_precompile_headers(Application
            PRIVATE
            ${PLANT_ARCHITECT_PCH_LOCAL}
            ${UNIENGINE_PCH}
            )
    if (BUILD_RAY_TRACER_FACILITY)
        target_include_directories(Application
                PRIVATE
                ${RAY_TRACER_FACILITY_INCLUDES_LOCAL}
                )
        target_link_libraries(Application
                uniengine
                RayTracerFacility
                )
        target_compile_definitions(Application
                PRIVATE
                RAYTRACERFACILITY
                )
    else ()
    endif ()
endif ()
# ------------------------------------------------------------------
# Copy Internal resources
# ------------------------------------------------------------------
file(COPY src/app/imgui.ini DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
if (BUILD_RAY_TRACER_FACILITY)
    if (NOT ${CMAKE_BINARY_DIR} STREQUAL ${PROJECT_BINARY_DIR})
        if (WIN32)
            add_custom_command(
                    TARGET RayTracerFacility
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy
                    ${PROJECT_BINARY_DIR}/RayTracerFacility.dll
                    ${CMAKE_BINARY_DIR}
            )
        endif ()
        set(RAY_TRACER_FACILITY_INCLUDES ${RAY_TRACER_FACILITY_INCLUDES_LOCAL} PARENT_SCOPE)
        set(RAY_TRACER_FACILITY_PCH ${PLANT_ARCHITECT_PCH_LOCAL} PARENT_SCOPE)
    endif ()
endif ()