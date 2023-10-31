#pragma once

#include <BTFBase.cuh>

#include <CUDABuffer.hpp>

#include <Optix7.hpp>

#include <Vertex.hpp>

#include <cuda.h>
#include <unordered_map>

#include "string"

#include "vector"

#include "map"

#include "Enums.hpp"
#include "MaterialProperties.hpp"
#include "filesystem"
#include "HitInfo.hpp"

namespace RayTracerFacility {
    enum class OutputType {
        Color, Normal, Albedo, Depth
    };

    struct RAY_TRACER_FACILITY_API CameraProperties {
#pragma region FrameBuffer
        /*! the color buffer we use during _rendering_, which is a bit
        larger than the actual displayed frame buffer (to account for
        the border), and in float4 format (the denoiser requires
        floats) */
        CudaBuffer m_frameBufferColor;
        CudaBuffer m_frameBufferNormal;
        CudaBuffer m_frameBufferAlbedo;
#pragma endregion
#pragma region Denoiser
        /*! output of the denoiser pass, in float4 */
        CudaBuffer m_denoisedBuffer;
        OptixDenoiser m_denoiser = nullptr;
        CudaBuffer m_denoiserScratch;
        CudaBuffer m_denoiserState;
        CudaBuffer m_denoiserIntensity;
#pragma endregion
        bool m_accumulate = true;

        float m_fov = 120;
        /*! camera position - *from* where we are looking */
        glm::vec3 m_from = glm::vec3(0.0f);
        /*! general up-vector */
        glm::vec3 m_horizontal;
        glm::vec3 m_vertical;
        glm::mat4 m_inverseProjectionView;

        float m_denoiserStrength = 1.0f;
        float m_maxDistance = 50.0f;
        unsigned m_outputTextureId = 0;
        OutputType m_outputType = OutputType::Color;
        float m_gamma = 2.2f;
        struct {
            glm::vec4 *m_colorBuffer;
            glm::vec4 *m_normalBuffer;
            glm::vec4 *m_albedoBuffer;
            /*! the size of the frame buffer to render */
            glm::ivec2 m_size;
            size_t m_frameId;
        } m_frame;

        bool m_modified = false;

        float m_aperture = 0.0f;
        float m_focalLength = 1.0f;

        void SetAperture(float value);

        void SetFocalLength(float value);

        void SetFov(float value);

        void SetGamma(float value);

        void SetMaxDistance(float value);

        void SetOutputType(OutputType value);

        void SetDenoiserStrength(float value);

        void Resize(const glm::ivec2 &newSize);

        void Set(const glm::vec3 &position, const glm::quat &rotation);

        void OnInspect();
    };

#pragma region MyRegion
    enum class EnvironmentalLightingType {
        Scene, Skydome, SingleLightSource
    };

    struct RAY_TRACER_FACILITY_API EnvironmentProperties {
        EnvironmentalLightingType m_environmentalLightingType =
                EnvironmentalLightingType::Scene;
        float m_skylightIntensity = 1.0f;
        float m_ambientLightIntensity = 0.1f;
        float m_lightSize = 0.0f;
        float m_gamma = 1.0f;
        glm::vec3 m_sunDirection = glm::vec3(0, 1, 0);
        glm::vec3 m_color = glm::vec3(1, 1, 1);
        unsigned m_environmentalMapId = 0;
        cudaTextureObject_t m_environmentalMaps[6];

        struct {
            float m_earthRadius =
                    6360; // In the paper this is usually Rg or Re (radius ground, eart)
            float m_atmosphereRadius =
                    6420; // In the paper this is usually R or Ra (radius atmosphere)
            float m_Hr =
                    7994; // Thickness of the atmosphere if density was uniform (Hr)
            float m_Hm = 1200; // Same as above but for Mie scattering (Hm)
            float m_g = 0.76f; // Mean cosine for Mie scattering
            int m_numSamples = 16;
            int m_numSamplesLight = 8;
        } m_atmosphere;

        [[nodiscard]] bool Changed(const EnvironmentProperties &properties) const {
            return properties.m_environmentalLightingType !=
                   m_environmentalLightingType ||
                   properties.m_lightSize != m_lightSize ||
                   properties.m_ambientLightIntensity != m_ambientLightIntensity ||
                   properties.m_skylightIntensity != m_skylightIntensity ||
                   properties.m_gamma != m_gamma ||
                   properties.m_sunDirection != m_sunDirection ||
                   properties.m_environmentalMapId != m_environmentalMapId ||
                   properties.m_color != m_color ||
                   properties.m_atmosphere.m_earthRadius !=
                   m_atmosphere.m_earthRadius ||
                   properties.m_atmosphere.m_atmosphereRadius !=
                   m_atmosphere.m_atmosphereRadius ||
                   properties.m_atmosphere.m_Hr != m_atmosphere.m_Hr ||
                   properties.m_atmosphere.m_Hm != m_atmosphere.m_Hm ||
                   properties.m_atmosphere.m_g != m_atmosphere.m_g ||
                   properties.m_atmosphere.m_numSamples != m_atmosphere.m_numSamples ||
                   properties.m_atmosphere.m_numSamplesLight !=
                   m_atmosphere.m_numSamplesLight;
        }

        void OnInspect();
    };

    struct RAY_TRACER_FACILITY_API RayProperties {
        int m_bounces = 4;
        int m_samples = 1;

        [[nodiscard]] bool Changed(const RayProperties &properties) const {
            return properties.m_bounces != m_bounces ||
                   properties.m_samples != m_samples;
        }

        void OnInspect();
    };

    struct RAY_TRACER_FACILITY_API RayTracerProperties {
        EnvironmentProperties m_environment;
        RayProperties m_rayProperties;

        [[nodiscard]] bool Changed(const RayTracerProperties &properties) const {
            return m_environment.Changed(properties.m_environment) ||
                   m_rayProperties.Changed(properties.m_rayProperties);
        }

        void OnInspect();
    };

    enum class RayType {
        Radiance, SpacialSampling, RayTypeCount
    };

    struct CameraRenderingLaunchParams {
        CameraProperties m_cameraProperties;
        RayTracerProperties m_rayTracerProperties;
        OptixTraversableHandle m_traversable;
    };

    template<typename T>
    struct RAY_TRACER_FACILITY_API IlluminationSampler {
				UniEngine::Vertex m_a;
				UniEngine::Vertex m_b;
				UniEngine::Vertex m_c;
        /**
         * \brief The calculated overall direction where the triangle received most
         * light.
         */
        glm::vec3 m_direction;
        /**
         * \brief The total energy received at this triangle.
         */
        T m_energy;
				bool m_frontFace = true;
				bool m_backFace = true;

				[[nodiscard]] float GetArea() const{
						const float a = glm::distance(m_a.m_position, m_b.m_position);
						const float b = glm::distance(m_b.m_position, m_c.m_position);
						const float c = glm::distance(m_c.m_position, m_a.m_position);
						const float p = (a + b + c) * 0.5f;
						return glm::sqrt(p * (p - a) * (p - b) * (p - c));
				}
    };

	struct IlluminationEstimationLaunchParams {
        unsigned m_seed = 0;
        float m_pushNormalDistance = 0.001f;
        size_t m_size;
        RayTracerProperties m_rayTracerProperties;
        IlluminationSampler<glm::vec3> *m_lightProbes;
        OptixTraversableHandle m_traversable;
    };

    struct RAY_TRACER_FACILITY_API PointCloudSample {
        // Input
        glm::vec3 m_direction = glm::vec3(0.0f);
        glm::vec3 m_start = glm::vec3(0.0f);

        // Output
        uint64_t m_handle = 0;
        bool m_hit = false;

        HitInfo m_hitInfo;
    };

    struct PointCloudScanningLaunchParams {
        size_t m_size;
        PointCloudSample *m_samples;
        OptixTraversableHandle m_traversable;
    };

#pragma endregion
    struct RAY_TRACER_FACILITY_API RayTracedTexture {
        unsigned m_textureId = 0;
    };

    struct SurfaceMaterial;

    struct RAY_TRACER_FACILITY_API RayTracedMaterial {
        MaterialType m_materialType = MaterialType::Default;

        BTFBase *m_btfBase;
        UniEngine::MaterialProperties m_materialProperties;

        RayTracedTexture m_albedoTexture;
        RayTracedTexture m_normalTexture;
        RayTracedTexture m_metallicTexture;
        RayTracedTexture m_roughnessTexture;

        size_t m_version = -1;
        uint64_t m_handle = 0;

        CudaBuffer m_materialBuffer;

        bool m_removeFlag = true;

        void UploadForSBT(std::vector<std::pair<unsigned, cudaTextureObject_t>> &boundTextures,
                          std::vector<cudaGraphicsResource_t> &boundResources);

        void BindTexture(unsigned int id, cudaGraphicsResource_t &graphicsResource, cudaTextureObject_t &textureObject);
    };

    enum class CurveMode {
        Linear,
        Quadratic,
        Cubic
    };

    struct RAY_TRACER_FACILITY_API RayTracedGeometry {
        RendererType m_rendererType = RendererType::Default;
        GeometryType m_geometryType = GeometryType::Triangle;
        union {
            std::vector<UniEngine::Vertex> *m_vertices = nullptr;
            std::vector<UniEngine::SkinnedVertex> *m_skinnedVertices;
            std::vector<UniEngine::StrandPoint> *m_curvePoints;
        };

        std::vector<glm::mat4> *m_boneMatrices = nullptr;
        std::vector<glm::vec4>* m_instanceColors = nullptr;
        std::vector<glm::mat4> *m_instanceMatrices = nullptr;
        union {
            std::vector<glm::uvec3> *m_triangles = nullptr;
            std::vector<glm::uint> *m_curveSegments;
        };
        
        OptixTraversableHandle m_traversableHandle = 0;

        CudaBuffer m_vertexDataBuffer;
        CudaBuffer m_curveStrandUBuffer;
        CudaBuffer m_curveStrandIBuffer;
        CudaBuffer m_curveStrandInfoBuffer;


        CudaBuffer m_triangleBuffer;
        CudaBuffer m_acceleratedStructureBuffer;
        size_t m_version = -1;
        uint64_t m_handle = 0;
        bool m_updateFlag = false;
        bool m_removeFlag = true;

        void BuildGAS(const OptixDeviceContext &context);

        void UploadForSBT();

        CudaBuffer m_geometryBuffer;
    };


    struct RAY_TRACER_FACILITY_API RayTracedInstance {
        uint64_t m_entityHandle = 0;
        size_t m_version = -1;
        size_t m_dataVersion = -1;
        uint64_t m_privateComponentHandle = 0;

        
        uint64_t m_geometryMapKey = 0;
        uint64_t m_materialMapKey = 0;
        glm::mat4 m_globalTransform;
        bool m_removeFlag = true;
    };

    struct RayTracerPipeline {
        std::string m_launchParamsName;
        OptixModule m_module;
        OptixModule m_quadraticCurveModule;
        OptixModule m_cubicCurveModule;
        OptixModule m_linearCurveModule;
        OptixModuleCompileOptions m_moduleCompileOptions = {};

        OptixPipeline m_pipeline;
        OptixPipelineCompileOptions m_pipelineCompileOptions = {};
        OptixPipelineLinkOptions m_pipelineLinkOptions = {};

        OptixProgramGroup m_rayGenProgramGroups;
        CudaBuffer m_rayGenRecordsBuffer;
        std::map<RayType, std::map<GeometryType, OptixProgramGroup>> m_hitGroupProgramGroups;
        CudaBuffer m_missRecordsBuffer;
        std::map<RayType, OptixProgramGroup> m_missProgramGroups;
        CudaBuffer m_hitGroupRecordsBuffer;
        OptixShaderBindingTable m_sbt = {};
        CudaBuffer m_launchParamsBuffer;
    };
    struct SurfaceCompressedBTF;
    struct MLVQMaterialStorage {
        std::shared_ptr<SurfaceCompressedBTF> m_material;
        CudaBuffer m_buffer;
    };


    class RayTracer {
    public:
        bool m_requireUpdate = false;
        std::unordered_map<uint64_t, RayTracedMaterial> m_materials;
        std::unordered_map<uint64_t, RayTracedGeometry> m_geometries;
        std::unordered_map<uint64_t, RayTracedInstance> m_instances;

        // ------------------------------------------------------------------
        // internal helper functions
        // ------------------------------------------------------------------
        [[nodiscard]] bool
        RenderToCamera(const EnvironmentProperties &environmentProperties,
                       CameraProperties &cameraProperties,
                       const RayProperties &rayProperties);

        void EstimateIllumination(const size_t &size,
                                  const EnvironmentProperties &environmentProperties,
                                  const RayProperties &rayProperties,
                                  CudaBuffer &lightProbes, unsigned seed,
                                  float pushNormalDistance);

        void ScanPointCloud(const size_t &size,
                            const EnvironmentProperties &environmentProperties,
                            CudaBuffer &samples);

        RayTracer();

        /*! build an acceleration structure for the given triangle mesh */
        void BuildIAS();

        /*! constructs the shader binding table */
        void BuildSBT(
                std::vector<std::pair<unsigned, cudaTextureObject_t>>
                &boundTextures,
                std::vector<cudaGraphicsResource_t> &boundResources);

        void LoadBtfMaterials(const std::vector<std::string> &folderPathes);

    protected:

#pragma region Device and context
        /*! @{ CUDA device context and stream that optix pipeline will run
                on, as well as device properties for this device */
        CUcontext m_cudaContext;
        CUstream m_stream;
        cudaDeviceProp m_deviceProps;
        /*! @} */
        //! the optix context that our pipeline will run in.
        OptixDeviceContext m_optixDeviceContext;

        friend class CameraProperties;

        /*! creates and configures a optix device context (in this simple
          example, only for the primary GPU device) */
        void CreateContext();

#pragma endregion
#pragma region Pipeline setup

        CameraRenderingLaunchParams m_cameraRenderingLaunchParams;
        IlluminationEstimationLaunchParams m_illuminationEstimationLaunchParams;
        PointCloudScanningLaunchParams m_pointCloudScanningLaunchParams;

        RayTracerPipeline m_cameraRenderingPipeline;
        RayTracerPipeline m_illuminationEstimationPipeline;
        RayTracerPipeline m_pointCloudScanningPipeline;

        /*! creates the module that contains all the programs we are going
          to use. in this simple example, we use a single module from a
          single .cu file, using a single embedded ptx string */
        void CreateModules();

        /*! does all setup for the rayGen program(s) we are going to use */
        void CreateRayGenPrograms();

        /*! does all setup for the miss program(s) we are going to use */
        void CreateMissPrograms();

        /*! does all setup for the hitGroup program(s) we are going to use */
        void CreateHitGroupPrograms();

        /*! assembles the full pipeline of all programs */
        void AssemblePipelines();

        void CreateRayGenProgram(RayTracerPipeline &targetPipeline,
                                 char entryFunctionName[]) const;

        void CreateModule(RayTracerPipeline &targetPipeline, char ptxCode[],
                          char launchParamsName[]) const;

        void AssemblePipeline(RayTracerPipeline &targetPipeline) const;

#pragma endregion

#pragma region Accleration structure
        /*! check if we have build the acceleration structure. */
        bool m_hasAccelerationStructure = false;
        //! buffer that keeps the (final, compacted) acceleration structure
        CudaBuffer m_iASBuffer;
#pragma endregion

        friend class RayTracerCamera;
    };

} // namespace RayTracerFacility
