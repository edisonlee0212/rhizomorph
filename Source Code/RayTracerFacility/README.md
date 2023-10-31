# RayTracerFacility
RayTracerFacility is a module of UniEngine that provides a simple ray tracer with path tracing based on OptiX 7 and CUDA. It also provide per-triangle illumination estimator for meshes that is used on my procedural sorghum projects.

## Getting Started
The project is a CMake project. For project editing, code inspections, Visual Studio 2017 or 2019 is recommanded. Simply clone/download the project files and open the folder as project in Visual Studio and you are ready.
To directly build the project, scripts under the root folder build.cmd (for Windows) and build.sh (for Linux) are provided for building with single command line.
E.g. For Linux, the command may be :
 - bash build.sh (build in default settings)
 - bash build.sh --clean release (clean and build in release mode)
Please visit script for further details.
## Main features
 - Simple ray tracer with Monte Carlo BRDF and efficient importance sampling, with auto scene syncronization with UniEngine - "You see in scene window, you see in ray tracer."
   - Screenshot: ![RayTracerScreenshot](/Resources/GitHub/BRDF.png?raw=true "BRDFScreenshot")
   - Screenshot: ![IndirectLightingScreenshot](/Resources/GitHub/IndirectLighting.png?raw=true "IndirectLightingScreenshot")
 - OptiX Ai Denoiser
 - CUDA version of CompressedBTF library that supports rendering compressed BTF materials with OptiX.
   - Screenshot: ![MLVQScreenshot](/Resources/GitHub/CompressedBTF.png?raw=true "MLVQScreenshot")
 - TriangularMesh surface illumination estimation
   - Screenshot: ![IlluminationScreenshot](/Resources/GitHub/Illumination.png?raw=true "IlluminationScreenshot")
 - Virtual Laser Scanner/LiDAR for point cloud generation in virtual scene 
   - Screenshot: ![PointCloudScreenshot](/Resources/GitHub/VirtualScan.png?raw=true "PointCloudScreenshot")
