# Rhizomorph: The Coordinated Function of Shoots and Roots

## [Project Page](https://storage.googleapis.com/pirk.io/projects/rhizomorph/index.html) | [Video](https://vimeo.com/819321601) | [Paper](https://dl.acm.org/doi/pdf/10.1145/3592145)
This repository is an offical implementation of the paper [Rhizomorph: The Coordinated Function of Shoots and Roots](https://storage.googleapis.com/pirk.io/projects/rhizomorph/index.html).

## Build instructions:
[! ! !]Prior to building process, please unzip the /Source Code/UniEngine/3rdParty/physx/physx.zip*.

### Step 1: Visual Studio
Please make sure you installed Visual Studio 2022 with “Desktop development with C++” selected:
![Pic1](/Resources/GitHub/Picture1.png?raw=true "Pic1")

### Step 2: Open as CMake project:
Open the EcoSysLab folder as a project in Visual Studio 2022. The Visual Studio will automatically recognize it as a 
CMake project. 
![Pic2](/Resources/GitHub/Picture2.png?raw=true "Pic2")

### Step 3: Setup building configurations:
Wait for a little while until Visual Studio loaded the CMake project, and you should see building configurations 
are set:
![Pic3](/Resources/GitHub/Picture3.png?raw=true "Pic3")

### Step 4: Build:
Open the drop-down menu for building target by clicking the button highlighted with blue box, and select 
EcoSysLab.exe and click start button marked with red box to start building
![Pic4](/Resources/GitHub/Picture4.png?raw=true "Pic4")
![Pic5](/Resources/GitHub/Picture5.png?raw=true "Pic5")

## Application instructions:
### 1. Once you have the framework opened, click the “Create or load New Project” button:
![Pic6](/Resources/GitHub/Picture6.png?raw=true "Pic6")

### 2. In the file dialog, select the “Project.ueproj” provided in the /SourceCode/Project folder
![Pic7](/Resources/GitHub/Picture7.png?raw=true "Pic7")

### 3. You should be able to see the project is loaded and the framework’s running. The project folder comes with a sample scene. Select EcoSysLab Layer panel and check “Auto grow” box to see the tree start growing. To move the camera in space, press and hold mouse right button and move with "W, A, S, D, Shift, Ctrl". To rotate camera, press and hold mouse right button and move the mouse.
![Pic8](/Resources/GitHub/Picture8.png?raw=true "Pic8")

You should be able to see the tree with its root system:
![Pic9](/Resources/GitHub/Picture9.png?raw=true "Pic9")

### 4. Click "Generate Meshes" button in EcoSysLab Layer panel and navigate to scene panel you will see the mesh for tree is generated.
![Pic10](/Resources/GitHub/Picture10.png?raw=true "Pic10")

### 5. To export mesh as OBJ, select (either from Entity Explorer panel or directly click the object in scene panel) the target entity named "Branch Mesh" as one of the children of "Butter" entity which contains the MeshRenderer that provides the rendering for branches.
Find the Temporary Mesh button which links to the branch mesh asset, double click it and it will show up in the Asset Inspector panel. You can then click Export as OBJ button to save branch mesh as OBJ on your disk.
![Pic11](/Resources/GitHub/Picture11.png?raw=true "Pic11")

### 6. You can export branch mesh, root mesh, and foliage mesh, but not fine root mesh. The fine root mesh is not stored as triangular mesh, but curves that is expand to mesh with geometry shader within the framework on GPU. Here's a screenshot of what does exported meshes look like in MeshLab:
![Pic12](/Resources/GitHub/Picture12.png?raw=true "Pic12")