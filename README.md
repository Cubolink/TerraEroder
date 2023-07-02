# TerraEroder
An approach to terrain erosion using C++ + CUDA and visualization with OpenGL.
It is also able to display the contour lines using shaders.

## Instrucciones de compilación

### Build Cmake Project
First, you need to build the CMake project. You can use the following command to create the project inside the `build` folder.

```
mkdir build
cmake -S . -B build
```

You may want or need to use a specific generator.

* For example, if you want to compile with MinGW, you can pass the flag `-G "MinGW Makefiles"`.
* Or if you want to use Visual Studio 2017, compiling for 64 bits, use `-G "Visual Studio 15 2017 Win64"`

Please refer to `cmake --help` to check the available generators. Make sure you use one compatible with the compiler you want to use.

### Build/Compile the project build sources and run the application.

Once you have successfully ran cmake, you can go to your build folder (`cd build`) and compile it. Different compilers use
different ways, but here we provide some common cases.

**Unix-like (g++)**
```
make
```
The executable should be inside the `build/src` folder, named as `terra_eroder.exe`.

**MinGw**
```
mingw32-make
```
The executable should be inside the `build/src` folder, named as `terra_eroder.exe`.

**Visual Studio desde la terminal**
```
msbuild terraeroder.sln
```
The executable should be inside the `build/src/Debug` folder, named as `terra_eroder.exe`. You will need, probably, an extra step:
to move the executable into `build/src`. This is due to the program trying to load the project `resources` and `data`
folders and failing because of being farther away than expected.

**Visual Studio desde el IDE**

Open the `terraeroder.sln` file with your Visual Studio and compile and execute the `terra_eroder` project from there.

## Executing instructions

When running the executable, you will find yourself with a 1st person camera, and a ImGui window.

*Camera controls*

```
WASD -> Move around (front/back/left/right)
Space -> Move up
L-Shift -> Move down

Flechas -> Look around
```

**Change the light position**

Using the ImGui window, you can move the light position using sliders for the three coordinates.

**Contour lines**

By pressing `TAB`, or selecting a shader in the ImGui window, you can choose how to display the terrain.
You can use this to alternate between displaying the terrain colorized by height, displaying the colored contour lines
over a gray terrain, or displaying the terrain triangles.

Also, in the ImGui window you can configure how many contour lines to render, and use its sliders to change their heights.

## Project structure

`dependencies` folder has the needed third party dependencies for the whole project (glad, glfw). `src/vendor` also has
some third party dependencies, but those work like external subprojects.

`resources` has the shaders and textures used during the program execution. The `data` folder contains the `terrain.obj`.

`src/renderer` is a custom library with abstractions to render with OpenGL.

The remaining files in `src` are part of the main program.