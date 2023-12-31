//
// Created by Cubolink on 19-06-2023.
//

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>

#include "renderer/renderer.h"
#include "shape_factory.h"
#include "renderer/CudaShape.h"
#include "obj_files.h"
#include "controller.h"
#include "display_controller.h"

// CUDA kernel interface
extern "C"
void cudaRunOscilateKernel(dim3 gridSize, dim3 blockSize, float t,
                           float4* verticesGrid);

extern "C"
void cudaRunNormalsKernel(dim3 gridSize, dim3 blockSize, float4* verticesGrid, float3* normals);


extern "C"
void cudaUpdateVBO(dim3 gridSize, dim3 blockSize, float4* cudaVerticesGrid, float3* cudaNormalsGrid, float* sedimentGrid,
                   float* verticesVBO, unsigned int width, unsigned int height);

extern "C"
void cudaRunErodeKernel(dim3 gridSize, dim3 blockSize, float dt, float dx, float dy, float4* verticesGrid,
                        float3* normals, float4* waterOutflowFlux, float* suspendedSediment);

int w_width = 1024;
int w_height = 576;
float w_proportion;

glm::mat4 projection_m;
glm::vec3 translation;
glm::mat4 model_m;

CameraController cameraController;
DisplayController displayController;

std::vector<float> terrain_vertices;
std::vector<std::vector<float4>> terrainGrid;
std::vector<std::vector<float3>> normalsGrid;
struct BoundingBox {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;

    BoundingBox()
    : min_x(std::numeric_limits<float>::infinity()),
    min_y(std::numeric_limits<float>::infinity()),
    min_z(std::numeric_limits<float>::infinity()),
    max_x(-std::numeric_limits<float>::infinity()),
    max_y(-std::numeric_limits<float>::infinity()),
    max_z(-std::numeric_limits<float>::infinity())
    {}
};

struct CudaParams {
    unsigned int numBlocks;
    unsigned int numBodies;
    dim3 blockSize;
    dim3 gridSize;
    float4* dataGrid;  // May have padding, may be bigger than the CPU and GL terrainGrid
    float3* normalsGrid;
    float4* waterOutFlowFluxGrid;
    float* suspendedSediment;

    // Constructors
    CudaParams(unsigned int numBlocks, unsigned int numBodies, dim3 blockSize, dim3 gridSize,
               float4* dataGrid, float3* normalsGrid, float4* waterOutFlowFluxGrid, float* suspendedSediment)
            : numBlocks(numBlocks), numBodies(numBodies), blockSize(blockSize), gridSize(gridSize),
            dataGrid(dataGrid), normalsGrid(normalsGrid), waterOutFlowFluxGrid(waterOutFlowFluxGrid), suspendedSediment(suspendedSediment)
    {
    }

    CudaParams()
            : numBlocks(0), numBodies(0), dataGrid(nullptr), normalsGrid(nullptr), waterOutFlowFluxGrid(nullptr),
            suspendedSediment(nullptr)
    {
    }
};
CudaParams cudaParams;

void resize_callback(GLFWwindow* window, int width, int height)
{
    w_width = width;
    w_height = height;

    glViewport(0, 0, width, height);
    w_proportion = ((float) w_height) / ((float) w_width);
    projection_m = glm::perspective(glm::radians(50.0f), (float) 1/w_proportion, 0.1f, 100.0f);
}

/**
 * Handles the glfw key_callback, changing parameters in the cameraController
 * @param window
 * @param key
 * @param scancode
 * @param action
 * @param mods
 */
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    switch (key) {
        case GLFW_KEY_W:
            cameraController.m_forth = (action == GLFW_PRESS || action == GLFW_REPEAT) ? 1: 0;
            break;
        case GLFW_KEY_S:
            cameraController.m_back = (action == GLFW_PRESS || action == GLFW_REPEAT) ? 1: 0;
            break;
        case GLFW_KEY_A:
            cameraController.m_left = (action == GLFW_PRESS || action == GLFW_REPEAT) ? 1: 0;
            break;
        case GLFW_KEY_D:
            cameraController.m_right = (action == GLFW_PRESS || action == GLFW_REPEAT) ? 1: 0;
            break;

        case GLFW_KEY_SPACE:
            cameraController.m_up = (action == GLFW_PRESS || action == GLFW_REPEAT) ? 1: 0;
            break;
        case GLFW_KEY_LEFT_SHIFT:
            cameraController.m_down = (action == GLFW_PRESS || action == GLFW_REPEAT) ? 1: 0;
            break;

        case GLFW_KEY_LEFT:
            cameraController.rot_left = (action == GLFW_PRESS || action == GLFW_REPEAT) ? 1: 0;
            break;
        case GLFW_KEY_RIGHT:
            cameraController.rot_right = (action == GLFW_PRESS || action == GLFW_REPEAT) ? 1: 0;
            break;
        case GLFW_KEY_UP:
            cameraController.rot_up = (action == GLFW_PRESS || action == GLFW_REPEAT) ? 1: 0;
            break;
        case GLFW_KEY_DOWN:
            cameraController.rot_down = (action == GLFW_PRESS || action == GLFW_REPEAT) ? 1: 0;
            break;

        case GLFW_KEY_TAB:
            if (action == GLFW_PRESS)
                displayController.toggleDisplay();
            break;
        case GLFW_KEY_N:
            if (action == GLFW_PRESS)
                displayController.toggleNormals();
            break;

        default:
            break;
    }
}


void initTerrainGrid()
{
    // terrain vertices is not necessarily a grid, it may have better triangulations.
    // but we do need a grid, in order to map it like m[x][y] = z in a matrix.

    unsigned int mn = terrain_vertices.size() / 8;  // {x,y,z,nx,ny,nz,water,sediment} -> 8 elements per vertex
    // assume that it's a square grid
    unsigned int m, n;
    m = n = (unsigned int) sqrt(mn);
    if (m * n != mn)
    {
        std::cout << "Unsupported non-square grid terrain" << std::endl;
        exit(1);
    }
    std::cout << "Loaded a squared grid terrain. "
                 "If that assumption is incorrect, the behaviour of this program is undefined." << std::endl;
    terrainGrid = std::vector<std::vector<float4>>(m, std::vector<float4>(n));
    normalsGrid = std::vector<std::vector<float3>>(m, std::vector<float3>(n));
    // ix from 0 to m
    // jy from 0 to n
    // matrix goes like      x y->
    //                       |
    //                      \|/
    for (unsigned int ix = 0; ix < m; ix++)
    {
        for (unsigned int jy = 0; jy < n; jy++)
        {
            unsigned int v = ix * n + jy;  // v contains info of x,y,z,normals,water,sediment, so we need 8*v+2 to get z
            /*
            std::cout << "terrainGrid["<<ix<<"]["<<jy<<"]="
                                                   "("<< terrain_vertices[8*v] <<", "
                                                   << terrain_vertices[8*v+1] << ", "
                                                   << terrain_vertices[8*v+2] << ")"<<std::endl;
                                                   */
            terrainGrid[ix][jy] = {
                    terrain_vertices[8 * v + 0],
                    terrain_vertices[8 * v + 1],
                    terrain_vertices[8 * v + 2],
                    terrain_vertices[8 * v + 6]
            };
            normalsGrid[ix][jy] = {
                    terrain_vertices[8 * v + 3],
                    terrain_vertices[8 * v + 4],
                    terrain_vertices[8 * v + 5],
            };
        }
    }

}


void initGL()
{
    // Configuring GL
    glad_glClearColor(0.43f, 0.39f, 0.48f, 1.f);
    glad_glEnable(GL_DEPTH_TEST);  // enable depth buffer
    glad_glEnable(GL_CLIP_DISTANCE0);  // clipping distance
    glad_glFrontFace(GL_CCW);  // set ccw mode, useful in particular for culling
    //glad_glEnable(GL_CULL_FACE);
    //glad_glCullFace(GL_BACK);
    glad_glEnable(GL_BLEND);  // enable transparency
    glad_glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  // set blending function
    glad_glBlendEquation(GL_FUNC_ADD);  // set blending reslts mixer function
}


void initCUDA() {
    unsigned int m = terrainGrid.size();
    unsigned int n = terrainGrid[0].size();

    cudaParams.blockSize.x = 16;
    cudaParams.blockSize.y = 16;

    cudaParams.gridSize.x = m / cudaParams.blockSize.x;
    if (cudaParams.gridSize.x * cudaParams.blockSize.x < m)
        cudaParams.gridSize.x++;  // x-padding
    cudaParams.gridSize.y = m / cudaParams.blockSize.y;
    if (cudaParams.gridSize.y * cudaParams.blockSize.y < n)
        cudaParams.gridSize.y++;  // y-padding

    cudaParams.numBlocks = (cudaParams.gridSize.x * cudaParams.gridSize.y);
    cudaParams.numBodies = (cudaParams.blockSize.x * cudaParams.blockSize.y) * cudaParams.numBlocks;  // this > m*n if there was any padding

    unsigned int cudaM = cudaParams.gridSize.x * cudaParams.blockSize.x;
    unsigned int cudaN = cudaParams.gridSize.y * cudaParams.blockSize.y;
    std::vector<float4> dataGrid1D;
    std::vector<float3> normalsGrid1D;
    for (unsigned int x = 0; x < m; x++)
    {
        // copy the n elements of the x-row
        dataGrid1D.insert(dataGrid1D.end(), terrainGrid[x].begin(), terrainGrid[x].end());
        normalsGrid1D.insert(normalsGrid1D.end(), normalsGrid[x].begin(), normalsGrid[x].end());
        // fill the rest until cudaN (padding)
        dataGrid1D.resize((x + 1) * cudaN);
        normalsGrid1D.resize((x + 1) * cudaN);
        for (unsigned int y = n; y < cudaN; y++)
        {
            dataGrid1D[x*cudaN + y] = terrainGrid[x][n-1];  //fill the padded values with the last non-padded in the row
            normalsGrid1D[x*cudaN + y] = normalsGrid[x][n-1];
        }
    }
    dataGrid1D.resize(cudaParams.numBodies);
    // fill the padded rows as copies of the last non-padded row
    for (unsigned int x = m; x < cudaM; x++)
        for (unsigned int y = 0; y < cudaN; y++)
            dataGrid1D[x*cudaN + y] = dataGrid1D[(m-1)*cudaN + y];

    // Initialize a matrix in CUDA
    cudaMalloc((void**) &(cudaParams.dataGrid), cudaParams.numBodies * sizeof(float4));
    cudaMemcpy(cudaParams.dataGrid, dataGrid1D.data(), dataGrid1D.size() * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &(cudaParams.normalsGrid), cudaParams.numBodies * sizeof(float3));
    cudaMemcpy(cudaParams.normalsGrid, normalsGrid1D.data(), normalsGrid1D.size() * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &(cudaParams.waterOutFlowFluxGrid), cudaParams.numBodies * sizeof(float4));
    cudaMemset(cudaParams.waterOutFlowFluxGrid, 0, cudaParams.numBodies * sizeof(float4));
    cudaMalloc((void**) &(cudaParams.suspendedSediment), cudaParams.numBodies * sizeof(float));
    cudaMemset(cudaParams.suspendedSediment, 0, cudaParams.numBodies * sizeof(float));
}


void updateModel(CudaShape& terrain, double t, double dt)
{
    // Run the kernel
    cudaRunErodeKernel(cudaParams.gridSize, cudaParams.blockSize, (float) dt,
                       terrainGrid[1][0].x - terrainGrid[0][0].x,
                       terrainGrid[0][1].y - terrainGrid[0][0].y,
                       cudaParams.dataGrid,
                       cudaParams.normalsGrid,
                       cudaParams.waterOutFlowFluxGrid,
                       cudaParams.suspendedSediment);
    cudaDeviceSynchronize();
    cudaRunNormalsKernel(cudaParams.gridSize, cudaParams.blockSize,
                         cudaParams.dataGrid,
                         cudaParams.normalsGrid);
    /*
    cudaDeviceSynchronize();
    cudaRunOscilateKernel(cudaParams.gridSize, cudaParams.blockSize, (float) t, cudaParams.dataGrid);
    cudaDeviceSynchronize();
    std::vector<float> download(cudaParams.numBodies);
    cudaMemcpy(download.data(), cudaParams.terrainGrid, cudaParams.numBodies * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    */
    glad_glFinish();
    float *dPtr = nullptr;
    size_t numBytes;
    terrain.cudaMap(&dPtr, &numBytes);

    cudaDeviceSynchronize();
    cudaUpdateVBO(cudaParams.gridSize, cudaParams.blockSize, cudaParams.dataGrid, cudaParams.normalsGrid,
                  cudaParams.suspendedSediment, dPtr, terrainGrid.size(), terrainGrid[0].size());
    cudaDeviceSynchronize();
    terrain.cudaUnmap();
}


int main() {

    GLFWwindow *window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(w_width, w_height, "TerraEroder", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);
    /*Make a key callback for the window */
    glfwSetKeyCallback(window, key_callback);
    glfwSetFramebufferSizeCallback(window, resize_callback);

    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress))
        return -1;
    std::cout << "Current GL Version: " << glad_glGetString(GL_VERSION) << std::endl;

    glfwMaximizeWindow(window);
    glfwGetWindowSize(window, &w_width, &w_height);
    initGL();

    // Init Camera
    Camera camera = Camera();
    camera.setEye(0, 0, 0);
    camera.setCenter(0, 0, -1);
    cameraController.setCamera(&camera);

    // Init transformations
    w_proportion = ((float) w_height) / ((float) w_width);
    projection_m = glm::perspective(glm::radians(50.0f), (float) 1/w_proportion, 0.1f, 100.0f);
    translation = glm::vec3(0, 0, -2.f);
    model_m = glm::translate(glm::mat4(1.0f), translation);

    // Init shaders
    Shader c_mpv_shaderProgram = Shader("../resources/shaders/color_mpv_shader.shader");
    Shader t_mpv_shaderProgram = Shader("../resources/shaders/texture_mpv_shader.shader");
    Shader gouraud_c_mpv_shaderProgram = Shader("../resources/shaders/gouraud_color_mpv.shader");
    Shader terrain_shaderProgram = Shader("../resources/shaders/gouraud_mpv_terrain_shader.shader");
    Shader terrainLines_shaderProgram = Shader("../resources/shaders/gouraud_mpv_terrain_trianglelines.shader");
    Shader grayTerrain_shaderProgram = Shader("../resources/shaders/gouraud_mpv_terrain_single_color.shader");
    Shader isolines_shaderProgram = Shader("../resources/shaders/isolines_gouraud_mpv_terrain.shader");
    Shader water_shaderProgram = Shader("../resources/shaders/gouraud_mpv_water_over_terrain.shader");
    Shader terrainNormals_shaderProgram = Shader("../resources/shaders/gouraud_mpv_terrain_face_normals.shader");

    // Init texture and shapes
    Texture texture = Texture("../resources/textures/red_yoshi.png");
    Shape square_shape = ShapeFactory::createTextureQuad();
    Shape axis_shape = ShapeFactory::createColorAxis(1);
    Shape normal_color_cube_shape = ShapeFactory::createColorNormalCube(.2f, .3f, .7f);
    CudaShape terrain(Obj::readFile("../../data/terrain(100x100-res5).obj"));
    BoundingBox terrainBB;

    terrain_vertices = terrain.getVertices();
    initTerrainGrid();
    initCUDA();
    for (int i = 0; i < terrain_vertices.size(); i += 8)
    {
        if (terrain_vertices[i] > terrainBB.max_x)
            terrainBB.max_x = terrain_vertices[i];
        if (terrain_vertices[i] < terrainBB.min_x)
            terrainBB.min_x = terrain_vertices[i];
        if (terrain_vertices[i+1] > terrainBB.max_y)
            terrainBB.max_y = terrain_vertices[i+1];
        if (terrain_vertices[i+1] < terrainBB.min_y)
            terrainBB.min_y = terrain_vertices[i+1];
        if (terrain_vertices[i+2] > terrainBB.max_z)
            terrainBB.max_z = terrain_vertices[i+2];
        if (terrain_vertices[i+2] < terrainBB.min_z)
            terrainBB.min_z = terrain_vertices[i+2];
    }
    float terrain_z_range = terrainBB.max_z - terrainBB.min_z;
    float terrain_water_level = terrain_z_range/3 + terrainBB.min_z;
    std::cout << "min/max (x, y, z): ("
            << terrainBB.min_x << "->" << terrainBB.max_x << ", "
            << terrainBB.min_y << "->" << terrainBB.max_y << ", "
            << terrainBB.min_z << "->" << terrainBB.max_z << ")"
            << std::endl;
    std::cout << "water_level: " << terrain_water_level;

    // Init materials
    Material cube_material = Material(0.5f, 0.6f, 0.4f, 100);
    Material water_material = Material(0.5f, 0.8f, 1.f, 100);
    glm::vec3 lightPos((terrainBB.min_x+terrainBB.max_x)/2,
                       (terrainBB.min_y+terrainBB.max_y)/2,
                       terrainBB.max_z+45);
    Light light = Light(1.0f, 1.0f, 1.0f, lightPos,
                        0.01f, 0.01f, 0.0001f);

    /* Starting main program */

    Renderer renderer = Renderer();

    // ImGui setup
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();
    const char* shaderSelectorOptions[] = {"Colored terrain", "Gray terrain + colored contour lines", "Terrain triangle lines"};
    static int currentShaderIndex = 0;
    // Set isoline levels
    static int nIsolines = 20;
    static std::vector<float> isoLines(nIsolines);
    for (int kIsoline = 0; kIsoline < nIsolines; kIsoline++) {
        isoLines[kIsoline] = terrainBB.min_z + terrain_z_range * float(kIsoline) / float(nIsolines);
    }
    static float sedimentColorIntensity = 1.f;

    double t0 = glfwGetTime();
    double t1, dt;

    glfwSwapInterval(1);  // vsync
    /* Loop until the user closes the window */
    while(!glfwWindowShouldClose(window))
    {
        t1 = glfwGetTime();
        dt = fmin(t1 - t0, 1.f/30.f);  // So if we lag too much, we will slow down. Improves simulation stability
        t0 = t1;

        updateModel(terrain, t1, dt);
        light.setPosition(lightPos);
        if (isoLines.size() != nIsolines)
        {
            // Update isolines and reset levels
            isoLines.resize(nIsolines);
            for (int kIsoline = 0; kIsoline < nIsolines; kIsoline++) {
                isoLines[kIsoline] = terrainBB.min_z + terrain_z_range * float(kIsoline) / float(nIsolines);
            }
        }

        renderer.Clear();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        cameraController.updateCameraProperties();
        camera.updateCoords((float) dt);

        t_mpv_shaderProgram.Bind();
        t_mpv_shaderProgram.SetUniformMat4f("u_projection", projection_m);
        t_mpv_shaderProgram.SetUniformMat4f("u_view", camera.getViewMatrix());
        t_mpv_shaderProgram.SetUniformMat4f("u_model", model_m);
        t_mpv_shaderProgram.SetUniform1i("u_texture", 0);

        c_mpv_shaderProgram.Bind();
        c_mpv_shaderProgram.SetUniformMat4f("u_projection", projection_m);
        c_mpv_shaderProgram.SetUniformMat4f("u_view", camera.getViewMatrix());
        c_mpv_shaderProgram.SetUniformMat4f("u_model", model_m);

        glm::vec3 cam_pos = camera.getEyeVec3();
        gouraud_c_mpv_shaderProgram.Bind();
        gouraud_c_mpv_shaderProgram.SetUniformMat4f("u_projection", projection_m);
        gouraud_c_mpv_shaderProgram.SetUniformMat4f("u_view", camera.getViewMatrix());
        gouraud_c_mpv_shaderProgram.SetUniformMat4f("u_model", model_m);
        gouraud_c_mpv_shaderProgram.SetUniform3f("u_viewPosition", cam_pos.x, cam_pos.y, cam_pos.z);

        terrain_shaderProgram.Bind();
        terrain_shaderProgram.SetUniformMat4f("u_projection", projection_m);
        terrain_shaderProgram.SetUniformMat4f("u_view", camera.getViewMatrix());
        terrain_shaderProgram.SetUniformMat4f("u_model", model_m);
        terrain_shaderProgram.SetUniform3f("u_viewPosition", cam_pos.x, cam_pos.y, cam_pos.z);
        terrain_shaderProgram.SetUniform1f("u_waterLevel", terrain_water_level);
        terrain_shaderProgram.SetUniform1f("u_deepestLevel", terrainBB.min_z);
        terrain_shaderProgram.SetUniform1f("u_levelRange", terrain_z_range);

        water_shaderProgram.Bind();
        water_shaderProgram.SetUniformMat4f("u_projection", projection_m);
        water_shaderProgram.SetUniformMat4f("u_view", camera.getViewMatrix());
        water_shaderProgram.SetUniformMat4f("u_model", model_m);
        water_shaderProgram.SetUniform3f("u_viewPosition", cam_pos.x, cam_pos.y, cam_pos.z);
        water_shaderProgram.SetUniform1f("u_sedimentColorIntensity", sedimentColorIntensity);

        terrainLines_shaderProgram.Bind();
        terrainLines_shaderProgram.SetUniformMat4f("u_projection", projection_m);
        terrainLines_shaderProgram.SetUniformMat4f("u_view", camera.getViewMatrix());
        terrainLines_shaderProgram.SetUniformMat4f("u_model", model_m);
        terrainLines_shaderProgram.SetUniform3f("u_viewPosition", cam_pos.x, cam_pos.y, cam_pos.z);
        terrainLines_shaderProgram.SetUniform1f("u_waterLevel", terrain_water_level);
        terrainLines_shaderProgram.SetUniform1f("u_deepestLevel", terrainBB.min_z);
        terrainLines_shaderProgram.SetUniform1f("u_levelRange", terrain_z_range);

        grayTerrain_shaderProgram.Bind();
        grayTerrain_shaderProgram.SetUniformMat4f("u_projection", projection_m);
        grayTerrain_shaderProgram.SetUniformMat4f("u_view", camera.getViewMatrix());
        grayTerrain_shaderProgram.SetUniformMat4f("u_model", model_m);
        grayTerrain_shaderProgram.SetUniform3f("u_viewPosition", cam_pos.x, cam_pos.y, cam_pos.z);
        grayTerrain_shaderProgram.SetUniform3f("color", 0.5f, 0.5f, 0.5f);

        isolines_shaderProgram.Bind();
        isolines_shaderProgram.SetUniformMat4f("u_projection", projection_m);
        isolines_shaderProgram.SetUniformMat4f("u_view", camera.getViewMatrix());
        isolines_shaderProgram.SetUniformMat4f("u_model", model_m);
        isolines_shaderProgram.SetUniform3f("u_viewPosition", cam_pos.x, cam_pos.y, cam_pos.z);
        isolines_shaderProgram.SetUniform1f("u_waterLevel", terrain_water_level);
        isolines_shaderProgram.SetUniform1f("u_deepestLevel", terrainBB.min_z);
        isolines_shaderProgram.SetUniform1f("u_levelRange", terrain_z_range);
        isolines_shaderProgram.SetUniform1i("u_nIsolines", nIsolines);
        isolines_shaderProgram.SetUniform1fv("u_isolines", (int) isoLines.size(), isoLines.data());

        terrainNormals_shaderProgram.Bind();
        terrainNormals_shaderProgram.SetUniformMat4f("u_projection", projection_m);
        terrainNormals_shaderProgram.SetUniformMat4f("u_view", camera.getViewMatrix());
        terrainNormals_shaderProgram.SetUniformMat4f("u_model", model_m);

        renderer.Draw(square_shape, texture, t_mpv_shaderProgram, GL_TRIANGLES);
        renderer.Draw(normal_color_cube_shape, cube_material, light, gouraud_c_mpv_shaderProgram, GL_TRIANGLES);
        renderer.Draw(axis_shape, texture, c_mpv_shaderProgram, GL_LINES);
        if (displayController.displayContourCurves())
        {
            renderer.Draw(terrain, cube_material, light, isolines_shaderProgram, GL_TRIANGLES);
            renderer.Draw(terrain, cube_material, light, grayTerrain_shaderProgram, GL_TRIANGLES);
            currentShaderIndex = 1;
        }
        else if (displayController.displayTriangles())
        {
            renderer.Draw(terrain, cube_material, light, terrainLines_shaderProgram, GL_TRIANGLES);
            currentShaderIndex = 2;
        }
        else
        {
            renderer.Draw(terrain, cube_material, light, terrain_shaderProgram, GL_TRIANGLES);
            renderer.Draw(terrain, water_material, light, water_shaderProgram, GL_TRIANGLES);
            currentShaderIndex = 0;
        }
        if (displayController.displayNormals())
            renderer.Draw(terrain, terrainNormals_shaderProgram, GL_TRIANGLES);

        ImGui::Begin("Variables");

        ImGui::Text("Camera");
        ImGui::Text("  CONTROLS:");
        ImGui::Text("  * WASD to move around, SPACE/Shift to move up/down");
        ImGui::Text("  * Arrows to look around");
        ImGui::Text("current values");
        ImGui::Text("-> eye-pos: (%.3f, %.3f, %.3f)", camera.getCX(), camera.getCY(), camera.getCZ());
        ImGui::Text("-> center(phi: %.3f, theta: %.3f)", camera.getPhi(), camera.getTheta());

        ImGui::Text("\nLight Position");
        ImGui::SliderFloat("x", &(lightPos.x), terrainBB.min_x, terrainBB.max_x);
        ImGui::SliderFloat("y", &(lightPos.y), terrainBB.min_y, terrainBB.max_y);
        ImGui::SliderFloat("z", &(lightPos.z), terrainBB.max_z+5, terrainBB.max_z+95);

        ImGui::Text("\nSediment color intensity upscale");
        ImGui::SliderFloat("intensity", &sedimentColorIntensity, 1.f, 1000.f);

        ImGui::Text("\nRenderer");
        ImGui::Text("  CONTROLS:");
        ImGui::Text("  * TAB to toggle shader or choose below \n");
        {
            ImGui::Combo("shader", &currentShaderIndex, shaderSelectorOptions, IM_ARRAYSIZE(shaderSelectorOptions));
            switch (currentShaderIndex) {
                case 1:
                    displayController.switchToContourCurves();
                    break;
                case 2:
                    displayController.switchToTriangleLines();
                    break;
                default:
                    displayController.switchToDefaultDisplay();
                    break;
            }
        }
        if (ImGui::Button("toggle normals"))
            displayController.toggleNormals();
        ImGui::Text("Isolines Setup");
        ImGui::SliderInt("n", &nIsolines, 0, 20);

        std::sort(isoLines.begin(), isoLines.end());
        if (ImGui::TreeNode("Isolines configuration"))
        {
            for (int i = 0; i < isoLines.size(); i++)
            {
                std::string name("Isoline ");
                if (i < 9)
                    name += "0";
                name += std::to_string(i+1);
                ImGui::SliderFloat(name.c_str(), &(isoLines[i]),
                                   terrainBB.min_z, terrainBB.max_z);
            }
            ImGui::TreePop();
        }

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();

    return 0;
}