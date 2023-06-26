//
// Created by Cubolink on 19-06-2023.
//

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>

#include "renderer/renderer.h"
#include "shape_factory.h"
#include "renderer/CudaShape.h"
#include "obj_files.h"
#include "controller.h"

// CUDA kernel interface
extern "C"
void cudaRunOscilateKernel(dim3 gridSize, dim3 blockSize, float t,
                          float* verticesGridVBO, unsigned int width, unsigned int height);

int w_width = 1024;
int w_height = 576;
float w_proportion;

glm::mat4 projection_m;
glm::vec3 translation;
glm::mat4 model_m;

CameraController cameraController;

std::vector<float> terrain_vertices;
std::vector<std::vector<float>> heightMap;

unsigned int cudaNumBlocks;
unsigned int cudaNumBodies;
dim3 cudaBlockSize;
dim3 cudaGridSize;
// float* cudaHeightMap;

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

        default:
            break;
    }
}


void createTerrainHeightMap()
{
    // terrain vertices is not necessarily a grid, it may have better triangulations.
    // but we do need a grid, in order to map it like m[x][y] = z in a matrix.

    unsigned int mn = terrain_vertices.size() / 6;
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
    heightMap = std::vector<std::vector<float>>(m, std::vector<float>(n));
    // x from 0 to m
    // y from 0 to n
    // matrix goes like      x y->
    //                       |
    //                      \|/
    for (unsigned int x = 0; x < m; x++)
    {
        for (unsigned int y = 0; y < n; y++)
        {
            unsigned int v = x * n + y;  // v contains info of x,y,z and normals, so we need 6*v+2 to get z
            /*
            std::cout << "heightMap["<<x<<"]["<<y<<"]="
                                                   "("<< terrain_vertices[6*v] <<", "
                                                   << terrain_vertices[6*v+1] << ", "
                                                   << terrain_vertices[6*v+2] << ")"<<std::endl;
                                                   */
            heightMap[x][y] = terrain_vertices[6 * v + 3];
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
    unsigned int m = heightMap.size();
    unsigned int n = heightMap[0].size();

    cudaBlockSize.x = 16;
    cudaBlockSize.y = 16;

    cudaGridSize.x = m / cudaBlockSize.x;
    if (cudaGridSize.x * cudaBlockSize.x < m)
        cudaGridSize.x++;  // x-padding
    cudaGridSize.y = m / cudaBlockSize.y;
    if (cudaGridSize.y * cudaBlockSize.y < n)
        cudaGridSize.y++;  // y-padding

    cudaNumBlocks = (cudaGridSize.x * cudaGridSize.y);
    cudaNumBodies = (cudaBlockSize.x * cudaBlockSize.x) * cudaNumBlocks;  // this > m*n if there was any padding

    /*
    std::vector<float> planeHeightMap;
    for (unsigned int x = 0; x < m; x++)
    {
        unsigned int cudaN = cudaGridSize.x * cudaBlockSize.x;

        // copy the n elements of the x-row
        planeHeightMap.insert(planeHeightMap.end(), heightMap[x].begin(), heightMap[x].end());
        // fill the rest until cudaN (padding)
        planeHeightMap.resize((x+1) * cudaN);
    }
    planeHeightMap.resize(cudaNumBodies);
     */
    /*
    // Initialize a matrix un CUDA
    unsigned int cudaM = cudaGridSize.x * cudaBlockSize.x;
    unsigned int cudaN = cudaGridSize.y * cudaBlockSize.y;

    auto** planeHeightMap = new float*[m];  // cudaM for padding
    for (unsigned int x = 0; x < m; x++)  // same
    {
        planeHeightMap[x] = new float[n];  // cudaN for padding
        for (unsigned int y = 0; y < n; y++)  // same
        {
            if (x < m && y < n)
                planeHeightMap[x][y] = heightMap[x][y];
            else
                planeHeightMap[x][y] = 0;
        }
    }
    size_t pitch;
    //cudaMalloc((void**) &cudaHeightMap, 10*sizeof(float));
    cudaMallocPitch((void**) &cudaHeightMap, &pitch,
                    cudaM * sizeof(float),
                    cudaN
                    );
    cudaMemcpy2D(cudaHeightMap, pitch, planeHeightMap, m * sizeof(float),
                 m * sizeof(float), n, cudaMemcpyHostToDevice);

    for (unsigned int x = 0; x < m; x++)
    {
        delete[] planeHeightMap[x];
    }
    delete[] planeHeightMap;
    */
}


void updateModel(CudaShape& terrain, double t)
{
    float *dPtr = nullptr;
    size_t numBytes;
    terrain.cudaMap(&dPtr, &numBytes);

    // Run the kernel
    cudaRunOscilateKernel(cudaGridSize, cudaBlockSize, (float) t, dPtr,
                         heightMap.size(), heightMap[0].size());

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
    // Init texture and shapes
    Texture texture = Texture("../resources/textures/red_yoshi.png");
    Shape square_shape = ShapeFactory::createTextureQuad();
    Shape axis_shape = ShapeFactory::createColorAxis(1);
    Shape normal_color_cube_shape = ShapeFactory::createColorNormalCube(.2f, .3f, .7f);
    CudaShape terrain(Obj::readFile("../../data/terrain.obj"));
    float terrain_max_z = -std::numeric_limits<float>::infinity();
    float terrain_min_z = std::numeric_limits<float>::infinity();

    terrain_vertices = terrain.getVertices();
    createTerrainHeightMap();
    initCUDA();
    for (int i = 2; i < terrain_vertices.size(); i += 6)
    {
        if (terrain_vertices[i] > terrain_max_z)
            terrain_max_z = terrain_vertices[i];
        if (terrain_vertices[i] < terrain_min_z)
            terrain_min_z = terrain_vertices[i];
    }
    float terrain_z_range = terrain_max_z - terrain_min_z;
    float terrain_water_level = terrain_z_range/3 + terrain_min_z;
    std::cout << "max/min: " << terrain_max_z << "/" << terrain_min_z << std::endl;
    std::cout << "water_level: " << terrain_water_level;

    // Init materials
    Material cube_material = Material(0.5f, 0.6f, 0.4f, 100);
    Light light = Light(1.0f, 1.0f, 1.0f, glm::vec3(0, 0, 50),
                        0.01f, 0.01f, 0.0001f);

    /* Starting main program */

    Renderer renderer = Renderer();

    double t0 = glfwGetTime();
    double t1, dt;

    glfwSwapInterval(1);  // vsync
    /* Loop until the user closes the window */
    while(!glfwWindowShouldClose(window))
    {
        t1 = glfwGetTime();
        dt = t1 - t0;
        t0 = t1;

        updateModel(terrain, t1);

        renderer.Clear();

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
        terrain_shaderProgram.SetUniform1f("u_deepestLevel", terrain_min_z);
        terrain_shaderProgram.SetUniform1f("u_levelRange", terrain_z_range);

        renderer.Draw(square_shape, texture, t_mpv_shaderProgram, GL_TRIANGLES);
        renderer.Draw(normal_color_cube_shape, cube_material, light, gouraud_c_mpv_shaderProgram, GL_TRIANGLES);
        renderer.Draw(axis_shape, texture, c_mpv_shaderProgram, GL_LINES);
        renderer.Draw(terrain, cube_material, light, terrain_shaderProgram, GL_TRIANGLES);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();

    return 0;
}