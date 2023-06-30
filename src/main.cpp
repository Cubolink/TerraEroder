//
// Created by Cubolink on 19-06-2023.
//

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

#include "renderer/renderer.h"
#include "shape_factory.h"
#include "obj_files.h"
#include "controller.h"
#include "display_controller.h"

int w_width = 1024;
int w_height = 576;
float w_proportion;

glm::mat4 projection_m;
glm::vec3 translation;
glm::mat4 model_m;

CameraController cameraController;
DisplayController displayController;

static int randNum;
static bool applyNoise = false;

std::vector<float> terrain_vertices;
std::vector<std::vector<float>> heightMap;
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

        case GLFW_KEY_T:
            if (action == GLFW_PRESS)
            {
                randNum = std::rand();
                applyNoise = true;
            }
            break;
        case GLFW_KEY_R:
            applyNoise = false;
            break;

        case GLFW_KEY_TAB:
            if (action == GLFW_PRESS)
                displayController.toggleDisplay();

        default:
            break;
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
    Shader terrain_shaderProgram = Shader("../resources/shaders/gouraud_mpv_terrain_shader.shader");
    Shader terrainLines_shaderProgram = Shader("../resources/shaders/gouraud_mpv_terrain_trianglelines.shader");
    Shader grayTerrain_shaderProgram = Shader("../resources/shaders/gouraud_mpv_terrain_single_color.shader");
    Shader isolines_shaderProgram = Shader("../resources/shaders/isolines_gouraud_mpv_terrain.shader");

    // Init texture and shapes
    Texture texture = Texture("../resources/textures/red_yoshi.png");
    Shape axis_shape = ShapeFactory::createColorAxis(1);
    Shape terrain = Obj::readFile("../../data/terrain.obj");

    terrain_vertices = terrain.getVertices();
    BoundingBox terrainBB;
    for (int i = 0; i < terrain_vertices.size(); i += 6)
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
    std::srand(123);
    randNum = std::rand();

    double t0 = glfwGetTime();
    double t1, dt;

    glfwSwapInterval(1);  // vsync
    /* Loop until the user closes the window */
    while(!glfwWindowShouldClose(window))
    {
        t1 = glfwGetTime();
        dt = t1 - t0;
        t0 = t1;

        //updateModel(terrain, t1);
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

        c_mpv_shaderProgram.Bind();
        c_mpv_shaderProgram.SetUniformMat4f("u_projection", projection_m);
        c_mpv_shaderProgram.SetUniformMat4f("u_view", camera.getViewMatrix());
        c_mpv_shaderProgram.SetUniformMat4f("u_model", model_m);

        glm::vec3 cam_pos = camera.getEyeVec3();

        terrain_shaderProgram.Bind();
        terrain_shaderProgram.SetUniformMat4f("u_projection", projection_m);
        terrain_shaderProgram.SetUniformMat4f("u_view", camera.getViewMatrix());
        terrain_shaderProgram.SetUniformMat4f("u_model", model_m);
        terrain_shaderProgram.SetUniform3f("u_viewPosition", cam_pos.x, cam_pos.y, cam_pos.z);
        terrain_shaderProgram.SetUniform1f("u_waterLevel", terrain_water_level);
        terrain_shaderProgram.SetUniform1f("u_deepestLevel", terrainBB.min_z);
        terrain_shaderProgram.SetUniform1f("u_levelRange", terrain_z_range);
        terrain_shaderProgram.SetUniform1i("u_applyNoise", (int) applyNoise);
        terrain_shaderProgram.SetUniform1f("u_randomNumber", (float) randNum);

        terrainLines_shaderProgram.Bind();
        terrainLines_shaderProgram.SetUniformMat4f("u_projection", projection_m);
        terrainLines_shaderProgram.SetUniformMat4f("u_view", camera.getViewMatrix());
        terrainLines_shaderProgram.SetUniformMat4f("u_model", model_m);
        terrainLines_shaderProgram.SetUniform3f("u_viewPosition", cam_pos.x, cam_pos.y, cam_pos.z);
        terrainLines_shaderProgram.SetUniform1f("u_waterLevel", terrain_water_level);
        terrainLines_shaderProgram.SetUniform1f("u_deepestLevel", terrainBB.min_z);
        terrainLines_shaderProgram.SetUniform1f("u_levelRange", terrain_z_range);
        terrainLines_shaderProgram.SetUniform1i("u_applyNoise", (int) applyNoise);
        terrainLines_shaderProgram.SetUniform1f("u_randomNumber", (float) randNum);

        grayTerrain_shaderProgram.Bind();
        grayTerrain_shaderProgram.SetUniformMat4f("u_projection", projection_m);
        grayTerrain_shaderProgram.SetUniformMat4f("u_view", camera.getViewMatrix());
        grayTerrain_shaderProgram.SetUniformMat4f("u_model", model_m);
        grayTerrain_shaderProgram.SetUniform3f("u_viewPosition", cam_pos.x, cam_pos.y, cam_pos.z);
        grayTerrain_shaderProgram.SetUniform3f("color", 0.5f, 0.5f, 0.5f);
        grayTerrain_shaderProgram.SetUniform1i("u_applyNoise", (int) applyNoise);
        grayTerrain_shaderProgram.SetUniform1f("u_randomNumber", (float) randNum);

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
        isolines_shaderProgram.SetUniform1i("u_applyNoise", (int) applyNoise);
        isolines_shaderProgram.SetUniform1f("u_randomNumber", (float) randNum);

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
            currentShaderIndex = 0;
        }

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

        if (ImGui::Button("Randomize"))
        {
            randNum = rand();
            applyNoise = true;
        }

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