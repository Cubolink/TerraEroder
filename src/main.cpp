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
#include "obj_files.h"
#include "controller.h"

int w_width = 1024;
int w_height = 576;
float w_proportion;

glm::mat4 projection_m;
glm::vec3 translation;
glm::mat4 model_m;

CameraController cameraController;

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
    Shape terrain = Obj::readFile("../../data/terrain.obj");
    float terrain_max_z = -std::numeric_limits<float>::infinity();
    float terrain_min_z = std::numeric_limits<float>::infinity();

    std::vector<float> terrain_vertices = terrain.getVertices();
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
    Material cube_material = Material(0.3f, 0.6f, 0.7f, 100, texture);
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