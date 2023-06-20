//
// Created by major on 19-06-2023.
//

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "renderer/renderer.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>

int w_width = 1024;
int w_height = 576;


void initGL()
{
    // Configuring GL
    glad_glClearColor(0.43f, 0.39f, 0.48f, 1.f);
    glad_glEnable(GL_DEPTH_TEST);  // enable depth buffer
    glad_glEnable(GL_CLIP_DISTANCE0);  // clipping distance
    glad_glFrontFace(GL_CCW);  // set ccw mode, useful in particular for culling
    glad_glEnable(GL_CULL_FACE);
    glad_glCullFace(GL_BACK);
    glad_glEnable(GL_BLEND);  // enable transparency
    glad_glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  // set blending function
    glad_glBlendEquation(GL_FUNC_ADD);  // set blending reslts mixer function
}


int main() {
    std::cout << "Hola mundo" << std::endl;

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
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress))
        return -1;

    initGL();

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

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();

    return 0;
}