#pragma once

#include <glad/glad.h>

#define ASSERT(x) if (!(x)) __debugbreak()
#define DEBUG  // comment/uncomment this macros definition to compile/not compile in debug mode

#ifdef DEBUG
#define GLCall(x) GLClearError(); x; ASSERT(GLLogCall(#x,__FILE__,__LINE__))
#else
#define GLCall(x) x
#endif


/**
* Clears the GL error trace.
*/
void GLClearError();

/**
* couts an OpenGL Error hex code in case there's one, indicating the file, function and line where the error was detected.
* @param function The name of the function that contains a code line.
* @param file Name of the file where the line is.
* @param line Number of the code line.
* @return false if there was an error, true is all was right.
*/
bool GLLogCall(const char* function, const char* file, int line);