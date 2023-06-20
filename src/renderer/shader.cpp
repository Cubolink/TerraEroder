#include "shader.h"
#include "glerrorhandler.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>




Shader::Shader(const std::string& filepath)
	: m_FilePath(filepath), m_RendererID(0)
{
    ShaderProgramSource source = ParseShaderSrcFile(filepath);
    m_RendererID = CreateShader(source.VertexSource, source.FragmentSource);

}

Shader::~Shader()
{
    GLCall(glDeleteProgram(m_RendererID));
}


ShaderProgramSource Shader::ParseShaderSrcFile(const std::string& filepath)
{
    enum class ShaderType
    {
        NONE = -1, VERTEX = 0, FRAGMENT = 1
    };
    std::ifstream stream(filepath);
    std::string line;
    std::stringstream ss[2];  // for vertex and shader strings
    ShaderType type = ShaderType::NONE;

    while (getline(stream, line))
    {
        if (line.find("#shader") != std::string::npos)
        {
            if (line.find("vertex") != std::string::npos)
            {
                type = ShaderType::VERTEX;
            }
            else if (line.find("fragment") != std::string::npos)
            {
                type = ShaderType::FRAGMENT;
            }
        }
        else
        {
            ss[(int)type] << line << '\n';
        }

    }
    return { ss[0].str(), ss[1].str() };
}


unsigned int Shader::CompileShader(unsigned int type, const std::string& source)
{
    unsigned int id = glCreateShader(type);
    const char* src = source.c_str();  // <=> &source[0]
    glShaderSource(id, // associate the id of the shader created
        1, // number of elements in the array of string source codes.
        &src, // array of pointers to strings with source code.
        nullptr);  // array of string lenghts, same size as the array of string source codes.
    glCompileShader(id);

    // here we handle compile errors
    int compile_result;
    glGetShaderiv(id, GL_COMPILE_STATUS, &compile_result);
    if (!compile_result)  // compile_result == GL_FALSE
    {
        int length;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
        // char errorMessage[length] doesn't work because length is not a constant, c++ things
        char* errorMessage = (char*)alloca(length * sizeof(char));  // alloc is like malloc, but on the stack and not the heap, we don't need to free like with malloc
        glGetShaderInfoLog(id, length, &length, errorMessage);
        std::cout << "Failed to compile " <<
            (type == GL_VERTEX_SHADER ? "vertex" : "fragment")  // works because we only have vertex and fragment shaders, but is not the perfect way
            << " shader." << std::endl;
        std::cout << errorMessage << std::endl;
        glDeleteShader(id);

        return 0;
    }
    else {
        std::cout << "The following shader:\n\"" << std::endl << src << "\n\"\nwas compiled successfully.\n";
    }

    return id;
}


unsigned int Shader::CreateShader(const std::string& vertexShader, const std::string& fragmentShader)
{
    unsigned int program = glCreateProgram();  // tell gl we want to create a program that will be attached to our shaders
    // then we compile the shaders we're going to attach
    unsigned int vs = CompileShader(GL_VERTEX_SHADER, vertexShader);
    unsigned int fs = CompileShader(GL_FRAGMENT_SHADER, fragmentShader);
    // and then we attach them to the program
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glValidateProgram(program);

    // as the shaders are now linked to the program, they are part of the program, we don't need the originals
    // so we can detach them and then delet them
    /*
    glDetachShader(vs);
    glDetachShader(fs);
    */ // but we aren't going to detach them. We should do it, but the gaining is minimum and it seems to complicate debugging.

    glDeleteShader(vs);  // the shaders are linked, they are part of the program now, we don't need them anymore
    glDeleteShader(fs);

    return program;
}

void Shader::Bind() const
{
    GLCall(glUseProgram(m_RendererID));
}

void Shader::Unbind() const
{
    GLCall(glUseProgram(0));
}

void Shader::SetUniform1i(const std::string& name, int value)
{
    GLCall(glUniform1i(GetUniformLocation(name), value));
}

void Shader::SetUniform1ui(const std::string &name, unsigned int value) {
    GLCall(glUniform1ui(GetUniformLocation(name), value));
}


void Shader::SetUniform1f(const std::string& name, float value)
{
    GLCall(glUniform1f(GetUniformLocation(name), value));
}

void Shader::SetUniform3f(const std::string &name, float v0, float v1, float v2) {
    GLCall(glUniform3f(GetUniformLocation(name), v0, v1, v2));
}

void Shader::SetUniform4f(const std::string& name, float v0, float v1, float v2, float v3)
{
    GLCall(glUniform4f(GetUniformLocation(name), v0, v1, v2, v3));
}

void Shader::SetUniformMat4f(const std::string& name, const glm::mat4& matrix)
{
    // Location of the uniform in the shader program, number of matrix, false if we don't need to transpose them, pointer to the first element
    GLCall(glUniformMatrix4fv(GetUniformLocation(name), 1, GL_FALSE, &matrix[0][0]));
}

int Shader::GetUniformLocation(const std::string& name)
{
    // Get the uniformlocation from the cache, if it is there.
    if (m_UniformLocationCache.find(name) != m_UniformLocationCache.end())
    {
        return m_UniformLocationCache[name];
    }
    // is not there, so we get that location from GPU

    GLCall(int location = glGetUniformLocation(m_RendererID, name.c_str()));  // retrieve the location of that uniform of the shader
    if (location == -1)
        std::cout << "Warning: uniform '" << name << "' doesn't exist!" << std::endl;

    m_UniformLocationCache[name] = location;  // store that location in a cache, so we don't have to ask the GPU again
    return location;
}
