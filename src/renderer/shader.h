#pragma once

#include <string>
#include <unordered_map>

#include "glm/glm.hpp"

struct ShaderProgramSource
{
	std::string VertexSource;
	std::string FragmentSource;
};

/**
 * Class to load, compile and handle shader programs
 */
class Shader
{
private:
	std::string m_FilePath;
	unsigned int m_RendererID;
	std::unordered_map<std::string, int> m_UniformLocationCache;
	
public:
	Shader(const std::string& filepath);
	~Shader();

	void Bind() const;
	void Unbind() const;

	// Set uniforms
	void SetUniform1i(const std::string& name, int value);
    void SetUniform1ui(const std::string& name, unsigned int value);
	void SetUniform1f(const std::string& name, float value);
    void SetUniform3f(const std::string& name, float v0, float v1, float v2);
	void SetUniform4f(const std::string& name, float v0, float v1, float v2, float v3);
	void SetUniformMat4f(const std::string& name, const glm::mat4& matrix);
private:
    /**
     * Returns a ShaderProgramSource structure with the Vertex shader string source code and the Fragment shader string source, readed from a file.
     *
     * @param filepath The path of the single file containing the vertex and fragment source codes.
     */
	ShaderProgramSource ParseShaderSrcFile(const std::string& filepath);

    /**
     * Compile a shaders.
     *
     * @param type Type of the shader to compile.
     * @param source Source code of the shader to compile.
     */
	unsigned int CompileShader(unsigned int type, const std::string& filepath);

    /**
     * Creates a shader program, with a vertexShader and a fragmentShader.
     * @param vertexShader Source code of the vertex shader
     * @param fragmentShader Source code of the fragment shader
     */
	unsigned int CreateShader(const std::string& vertexShader, const std::string& fragmentShader);

    /**
     * Gets the location of a uniform in the shader
     * @param name
     * @return
     */
	int GetUniformLocation(const std::string& name);

};