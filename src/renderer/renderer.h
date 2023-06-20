#pragma once

#include "glerrorhandler.h"

#include "indexbuffer.h"
#include "vertexarray.h"
#include "shader.h"
#include "light.h"
#include "material.h"
#include "texture.h"
#include "shape.h"

/**
 * Class to handle drawings using different parameters like shapes and shaders,
 * binding whatever is necessary to draw, and the drawing
 */
class Renderer
{
public:
	void Draw(const VertexArray& vao, const IndexBuffer& ibo, const Shader& shader) const;
    void Draw(const VertexArray& vao, const IndexBuffer& ibo, const Texture& texture, const Shader& shader) const;
    void Draw(const Shape& shape, const Texture& texture, const Shader& shader) const;
    void Draw(const Shape& shape, const Texture& texture, const Shader& shader, int gl_mode) const;
    void Draw(const Shape& shape, const Material& material, const Light& light, Shader& shader, int gl_mode) const;

	void Clear() const;
};