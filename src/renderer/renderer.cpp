#include "renderer.h"

void Renderer::Draw(const VertexArray& vao, const IndexBuffer& ibo, const Shader& shader) const
{
	shader.Bind();
	vao.Bind();
	ibo.Bind();

	GLCall(glDrawElements(GL_TRIANGLES, ibo.GetCount(), GL_UNSIGNED_INT, nullptr));

	// We should unbind, but we're not doing it because when binding other things it will be like the same.
	// So, unbinding may low a little the performance. But, it may fix some errors we could have.
	// We are not doing it, but we should.
}

void Renderer::Draw(const VertexArray &vao, const IndexBuffer &ibo, const Texture &texture, const Shader &shader) const {
    shader.Bind();
    vao.Bind();
    ibo.Bind();
    texture.Bind(0);

    GLCall(glDrawElements(GL_TRIANGLES, ibo.GetCount(), GL_UNSIGNED_INT, nullptr));
}

void Renderer::Draw(const Shape &shape, const Texture &texture, const Shader &shader) const {
    shader.Bind();
    shape.Bind();
    texture.Bind(0);

    GLCall(glDrawElements(GL_TRIANGLES, shape.getIBOCount(), GL_UNSIGNED_INT, nullptr));
}

void Renderer::Clear() const
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Renderer::Draw(const Shape &shape, const Texture &texture, const Shader &shader, int gl_mode) const {
    shader.Bind();
    shape.Bind();
    texture.Bind(0);

    GLCall(glDrawElements(gl_mode, shape.getIBOCount(), GL_UNSIGNED_INT, nullptr));
}

void Renderer::Draw(const Shape &shape, const Material &material, const Light &light, Shader &shader, int gl_mode) const {
    shader.Bind();
    shape.Bind();
    material.Bind(shader);
    light.Bind(shader);

    GLCall(glDrawElements(gl_mode, shape.getIBOCount(), GL_UNSIGNED_INT, nullptr));
}
