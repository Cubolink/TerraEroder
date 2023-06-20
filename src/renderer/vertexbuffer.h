#pragma once

#include <vector>

/**
 * VertexBuffer abstraction implementations.
 *
 * VertexBuffer Objects handle binding and unbinding the GL vertex buffer objects.
 */
class VertexBuffer
{
private:
	unsigned int m_RendererID;  // ID of the GL buffer generated
public:
	explicit VertexBuffer(std::vector<float> &data);
	~VertexBuffer();

	void Bind() const;
	void Unbind() const;

    /**
     * Updates the vertices of the vertex buffer
     * @param data
     */
    void updateData(std::vector<float> &data);
};