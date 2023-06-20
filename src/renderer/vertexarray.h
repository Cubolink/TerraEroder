#pragma once

#include "vertexbuffer.h"
#include "vertexbufferlayout.h"


/**
 * Vertex Array abstraction implementations.
 *
 * VertexArray Objects handles binding and unbinding the GL vertex arrays, which are binding multiple vertex buffers and layouts.
 */
class VertexArray
{
private:
	unsigned int m_RendererID;  // ID of the GL vertex array generated.
public:
	VertexArray();
	~VertexArray();

    /**
     * Adds a vertex buffer with a layout to the vertex array
     * @param vb
     * @param layout
     */
	void AddBuffer(const VertexBuffer& vb, const VertexBufferLayout& layout);

	void Bind() const;
	void Unbind() const;
};