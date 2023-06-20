#include "vertexbuffer.h"

#include "glerrorhandler.h"


/**
* Default constructor for VertexBuffer. Generates a GL (Array) Buffer.
*
* @param data Pointer to the index data array, which contains the indices.
* @param size Size (in bytes) of the index data.
*/
VertexBuffer::VertexBuffer(std::vector<float> &data)
{
    GLCall(glGenBuffers(1, &m_RendererID));  // Tells GL to generate 1 Buffer, and store its id in m_RendererID.
    GLCall(glBindBuffer(GL_ARRAY_BUFFER, m_RendererID));  // Binds an Array Buffer to the generated buffer.
    GLCall(glBufferData(GL_ARRAY_BUFFER, (int) (data.size() * sizeof(float)), &data[0], GL_STATIC_DRAW));  // Tells GL that the binded buffer has some characteristics.
}

/**
* Default destructor of VertexBuffer.
* Tells GL to delete the buffer.
*/
VertexBuffer::~VertexBuffer()
{
    GLCall(glDeleteBuffers(1, &m_RendererID));
}

/**
* Tells GL to bind the index buffer.
*/
void VertexBuffer::Bind() const
{
    GLCall(glBindBuffer(GL_ARRAY_BUFFER, m_RendererID));
}

/**
* Tells GL to unbind the index buffer.
*/
void VertexBuffer::Unbind() const
{
    GLCall(glBindBuffer(GL_ARRAY_BUFFER, 0));
}

void VertexBuffer::updateData(std::vector<float> &data) {
    GLCall(glBindBuffer(GL_ARRAY_BUFFER, m_RendererID));
    GLCall(glBufferData(GL_ARRAY_BUFFER, (int) (data.size() * sizeof(float)), data.data(), GL_DYNAMIC_DRAW));
}
