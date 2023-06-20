#include "indexbuffer.h"

#include "glerrorhandler.h"


IndexBuffer::IndexBuffer(std::vector<unsigned int> &data)
	: m_Count(data.size())
{
	ASSERT(sizeof(unsigned int) == sizeof(GLuint));

	GLCall(glGenBuffers(1, &m_RendererID));  // Tells GL to generate 1 Buffer, and store its id in m_RendererID.
	GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_RendererID));  // Binds an Element Array Buffer to the generated buffer.
	GLCall(glBufferData(GL_ELEMENT_ARRAY_BUFFER, data.size() * sizeof(unsigned int), data.data(), GL_STATIC_DRAW));  // Tells GL that the binded buffer has some characteristics.
}


IndexBuffer::~IndexBuffer()
{
	GLCall(glDeleteBuffers(1, &m_RendererID));
}


void IndexBuffer::Bind() const
{
	GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_RendererID));
}


void IndexBuffer::Unbind() const
{
	GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
}

void IndexBuffer::updateData(std::vector<unsigned int> data) {
    m_Count = data.size();
    GLCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_RendererID));
    GLCall(glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_Count * sizeof(unsigned int), data.data(), GL_DYNAMIC_DRAW));
}
