#include "vertexarray.h"
#include "glerrorhandler.h"


/**
* Default constructor for VertexArrays. Generates a GL Vertex Array.
*/
VertexArray::VertexArray()
{
	GLCall(glGenVertexArrays(1, &m_RendererID));  // Tells GL to generate a Vertex Array. Stores the ID of the generated array in m_RendererID
}

/**
* Default destructor for Vertex Arrays. Tells GL to delete the VertexArray.
*/
VertexArray::~VertexArray()
{
	GLCall(glDeleteVertexArrays(1, &m_RendererID));  // Tells GL to delete the VertexArray associated with the m_RendererID.
}

/**
* Adds to the Vertex Array a Vertex Buffer with an associated Vertex Buffer Layout.
*/
void VertexArray::AddBuffer(const VertexBuffer& vb, const VertexBufferLayout& layout)
{
	Bind();  // Binds the current Vertex Array, in order to tell GL that we're going to add things to the GL Vertex Array associated.
	vb.Bind();  // Binds the Vertex Buffer we're adding.
	const auto& elements = layout.GetElements();
	unsigned int offset = 0;
	for (unsigned int i = 0; i < elements.size(); i++)  // Iterates over the elements, counting an accumulated offset.
	{
		const auto& element = elements[i];
		glEnableVertexAttribArray(i);  // tells GL to enable the VertexAttributeArray, and then we set a VertexAttributePointer for this element.
		glVertexAttribPointer(i, element.count, element.type, element.normalized, layout.GetStride(), (const void*) offset);
		offset += element.count * VertexBufferElement::GetSizeOfType(element.type);
	}
	
}

/**
* Binds the Vertex Array
*/
void VertexArray::Bind() const
{
	GLCall(glBindVertexArray(m_RendererID));  // Tells GL to bind the Vertex Array associated to m_RendererID
}
void VertexArray::Unbind() const
{
	GLCall(glBindVertexArray(0));  // Tells GL to unbind the Vertex Array.
}