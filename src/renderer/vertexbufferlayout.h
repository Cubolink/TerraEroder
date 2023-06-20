#pragma once

#include <vector>
#include <glad/glad.h>

#include "glerrorhandler.h"
/**
* Module tp help Vertex Arrays to determine the GL Attribute Pointers.
*/

template<typename T>
struct vbl_identity {
    typedef T type;
};


/**
* Structure to store information about a layout of a vertex buffer.
* It stores the type of the vertex buffer, number of elements in the vertex buffer and GL_NORMALIZED or not normalized.
*/
struct VertexBufferElement
{
	unsigned int type;  // GL type of the vertex buffer: GL_FLOAT, GL_UNSIGNED_INT and GL_UNSIGNED_BYTE are supported.
	unsigned int count;  // number of values that conform a single vertex
	unsigned char normalized;  // GL_FALSE, GL_TRUE

	static unsigned int GetSizeOfType(unsigned int type)
	{
		switch (type)
		{
			case GL_FLOAT: return 4;
			case GL_UNSIGNED_INT: return 4;
			case GL_UNSIGNED_BYTE: return 1;
		}
		return 0;
	}
};


/**
* Class to handle the Vertex Buffer Layout logic.
*/
class VertexBufferLayout
{
private:
	std::vector<VertexBufferElement> m_Elements;  // Array of different VertexBufferElement structures, to handle multiple types of GL Vertex Attribute Pointer in the same layout
	unsigned int m_Stride;  // Size of a single vertex.
public:
	VertexBufferLayout()
		: m_Stride(0) {}

	template<typename T>
	/**
	* Adds `count` elements to the layout.
	*/
	void Push(unsigned int count)
	{
		//static_assert(false);
        Push(count, vbl_identity<T>());
	}

	inline const std::vector<VertexBufferElement> GetElements() const { return m_Elements; }
	inline unsigned int GetStride() const { return m_Stride; }

    VertexBufferLayout& operator= (VertexBufferLayout vbl)
    {
        if (this == &vbl)
            return *this;

        m_Stride = vbl.m_Stride;
        m_Elements = vbl.m_Elements;

        return *this;
    }

private:
	template<typename T>
	void Push(unsigned int count, vbl_identity<T>)
    {

    }

	void Push(unsigned int count, vbl_identity<float>)
	{
		VertexBufferElement v = {GL_FLOAT, count, GL_FALSE};
		m_Elements.push_back(v);
		m_Stride += count * VertexBufferElement::GetSizeOfType(GL_FLOAT);
	}

	void Push(unsigned int count, vbl_identity<unsigned int>)
	{
		VertexBufferElement v = { GL_UNSIGNED_INT, count, GL_FALSE };
		m_Elements.push_back(v);
		m_Stride += count * VertexBufferElement::GetSizeOfType(GL_UNSIGNED_INT);
	}

	void Push(unsigned int count, vbl_identity<unsigned char>)
	{
		VertexBufferElement v = { GL_UNSIGNED_BYTE, count, GL_TRUE };
		m_Elements.push_back(v);
		m_Stride += count * VertexBufferElement::GetSizeOfType(GL_UNSIGNED_BYTE);
	}

};