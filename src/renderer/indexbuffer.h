#pragma once

#include <vector>

/**
 * IndexBuffer abstraction implementations.
 * 
 * IndexBuffer Objects handle binding and unbinding the GL index buffer objects.
 */
class IndexBuffer
{
private:
	unsigned int m_RendererID;  // The ID of the generated buffer.
	unsigned int m_Count;  // Number of indices in the buffer.

public:
    /**
     * Default constructor for IndexBuffer. Generates a GL (ELEMENT ARRAY) Buffer. It assumes they contain unsigned ints.
     *
     * @param data Pointer to the index data array, which contains the indices.
     * @param count Number of indices in the index data.
     */
	explicit IndexBuffer(std::vector<unsigned int> &data);

    /**
     * Destructor of IndexBuffer.
     * Tells GL to delete the buffer.
     */
    ~IndexBuffer();

    /**
     * Tells GL to bind the index buffer.
     */
	void Bind() const;

    /**
     * Tells GL to unbind the index buffer.
     */
	void Unbind() const;

    /**
     * @return the number of indices in the index buffer
     */
	inline unsigned int GetCount() const { return m_Count; }

    /**
     * Updates the indices in the index buffer
     * @param data
     */
    void updateData(std::vector<unsigned int> data);
};