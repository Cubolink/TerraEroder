//
// Created by Cubolink on 22-06-2023.
//

#include "CudaShape.h"

CudaShape::CudaShape(const std::vector<float> &vertices, const std::vector<unsigned int> &indices,
                     const std::vector<int> &count_layouts)
: Shape(vertices, indices, count_layouts), cudaVBOResource(nullptr)
{
    cudaGraphicsGLRegisterBuffer(&cudaVBOResource, vbo.getGLBufferID(), cudaGraphicsMapFlagsWriteDiscard);
}


CudaShape::CudaShape(const Shape& shape)
: Shape(shape), cudaVBOResource(nullptr)
{
    cudaGraphicsGLRegisterBuffer(&cudaVBOResource, vbo.getGLBufferID(), cudaGraphicsMapFlagsWriteDiscard);
}

void CudaShape::cudaMap(float4 *devicePointer, size_t *numBytes)
{
    cudaGraphicsMapResources(1, &cudaVBOResource, nullptr);
    cudaGraphicsResourceGetMappedPointer((void**) &devicePointer, numBytes, cudaVBOResource);
}

void CudaShape::cudaUnmap()
{
    cudaGraphicsUnmapResources(1, &cudaVBOResource, nullptr);
}

CudaShape::~CudaShape() {
    cudaGraphicsUnregisterResource(cudaVBOResource);
}
