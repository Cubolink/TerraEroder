//
// Created by Cubolink on 22-06-2023.
//

#include "CudaShape.h"

CudaShape::CudaShape(std::vector<float> vertices, std::vector<unsigned int> indices,
                     const std::vector<int> &count_layouts)
: Shape(std::move(vertices), std::move(indices), count_layouts), cudaVBOResource(nullptr)
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
