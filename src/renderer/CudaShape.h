//
// Created by Cubolink on 22-06-2023.
//

#ifndef TERRAERODER_CUDASHAPE_H
#define TERRAERODER_CUDASHAPE_H

#include "shape.h"
#include <cuda_gl_interop.h>

#include <utility>

class CudaShape : Shape {
private:
    cudaGraphicsResource *cudaVBOResource;
public:
    CudaShape(std::vector<float> vertices, std::vector<unsigned int> indices, const std::vector<int>& count_layouts);

    void cudaMap(float4 *devicePointer, size_t *numBytes);

    void cudaUnmap();
};


#endif //TERRAERODER_CUDASHAPE_H
