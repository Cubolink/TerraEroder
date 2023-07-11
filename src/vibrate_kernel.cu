#ifndef _VIBRATE_KERNEL_H_
#define _VIBRATE_KERNEL_H_


__global__ void 
oscilateKernel(float t, float3* verticesGrid)
{
	// shared memory
    //	extern __shared__ float4 shPosition[];
	
	// index of my body	
	unsigned int cuX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int cuY = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int cudaWidth = blockDim.x * gridDim.x;
    //unsigned int cudaHeight = blockDim.cuY * gridDim.cuY;
    unsigned int cudaIdx = cuX * cudaWidth + cuY;

    float x = verticesGrid[cudaIdx].x;
    float y = verticesGrid[cudaIdx].y;
    verticesGrid[cudaIdx].z = 2+2*sinf(t + x + y);

}


__global__ void
updateVBOKernel(float3* verticesGrid, float* verticesVBO, unsigned int width, unsigned  int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int cudaWidth = blockDim.x * gridDim.x;
    //unsigned int cudaHeight = blockDim.y * gridDim.y;
    unsigned int cudaIdx = x * cudaWidth + y;

    if (x < width && y < height)  // Avoid padded objects
    {
        unsigned int idx = x * width + y;

        // Z coord
        verticesVBO[6 * idx + 2] = verticesGrid[cudaIdx].z;
    }

}

extern "C" 
void cudaRunOscilateKernel(dim3 gridSize, dim3 blockSize, float t,
                          float3* verticesGrid)
{
    //dim3 block(16, 16, 1);
    //dim3 grid(width / block.x, height / block.y, 1);
    //int sharedMemSize = 256 * sizeof(float3);
    oscilateKernel<<<gridSize, blockSize>>>(t, verticesGrid);
}

extern "C"
void cudaUpdateVBO(dim3 gridSize, dim3 blockSize, float3* cudaVerticesGrid,
                   float* verticesVBO, unsigned int width, unsigned int height)
{
    updateVBOKernel<<<gridSize, blockSize>>>(cudaVerticesGrid, verticesVBO, width, height);
}

#endif