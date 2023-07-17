#ifndef _VIBRATE_KERNEL_CU_
#define _VIBRATE_KERNEL_CU_


__global__ void 
oscilateKernel(float t, float4* verticesGrid)
{
	// shared memory
    //	extern __shared__ float4 shPosition[];
	
	// index of my body	
	unsigned int cuX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int cuY = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int cuWidth = blockDim.x * gridDim.x;
    //unsigned int cudaHeight = blockDim.cuY * gridDim.cuY;
    unsigned int cuIdx = cuX * cuWidth + cuY;

    float x = verticesGrid[cuIdx].x;
    float y = verticesGrid[cuIdx].y;
    verticesGrid[cuIdx].z = 2 + 2 * sinf(t + x + y);
    verticesGrid[cuIdx].w = max(0.f, -sinf(t + x + y));

}


__global__ void
updateVBOKernel(float4* verticesGrid, float3* normalsGrid, float* sedimentGrid, float* verticesVBO, unsigned int width, unsigned  int height)
{
    unsigned int cuX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int cuY = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int cuWidth = blockDim.x * gridDim.x;
    //unsigned int cudaHeight = blockDim.cuY * gridDim.cuY;
    unsigned int cudaIdx = cuX * cuWidth + cuY;

    if (cuX < width && cuY < height)  // Avoid padded objects
    {
        unsigned int idx = cuX * width + cuY;

        // Z coord
        verticesVBO[8 * idx + 2] = verticesGrid[cudaIdx].z;
        // Normals
        verticesVBO[8 * idx + 3] = normalsGrid[cudaIdx].x;
        verticesVBO[8 * idx + 4] = normalsGrid[cudaIdx].y;
        verticesVBO[8 * idx + 5] = normalsGrid[cudaIdx].z;
        // Water level
        verticesVBO[8 * idx + 6] = verticesGrid[cudaIdx].w;
        // Sediment
        verticesVBO[8 * idx + 7] = sedimentGrid[cudaIdx];
    }

}

extern "C" 
void cudaRunOscilateKernel(dim3 gridSize, dim3 blockSize, float t,
                          float4* verticesGrid)
{
    //dim3 block(16, 16, 1);
    //dim3 grid(width / block.x, height / block.y, 1);
    //int sharedMemSize = 256 * sizeof(float3);
    oscilateKernel<<<gridSize, blockSize>>>(t, verticesGrid);
}

extern "C"
void cudaUpdateVBO(dim3 gridSize, dim3 blockSize, float4* cudaVerticesGrid, float3* cudaNormalsGrid, float* sedimentGrid,
                   float* verticesVBO, unsigned int width, unsigned int height)
{
    updateVBOKernel<<<gridSize, blockSize>>>(cudaVerticesGrid, cudaNormalsGrid, sedimentGrid, verticesVBO, width, height);
}

#endif