#ifndef _VIBRATE_KERNEL_H_
#define _VIBRATE_KERNEL_H_


__global__ void 
oscilateKernel(float t, float* verticesGrid, unsigned int width, unsigned int height)
{
	// shared memory
    //	extern __shared__ float4 shPosition[];
	
	// index of my body	
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int cudaWidth = blockDim.x * gridDim.x;
    unsigned int cudaHeight = blockDim.y * gridDim.y;

    unsigned int cudaIndex = x * cudaWidth + y;
    cudaIndex *= 6;  // verticesGrid vbo contains x, y, z

    unsigned int index = x * width + y;
    index *= 6;
    if (x < width && y < height)
    {
        verticesGrid[index+2] = 2+2*sinf(t + verticesGrid[index] + verticesGrid[index+1]);  // z
        //verticesGrid[index+3] = 0;
        //verticesGrid[index+4] = 0;
        //verticesGrid[index+5] = 1;
    }

}

extern "C" 
void cudaRunOscilateKernel(dim3 gridSize, dim3 blockSize, float t,
                          float* verticesGridVBO, unsigned int width, unsigned int height)
{
    //dim3 block(16, 16, 1);
    //dim3 grid(width / block.x, height / block.y, 1);
    int sharedMemSize = 256 * sizeof(float3);
    oscilateKernel<<<gridSize, blockSize>>>(t, verticesGridVBO, width, height);
}

#endif