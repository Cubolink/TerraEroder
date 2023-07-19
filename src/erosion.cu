#ifndef _EROSION_CU
#define _EROSION_CU

#define g 9.8067f


/**
 * Computes the normals of the grid.
 *
 * @param verticesGrid Contains info about x,y,z position of a cell of the terrain, and a water level above the z-height
 * @param normalsGrid Contains the normal of the cells of the terrain
 */
__global__ void
normalsKernel(float4* verticesGrid, float3* normalsGrid)
{
    unsigned int cuX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int cuY = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int cuWidth = blockDim.x * gridDim.x;
    unsigned int cuHeight = blockDim.y * gridDim.y;
    unsigned int cuIdx = cuX * cuHeight + cuY;

    // compute indices of neighbor cells. When an index out of bonds, set it as the current cell.
    unsigned int cuIdxL = cuX > 0 ? (cuX-1) * cuHeight + cuY : cuIdx;
    unsigned int cuIdxR = (cuX+1) < cuWidth ? (cuX+1) * cuHeight + cuY : cuIdx;
    unsigned int cuIdxB = cuY > 0 ? cuX * cuHeight + (cuY-1) : cuIdx;
    unsigned int cuIdxF = (cuY+1) < cuHeight ? cuX * cuHeight + (cuY+1) : cuIdx;

    float3 v1 = {verticesGrid[cuIdxR].x - verticesGrid[cuIdxL].x,
                 verticesGrid[cuIdxR].y - verticesGrid[cuIdxL].y,
                 verticesGrid[cuIdxR].z - verticesGrid[cuIdxL].z};
    float3 v2 = {verticesGrid[cuIdxF].x - verticesGrid[cuIdxB].x,
                 verticesGrid[cuIdxF].y - verticesGrid[cuIdxB].y,
                 verticesGrid[cuIdxF].z - verticesGrid[cuIdxB].z};
    float3 n = {v1.y * v2.z - v1.z * v2.y,
                v1.z * v2.x - v1.x * v2.z,
                v1.x * v2.y - v1.y * v2.x
    };
    float nn = normf(3, (float*) &n);
    n.x /= nn;
    n.y /= nn;
    n.z /= nn;

    normalsGrid[cuIdx] = n;
}


/**
 *
 * @param dt time interval from the previous to the current time
 * @param dx distance between cells in the x direction
 * @param dy distance between cells in the y direction
 * @param verticesGrid Contains info about x,y,z position of a cell of the terrain, and a water level above the z-height
 * @param normalsGrid Contains the normal of the cells of the terrain
 * @param waterOutflowFlux Flux from a cell to the neighbors in this order: Left (x-), Right (x+), Back (y-), Front (y+)
 * @param suspendedSediment Suspended sediment amount in the water
 */
__global__ void
erodeKernel(float dt, float dx, float dy, float4* verticesGrid, float3* normalsGrid, float4* waterOutflowFlux, float* suspendedSediment)
{
    unsigned int cuX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int cuY = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int cuWidth = blockDim.x * gridDim.x;
    unsigned int cuHeight = blockDim.y * gridDim.y;
    unsigned int cuIdx = cuX * cuHeight + cuY;

    // compute indices of neighbor cells. When an index out of bonds, set it as the current cell.

    unsigned int cuIdxL = cuX > 0 ? (cuX-1) * cuHeight + cuY : cuIdx;
    unsigned int cuIdxR = (cuX+1) < cuWidth ? (cuX+1) * cuHeight + cuY : cuIdx;
    unsigned int cuIdxB = cuY > 0 ? cuX * cuHeight + (cuY-1) : cuIdx;
    unsigned int cuIdxF = (cuY+1) < cuHeight ? cuX * cuHeight + (cuY+1) : cuIdx;

    const float r = 0.0001f;  // rain rate for each cell
    const float kr = 1.f;  // rain rate scale
    const float kc = 0.1f;  // sediment transportation capacity
    const float l = (dx+dy)/2;  // length of virtual pipes
    const float A = pi * l * l / 4.f;  // cross-section area of virtual pipes. I'm using a circle with radius=l/2

    float x = verticesGrid[cuIdx].x;
    float y = verticesGrid[cuIdx].y;
    float z = verticesGrid[cuIdx].z;  // terrain height
    float w = verticesGrid[cuIdx].w;  // water level

    // update water height due to rain
    w += r * kr * dt;

    // compute water OUT-flux from cell

    float fl = waterOutflowFlux[cuIdx].x;
    float fr = waterOutflowFlux[cuIdx].y;
    float fb = waterOutflowFlux[cuIdx].z;
    float ff = waterOutflowFlux[cuIdx].w;
    float dhL = (z+w) - (verticesGrid[cuIdxL].z + verticesGrid[cuIdxL].w);  // positive means current cell is higher
    float dhR = (z+w) - (verticesGrid[cuIdxR].z + verticesGrid[cuIdxR].w);
    float dhB = (z+w) - (verticesGrid[cuIdxB].z + verticesGrid[cuIdxB].w);
    float dhF = (z+w) - (verticesGrid[cuIdxF].z + verticesGrid[cuIdxF].w);

    fl = max(0.f, fl + dt * A * (g * dhL / l));  // so the outflow flux can decrease, but will never be negative
    fr = max(0.f, fr + dt * A * (g * dhR / l));
    fb = max(0.f, fb + dt * A * (g * dhB / l));
    ff = max(0.f, ff + dt * A * (g * dhF / l));
    // scale down the fluxes such that the total flux is, at most, the amount of water
    float k = min(1.f, (w * dx * dy) / ((fl + fr + fb + ff) * dt + 0.00001f));  // avoid division by 0 (should not happen tho)
    fl *= k;
    fr *= k;
    fb *= k;
    ff *= k;

    __syncthreads();
    waterOutflowFlux[cuIdx].x = fl;
    waterOutflowFlux[cuIdx].y = fr;
    waterOutflowFlux[cuIdx].z = fb;
    waterOutflowFlux[cuIdx].w = ff;
    // Sync block threads, so the cells can access the updated flux of its neighbors.
    __syncthreads();  // This is not enough tho, since the border-cells need to read info from other blocks. TODO

    // compute the water height change using fluxes.
    float outflow = fl + fr + fb + ff;
    float inflow = waterOutflowFlux[cuIdxL].y  // Left cell's Right flux
                   + waterOutflowFlux[cuIdxR].x  // Right cell's Left flux
                   + waterOutflowFlux[cuIdxB].w  // Back cell's Front flux
                   + waterOutflowFlux[cuIdxF].z;  // Front cell's Back flux
    float dV = (inflow - outflow) * dt;

    // Update the water height

    float w2 = max(0.f, w + (dV / (dx * dy)));
    verticesGrid[cuIdx].w = w2;

}

extern "C"
void cudaRunNormalsKernel(dim3 gridSize, dim3 blockSize, float4* verticesGrid, float3* normals)
{
    normalsKernel<<<gridSize, blockSize>>>(verticesGrid, normals);
}

extern "C"
void cudaRunErodeKernel(dim3 gridSize, dim3 blockSize, float dt, float dx, float dy, float4* verticesGrid, float3* normals, float4* waterOutflowFlux, float* suspendedSediment)
{
    erodeKernel<<<gridSize, blockSize>>>(dt, dx, dy, verticesGrid, normals, waterOutflowFlux, suspendedSediment);
}

#endif