#ifndef _EROSION_CU
#define _EROSION_CU

#define g 9.8067f
#define pi 3.1415f


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
    const float kd = 0.01f;  // sediment deposition
    const float ks = 0.005f;  // sediment dissolving
    const float kh = 1.f;  // sediment softness
    const float l = (dx+dy)/2;  // length of virtual pipes
    const float A = pi * l * l / 4.f;  // cross-section area of virtual pipes. I'm using a circle with radius=l/2
    const float lmax = 0.5f;  // limit to the water depth's erode capability (to avoid the deep water erode too much)

    float x = verticesGrid[cuIdx].x;
    float y = verticesGrid[cuIdx].y;
    float z = verticesGrid[cuIdx].z;  // terrain height
    float w = verticesGrid[cuIdx].w;  // water level
    float s = suspendedSediment[cuIdx];  // suspended sediment

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

    // Compute the xy-velocity field and use it to compute the water sediment

    // Average xy-flux
    float wVX = ((waterOutflowFlux[cuIdxL].y - fl)  // Left cell's right - current's left
            + (fr -waterOutflowFlux[cuIdxR].x)) / 2.f;  // Right cell's left - current's right
    float wVY = ((waterOutflowFlux[cuIdxB].w - fb)  // Back cell's front - current's back
            + (ff - waterOutflowFlux[cuIdxF].z)) / 2.f;  // Front cell's back - current's front
    // Average xy linear velocity
    float u = wVX / A;//(l * ((w + w2) / 2.f));
    float v = wVY / A;//(l * ((w + w2) / 2.f));
    // Sediment capacity
    float C = -kc * (normalsGrid[cuIdx].x * u + normalsGrid[cuIdx].y * v + normalsGrid[cuIdx].z) * norm3df(u, v, 0) * min(lmax, w2);
    float w3 = w2;
    if (C > s)
    {
        float ds = min(dt * ks * (C - s), z);  // dissolve, at most, the amount of terrain
        z -= ds;
        s += ds;
        w3 += ds;
    }
    else
    {
        float ds = min(dt * kd * (s - C), s);  // deposit, at most, the amount of dissolved sediment
        z += ds;
        s -= ds;
        w3 -= ds;
    }
    /*
    // Ultra-optimized code
    float ds = C - s;
    ds = (ds > 0) ?
            ks * ds : // Dissolve some soil in the water
            kd * ds;  // Deposite some sediment in the soil
    z -= ds * dt;  // this will add or subtract depending on the sign of ds
    s += ds * dt;
    float w3 = max(0.f, w2 + ds * dt);  // to improve stability according to paper, we have to increment the water level when dissolving soil
    */
    verticesGrid[cuIdx].z = max(z, 0.f);
    verticesGrid[cuIdx].w = max(w3, 0.f);
    suspendedSediment[cuIdx] = max(s, 0.f);
    __syncthreads();

    // Move dissolved sediment along the water

    // use xy coords of a cell to get the ids of the 4 cells nearer to that coordinate
    float fromX = max(0.f, x - u * dt);  // x-coord of the cell from where we're taking the sediment.
    float fromY = max(0.f, y - v * dt);  // y-coord of the cell from where we're taking the sediment
    // the (idx, idy) will be around 4 cells, so we interpolate them
    // | <-y->
    // x  A  C
    // |  B  D
    //
    // cuFromIdxX = (x - x0) / dx, but I'm assuming (x0, y0) is (0, 0) in the grid
    unsigned int cuFromIdX = max(0, min(cuWidth - 1, int (fromX / dx)));
    unsigned int cuFromIdY = max(0, min(cuHeight - 1, int (fromY / dy)));
    unsigned int cuFromIdX2 = min(cuWidth, cuFromIdX + 1);
    unsigned int cuFromIdY2 = min(cuHeight - 1, cuFromIdY + 1);
    // Finally get the ids
    unsigned int cuFromAIdx = cuFromIdX * cuHeight + cuFromIdY;
    unsigned int cuFromBIdx = cuFromIdX2 * cuHeight + cuFromIdY;
    unsigned int cuFromCIdx = cuFromIdX * cuHeight + cuFromIdY2;
    unsigned int cuFromDIdx = cuFromIdX2* cuHeight + cuFromIdY2;
    // and get the weights of those cells
    float px = (fromX / dx) - (float) cuFromIdX;  // get decimal part. 0 means is nearer to AB, 1 is nearer to CD
    float py = (fromY / dy) - (float) cuFromIdY;  // 0 means nearer to AC, 1 nearer to BD
    float incomingSediment = (1 - py) * (
            (1 - px) * suspendedSediment[cuFromAIdx] + px * suspendedSediment[cuFromBIdx]
    )
                             + py * (
            (1 - px) * suspendedSediment[cuFromCIdx] + px * suspendedSediment[cuFromDIdx]
    );
    __syncthreads();
    suspendedSediment[cuIdx] = incomingSediment;
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