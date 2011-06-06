#include "wavesimulator.h"

#include <glog/logging.h>

#include "types.h"

const int UNINTIALISED = 0;
const int INITIALISED = 1;
const float GRAVITY = 9.8;

gridpoint* device_grid;
vertex* device_heightmap;
vertex* device_watersurfacevertices;
rgb* device_watersurfacecolors;

int f;


int state = UNINTIALISED;
/*
__host__ __device__ float3 U(float h, velocity v)
{
    float3 U;
    U.x = h;
    U.y = v.x * h;
    U.z = v.y * h;
    return U;
}

__host__ __device__ float3 F(float h, velocity v)
{
    float3 F;
    F.x = v.x * h;
    F.y = (v.x * v.x * h) + ((1 / 2) * GRAVITY * h * h);
    F.z = v.x * v.y * h;
    return F;
}

__host__ __device__ float3 G(float h, velocity v)
{
    float3 G;
    G.x = v.y * h;
    G.y = v.x * v.y * h;
    G.z = (v.y * v.y * h) + ((1 / 2) * GRAVITY * h * h);
    return G;
}
*/
#if __GPUVERSION__
__global__ void simulateWaveStep(int frame, gridpoint* device_grid, vertex* device_heightmap, 
                            vertex* device_watersurfacevertices, rgb* device_watersurfacecolors, 
                            int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int gridx = x + 1;
    int gridy = y + 1;
    
    if(x < width && y < height)
    {
       vertex v;
       v.x = x/float(width)*16-8;
       v.z = y/float(height)*16-8;
       v.y = 0.05 * sin(frame / 10.0f + v.x * 20 - 5 * v.z) + 0.9;
       device_watersurfacevertices[y*width+x] = v;
       
       rgb c;
       c.x = 100 + 50 * (v.y - 0.9) * 20;
       c.y = 150 + 50 * (v.y - 0.9) * 20;
       c.z = 255;
       device_watersurfacecolors[y * width + x] = c;
	}
}
#endif
void initWaterSurface(int width, int height, vertex* heightmapvertices)
{

    if(state != UNINTIALISED)
    {
        return;
    }
#if __GPUVERSION__
    size_t sizeInBytes;
    cudaError_t error;
          
    // malloc memory for device_grid
    sizeInBytes = (height+2) * (width+2) * sizeof(gridpoint);
    error = cudaMalloc(&device_grid, sizeInBytes);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
    // malloc memory for device_watersurfacevertices
    sizeInBytes = height * width * sizeof(vertex);
    error = cudaMalloc(&device_watersurfacevertices, sizeInBytes);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
    // malloc memory for watersurfacecolors
    sizeInBytes = height * width * sizeof(rgb);
    error = cudaMalloc(&device_watersurfacecolors, sizeInBytes);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
    // copy heightmapdata data to device
    sizeInBytes = height * width * sizeof(vertex);
    error = cudaMalloc(&device_heightmap, sizeInBytes);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
     
#endif
    
    state = INITIALISED;
}

void computeNext(float time, int width, int height, vertex* watersurfacevertices, rgb* watersurfacecolors)
{
    if(state != INITIALISED)
    {
        return;
    }
    #if __GPUVERSION__
    cudaError_t error;
    
     // make dimension
    int x = (width + 16 - 1) / 16;
    int y = (height + 16 - 1) / 16;
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(x, y);
    
    //gitter 1 zeitschritt
    simulateWaveStep<<<blocksPerGrid, threadsPerBlock>>>(f++ ,device_grid, device_heightmap, device_watersurfacevertices, 
                     device_watersurfacecolors, width, height);
    
    error = cudaGetLastError();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    

    
    // copy back data
    cudaMemcpy(watersurfacevertices, device_watersurfacevertices, width*height*sizeof(vertex), cudaMemcpyDeviceToHost);
    cudaMemcpy(watersurfacecolors, device_watersurfacecolors, width*height*sizeof(rgb), cudaMemcpyDeviceToHost);
    #endif
}

void destroyWaterSurface()
{
    if(state != INITIALISED)
    {
        return;
    }

    state = UNINTIALISED;
}

















