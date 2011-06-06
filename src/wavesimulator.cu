#include "wavesimulator.h"

#include <glog/logging.h>

#include "types.h"

const float GRAVITY = 9.8f;
const float dt = 0.01f;
const float dx = 1.0f;
const float dy = 1.0f;

const int UNINTIALISED = 0;
const int INITIALISED = 1;

gridpoint* device_grid;
gridpoint* device_grid_next;
vertex* device_heightmap;
vertex* device_watersurfacevertices;
rgb* device_watersurfacecolors;

int f;


int state = UNINTIALISED;

__host__ __device__ float3 U(gridpoint gp)
{
    float3 U;
    U.x = gp.y;
    U.y = gp.x * gp.y;
    U.z = gp.z * gp.y;
    return U;
}

__host__ __device__ float3 F(gridpoint gp)
{
    float3 F;
    F.x = gp.x * gp.y;
    F.y = (gp.x * gp.x * gp.y) + (0.5f * GRAVITY * gp.y * gp.y);
    F.z = gp.x * gp.z * gp.y;
    return F;
}

__host__ __device__ float3 G(gridpoint gp)
{
    float3 G;
    G.x = gp.z * gp.y;
    G.y = gp.x * gp.z * gp.y;
    G.z = (gp.z * gp.z * gp.y) + (0.5f * GRAVITY * gp.y * gp.y);
    return G;
}

__host__ __device__ gridpoint reverseU(gridpoint point)
{
    gridpoint n_point;
    n_point.x = point.y/point.x;
    n_point.y = point.x;
    n_point.z = point.z/point.x;
    return n_point;
} 



#if __GPUVERSION__
__global__ void simulateWaveStep(int frame, gridpoint* device_grid, gridpoint* device_grid_next, vertex* device_heightmap, 
                            vertex* device_watersurfacevertices, rgb* device_watersurfacecolors, 
                            int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int gridx = x + 1;
    int gridy = y + 1;
    
    if(x < width && y < height)
    {
        gridpoint north = device_grid[gridx+(gridy-1)*width];
        gridpoint west = device_grid[gridx-1+(gridy)*width];
        gridpoint south = device_grid[gridx+(gridy+ 1)*width];
        gridpoint east = device_grid[gridx+1+(gridy)*width];
        gridpoint center = device_grid[gridx+(gridy)*width];
        
        gridpoint u_south = 0.5f*(U(south) + U(center) - 0.5f*dt/dy*( G(south)-G(center) );
        gridpoint u_west = 0.5f*(U(west) + U(center) - 0.5f*dt/dx*( F(west)-F(center) );
        gridpoint u_north = 0.5f*(U(north) + U(center) - 0.5f*dt/dy*( G(north)-G(center) );
        gridpoint u_east = 0.5f*(U(east) + U(center) - 0.5f*dt/dx*( F(east)-F(center) );
        
        gridpoint n_east = reverseU(u_east);      
        gridpoint n_south = reverseU(u_south);
        gridpoint n_north = reverseU(u_north);
        gridpoint n_west = reverseU(u_west);
        
        gridpoint u_new_point = center - dt/dx*(F(n_east)-F(n_west)) - dt/dy * (G(n_south) - G(n_north))
        gridpoint new_point = reverseU(u_new_point);
        
        device_grid_next[gridx+(gridy)*width] = new_point;
        
        vertex new_vertex;
        new_vertex.x = x/float(width)*16-8;
        new_vertex.z = y/float(height)*16-8;
        new_vertex.y = new_point.y;
        device_watersurfacevertices[x+y*width] = new_vertex;
       
        rgb c;
        c.x = 100 + 50 * (v.y - 0.9) * 20;
        c.y = 150 + 50 * (v.y - 0.9) * 20;
        c.z = 255;
        device_watersurfacecolors[y * width + x] = c;
	}
}

__global__ void initWaterSurface(vertex* device_watersurfacevertices, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(x < width && y < height)
    {
        vertex new_vertex;
        new_vertex.x = 0.0f;
        new_vertex.y = 1.0f;
        new_vertex.z = 0.0f;
        device_watersurfacevertices[x+y*width] = new_vertex;
        
        if(x > 120 && x < 130 && y > 120 && y < 130)
        {
            new_vertex.y = 2.0f;
        }
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
    
    // malloc memory for device_grid_next
    error = cudaMalloc(&device_grid_next, sizeInBytes);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
    error = cudaMemset(device_grid, 0 , sizeInBytes);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
    // copy heightmapdata data to device
    sizeInBytes = height * width * sizeof(vertex);
    error = cudaMalloc(&device_heightmap, sizeInBytes);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
    // malloc memory for device_watersurfacevertices
    error = cudaMalloc(&device_watersurfacevertices, sizeInBytes);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
    // malloc memory for watersurfacecolors
    sizeInBytes = height * width * sizeof(rgb);
    error = cudaMalloc(&device_watersurfacecolors, sizeInBytes);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
     // make dimension
    int x = (width + 16 - 1) / 16;
    int y = (height + 16 - 1) / 16;
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(x, y);
    
    initWaterSurface<<<blocksPerGrid, threadsPerBlock>>>(device_watersurfacevertices, width, height);
    error = cudaGetLastError();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    error = cudaThreadSynchronize();
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
    simulateWaveStep<<<blocksPerGrid, threadsPerBlock>>>(f++ ,device_grid, device_grid_next, device_heightmap, device_watersurfacevertices, 
                     device_watersurfacecolors, width, height);
    
    error = cudaGetLastError();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    

    
    // copy back data
    cudaMemcpy(watersurfacevertices, device_watersurfacevertices, width*height*sizeof(vertex), cudaMemcpyDeviceToHost);
    cudaMemcpy(watersurfacecolors, device_watersurfacecolors, width*height*sizeof(rgb), cudaMemcpyDeviceToHost);
    
    grindpoint *grid_helper = device_grid;
    device_grid = device_grid_next;
    device_grid_next = grid_helper;
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

















