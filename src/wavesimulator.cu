#include "wavesimulator.h"

#include <iostream>
#include <glog/logging.h>

#include "types.h"
#include "stdlib.h"

const float GRAVITY = 4.9f; //0.5f * Fallbeschleunigung
const float dt = 0.1f;
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


__host__ __device__ gridpoint F(gridpoint u)
{
    gridpoint F;
    F.x = u.y;
    F.y = (u.y * u.y / u.x) + (GRAVITY * u.x * u.x);
    F.z = u.z * u.y / u.x;
    return F;
}

__host__ __device__ gridpoint G(gridpoint u)
{
    gridpoint G;
    G.x = u.z;
    G.y = u.z * u.y / u.x;
    G.z = (u.z * u.z / u.x) + (GRAVITY * u.x * u.x);
    return G;
}

__host__ __device__ gridpoint operator +(const gridpoint& x, const gridpoint& y)
{
    gridpoint z;
    z.x = x.x + y.x;
    z.y = x.y + y.y;
    z.z = x.z + y.z;
    return z;
}
__host__ __device__ gridpoint operator -(const gridpoint& x, const gridpoint& y)
{
    gridpoint z;
    z.x = x.x - y.x;
    z.y = x.y - y.y;
    z.z = x.z - y.z;
    return z;
}
__host__ __device__ gridpoint operator *(const gridpoint& x, const float& c)
{
    gridpoint z;
    z.x = c * x.x;
    z.y = c * x.y;
    z.z = c * x.z;
    return z;
}
__host__ __device__ gridpoint operator *(const float& c, const gridpoint& x)
{
    return x * c;
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
    
    int gridwidth = width+2;
    if(x < width && y < height)
    {
        
        gridpoint center = device_grid[gridx+gridy*gridwidth];
        
        
        gridpoint north = device_grid[gridx+(gridy-1)*gridwidth];
        gridpoint west = device_grid[gridx-1+gridy*gridwidth];
        gridpoint south = device_grid[gridx+(gridy+1)*gridwidth];
        gridpoint east = device_grid[gridx+1+gridy*gridwidth];
        
        
        gridpoint u_south = 0.5f*( south + center ) - dt/(2*dy) *( G(south) - G(center) );
        gridpoint u_north = 0.5f*( north + center ) - dt/(2*dy) *( G(north) - G(center) );
        gridpoint u_west = 0.5f*( west + center ) - dt/(2*dx) *( F(west) -F(center) );
        gridpoint u_east = 0.5f*( east + center ) - dt/(2*dx) *( F(east) -F(center) );
        
        gridpoint u_center = center - dt/dx * ( F(u_east)-F(u_west) ) - dt/dy * ( G(u_south) - G(u_north) );

        int n = 100;
        u_center = (u_center + 1.0f/n * south + 1.0f/n * north + 1.0f/n * east + 1.0f/n * west) * (1.0f/(1.0f+4.0f/n));

        device_grid_next[gridx+gridy*gridwidth] = u_center;
        
        vertex v;
        v.x = x/float(width)*16.0f-8.0f;
        v.z = y/float(height)*16.0f-8.0f;
        v.y = (u_center.x-0.2f)*10.0f+1.0f;
        device_watersurfacevertices[x+y*width] = v;
       
        rgb c;
        c.x = (v.y-0.2f)/0.06f*80.0f;
        c.y = (v.y-0.2f)/0.06f*80.0f;
        c.z = 255;
        device_watersurfacecolors[y * width + x] = c;
	}
}

__global__ void initWaterSurface(gridpoint *device_grid, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(x < width && y < height)
    {
        gridpoint gp;
        gp.x = 0.2f;
        gp.y = 0.0f;
        gp.z = 0.0f;
        if(x < 300 && x > 200 && y > 200 && y < 300)
        {
            gp.x = 0.2f+0.05f*cos(0.031415f*(x-50))+0.03f*sin(0.031415f*(y-25));
            gp.y = 0.0f;
            gp.z = 0.0f;
        }
        device_grid[x+y*width] = gp;
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
    
    
    // copy heightmapdata data to device
    sizeInBytes = height * width * sizeof(vertex);
    error = cudaMalloc(&device_heightmap, sizeInBytes);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
    error = cudaMemcpy(device_heightmap, heightmapvertices, sizeInBytes, cudaMemcpyHostToDevice);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
    // malloc memory for device_watersurfacevertices
    error = cudaMalloc(&device_watersurfacevertices, sizeInBytes);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
    // malloc memory for watersurfacecolors
    sizeInBytes = height * width * sizeof(rgb);
    error = cudaMalloc(&device_watersurfacecolors, sizeInBytes);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
     // make dimension
    int x = (width + 18 - 1) / 16;
    int y = (height + 18 - 1) / 16;
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(x, y);
    
    initWaterSurface<<<blocksPerGrid, threadsPerBlock>>>(device_grid, width + 2, height + 2);
    error = cudaGetLastError();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
    initWaterSurface<<<blocksPerGrid, threadsPerBlock>>>(device_grid_next, width + 2, height + 2);
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
    error = cudaMemcpy(watersurfacevertices, device_watersurfacevertices, width*height*sizeof(vertex), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    error = cudaMemcpy(watersurfacecolors, device_watersurfacecolors, width*height*sizeof(rgb), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
    /*
    gridpoint *grid;
    grid = (gridpoint*) malloc((width+2)*(height+2)*sizeof(gridpoint));
    CHECK_NOTNULL(grid);
    
    error = cudaMemcpy(grid, device_grid_next, (width+2)*(height+2)*sizeof(gridpoint), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
    
    float sum = 0.0f;
    for(int y = 0;y < height;y++)
    {
        for(int x = 0;x < width;x++)
        {
            sum += grid[(y+1)*(width+2)+x+1].z;
        }
    }

    std::cout << sum << std::endl;
    */
    
    gridpoint *grid_helper = device_grid;
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
