#include "wavesimulator.h"

#include <iostream>
#include <glog/logging.h>

#include "types.h"
#include "stdlib.h"

#ifndef min
    #define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

const float GRAVITY = 9.83219f/2.0f; //0.5f * Fallbeschleunigung
const float dt = 0.01f;
const float dx = 1.0f;
const float dy = 1.0f;
const float NN = 0.2f;

const int UNINTIALISED = 0;
const int INITIALISED = 1;
const int stepsperframe = 10;

gridpoint* device_grid;
gridpoint* device_grid_next;
vertex* device_heightmap;
vertex* device_watersurfacevertices;
rgb* device_watersurfacecolors;

int state = UNINTIALISED;


__host__ __device__ gridpoint F(gridpoint u)
{
    gridpoint F;
    F.x = u.y;
    F.y = (u.y * u.y) / u.x + (GRAVITY * u.x * u.x);
    F.z = u.z * u.y / u.x;
    return F;
}

__host__ __device__ gridpoint G(gridpoint u)
{
    gridpoint G;
    G.x = u.z;
    G.y = u.z * u.y / u.x;
    G.z = (u.z * u.z) / u.x + (GRAVITY * u.x * u.x);
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

__host__ __device__ vertex gridpointToVertex(gridpoint gp, float x, float y)
{
    vertex v;
    v.x = x*16.0f-8.0f;
    v.z = y*16.0f-8.0f;
    v.y = (gp.x-NN)*10.0f+1.0f;
    return v;
}

__host__ __device__ rgb gridpointToColor(gridpoint gp)
{
    rgb c;
    c.x = min(50+(gp.x-NN)/0.08f*150.0f,255);
    c.y = min(70+(gp.x-NN)/0.08f*150.0f,255);
    c.z = min(100+(gp.x-NN)/0.08f*150.0f,255);
    c.w = 235;
    return c;
}

#if __GPUVERSION__
__global__ void simulateWaveStep(gridpoint* device_grid, gridpoint* device_grid_next, vertex* device_heightmap, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int gridx = x + 1;
    int gridy = y + 1;
    
    int gridwidth = width+2;
    if(x < width && y < height)
    {
        // is point offshore
                
        gridpoint center = device_grid[gridx+gridy*gridwidth];
        bool offshore = device_heightmap[y*width + x].y < center.x - NN + 1.0f;
		
		
        gridpoint north = device_grid[gridx+(gridy-1)*gridwidth];
        gridpoint west = device_grid[gridx-1+gridy*gridwidth];
        gridpoint south = device_grid[gridx+(gridy+1)*gridwidth];
        gridpoint east = device_grid[gridx+1+gridy*gridwidth];
        
        
        gridpoint u_south = 0.5f*( south + center ) - dt/(2*dy) *( G(south) - G(center) );
        gridpoint u_north = 0.5f*( north + center ) - dt/(2*dy) *( G(north) - G(center) );
        gridpoint u_west = 0.5f*( west + center ) - dt/(2*dx) *( F(west) -F(center) );
        gridpoint u_east = 0.5f*( east + center ) - dt/(2*dx) *( F(east) -F(center) );
        
        gridpoint u_center = center - dt/dx * ( F(u_east)-F(u_west) ) - dt/dy * ( G(u_south) - G(u_north) );
        
        
        /*
         * if point is onshore write in grid(0,0)
         */
        device_grid_next[offshore*(gridx+gridy*gridwidth)] = u_center;
          
	}
}

__global__ void visualise(	gridpoint* device_grid, vertex* device_watersurfacevertices, 
							rgb* device_watersurfacecolors, int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int gridx = x + 1;
    int gridy = y + 1;
    
    int gridwidth = width+2;
	if(x < width && y < height)
    {
		device_watersurfacevertices[y * width + x] = gridpointToVertex(device_grid[(gridx+gridy*gridwidth)], 
		x / float(width), y / float(height));
		device_watersurfacecolors[y * width + x] = gridpointToColor(device_grid[(gridx+gridy*gridwidth)]);
	}
}

__global__ void initWaterSurface(gridpoint *device_grid, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(x < width && y < height)
    {
        gridpoint gp;
        gp.x = NN;
        gp.y = 0.0f;
        gp.z = 0.0f;
        if(x < 300 && x > 200 && y > 100 && y < 200)
        {
            gp.x = NN+(NN/20)*cos(0.031415f*(x-50))+(NN/30)*sin(0.031415f*(y-25));
            gp.y = 0;
            gp.z = 0;
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
    //gitter "stepsperframe" zeitschritt
    for(int x=0; x < stepsperframe; x++)
	{
		simulateWaveStep<<<blocksPerGrid, threadsPerBlock>>>(device_grid, device_grid_next, device_heightmap, width, height);
		error = cudaGetLastError();
		CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
		error = cudaThreadSynchronize();
		CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
		
		gridpoint *grid_helper = device_grid;
		device_grid = device_grid_next;
		device_grid_next = grid_helper;
	}
	visualise<<<blocksPerGrid, threadsPerBlock>>>(device_grid, device_watersurfacevertices, device_watersurfacecolors, width, height);
    
    error = cudaGetLastError();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
    // copy back data
    error = cudaMemcpy(watersurfacevertices, device_watersurfacevertices, width*height*sizeof(vertex), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    error = cudaMemcpy(watersurfacecolors, device_watersurfacecolors, width*height*sizeof(rgb), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
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
