#include "wavesimulator.h"

#include <glog/logging.h>

#include "types.h"

const float GRAVITY = 0.0001f;
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

__host__ __device__ gridpoint U(gridpoint gp)
{
    gridpoint U;
    U.x = gp.y;
    U.y = gp.x * gp.y;
    U.z = gp.z * gp.y;
    return U;
}

__host__ __device__ gridpoint F(gridpoint gp)
{
    gridpoint F;
    F.x = gp.x * gp.y;
    F.y = (gp.x * gp.x * gp.y) + (0.5f * GRAVITY * gp.y * gp.y);
    F.z = gp.x * gp.z * gp.y;
    return F;
}

__host__ __device__ gridpoint G(gridpoint gp)
{
    gridpoint G;
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
    v.y = gp.y;
    return v;
}

__host__ __device__ vertex vertexToColor(gridpoint gp)
{
    rgb c;
    c.x = 50+(gp.y-1.0f)*100;
    c.y = 100+(gp.y-1.0f)*100;
    c.z = 255;
    return c;
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
        // check if offshore
        bool offshore = device_heightmap[y*width + x].y < 1.0f;
        
        gridpoint center = device_grid[gridx+gridy*gridwidth];        
        gridpoint north = device_grid[gridx+(gridy-1)*gridwidth];
        gridpoint west = device_grid[gridx-1+gridy*gridwidth];
        gridpoint south = device_grid[gridx+(gridy+1)*gridwidth];
        gridpoint east = device_grid[gridx+1+gridy*gridwidth];
        
        
        gridpoint u_south = 0.5f*( U(south) + U(center) ) - dt/(2*dy) *( G(south)-G(center) );
        gridpoint u_north = 0.5f*( U(north) + U(center) ) - dt/(2*dy) *( G(north)-G(center) );
        gridpoint u_west = 0.5f*( U(west) + U(center) ) - dt/(2*dx) *( F(west)-F(center) );
        gridpoint u_east = 0.5f*( U(east) + U(center) ) - dt/(2*dx) *( F(east)-F(center) );
        
        gridpoint n_east = reverseU(u_east);      
        gridpoint n_south = reverseU(u_south);
        gridpoint n_north = reverseU(u_north);
        gridpoint n_west = reverseU(u_west);
        
        gridpoint u_center = U(center) -  dt/dx * ( F(n_east)-F(n_west) ) - dt/dy * ( G(n_south) - G(n_north) );
        
        gridpoint n_center = reverseU(u_center);
        
        /* 
         * write if point is offshore
         * onshore points are mapped to grid(0,0)
         */
        device_grid_next[offshore*(gridx+gridy*gridwidth)] = n_center;
          
        device_watersurfacevertices[x+y*width] = gridpointToVertex(n_center, x/float(width), y/float(height));
       
        device_watersurfacecolors[y * width + x] = vertexToColor(n_center);
	}
}

__global__ void initWaterSurface(gridpoint *device_grid, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(x < width && y < height)
    {
        gridpoint gp;
        gp.x = 0.0f;
        gp.y = 1.0f;
        gp.z = 0.0f;
        if(x > 50 && x < 100 && y <= 200 && y >= 90)
        {
            gp.y = 1.5f;
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
    cudaMemcpy(watersurfacevertices, device_watersurfacevertices, width*height*sizeof(vertex), cudaMemcpyDeviceToHost);
    cudaMemcpy(watersurfacecolors, device_watersurfacecolors, width*height*sizeof(rgb), cudaMemcpyDeviceToHost);
    
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
