#include "wavesimulator.h"

#include <iostream>
#include <glog/logging.h>
#include <gflags/gflags.h>

#include "types.h"
#include "stdlib.h"

#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 16

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

static bool validateTimestep(const char* flagname, double value)
{
    if (value > 0 && value < 1.0f)
        return true;
    printf("Invalid value for --%s: %f\n", flagname, (double)value);
    return false;
}

DEFINE_bool(smooth, false, "smoothing disabled / enabled");
DEFINE_double(timestep, 0.002f, "timestep");

static const bool timestep_dummy = google::RegisterFlagValidator(&FLAGS_timestep, &validateTimestep);

float timestep;

const float GRAVITY = 9.83219f / 2.0f; //0.5f * Fallbeschleunigung

const float NN = 0.2f;

const int UNINTIALISED = 0;
const int INITIALISED = 1;
const int stepsperframe = 20;

int f;

gridpoint* device_grid;
gridpoint* device_grid_next;

vertex* device_heightmap;
vertex* device_watersurfacevertices;

float* device_waves;
rgb* device_watersurfacecolors;

int state = UNINTIALISED;

__host__ __device__ gridpoint F(gridpoint u)
{
    float uyx = u.y / u.x;

    gridpoint F;
    F.x = u.y;
    F.y = u.y * uyx + GRAVITY * (u.x * u.x);
    F.z = u.z * uyx;
    return F;
}

__host__ __device__ gridpoint G(gridpoint u)
{
    float uzx = u.z / u.x;
    gridpoint G;
    G.x = u.z;
    G.y = u.y * uzx;
    G.z = u.z * uzx + GRAVITY * (u.x * u.x);
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
    v.x = x * 16.0f - 8.0f;
    v.z = y * 16.0f - 8.0f;
    v.y = (gp.x - NN) * 10.0f + 1.0f;
    return v;
}

__host__ __device__ rgb gridpointToColor(gridpoint gp)
{
    rgb c;
    c.x = min(20 + (gp.x - NN) / (NN/2) * 150.0f, 255);
    c.y = min(40 + (gp.x - NN) / (NN/2) * 150.0f, 255);
    c.z = min(100 + (gp.x - NN) / (NN/2) * 150.0f, 255);
    c.w = 235;
    return c;
}

#if __GPUVERSION__
__global__ void simulateWaveStep(gridpoint* device_grid, gridpoint* device_grid_next, vertex* device_heightmap, int width, int height, float timestep)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    int gridwidth = width + 2;
    
    if(x < width && y < height)
#else
void simulateWaveStep(gridpoint* device_grid, gridpoint* device_grid_next, vertex* device_heightmap, int width, int height, float timestep)
{
    int gridwidth = width + 2;
    
    for(int y = 0;y < height;y++)
        for(int x = 0;x < width;x++)
#endif
    {
        int gridx = x + 1;
        int gridy = y + 1;

        int gridposition = gridx + gridy * gridwidth;
        
        gridpoint center = device_grid[gridposition];

        // is point offshore
        bool offshore = device_heightmap[y * width + x].y < center.x - NN + 1.0f;

        gridpoint north = device_grid[gridposition - gridwidth];
        gridpoint west = device_grid[gridposition - 1];
        gridpoint south = device_grid[gridposition + gridwidth];
        gridpoint east = device_grid[gridposition + 1];



        gridpoint u_south = 0.5f * ( south + center ) - 0.5f * timestep * ( G(south) - G(center) );
        gridpoint u_north = 0.5f * ( north + center ) - 0.5f * timestep * ( G(north) - G(center) );
        gridpoint u_west = 0.5f * ( west + center ) - 0.5f * timestep * ( F(west) - F(center) );
        gridpoint u_east = 0.5f * ( east + center ) - 0.5f * timestep * ( F(east) - F(center) );

        gridpoint u_center = center - timestep * ( F(u_east) - F(u_west) ) -  timestep * ( G(u_south) - G(u_north) );
        
        /*
         * if point is onshore write in grid(0,0)
         */
        device_grid_next[offshore * (gridposition)] = u_center;
    }
}

#if __GPUVERSION__
__global__ void smooth(gridpoint* device_grid, int width, int height, int n)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    int gridwidth = width + 2;
    
    if(x < width && y < height)
#else
void smooth(gridpoint* device_grid, int width, int height, int n)
{
    int gridwidth = width + 2;
    
    for(int y = 0;y < height;y++)
        for(int x = 0;x < width;x++)
#endif
    {
        int gridx = x + 1;
        int gridy = y + 1;

        int gridposition = gridx + gridy * gridwidth;
        
        gridpoint center = device_grid[gridposition];

        gridpoint north = device_grid[gridposition - gridwidth];
        gridpoint west = device_grid[gridposition - 1];
        gridpoint south = device_grid[gridposition + gridwidth];
        gridpoint east = device_grid[gridposition + 1];
        
        center = (n * center + south + north + east + west) * (1.0f/(n+4));
        
        device_grid[gridposition] = center;
    }
}


#if __GPUVERSION__
__global__ void visualise(  gridpoint* device_grid, vertex* device_watersurfacevertices,
                            rgb* device_watersurfacecolors, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int gridwidth = width + 2;
    if(x < width && y < height)
#else
void visualise(  gridpoint* device_grid, vertex* device_watersurfacevertices,
                                    rgb* device_watersurfacecolors, int width, int height)
{
    int gridwidth = width + 2;
    
    for(int y = 0;y < height;y++)
        for(int x = 0;x < width;x++)
#endif
    {
        int gridx = x + 1;
        int gridy = y + 1;
        device_watersurfacevertices[y * width + x] = gridpointToVertex(device_grid[(gridx + gridy * gridwidth)],
                x / float(width), y / float(height));
        device_watersurfacecolors[y * width + x] = gridpointToColor(device_grid[(gridx + gridy * gridwidth)]);
    }
}


#if __GPUVERSION__
__global__ void addWave(gridpoint* device_grid, float* device_wave, vertex* device_heightmap, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < width && y < height)
#else
void addWave(gridpoint* device_grid, float* device_wave, vertex* device_heightmap, int width, int height)
{
    for(int y = 0;y < height;y++)
        for(int x = 0;x < width;x++)
#endif
    {
        int gridx = x + 1;
        int gridy = y + 1;
        
        float waveheight = device_grid[gridx+gridy*(width+2)].x;
        
        bool offshore = device_heightmap[y * width + x].y < waveheight - NN + 1.0f;
        
        waveheight += device_wave[x+y*width];
        
        device_grid[ offshore*(gridx+gridy*(width+2)) ].x = min(max(waveheight, 0.0001f), 2*NN);
    }
}

#if __GPUVERSION__
__global__ void initWaterSurface(gridpoint *device_grid, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < width && y < height)
#else
void initWaterSurface(gridpoint *device_grid, int width, int height)
{
    for(int y = 0;y < height;y++)
        for(int x = 0;x < width;x++)
#endif
    {
        gridpoint gp;
        gp.x = NN;
        gp.y = 0.0f;
        gp.z = 0.0f;
        device_grid[x + y * width] = gp;
    }
}


float* generateWave(int width, int height, float centerx, float centery, float r, float maxheight)
{
    size_t sizeInBytes = width*height*sizeof(float);
    float* wave = (float *) malloc(sizeInBytes);
    CHECK_NOTNULL(wave);
    
    float radius = width * r;
    
    for(int y = 0;y < height;y++)
    {
        for(int x = 0;x < width;x++)
        {
            float rx = x - (width*centerx);
            float ry = y - (height*centery);
            float rr = radius*radius-rx*rx-ry*ry;
            if(rr > 0)
            {
                wave[y*width+x] = rr/(radius*radius)*maxheight;
            }
            else{
                wave[y*width+x] = 0.0f;
            }
        }
    }
    return wave;
}

void addWave(float* wave, int width, int height)
{
    #if __GPUVERSION__
    cudaError_t error;
    size_t sizeInBytes = width*height*sizeof(float);
    
    error = cudaMemcpy(device_waves, wave, sizeInBytes, cudaMemcpyHostToDevice);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
    int x = (width + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
    int y = (height + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
    dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 blocksPerGrid(x, y);
    
    addWave<<< blocksPerGrid, threadsPerBlock>>>(device_grid, device_waves, device_heightmap, width, height);
    
    error = cudaGetLastError();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    #else
    size_t sizeInBytes = width*height*sizeof(float);
    memcpy(device_waves, wave, sizeInBytes);
    addWave(device_grid, device_waves, device_heightmap, width, height);
    #endif
}

void initWaterSurface(int width, int height, vertex* heightmapvertices)
{

    if(state != UNINTIALISED)
    {
        return;
    }
    timestep = FLAGS_timestep;
#if __GPUVERSION__
    size_t sizeInBytes;
    cudaError_t error;

    // malloc memory for device_grid
    sizeInBytes = (height + 2) * (width + 2) * sizeof(gridpoint);
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
    
    // malloc memory for waves
    sizeInBytes = height * width * sizeof(float);
    error = cudaMalloc(&device_waves, sizeInBytes);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);

    // make dimension
    int x = (width + BLOCKSIZE_X + 1) / BLOCKSIZE_X;
    int y = (height + BLOCKSIZE_Y + 1) / BLOCKSIZE_Y;
    dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 blocksPerGrid(x, y);

    initWaterSurface <<< blocksPerGrid, threadsPerBlock>>>(device_grid, width + 2, height + 2);
    error = cudaGetLastError();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);

    initWaterSurface <<< blocksPerGrid, threadsPerBlock>>>(device_grid_next, width + 2, height + 2);
    error = cudaGetLastError();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
    float* wave  = generateWave(width, height, 0.3f, 0.8f, 0.1f, NN/2);
    addWave(wave, width, height);
    free(wave);
#else
    size_t sizeInBytes;

    sizeInBytes = (height + 2) * (width + 2) * sizeof(gridpoint);
    device_grid = (gridpoint *) malloc(sizeInBytes);
    CHECK_NOTNULL(device_grid);


    device_grid_next = (gridpoint *) malloc(sizeInBytes);
    CHECK_NOTNULL(device_grid_next);

    sizeInBytes = width*height*sizeof(vertex);
    device_heightmap = (vertex *) malloc(sizeInBytes);
    CHECK_NOTNULL(device_heightmap);

    memcpy(device_heightmap, heightmapvertices, sizeInBytes);

    device_watersurfacevertices = (vertex *) malloc(sizeInBytes);
    CHECK_NOTNULL(device_watersurfacevertices);

    sizeInBytes = height * width * sizeof(rgb);
    device_watersurfacecolors = (rgb *) malloc(sizeInBytes);
    CHECK_NOTNULL(device_watersurfacecolors);
    
    sizeInBytes = height * width * sizeof(float);
    device_waves = (float *) malloc(sizeInBytes);
    CHECK_NOTNULL(device_waves);

    initWaterSurface (device_grid, width + 2, height + 2);

    initWaterSurface (device_grid_next, width + 2, height + 2);
    
    float* wave  = generateWave(width, height, 0.3f, 0.8f, 0.1f, NN/2);
    addWave(wave, width, height);
    free(wave);
#endif

    state = INITIALISED;
}

void computeNext(int width, int height, vertex* watersurfacevertices, rgb* watersurfacecolors)
{
    if(state != INITIALISED)
    {
        return;
    }

    f++;
    /*
    if(f % 1 == 0){
        float cx = (float) rand()/RAND_MAX;
        float cy = (float) rand()/RAND_MAX;
        float r = (float) rand()/RAND_MAX*0.01f;
        float h = (float) rand()/RAND_MAX*NN/10;
        float* wave  = generateWave(width, height, cx, cy, r, h);
        addWave(wave, width, height);
        free(wave);
    }*/
#if __GPUVERSION__    
    cudaError_t error;
    // make dimension
    int x = (width + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
    int y = (height + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
    dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 blocksPerGrid(x, y);
    
    //gitter "stepsperframe" zeitschritt
    for(int x = 0; x < stepsperframe; x++)
    {
        simulateWaveStep <<< blocksPerGrid, threadsPerBlock>>>(device_grid, device_grid_next, device_heightmap, width, height, timestep);
        error = cudaGetLastError();
        CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
        error = cudaThreadSynchronize();
        CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);

        gridpoint *grid_helper = device_grid;
        device_grid = device_grid_next;
        device_grid_next = grid_helper;
    }
    
    if(FLAGS_smooth)
    {
        smooth <<< blocksPerGrid, threadsPerBlock>>>(device_grid, width, height, 20000/stepsperframe);
        
        error = cudaGetLastError();
        CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
        error = cudaThreadSynchronize();
        CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    }
    
    visualise <<< blocksPerGrid, threadsPerBlock>>>(device_grid, device_watersurfacevertices, device_watersurfacecolors, width, height);

    error = cudaGetLastError();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);

    // copy back data
    error = cudaMemcpy(watersurfacevertices, device_watersurfacevertices, width * height * sizeof(vertex), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    error = cudaMemcpy(watersurfacecolors, device_watersurfacecolors, width * height * sizeof(rgb), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
#else
    for(int x = 0; x < stepsperframe; x++)
    {
        simulateWaveStep(device_grid, device_grid_next, device_heightmap, width, height, timestep);

        gridpoint *grid_helper = device_grid;
        device_grid = device_grid_next;
        device_grid_next = grid_helper;
    }
    
    if(FLAGS_smooth)
    {
        smooth(device_grid, width, height, 20000/stepsperframe);
    }
    visualise(device_grid, device_watersurfacevertices, device_watersurfacecolors, width, height);


    // copy back data
    memcpy(watersurfacevertices, device_watersurfacevertices, width * height * sizeof(vertex));
    memcpy(watersurfacecolors, device_watersurfacecolors, width * height * sizeof(rgb));
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
