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

#define grid2Dwrite(array, x, y, pitch, value) array[(y)*pitch+x] = value
#define grid2Dread(array, x, y, pitch) array[(y)*pitch+x]

static bool validateTimestep(const char* flagname, double value)
{
    if (value > 0 && value < 1.0f)
        return true;
    printf("Invalid value for --%s: %f\n", flagname, (double)value);
    return false;
}
DEFINE_double(timestep, 0.01f, "timestep");

static const bool timestep_dummy = google::RegisterFlagValidator(&FLAGS_timestep, &validateTimestep);

float timestep;

const float GRAVITY = 9.83219f * 0.5f; //0.5f * Fallbeschleunigung

const float NN = 0.2f;

const int UNINTIALISED = 0;
const int INITIALISED = 1;
const int stepsperframe = 50;

int f;

#if __GPUVERSION__
texture<gridpoint, 2, cudaReadModeElementType> texture_grid;
size_t pitch;
int pitch_elements;
cudaChannelFormatDesc channelDesc;
#endif

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
    F.y = u.y * uyx + GRAVITY * u.x * u.x;
    F.z = u.z * uyx;
    return F;
}

__host__ __device__ gridpoint G(gridpoint u)
{
    float uzx = u.z / u.x;
    
    gridpoint G;
    G.x = u.z;
    G.y = u.y * uzx;
    G.z = u.z * uzx + GRAVITY * u.x * u.x;
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
    v.y = (gp.x - NN)/NN * 2.0f + 1.0f;
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
__global__ void simulateWaveStep(gridpoint* grid_next, vertex* device_heightmap, int width, int height, float timestep, int pitch)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(x < width && y < height)
#else
void simulateWaveStep(gridpoint* grid, gridpoint* grid_next, vertex* device_heightmap, int width, int height, float timestep)
{
    int gridwidth = width + 2;
    int pitch = gridwidth;
    
    for(int y = 0;y < height;y++)
        for(int x = 0;x < width;x++)
#endif
    {
        int gridx = x + 1;
        int gridy = y + 1;
        
#if __GPUVERSION__
        gridpoint center = tex2D(texture_grid, gridx, gridy);
        gridpoint north = tex2D(texture_grid, gridx, gridy - 1);
        gridpoint west = tex2D(texture_grid, gridx - 1, gridy);
        gridpoint south = tex2D(texture_grid, gridx, gridy + 1);
        gridpoint east = tex2D(texture_grid, gridx + 1, gridy);
#else
        gridpoint center = grid2Dread(grid, gridx, gridy, gridwidth);
        gridpoint north = grid2Dread(grid, gridx, gridy-1, gridwidth);
        gridpoint west = grid2Dread(grid, gridx-1, gridy, gridwidth);
        gridpoint south = grid2Dread(grid, gridx, gridy+1, gridwidth);
        gridpoint east = grid2Dread(grid, gridx+1, gridy, gridwidth);
#endif

        gridpoint u_south = 0.5f * ( south + center ) - 0.5f * timestep * ( G(south) - G(center) );
        gridpoint u_north = 0.5f * ( north + center ) - 0.5f * timestep * ( G(center) - G(north) );
        gridpoint u_west = 0.5f * ( west + center ) - 0.5f * timestep * ( F(center) - F(west) );
        gridpoint u_east = 0.5f * ( east + center ) - 0.5f * timestep * ( F(east) - F(center) );
        
        gridpoint u_center = center - timestep * ( F(u_east) - F(u_west) ) - timestep * ( G(u_south) - G(u_north) );
        
        grid2Dwrite(grid_next, gridx, gridy, pitch, u_center);
    }
}

#if __GPUVERSION__
__global__ void setHorBorder(gridpoint* grid, int width, int height, int pitch)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int gridheight = height + 2;
    int gridwidth = width + 2;
    
    if(x < gridwidth)
    {
        gridpoint src = tex2D(texture_grid, x, 1);
        src.z = -src.z;
        grid2Dwrite(grid, x, 0, pitch, src);
        
        src = tex2D(texture_grid, x, gridheight-2);
        src.z = -src.z;
        grid2Dwrite(grid, x, gridheight-1, pitch, src);
    }
}

__global__ void setVertBorder(gridpoint* grid, int width, int height, int pitch)
{
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int gridheight = height + 2;
    int gridwidth = width + 2;
    if(y < gridheight)
    {
        gridpoint src = tex2D(texture_grid, 1, y);
        src.y = -src.y;
        grid2Dwrite(grid, 0, y, pitch, src);
        
        src = tex2D(texture_grid, gridwidth-2, y);
        src.y = -src.y;
        grid2Dwrite(grid, gridwidth-1, y, pitch, src);
    }
}
#endif

#if __GPUVERSION__
__global__ void visualise(vertex* watersurfacevertices,
                            rgb* watersurfacecolors, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(x < width && y < height)
#else
void visualise(  gridpoint* grid, vertex* watersurfacevertices,
                                    rgb* watersurfacecolors, int width, int height)
{
    int gridwidth = width + 2;
    
    for(int y = 0;y < height;y++)
        for(int x = 0;x < width;x++)
#endif
    {
        int gridx = x + 1;
        int gridy = y + 1;
        #if __GPUVERSION__
        gridpoint gp = tex2D(texture_grid, gridx, gridy);
        #else
        gridpoint gp = grid2Dread(grid, gridx, gridy, gridwidth);
        #endif
        watersurfacevertices[y * width + x] = gridpointToVertex(gp, x / float(width-1), y / float(height-1));
        watersurfacecolors[y * width + x] = gridpointToColor(gp);
    }
}


#if __GPUVERSION__
__global__ void addWave(gridpoint* grid, float* wave, vertex* heightmap, int width, int height, int pitch)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < width && y < height)
#else
void addWave(gridpoint* grid, float* wave, vertex* heightmap, int width, int height)
{
    int pitch = width + 2;
    for(int y = 0;y < height;y++)
        for(int x = 0;x < width;x++)
#endif
    {
        int gridx = x + 1;
        int gridy = y + 1;
        
        float waveheight = grid2Dread(grid, gridx, gridy, pitch).x;
        
        bool offshore = grid2Dread(heightmap, x, y, width).y < waveheight - NN + 1.0f;
        
        waveheight += grid2Dread(wave, x, y, width);
        
        grid[ offshore*(gridx+gridy*pitch) ].x = max(waveheight, 0.0001f);
    }
}

#if __GPUVERSION__
__global__ void initWaterSurface(gridpoint *grid, int gridwidth, int gridheight, int pitch)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < gridwidth && y < gridheight)
#else
void initWaterSurface(gridpoint *grid, int gridwidth, int gridheight)
{
    for(int y = 0;y < gridheight;y++)
        for(int x = 0;x < gridwidth;x++)
#endif
    {
        gridpoint gp;
        gp.x = NN;
        gp.y = 0.0f;
        gp.z = 0.0f;
        #if __GPUVERSION__
        grid[y * pitch + x] = gp;
        #else
        grid[y * gridwidth + x] = gp;
        #endif
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

#if __GPUVERSION__
void addWave(float* wave, int width, int height, int pitch_elements)
{
    cudaError_t error;
    size_t sizeInBytes = width*height*sizeof(float);
    
    error = cudaMemcpy(device_waves, wave, sizeInBytes, cudaMemcpyHostToDevice);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
    int x = (width + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
    int y = (height + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
    dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 blocksPerGrid(x, y);
    
    addWave<<< blocksPerGrid, threadsPerBlock>>>(device_grid_next, device_waves, device_heightmap, width, height, pitch_elements);
    
    error = cudaGetLastError();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
    gridpoint *grid_helper = device_grid;
    device_grid = device_grid_next;
    device_grid_next = grid_helper;
    
    error = cudaBindTexture2D(0, &texture_grid, device_grid, &channelDesc, width+2, height+2, pitch);
    CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);
}
#else
void addWave(float* wave, int width, int height)
{
    size_t sizeInBytes = width*height*sizeof(float);
    memcpy(device_waves, wave, sizeInBytes);
    addWave(device_grid_next, device_waves, device_heightmap, width, height);
    
    gridpoint *grid_helper = device_grid;
    device_grid = device_grid_next;
    device_grid_next = grid_helper;
}
#endif

void initWaterSurface(int width, int height, vertex* heightmapvertices)
{

    if(state != UNINTIALISED)
    {
        return;
    }
    timestep = FLAGS_timestep;
    int gridwidth = width + 2;
    int gridheight = height + 2;
    
#if __GPUVERSION__
    size_t sizeInBytes;
    cudaError_t error;
    channelDesc = cudaCreateChannelDesc<float4>();
    
    error = cudaMallocPitch(&device_grid, &pitch, gridwidth * sizeof(gridpoint), gridheight);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    CHECK_NOTNULL(device_grid);
    
    size_t oldpitch = pitch;
    
    error = cudaMallocPitch(&device_grid_next, &pitch, gridwidth * sizeof(gridpoint), gridheight);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    CHECK_NOTNULL(device_grid_next);
    
    CHECK_EQ(oldpitch, pitch);
    
    pitch_elements = pitch / sizeof(gridpoint);
    
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
    int x = (gridwidth + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
    int y = (gridheight + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
    dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 blocksPerGrid(x, y);

    initWaterSurface <<< blocksPerGrid, threadsPerBlock>>>(device_grid, gridwidth, gridheight, pitch_elements);
    error = cudaGetLastError();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);

    initWaterSurface <<< blocksPerGrid, threadsPerBlock>>>(device_grid_next, gridwidth, gridheight, pitch_elements);
    error = cudaGetLastError();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
    error = cudaBindTexture2D(0, &texture_grid, device_grid, &channelDesc, gridwidth, gridheight, pitch);
    CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);
    
    float* wave  = generateWave(width, height, 0.3f, 0.8f, 0.1f, NN);
    addWave(wave, width, height, pitch_elements);
    free(wave);
#else
    size_t sizeInBytes;

    sizeInBytes = gridheight * gridwidth * sizeof(gridpoint);
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

    initWaterSurface (device_grid, gridwidth, gridheight);

    initWaterSurface (device_grid_next, gridwidth, gridheight);
    
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
    
    
    
#if __GPUVERSION__   
    if(f < 100)
    {
        float cx = (float) rand()/RAND_MAX;
        float cy = (float) rand()/RAND_MAX;
        float r = (float) rand()/RAND_MAX*0.01f;
        float h = (float) rand()/RAND_MAX*NN;
        float* wave  = generateWave(width, height, cx, cy, r, h);
        addWave(wave, width, height, pitch_elements);
        free(wave);
    }
    int gridwidth = width + 2;
    int gridheight = height + 2; 
    
    cudaError_t error;
    // make dimension
    int x = (width + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
    int y = (height + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
    dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 blocksPerGrid(x, y);
    
    int x1 = (gridwidth + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
    dim3 threadsPerBlock1(BLOCKSIZE_X, 1);
    dim3 blocksPerGrid1(x1, 1);
    
    int y1 = (gridheight + BLOCKSIZE_Y -  1) / BLOCKSIZE_Y;
    dim3 threadsPerBlock2(1, BLOCKSIZE_Y);
    dim3 blocksPerGrid2(1, y1);
    
    //gitter "stepsperframe" zeitschritt
    for(int x = 0; x < stepsperframe; x++)
    {        
        simulateWaveStep <<< blocksPerGrid, threadsPerBlock>>>(device_grid_next, device_heightmap, width, height, timestep, pitch_elements);

        setHorBorder<<< blocksPerGrid1, threadsPerBlock1>>>(device_grid_next, width, height, pitch_elements);

        setVertBorder<<< blocksPerGrid2, threadsPerBlock2>>>(device_grid_next, width, height, pitch_elements);
        
        error = cudaThreadSynchronize();
        CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);

        gridpoint *grid_helper = device_grid;
        device_grid = device_grid_next;
        device_grid_next = grid_helper;
        
        error = cudaBindTexture2D(0, &texture_grid, device_grid, &channelDesc, gridwidth, gridheight, pitch);
        CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);
    }
    visualise <<< blocksPerGrid, threadsPerBlock>>>(device_watersurfacevertices, device_watersurfacecolors, width, height);

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
