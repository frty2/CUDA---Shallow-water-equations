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
        { return true; }
    std::cout << "Invalid value for --" << flagname << ": "
              << value << std::endl;
    return false;
}

static bool validateWSPF(const char* flagname, int value)
{
    if (value > 0)
        { return true; }
    std::cout << "Invalid value for --" << flagname << ": "
              << value << std::endl;
    return false;
}


DEFINE_int32(wspf, 50, "wavesteps per frame.");
DEFINE_double(timestep, 0.01f, "timestep between 2 wavesteps.");

static const bool timestep_dummy = google::RegisterFlagValidator(&FLAGS_timestep, &validateTimestep);
static const bool wspf_dummy = google::RegisterFlagValidator(&FLAGS_wspf, &validateWSPF);

float timestep;

const float GRAVITY = 9.83219f * 0.5f; //0.5f * Fallbeschleunigung

const float NN = 0.5f;

const int UNINTIALISED = 0;
const int INITIALISED = 1;
int stepsperframe = 50;

int f;

#if __GPUVERSION__
texture<gridpoint, 2, cudaReadModeElementType> texture_grid;
texture<char, 2, cudaReadModeElementType> texture_reflections;
texture<float, 2, cudaReadModeElementType> texture_treshholds;
texture<vertex, 2, cudaReadModeElementType> texture_landscape;

int grid_pitch_elements;
int treshholds_pitch_elements;
cudaChannelFormatDesc grid_channeldesc;
#endif

gridpoint* device_grid;
gridpoint* device_grid_next;

vertex* device_heightmap;
vertex* device_watersurfacevertices;
float* device_treshholds;

float* device_waves;
rgb* device_watersurfacecolors;

reflection* device_reflections;

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
__host__ __device__ gridpoint operator *(const gridpoint& x, const gridpoint& y)
{
    gridpoint z;
    z.x = y.x * x.x;
    z.y = y.y * x.y;
    z.z = y.z * x.z;
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
    v.y = (gp.x - NN) * 3 + NN;
    return v;
}

__host__ __device__ rgb gridpointToColor(gridpoint gp)
{
    rgb c;
    c.x = min(20 + (gp.x - NN) / (NN / 2) * 150.0f, 255);
    c.y = min(40 + (gp.x - NN) / (NN / 2) * 150.0f, 255);
    c.z = min(100 + (gp.x - NN) / (NN / 2) * 150.0f, 255);
    c.w = 235;
    return c;
}

__host__ __device__ gridpoint reflect(char r, gridpoint center, gridpoint point, int dir)
{
    gridpoint mult;
    mult.x = 1.0f;
    mult.y = 1 - 2 * dir;
    mult.z = -1 + 2 * dir;
    gridpoint result = r ? mult * center : point;
    return result;
}

#if __GPUVERSION__
__global__ void simulateWaveStep(gridpoint* grid_next, int width, int height, float timestep, int pitch)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < width && y < height)
#else
void simulateWaveStep(gridpoint* grid, gridpoint* grid_next, int width, int height, float timestep)
{
    int gridwidth = width + 2;
    int pitch = gridwidth;

    for(int y = 0; y < height; y++)
        for(int x = 0; x < width; x++)
#endif
    {
        int gridx = x + 1;
        int gridy = y + 1;

#if __GPUVERSION__
        gridpoint center = tex2D(texture_grid, gridx, gridy);
        char r = tex2D(texture_reflections, gridx, gridy);

        gridpoint north = tex2D(texture_grid, gridx, gridy - 1);
        north = reflect(tex2D(texture_reflections, gridx, gridy - 1), center, north, 0);

        gridpoint west = tex2D(texture_grid, gridx - 1, gridy);
        west = reflect(tex2D(texture_reflections, gridx - 1, gridy), center, west, 1);

        gridpoint south = tex2D(texture_grid, gridx, gridy + 1);
        south = reflect(tex2D(texture_reflections, gridx, gridy + 1), center, south, 0);

        gridpoint east = tex2D(texture_grid, gridx + 1, gridy);
        east = reflect(tex2D(texture_reflections, gridx + 1, gridy ), center, east, 1);

#else
        char r = 1;
        gridpoint center = grid2Dread(grid, gridx, gridy, gridwidth);
        gridpoint north = grid2Dread(grid, gridx, gridy - 1, gridwidth);
        gridpoint west = grid2Dread(grid, gridx - 1, gridy, gridwidth);
        gridpoint south = grid2Dread(grid, gridx, gridy + 1, gridwidth);
        gridpoint east = grid2Dread(grid, gridx + 1, gridy, gridwidth);
#endif

        gridpoint u_south = 0.5f * ( south + center ) - timestep * ( G(south) - G(center) );
        gridpoint u_north = 0.5f * ( north + center ) - timestep * ( G(center) - G(north) );
        gridpoint u_west = 0.5f * ( west + center ) - timestep * ( F(center) - F(west) );
        gridpoint u_east = 0.5f * ( east + center ) -  timestep * ( F(east) - F(center) );

        gridpoint u_center = center - timestep * ( F(u_east) - F(u_west) ) - timestep * ( G(u_south) - G(u_north) );

        grid2Dwrite(grid_next, gridx, gridy, pitch, (1-r)*u_center);

    }
}

#if __GPUVERSION__
__global__ void initReflectionGrid(reflection* reflections, int width, int height, int pitch)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int gridx = x + 1;
    int gridy = y + 1;

    if(x < width && y < height)
    {
        gridpoint gp = tex2D(texture_grid, gridx, gridy);
        int onshore = tex2D(texture_landscape, x, y).y >= gp.x;
        grid2Dwrite(reflections, gridx, gridy, pitch, onshore);
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

    for(int y = 0; y < height; y++)
        for(int x = 0; x < width; x++)
#endif
    {
        int gridx = x + 1;
        int gridy = y + 1;
        
#if __GPUVERSION__
        gridpoint gp = tex2D(texture_grid, gridx, gridy);
#else
        gridpoint gp = grid2Dread(grid, gridx, gridy, gridwidth);
#endif

        watersurfacevertices[y * width + x] = gridpointToVertex(gp, x / float(width - 1), y / float(height - 1));
        watersurfacecolors[y * width + x] = gridpointToColor(gp);
    }
}


#if __GPUVERSION__
__global__ void addWave(gridpoint* grid, float* wave, float norm, int width, int height, int pitch)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < width && y < height)
#else
void addWave(gridpoint* grid, float* wave, float norm, int width, int height)
{
    int pitch = width + 2;
    for(int y = 0; y < height; y++)
        for(int x = 0; x < width; x++)
#endif
    {
        int gridx = x + 1;
        int gridy = y + 1;

        float waveheight = grid2Dread(grid, gridx, gridy, pitch).x;

        waveheight += (grid2Dread(wave, x, y, width) - 1.5f) / norm;

#if __GPUVERSION__
        bool offshore = tex2D(texture_landscape, x, y).y < waveheight;
#else
        bool offshore = 1;
#endif
        grid[ offshore * (gridx + gridy * pitch) ].x = max(waveheight, 0.0001f);

    }
}

#if __GPUVERSION__
void addWave(float* wave, float norm, int width, int height, int pitch_elements)
{
    cudaError_t error;
    size_t sizeInBytes = width * height * sizeof(float);

    error = cudaMemcpy(device_waves, wave, sizeInBytes, cudaMemcpyHostToDevice);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);

    int x = (width + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
    int y = (height + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
    dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 blocksPerGrid(x, y);

    addWave <<< blocksPerGrid, threadsPerBlock>>>(device_grid_next, device_waves, norm, width, height, pitch_elements);

    error = cudaGetLastError();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);

    gridpoint *grid_helper = device_grid;
    device_grid = device_grid_next;
    device_grid_next = grid_helper;

    error = cudaBindTexture2D(0, &texture_grid, device_grid, &grid_channeldesc, width + 2, height + 2, grid_pitch_elements * sizeof(gridpoint));
    CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);
}
#else
void addWave(float* wave, float norm, int width, int height)
{
    size_t sizeInBytes = width * height * sizeof(float);
    memcpy(device_waves, wave, sizeInBytes);
    addWave(device_grid_next, device_waves, norm, width, height);

    gridpoint *grid_helper = device_grid;
    device_grid = device_grid_next;
    device_grid_next = grid_helper;
}
#endif

#if __GPUVERSION__
__global__ void initGrid(gridpoint *grid, int gridwidth, int gridheight, int pitch)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < gridwidth && y < gridheight)
#else
void initGrid(gridpoint *grid, int gridwidth, int gridheight)
{
    int pitch = gridwidth;
    for(int y = 0; y < gridheight; y++)
        for(int x = 0; x < gridwidth; x++)
#endif
    {
        gridpoint gp;
        gp.x = NN;
        gp.y = 0.0f;
        gp.z = 0.0f;
        grid[y * pitch + x] = gp;
    }
}

void initWaterSurface(int width, int height, vertex *heightmapvertices, float *wave, float *treshholds)
{

    if(state != UNINTIALISED)
    {
        return;
    }
    stepsperframe = FLAGS_wspf;
    timestep = FLAGS_timestep;
    int gridwidth = width + 2;
    int gridheight = height + 2;

#if __GPUVERSION__
    size_t sizeInBytes;
    size_t grid_pitch, reflections_pitch, treshholds_pitch;
    cudaError_t error;

    grid_channeldesc = cudaCreateChannelDesc<float4>();
    cudaChannelFormatDesc treshholds_channeldesc = cudaCreateChannelDesc<float>();
    cudaChannelFormatDesc reflections_channeldesc = cudaCreateChannelDesc<int>();

    //alloc pitched memory for device_grid
    error = cudaMallocPitch(&device_grid, &grid_pitch, gridwidth * sizeof(gridpoint), gridheight);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    CHECK_NOTNULL(device_grid);

    size_t oldpitch = grid_pitch;

    //alloc pitched memoty for device_grid_next
    error = cudaMallocPitch(&device_grid_next, &grid_pitch, gridwidth * sizeof(gridpoint), gridheight);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    CHECK_NOTNULL(device_grid_next);

    CHECK_EQ(oldpitch, grid_pitch);

    grid_pitch_elements = grid_pitch / sizeof(gridpoint);

    //alloc pitched memory for reflection grid
    error = cudaMallocPitch(&device_reflections, &reflections_pitch, gridwidth * sizeof(reflection), gridheight);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    CHECK_NOTNULL(device_reflections);

    int reflections_pitch_elements = reflections_pitch / sizeof(reflection);

    //set all entries of reflection grid to 0
    error = cudaMemset2D(device_reflections, reflections_pitch_elements * sizeof(reflection), 0, gridwidth * sizeof(reflection), gridheight);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);

    //alloc pitched memory for treshhold values
    error = cudaMallocPitch(&device_treshholds, &treshholds_pitch, gridwidth * sizeof(float), gridheight);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    CHECK_NOTNULL(device_treshholds);

    //copy the threshhold values to the device
    /*
     * Bug, braucht eigenen Kernel
     */
    error = cudaMemcpy2D(device_treshholds, treshholds_pitch, treshholds, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);

    treshholds_pitch_elements = treshholds_pitch / sizeof(float);

    //alloc pitched memory for landscape data on device
    size_t heightmap_pitch;
    error = cudaMallocPitch(&device_heightmap, &heightmap_pitch, width * sizeof(vertex), height);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    CHECK_NOTNULL(device_heightmap);

    // copy landscape data to device
    error = cudaMemcpy2D(device_heightmap, heightmap_pitch, heightmapvertices, width * sizeof(vertex), width * sizeof(vertex), height, cudaMemcpyHostToDevice);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    
    // bind heightmap to texture_landscape
    cudaChannelFormatDesc heightmap_channeldesc = cudaCreateChannelDesc<float4>();
    error = cudaBindTexture2D(0, &texture_landscape, device_heightmap, &heightmap_channeldesc, width, height, heightmap_pitch);
    CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);
    
    // malloc memory for watersurface vertices
    sizeInBytes = width*height*sizeof(vertex);
    error = cudaMalloc(&device_watersurfacevertices, sizeInBytes);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);

    // malloc memory for watersurface colors
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

    int x1 = (gridwidth + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
    dim3 threadsPerBlock1(BLOCKSIZE_X, 1);
    dim3 blocksPerGrid1(x1, 1);

    int y1 = (gridheight + BLOCKSIZE_Y -  1) / BLOCKSIZE_Y;
    dim3 threadsPerBlock2(1, BLOCKSIZE_Y);
    dim3 blocksPerGrid2(1, y1);

    //init grid with initial values
    initGrid <<< blocksPerGrid, threadsPerBlock>>>(device_grid, gridwidth, gridheight, grid_pitch_elements);

    //init grid_next with initial values
    initGrid <<< blocksPerGrid, threadsPerBlock>>>(device_grid_next, gridwidth, gridheight, grid_pitch_elements);

    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);


    //bind the grid to texture_grid
    error = cudaBindTexture2D(0, &texture_grid, device_grid, &grid_channeldesc, gridwidth, gridheight, grid_pitch);
    CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);

    //add the initial wave to the grid
    addWave(wave, 6.0f, width, height, grid_pitch_elements);

    //init the reflection grid
    initReflectionGrid <<< blocksPerGrid, threadsPerBlock>>>(device_reflections, width, height, reflections_pitch_elements);

    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);

    //bind reflection grid to texture_reflections
    error = cudaBindTexture2D(0, &texture_reflections, device_reflections, &reflections_channeldesc, gridwidth, gridheight, reflections_pitch);
    CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);

    //bin the threshholds to texture_treshholds
    error = cudaBindTexture2D(0, &texture_treshholds, device_treshholds, &treshholds_channeldesc, gridwidth, gridheight, grid_pitch);
    CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);

#else
    size_t sizeInBytes;

    sizeInBytes = gridheight * gridwidth * sizeof(gridpoint);
    device_grid = (gridpoint *) malloc(sizeInBytes);
    CHECK_NOTNULL(device_grid);


    device_grid_next = (gridpoint *) malloc(sizeInBytes);
    CHECK_NOTNULL(device_grid_next);

    sizeInBytes = width * height * sizeof(vertex);
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

    initGrid (device_grid, gridwidth, gridheight);

    initGrid (device_grid_next, gridwidth, gridheight);

    addWave(wave, 6.0f, width, height);
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
    int gridwidth = width + 2;
    int gridheight = height + 2;

    cudaError_t error;
    // make dimension
    int x = (width + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
    int y = (height + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
    dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 blocksPerGrid(x, y);

    //gitter "stepsperframe" zeitschritt
    for(int x = 0; x < stepsperframe; x++)
    {
        simulateWaveStep <<< blocksPerGrid, threadsPerBlock>>>(device_grid_next, width, height, timestep, grid_pitch_elements);

        error = cudaThreadSynchronize();
        CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);

        gridpoint *grid_helper = device_grid;
        device_grid = device_grid_next;
        device_grid_next = grid_helper;

        error = cudaBindTexture2D(0, &texture_grid, device_grid, &grid_channeldesc, gridwidth, gridheight, grid_pitch_elements * sizeof(gridpoint));
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
        simulateWaveStep(device_grid, device_grid_next, width, height, timestep);

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
