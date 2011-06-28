#include "wavesimulator.h"

#include <iostream>
#include <glog/logging.h>
#include <gflags/gflags.h>

#include "types.h"
#include "stdlib.h"

#define BLOCKSIZE_X 8
#define BLOCKSIZE_Y 8

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

const float NN = 1.2f;

const int UNINTIALISED = 0;
const int INITIALISED = 1;
int stepsperframe = 50;

int f;


texture<gridpoint, 2, cudaReadModeElementType> texture_grid;
texture<char, 2, cudaReadModeElementType> texture_reflections;
texture<float, 2, cudaReadModeElementType> texture_treshholds;
texture<vertex, 2, cudaReadModeElementType> texture_landscape;

int grid_pitch_elements;
int treshholds_pitch_elements;
int reflections_pitch_elements;
cudaChannelFormatDesc grid_channeldesc;

gridpoint* device_grid;
gridpoint* device_grid_next;

vertex* device_heightmap;
vertex* device_watersurfacevertices;
float* device_treshholds;

float* device_waves;
rgb* device_watersurfacecolors;

reflection* device_reflections;

int state = UNINTIALISED;

#define EPSILON 0.001f

__host__ __device__ gridpoint F(gridpoint u)
{
    float h4 = u.x*u.x*u.x*u.x;
    float v = u.x+u.y/(sqrtf(h4 + max(h4, EPSILON)));

    gridpoint F;
    F.x = u.y;
    F.y = u.y * v+ GRAVITY * u.x * u.x;
    F.z = u.z * v;
    F.w = 0;
    return F;
}

__host__ __device__ gridpoint G(gridpoint u)
{
    float h4 = u.x*u.x*u.x*u.x;
    float v = sqrtf(2)*u.x+u.z/(sqrtf(h4 + max(h4, EPSILON)));

    gridpoint G;
    G.x = u.z;
    G.y = u.y * v;
    G.z = u.z * v + GRAVITY * u.x * u.x;
    G.w = 0;
    return G;
}

__host__ __device__ gridpoint H(gridpoint c, gridpoint n, gridpoint e, gridpoint s, gridpoint w)
{
    gridpoint H;
    H.x = 0;
    H.y = -GRAVITY * c.x * (e.w-w.w);
    H.z = -GRAVITY * c.x * (s.w-n.w);
    H.w = 0;
    return H;
}

__host__ __device__ gridpoint operator +(const gridpoint& x, const gridpoint& y)
{
    gridpoint z;
    z.x = x.x + y.x;
    z.y = x.y + y.y;
    z.z = x.z + y.z;
    z.w = x.w + y.w;
    return z;
}
__host__ __device__ gridpoint operator -(const gridpoint& x, const gridpoint& y)
{
    gridpoint z;
    z.x = x.x - y.x;
    z.y = x.y - y.y;
    z.z = x.z - y.z;
    z.w = x.w - y.w;
    return z;
}
__host__ __device__ gridpoint operator *(const gridpoint& x, const gridpoint& y)
{
    gridpoint z;
    z.x = y.x * x.x;
    z.y = y.y * x.y;
    z.z = y.z * x.z;
    z.w = y.w * x.w;
    return z;
}
__host__ __device__ gridpoint operator *(const gridpoint& x, const float& c)
{
    gridpoint z;
    z.x = c * x.x;
    z.y = c * x.y;
    z.z = c * x.z;
    z.w = c * x.w;
    return z;
}
__host__ __device__ gridpoint operator *(const float& c, const gridpoint& x)
{
    return x * c;
}

__host__ __device__ gridpoint reflect(gridpoint& center, gridpoint point)
{
    float cor = 0.00001f;
    float diff = point.x-cor;
    if(diff <= 0)
    {
        //point.w = center.w+center.x;
        point.x = cor;
        center.x += diff;
        diff = center.x - cor;
        if(diff <= 0)
        {
            center.x = cor;
            point.w += diff;
        }
        //center.x = max(0, center.x+diff);
        //point.w = center.w + center.x;
    }
    return point;
}

__global__ void simulateWaveStep(gridpoint* grid_next, int width, int height, float timestep, int pitch)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < width && y < height)
    {
        int gridx = x + 1;
        int gridy = y + 1;
        gridpoint center = tex2D(texture_grid, gridx, gridy);

        center = reflect(center, center);
        
        gridpoint north = tex2D(texture_grid, gridx, gridy - 1);
        north = reflect(center, north);
        
        gridpoint west = tex2D(texture_grid, gridx - 1, gridy);
        west = reflect(center, west);
        
        gridpoint south = tex2D(texture_grid, gridx, gridy + 1);
        south = reflect(center, south);
        
        gridpoint east = tex2D(texture_grid, gridx + 1, gridy);
        east = reflect(center, east);
        
        gridpoint u_south = 0.5f * ( south + center ) - timestep * ( G(south) - G(center) );
        gridpoint u_north = 0.5f * ( north + center ) - timestep * ( G(center) - G(north) );
        gridpoint u_west = 0.5f * ( west + center ) - timestep * ( F(center) - F(west) );
        gridpoint u_east = 0.5f * ( east + center ) - timestep * ( F(east) - F(center) );
        
        gridpoint u_center = center + timestep * H(center, north, east, south, west) - timestep *( F(u_east) - F(u_west) ) - timestep * ( G(u_south) - G(u_north) );
        
        grid2Dwrite(grid_next, gridx, gridy, pitch, u_center);
    }
}

__global__ void initGrid(gridpoint *grid, int gridwidth, int gridheight, int pitch)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x < gridwidth && y < gridheight)
    {
        float a = tex2D(texture_landscape, x-1, y-1).y;
        
        gridpoint gp;
        gp.x = max(NN-a, 0.0f);
        gp.y = 0.0f;
        gp.z = 0.0f;
        gp.w = a;
        grid2Dwrite(grid, x, y, pitch, gp);
    }
}

__global__ void initReflectionGrid(reflection* reflections, int gridwidth, int gridheight, int pitch)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < gridwidth && y < gridheight)
    {
        gridpoint gp = tex2D(texture_grid, x, y);
        int onshore = gp.x <= 0.0f;
        grid2Dwrite(reflections, x, y, pitch, onshore);
    }
}

__host__ __device__ vertex gridpointToVertex(gridpoint gp, float x, float y)
{
    vertex v;
    v.x = x * 16.0f - 8.0f;
    v.z = y * 16.0f - 8.0f;
    v.y = gp.x+gp.w-0.0001f;
    return v;
}

__host__ __device__ rgb gridpointToColor(gridpoint gp)
{
    rgb c;
    c.x = min(20 + (gp.x+gp.w - NN) / (NN / 2) * 150.0f, 255);
    c.y = min(40 + (gp.x+gp.w - NN) / (NN / 2) * 150.0f, 255);
    c.z = min(100 + (gp.x+gp.w - NN) / (NN / 2) * 150.0f, 255);
    c.w = 235;
    return c;
}

__global__ void visualise(vertex* watersurfacevertices,
                          rgb* watersurfacecolors, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < width && y < height)
    {
        int gridx = x + 1;
        int gridy = y + 1;
        
        gridpoint gp = tex2D(texture_grid, gridx, gridy);

        watersurfacevertices[y * width + x] = gridpointToVertex(gp, x / float(width - 1), y / float(height - 1));
        watersurfacecolors[y * width + x] = gridpointToColor(gp);
    }
}


__global__ void addWave(gridpoint* grid, float* wave, float norm, int width, int height, int pitch)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < width && y < height)
    {
        int gridx = x + 1;
        int gridy = y + 1;

        float waveheight = grid2Dread(grid, gridx, gridy, pitch).x;

        waveheight += (grid2Dread(wave, x, y, width) - 1.5f) / norm;

        bool offshore = tex2D(texture_grid, x, y).x > 0.0f;
        //if(offshore)
        grid[ gridx + gridy * pitch ].x = waveheight;

    }
}

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

    reflections_pitch_elements = reflections_pitch / sizeof(reflection);

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
    addWave(wave, 10.0f, width, height, grid_pitch_elements);

    //init the reflection grid
    initReflectionGrid <<< blocksPerGrid, threadsPerBlock>>>(device_reflections, gridwidth, gridheight, reflections_pitch_elements);

    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);

    //bind reflection grid to texture_reflections
    error = cudaBindTexture2D(0, &texture_reflections, device_reflections, &reflections_channeldesc, gridwidth, gridheight, reflections_pitch);
    CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);

    //bin the threshholds to texture_treshholds
    error = cudaBindTexture2D(0, &texture_treshholds, device_treshholds, &treshholds_channeldesc, gridwidth, gridheight, grid_pitch);
    CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);

    state = INITIALISED;
}

void computeNext(int width, int height, vertex* watersurfacevertices, rgb* watersurfacecolors)
{
    if(state != INITIALISED)
    {
        return;
    }

    f++;

    int gridwidth = width + 2;
    int gridheight = height + 2;

    cudaError_t error;
    // make dimension
    int x = (gridwidth + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
    int y = (gridheight + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
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
        
         //init the reflection grid
        initReflectionGrid <<< blocksPerGrid, threadsPerBlock>>>(device_reflections, gridwidth, gridheight, reflections_pitch_elements);
        
        error = cudaThreadSynchronize();
        CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    }
    visualise <<< blocksPerGrid, threadsPerBlock >>>(device_watersurfacevertices, device_watersurfacecolors, width, height);

    error = cudaGetLastError();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);

    // copy back data
    error = cudaMemcpy(watersurfacevertices, device_watersurfacevertices, width * height * sizeof(vertex), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    error = cudaMemcpy(watersurfacecolors, device_watersurfacecolors, width * height * sizeof(rgb), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
}

void destroyWaterSurface()
{
    if(state != INITIALISED)
    {
        return;
    }

    state = UNINTIALISED;
}
