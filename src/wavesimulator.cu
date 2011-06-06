#include "wavesimulator.h"

#include "glog/logging.h"

const int UNINTIALISED = 0;
const int INITIALISED = 1;

int state = UNINTIALISED;

__device__ float3 U(float h, velocity v)
{
    float3 U;
    U.x = h;
    U.y = v.x * h;
    U.z = v.y * h;
    return U;
}

__device__ float3 F(float h, velocity v, float gravity)
{
    float3 F;
    F.x = v.x * h;
    F.y = (v.x * v.x * h) + ((1 / 2) * gravity * h * h);
    F.z = v.x * v.y * h;
    return F;
}

__device__ float3 G(float h, velocity v, float gravity)
{
    float3 G;
    G.x = v.y * h;
    G.y = v.x * v.y * h;
    G.z = (v.y * v.y * h) + ((1 / 2) * gravity * h * h);
    return G;
}

void initWaterSurface(int width, int height, float *&heights, velocity *velocities)
{
    if(state != UNINTIALISED)
    {
        return;
    }

    state = INITIALISED;
}

void computeNext(float time)
{
    if(state != INITIALISED)
    {
        return;
    }
}

void destroyWaterSurface()
{
    if(state != INITIALISED)
    {
        return;
    }

    state = UNINTIALISED;
}

__global__ void simulateWave(rgb* image, vertex* vert, velocity* velo)
{
    // TODO
}
#if __GPUVERSION__
void startWaveSimulator(int width, int height, vertex* vert, velocity* velo, rgb* image)
{
    // make device pointer
    rgb* imageDevPointer;
    vertex* vertexDevPointer;
    velocity* velocityDevPointer;

    // copy vertex data to device
    size_t sizeInBytes = height * width * sizeof(vertex);
    cudaError_t error = cudaMalloc(&vertexDevPointer, sizeInBytes);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);

    error = cudaMemcpy(vertexDevPointer, vert, sizeInBytes, cudaMemcpyHostToDevice);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    CHECK_NOTNULL(vertexDevPointer);

    // copy velocity data to device
    error = cudaMalloc(&velocityDevPointer, height * width * sizeof(velocity));
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);

    error = cudaMemcpy(velocityDevPointer, velo, sizeInBytes, cudaMemcpyHostToDevice);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    CHECK_NOTNULL(velocityDevPointer);

    // malloc space for image
    sizeInBytes = height * width * sizeof(rgb);

    error = cudaMalloc(&imageDevPointer, sizeInBytes);
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    CHECK_NOTNULL(imageDevPointer);

    // make dimension
    int x = (width + 16 - 1) / 16;
    int y = (height + 16 - 1) / 16;
    dim3 dimBlock(16, 16);
    dim3 dimGrid(x, y);

    // start kernel
    simulateWave <<< dimGrid, dimBlock>>>(imageDevPointer, vertexDevPointer, velocityDevPointer);
    error = cudaGetLastError();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);

    // copy image data back
    cudaMemcpy(image, imageDevPointer, width*height, cudaMemcpyDeviceToHost);

    // free Data
    free(imageDevPointer);
}
#endif















